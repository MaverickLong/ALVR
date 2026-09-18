use crate::interaction::{HandJointsPoll, InteractionContext};
use alvr_common::{info, parking_lot::RwLock, warn, DeviceMotion, Pose};
use alvr_system_info::Platform;
use std::time::{Duration, Instant};

// Hand tracking on the Focus Vision runs at 60Hz, polling faster only returns duplicate samples
const HAND_JOINTS_POLL_INTERVAL: Duration = Duration::from_micros(16_667);
// Both controllers must be in use for this long before the hand trackers are destroyed
// (hysteresis against brief controller pickups)
const HAND_TRACKERS_RELEASE_DELAY: Duration = Duration::from_secs(10);
// Controllers resting still for this long were probably put down: recreate the trackers now so
// that the runtime pipeline is warm by the time the controllers go idle and the hands take over
const CONTROLLERS_STILL_PREWARM_DELAY: Duration = Duration::from_millis(1500);
const CONTROLLER_STILL_POSITION_EPSILON_M: f32 = 0.02;
const CONTROLLER_STILL_ANGLE_EPSILON_RAD: f32 = 5.0 * std::f32::consts::PI / 180.0;
// Also rate-limits the warning emitted by create_ext_object on each failed attempt
const HAND_TRACKERS_CREATE_RETRY_INTERVAL: Duration = Duration::from_secs(5);

struct StillnessDetector {
    anchor: Pose,
    still_since: Instant,
}

impl StillnessDetector {
    fn new(now: Instant) -> Self {
        Self {
            anchor: Pose::default(),
            still_since: now,
        }
    }

    // Returns for how long the pose stayed within the epsilons of the anchor
    fn update(&mut self, pose: Pose, now: Instant) -> Duration {
        let moved = pose.position.distance(self.anchor.position)
            > CONTROLLER_STILL_POSITION_EPSILON_M
            || pose.orientation.angle_between(self.anchor.orientation)
                > CONTROLLER_STILL_ANGLE_EPSILON_RAD;
        if moved {
            self.anchor = pose;
            self.still_since = now;
        }

        now.saturating_duration_since(self.still_since)
    }
}

#[derive(Clone, Copy)]
enum TrackersState {
    // The XrHandTrackerEXT objects exist (runtime pipeline running). `controllers_busy_since` is
    // set while both controllers are in use.
    Created {
        controllers_busy_since: Option<Instant>,
    },
    // The objects were destroyed (runtime pipeline stopped)
    Released {
        next_create_attempt: Option<Instant>,
    },
    // The runtime refused to create a hand tracker while the session runs: the trackers created
    // at startup are kept for the whole session, only the poll rate limiter stays active
    Pinned,
}

pub struct FocusVisionGate<'a> {
    interaction_context: &'a RwLock<InteractionContext>,
    state: TrackersState,
    mid_session_creation_verified: bool,
    next_poll: Instant,
    cached_skeletons: [Option<[Pose; 26]>; 2],
    stillness: [StillnessDetector; 2],
}

impl FocusVisionGate<'_> {
    // Returns false if the runtime cannot create hand trackers while the session runs
    fn release_trackers(&mut self) -> bool {
        // Before destroying the trackers created at startup, verify that they can be recreated
        // later. The probe object is destroyed together with them.
        if !self.mid_session_creation_verified {
            if self.interaction_context.read().can_create_hand_tracker() {
                self.mid_session_creation_verified = true;
            } else {
                warn!("Hand trackers cannot be recreated mid-session, keeping them");

                return false;
            }
        }

        self.interaction_context.write().release_hand_trackers();
        self.cached_skeletons = [None, None];

        true
    }

    // Must be called with the InteractionContext read guard already dropped
    fn update(&mut self, controllers: [Option<DeviceMotion>; 2], now: Instant) {
        let hands_wanted = match controllers {
            [Some(left), Some(right)] => {
                let left_still =
                    self.stillness[0].update(left.pose, now) >= CONTROLLERS_STILL_PREWARM_DELAY;
                let right_still =
                    self.stillness[1].update(right.pose, now) >= CONTROLLERS_STILL_PREWARM_DELAY;

                left_still && right_still
            }
            // At least one controller is idle or off: the runtime tracks that hand
            _ => true,
        };

        match self.state {
            TrackersState::Created {
                controllers_busy_since,
            } => {
                if hands_wanted {
                    self.state = TrackersState::Created {
                        controllers_busy_since: None,
                    };
                } else {
                    let busy_since = controllers_busy_since.unwrap_or(now);

                    if now.saturating_duration_since(busy_since) >= HAND_TRACKERS_RELEASE_DELAY {
                        self.state = if self.release_trackers() {
                            info!("Hand trackers released: controllers in use");

                            TrackersState::Released {
                                next_create_attempt: None,
                            }
                        } else {
                            TrackersState::Pinned
                        };
                    } else {
                        self.state = TrackersState::Created {
                            controllers_busy_since: Some(busy_since),
                        };
                    }
                }
            }
            TrackersState::Released {
                next_create_attempt,
            } => {
                let attempt_due = next_create_attempt.is_none_or(|t| now >= t);

                if hands_wanted && attempt_due {
                    if self.interaction_context.write().ensure_hand_trackers() {
                        self.next_poll = now;
                        self.state = TrackersState::Created {
                            controllers_busy_since: None,
                        };
                        info!("Hand trackers created: hands wanted");
                    } else {
                        // create_ext_object already logged the failure
                        self.state = TrackersState::Released {
                            next_create_attempt: Some(now + HAND_TRACKERS_CREATE_RETRY_INTERVAL),
                        };
                    }
                }
            }
            TrackersState::Pinned => (),
        }
    }
}

impl Drop for FocusVisionGate<'_> {
    // The lobby expects the trackers to exist whenever the stream input thread is not running
    fn drop(&mut self) {
        if let TrackersState::Released { .. } = self.state {
            self.interaction_context.write().ensure_hand_trackers();
        }
    }
}

// Rate-limits the hand joints polling and destroys the hand tracker objects while both
// controllers are in use, because the HTC runtime runs its camera hand tracking pipeline for as
// long as the objects exist. Every platform except the Focus Vision gets the pass-through variant,
// which keeps the pre-existing behavior: joints located on every tick and trackers never touched.
pub enum HandTrackingGate<'a> {
    Passthrough,
    FocusVision(Box<FocusVisionGate<'a>>),
}

impl<'a> HandTrackingGate<'a> {
    pub fn new(platform: Platform, interaction_context: &'a RwLock<InteractionContext>) -> Self {
        if platform != Platform::FocusVision || !interaction_context.read().has_hand_trackers() {
            return Self::Passthrough;
        }

        let now = Instant::now();

        Self::FocusVision(Box::new(FocusVisionGate {
            interaction_context,
            state: TrackersState::Created {
                controllers_busy_since: None,
            },
            mid_session_creation_verified: false,
            next_poll: now,
            cached_skeletons: [None, None],
            stillness: [StillnessDetector::new(now), StillnessDetector::new(now)],
        }))
    }

    pub fn joints_poll(&mut self) -> HandJointsPoll {
        let gate = match self {
            Self::Passthrough => return HandJointsPoll::Locate,
            Self::FocusVision(gate) => gate,
        };

        if let TrackersState::Released { .. } = gate.state {
            return HandJointsPoll::Skip;
        }

        let now = Instant::now();
        if now >= gate.next_poll {
            gate.next_poll += HAND_JOINTS_POLL_INTERVAL;
            if gate.next_poll < now {
                // Fell behind after a stall: re-anchor instead of bursting polls
                gate.next_poll = now + HAND_JOINTS_POLL_INTERVAL;
            }

            HandJointsPoll::Locate
        } else {
            HandJointsPoll::Skip
        }
    }

    // Skeletons to send this tick: the freshly located ones when polled, the last located ones
    // otherwise. The server keys skeletons by packet timestamp, so every packet needs them.
    pub fn skeletons(
        &mut self,
        joints_poll: HandJointsPoll,
        located: [Option<[Pose; 26]>; 2],
    ) -> [Option<[Pose; 26]>; 2] {
        match (self, joints_poll) {
            (Self::Passthrough, _) => located,
            (Self::FocusVision(gate), HandJointsPoll::Locate) => {
                gate.cached_skeletons = located;

                located
            }
            (Self::FocusVision(gate), HandJointsPoll::Skip) => gate.cached_skeletons,
        }
    }

    // Must be called with the InteractionContext read guard already dropped: takes the write lock
    // on state transitions.
    pub fn update(&mut self, controllers: [Option<DeviceMotion>; 2]) {
        if let Self::FocusVision(gate) = self {
            gate.update(controllers, Instant::now());
        }
    }
}
