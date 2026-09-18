//! Streams the raw infrared eye tracking cameras of the VIVE Focus Vision to the streamer.
//!
//! The cameras are a single USB video class device (OmniVision OV580 stereo bridge) that outputs
//! both eyes side by side. The device is opened through the Android USB host API (which owns the
//! permission) and then driven directly with usbfs ioctls, so no JNI call happens per frame.
use alvr_client_core::ClientCoreContext;
use alvr_common::{error, info, warn, RelaxedAtomic};
use alvr_session::EyeCamerasConfig;
use image::{codecs::jpeg::JpegEncoder, ExtendedColorType, ImageEncoder};
use jni::{
    errors::Error as JniError,
    objects::{GlobalRef, JByteArray, JObject, JString, JValue},
    JNIEnv,
};
use std::{
    ffi::c_void,
    mem, ptr,
    sync::{
        mpsc::{self, Receiver, Sender, TrySendError},
        Arc,
    },
    thread::{self, JoinHandle},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const EYE_CAMERA_VENDOR_ID: i32 = 0x0bb4;
const EYE_CAMERA_PRODUCT_ID: i32 = 0x06a7;
const CAMERA_PERMISSION: &str = "android.permission.CAMERA";
const USB_PERMISSION_ACTION: &str = "alvr.USB_PERMISSION";
const RETRY_INTERVAL: Duration = Duration::from_secs(2);
const USB_TIMEOUT_MS: u32 = 1000;
const TRANSFER_POLL_TIMEOUT: Duration = Duration::from_millis(250);
// Consecutive transfer timeouts after which the camera is reopened
const MAX_TRANSFER_TIMEOUTS: u32 = 20;
// The camera has a small internal buffer and corrupts frames as soon as the host stops draining
// it, so enough bulk transfers are kept queued in the kernel to cover this much camera output
// while the reader thread is busy or not scheduled
const TRANSFER_QUEUE_DURATION: Duration = Duration::from_millis(50);
const MIN_QUEUED_TRANSFERS: usize = 8;
const MAX_QUEUED_TRANSFERS: usize = 64;

// The stream is declared as 400x401 YUY2 but it actually carries two 400x400 8-bit images side by
// side (right eye then left eye on each row) plus one row of metadata
const EYE_IMAGE_WIDTH: usize = 400;
const EYE_IMAGE_HEIGHT: usize = 400;
const FRAME_ROW_BYTES: usize = EYE_IMAGE_WIDTH * 2;
const FRAME_BYTES: usize = FRAME_ROW_BYTES * (EYE_IMAGE_HEIGHT + 1);

// USB video class
const UVC_SET_CUR: u8 = 0x01;
const UVC_GET_CUR: u8 = 0x81;
const UVC_VS_PROBE_CONTROL: u16 = 0x01;
const UVC_VS_COMMIT_CONTROL: u16 = 0x02;
const UVC_PAYLOAD_HEADER_FID: u8 = 1 << 0;
const UVC_PAYLOAD_HEADER_EOF: u8 = 1 << 1;
const UVC_PAYLOAD_HEADER_ERROR: u8 = 1 << 6;

// usbfs ioctls (linux/usbdevice_fs.h)
#[repr(C)]
struct UsbdevfsCtrlTransfer {
    request_type: u8,
    request: u8,
    value: u16,
    index: u16,
    length: u16,
    timeout: u32,
    data: *mut c_void,
}

#[repr(C)]
struct UsbdevfsUrb {
    urb_type: u8,
    endpoint: u8,
    status: i32,
    flags: u32,
    buffer: *mut c_void,
    buffer_length: i32,
    actual_length: i32,
    start_frame: i32,
    number_of_packets: i32,
    error_count: i32,
    signr: u32,
    usercontext: *mut c_void,
}

const USBDEVFS_URB_TYPE_BULK: u8 = 3;

const IOC_WRITE: u32 = 1;
const IOC_READ: u32 = 2;
const fn usbdevfs_ioctl(direction: u32, number: u32, size: usize) -> libc::c_int {
    ((direction << 30) | ((size as u32) << 16) | ((b'U' as u32) << 8) | number) as libc::c_int
}
const USBDEVFS_CONTROL: libc::c_int = usbdevfs_ioctl(
    IOC_READ | IOC_WRITE,
    0,
    mem::size_of::<UsbdevfsCtrlTransfer>(),
);
const USBDEVFS_SUBMITURB: libc::c_int = usbdevfs_ioctl(IOC_READ, 10, mem::size_of::<UsbdevfsUrb>());
const USBDEVFS_DISCARDURB: libc::c_int = usbdevfs_ioctl(0, 11, 0);
const USBDEVFS_REAPURBNDELAY: libc::c_int =
    usbdevfs_ioctl(IOC_WRITE, 13, mem::size_of::<*mut c_void>());
const USBDEVFS_CLAIMINTERFACE: libc::c_int = usbdevfs_ioctl(IOC_READ, 15, mem::size_of::<u32>());
const USBDEVFS_RELEASEINTERFACE: libc::c_int = usbdevfs_ioctl(IOC_READ, 16, mem::size_of::<u32>());

struct StreamingInterface {
    interface_number: u16,
    endpoint_address: u32,
    format_index: u8,
    frame_index: u8,
    // Supported frame intervals in 100ns units
    frame_intervals: Vec<u32>,
    uvc_version: u16,
}

// Parse the class-specific descriptors and return the first video streaming interface
fn parse_streaming_interface(descriptors: &[u8]) -> Option<StreamingInterface> {
    let mut interface = None;
    let mut current_interface = (0u16, 0u8, 0u8);
    let mut uvc_version = 0x0100;
    let mut i = 0;
    while i + 2 <= descriptors.len() {
        let len = descriptors[i] as usize;
        if len < 2 || i + len > descriptors.len() {
            break;
        }
        let d = &descriptors[i..i + len];
        let is_video_control = current_interface.1 == 14 && current_interface.2 == 1;
        let is_video_streaming = current_interface.1 == 14 && current_interface.2 == 2;
        match d[1] {
            // interface descriptor: number, class, subclass
            0x04 => current_interface = (d[2] as u16, d[5], d[6]),
            0x24 if is_video_control && d[2] == 0x01 && len >= 5 => {
                uvc_version = u16::from_le_bytes([d[3], d[4]]);
            }
            // VS_INPUT_HEADER: endpoint address
            0x24 if is_video_streaming && d[2] == 0x01 && len >= 7 && interface.is_none() => {
                interface = Some(StreamingInterface {
                    interface_number: current_interface.0,
                    endpoint_address: d[6] as u32,
                    format_index: 0,
                    frame_index: 0,
                    frame_intervals: vec![],
                    uvc_version,
                });
            }
            // VS_FORMAT_UNCOMPRESSED: take the first format
            0x24 if is_video_streaming && d[2] == 0x04 && len >= 5 => {
                if let Some(interface) = &mut interface {
                    if interface.format_index == 0 {
                        interface.format_index = d[3];
                    }
                }
            }
            // VS_FRAME_UNCOMPRESSED: take the first frame of the selected format
            0x24 if is_video_streaming && d[2] == 0x05 && len >= 26 => {
                if let Some(interface) = &mut interface {
                    if interface.frame_index == 0 {
                        interface.frame_index = d[3];
                        let mut k = 26;
                        while k + 4 <= len {
                            let interval = u32::from_le_bytes([d[k], d[k + 1], d[k + 2], d[k + 3]]);
                            if interval > 0 {
                                interface.frame_intervals.push(interval);
                            }
                            k += 4;
                        }
                    }
                }
            }
            _ => (),
        }
        i += len;
    }

    interface.filter(|i| i.format_index != 0 && i.frame_index != 0 && !i.frame_intervals.is_empty())
}

// The slowest camera frame rate that still satisfies the requested one
fn select_frame_interval(interface: &StreamingInterface, fps: u32) -> u32 {
    let max_interval = 10_000_000 / fps.max(1);

    interface
        .frame_intervals
        .iter()
        .copied()
        .filter(|interval| *interval <= max_interval)
        .max()
        .unwrap_or_else(|| interface.frame_intervals.iter().copied().min().unwrap())
}

fn jni_context<'a>() -> JObject<'a> {
    unsafe { JObject::from_raw(alvr_system_info::android::context()) }
}

fn has_runtime_permission(env: &mut JNIEnv, permission: &str) -> bool {
    let Ok(permission) = env.new_string(permission) else {
        return false;
    };

    env.call_method(
        jni_context(),
        "checkSelfPermission",
        "(Ljava/lang/String;)I",
        &[(&permission).into()],
    )
    .map(|status| status.i().unwrap_or(-1) == 0)
    .unwrap_or(false)
}

// USB device connection driven with usbfs ioctls on the file descriptor obtained from the Android
// USB host API
struct Camera {
    fd: libc::c_int,
    interface: StreamingInterface,
    claimed: bool,
}

impl Camera {
    fn control_transfer(
        &self,
        request_type: u8,
        request: u8,
        value: u16,
        index: u16,
        data: &mut [u8],
    ) -> Result<(), String> {
        let mut transfer = UsbdevfsCtrlTransfer {
            request_type,
            request,
            value,
            index,
            length: data.len() as u16,
            timeout: USB_TIMEOUT_MS,
            data: data.as_mut_ptr().cast(),
        };
        let result = unsafe { libc::ioctl(self.fd, USBDEVFS_CONTROL, &mut transfer) };
        if result < 0 {
            return Err(format!(
                "control transfer failed: {}",
                std::io::Error::last_os_error()
            ));
        }

        Ok(())
    }

    // Claims the streaming interface. Returns Ok(false) when another process holds it.
    fn claim_interface(&mut self) -> Result<bool, String> {
        let mut number = self.interface.interface_number as u32;
        if unsafe { libc::ioctl(self.fd, USBDEVFS_CLAIMINTERFACE, &mut number) } >= 0 {
            self.claimed = true;

            return Ok(true);
        }
        let error = std::io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::EBUSY) {
            Ok(false)
        } else {
            Err(format!("cannot claim the interface: {error}"))
        }
    }

    // UVC probe/commit negotiation. Returns the maximum payload transfer size.
    fn commit_stream(&self, frame_interval: u32) -> Result<usize, String> {
        let control_len = if self.interface.uvc_version >= 0x0150 {
            48
        } else if self.interface.uvc_version >= 0x0110 {
            34
        } else {
            26
        };
        let mut control = vec![0u8; control_len];
        control[0] = 1; // bmHint: keep dwFrameInterval fixed
        control[2] = self.interface.format_index;
        control[3] = self.interface.frame_index;
        control[4..8].copy_from_slice(&frame_interval.to_le_bytes());

        let interface_number = self.interface.interface_number;
        self.control_transfer(
            0x21,
            UVC_SET_CUR,
            UVC_VS_PROBE_CONTROL << 8,
            interface_number,
            &mut control,
        )?;
        self.control_transfer(
            0xa1,
            UVC_GET_CUR,
            UVC_VS_PROBE_CONTROL << 8,
            interface_number,
            &mut control,
        )?;
        let max_payload_size =
            u32::from_le_bytes([control[22], control[23], control[24], control[25]]);
        self.control_transfer(
            0x21,
            UVC_SET_CUR,
            UVC_VS_COMMIT_CONTROL << 8,
            interface_number,
            &mut control,
        )?;

        Ok((max_payload_size as usize).clamp(1024, 1 << 20))
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        if self.claimed {
            let mut number = self.interface.interface_number as u32;
            unsafe { libc::ioctl(self.fd, USBDEVFS_RELEASEINTERFACE, &mut number) };
        }
    }
}

enum TransferStatus {
    Completed(usize),
    // The payload was lost but the endpoint keeps working
    Failed,
}

// Queue of asynchronous bulk transfers: several are always pending in the kernel, which drains the
// camera even while this thread is busy. usbfs copies the data into `buffers` on reap only.
struct TransferQueue<'a> {
    camera: &'a Camera,
    urbs: Box<[UsbdevfsUrb]>,
    buffers: Box<[Box<[u8]>]>,
    pending: usize,
}

impl<'a> TransferQueue<'a> {
    fn new(camera: &'a Camera, payload_size: usize, count: usize) -> Result<Self, String> {
        let mut buffers = (0..count)
            .map(|_| vec![0u8; payload_size].into_boxed_slice())
            .collect::<Box<[_]>>();
        let urbs = buffers
            .iter_mut()
            .map(|buffer| UsbdevfsUrb {
                urb_type: USBDEVFS_URB_TYPE_BULK,
                endpoint: camera.interface.endpoint_address as u8,
                status: 0,
                flags: 0,
                buffer: buffer.as_mut_ptr().cast(),
                buffer_length: payload_size as i32,
                actual_length: 0,
                start_frame: 0,
                number_of_packets: 0,
                error_count: 0,
                signr: 0,
                usercontext: ptr::null_mut(),
            })
            .collect();

        let mut queue = Self {
            camera,
            urbs,
            buffers,
            pending: 0,
        };
        for index in 0..count {
            queue.submit(index)?;
        }

        Ok(queue)
    }

    fn submit(&mut self, index: usize) -> Result<(), String> {
        let urb = &mut self.urbs[index];
        urb.actual_length = 0;
        if unsafe { libc::ioctl(self.camera.fd, USBDEVFS_SUBMITURB, urb as *mut UsbdevfsUrb) } < 0 {
            return Err(format!(
                "cannot submit a transfer: {}",
                std::io::Error::last_os_error()
            ));
        }
        self.pending += 1;

        Ok(())
    }

    // Waits for the next completed transfer. Returns its index and status, or None on timeout.
    fn reap(&mut self, timeout: Duration) -> Result<Option<(usize, TransferStatus)>, String> {
        let mut poll_fd = libc::pollfd {
            fd: self.camera.fd,
            events: libc::POLLOUT,
            revents: 0,
        };
        let ready = unsafe { libc::poll(&mut poll_fd, 1, timeout.as_millis() as libc::c_int) };
        if ready < 0 {
            let error = std::io::Error::last_os_error();
            return if error.raw_os_error() == Some(libc::EINTR) {
                Ok(None)
            } else {
                Err(format!("cannot wait for transfers: {error}"))
            };
        }
        if ready == 0 {
            return Ok(None);
        }
        if poll_fd.revents & (libc::POLLERR | libc::POLLHUP) != 0 {
            return Err("device disconnected".into());
        }

        let mut completed: *mut UsbdevfsUrb = ptr::null_mut();
        if unsafe { libc::ioctl(self.camera.fd, USBDEVFS_REAPURBNDELAY, &mut completed) } < 0 {
            let error = std::io::Error::last_os_error();
            return if error.raw_os_error() == Some(libc::EAGAIN) {
                Ok(None)
            } else {
                Err(format!("cannot reap a transfer: {error}"))
            };
        }
        self.pending -= 1;
        let index = unsafe { completed.offset_from(self.urbs.as_ptr()) };
        let Some(urb) = usize::try_from(index).ok().and_then(|i| self.urbs.get(i)) else {
            return Err("reaped an unknown transfer".into());
        };
        let index = index as usize;

        match -urb.status {
            0 => Ok(Some((
                index,
                TransferStatus::Completed(urb.actual_length.max(0) as usize),
            ))),
            // Transmission errors and cancelled transfers
            libc::EOVERFLOW
            | libc::EPROTO
            | libc::EILSEQ
            | libc::ETIME
            | libc::EREMOTEIO
            | libc::ENOENT
            | libc::ECONNRESET => Ok(Some((index, TransferStatus::Failed))),
            error => Err(format!(
                "transfer failed: {}",
                std::io::Error::from_raw_os_error(error)
            )),
        }
    }
}

impl Drop for TransferQueue<'_> {
    fn drop(&mut self) {
        for urb in self.urbs.iter_mut() {
            unsafe { libc::ioctl(self.camera.fd, USBDEVFS_DISCARDURB, urb as *mut UsbdevfsUrb) };
        }
        let deadline = Instant::now() + Duration::from_secs(1);
        while self.pending > 0 && Instant::now() < deadline {
            if self.reap(Duration::from_millis(100)).is_err() {
                break;
            }
        }
    }
}

fn find_eye_camera<'a>(
    env: &mut JNIEnv<'a>,
    usb_manager: &JObject,
) -> Result<Option<JObject<'a>>, JniError> {
    let device_map = env
        .call_method(usb_manager, "getDeviceList", "()Ljava/util/HashMap;", &[])?
        .l()?;
    let values = env
        .call_method(&device_map, "values", "()Ljava/util/Collection;", &[])?
        .l()?;
    let iterator = env
        .call_method(&values, "iterator", "()Ljava/util/Iterator;", &[])?
        .l()?;

    loop {
        if !env.call_method(&iterator, "hasNext", "()Z", &[])?.z()? {
            return Ok(None);
        }
        let device = env
            .call_method(&iterator, "next", "()Ljava/lang/Object;", &[])?
            .l()?;
        let vendor_id = env.call_method(&device, "getVendorId", "()I", &[])?.i()?;
        let product_id = env.call_method(&device, "getProductId", "()I", &[])?.i()?;
        if vendor_id == EYE_CAMERA_VENDOR_ID && product_id == EYE_CAMERA_PRODUCT_ID {
            return Ok(Some(device));
        }
    }
}

// The device node path (/dev/bus/usb/<bus>/<address>), which changes at every re-enumeration
fn device_name(env: &mut JNIEnv, device: &JObject) -> Result<String, JniError> {
    let name: JString = env
        .call_method(device, "getDeviceName", "()Ljava/lang/String;", &[])?
        .l()?
        .into();
    let name = env.get_string(&name)?.into();

    Ok(name)
}

fn has_usb_permission(
    env: &mut JNIEnv,
    usb_manager: &JObject,
    device: &JObject,
) -> Result<bool, JniError> {
    env.call_method(
        usb_manager,
        "hasPermission",
        "(Landroid/hardware/usb/UsbDevice;)Z",
        &[device.into()],
    )?
    .z()
}

fn request_usb_permission(
    env: &mut JNIEnv,
    usb_manager: &JObject,
    device: &JObject,
) -> Result<(), JniError> {
    const FLAG_IMMUTABLE: i32 = 0x0400_0000;

    let action = env.new_string(USB_PERMISSION_ACTION)?;
    let intent = env.new_object(
        "android/content/Intent",
        "(Ljava/lang/String;)V",
        &[(&action).into()],
    )?;
    let pending_intent = env
        .call_static_method(
            "android/app/PendingIntent",
            "getBroadcast",
            "(Landroid/content/Context;ILandroid/content/Intent;I)Landroid/app/PendingIntent;",
            &[
                (&jni_context()).into(),
                JValue::Int(0),
                (&intent).into(),
                JValue::Int(FLAG_IMMUTABLE),
            ],
        )?
        .l()?;
    env.call_method(
        usb_manager,
        "requestPermission",
        "(Landroid/hardware/usb/UsbDevice;Landroid/app/PendingIntent;)V",
        &[device.into(), (&pending_intent).into()],
    )?;

    Ok(())
}

enum AttemptOutcome {
    CameraNotFound,
    WaitingForPermission,
    OpenFailed(String),
    CameraBusy,
    Streamed,
}

// Streams from an open device. Returns when the stream ends or fails.
fn stream_from_device(
    fd: libc::c_int,
    descriptors: &[u8],
    core_ctx: &Arc<ClientCoreContext>,
    config: &EyeCamerasConfig,
    running: &RelaxedAtomic,
) -> AttemptOutcome {
    let Some(interface) = parse_streaming_interface(descriptors) else {
        return AttemptOutcome::OpenFailed("no video streaming interface found".into());
    };

    let mut camera = Camera {
        fd,
        interface,
        claimed: false,
    };
    match camera.claim_interface() {
        Ok(true) => {
            info!("Eye cameras: camera opened");
            stream_frames(&camera, core_ctx, config, running);

            AttemptOutcome::Streamed
        }
        Ok(false) => AttemptOutcome::CameraBusy,
        Err(e) => AttemptOutcome::OpenFailed(e),
    }
}

// One attempt to open and stream the camera. Returns when the stream ends or fails.
fn attempt_streaming(
    env: &mut JNIEnv,
    usb_manager: &JObject,
    permission_requested_for: &mut Option<String>,
    core_ctx: &Arc<ClientCoreContext>,
    config: &EyeCamerasConfig,
    running: &RelaxedAtomic,
) -> Result<AttemptOutcome, JniError> {
    let Some(device) = find_eye_camera(env, usb_manager)? else {
        return Ok(AttemptOutcome::CameraNotFound);
    };

    if !has_usb_permission(env, usb_manager, &device)? {
        // The permission is tied to the device instance: it is voided whenever the camera is
        // power-cycled (re-enumerated), so ask again for every new instance
        let name = device_name(env, &device)?;
        if permission_requested_for.as_deref() != Some(name.as_str()) {
            request_usb_permission(env, usb_manager, &device)?;
            info!("Eye cameras: waiting for the USB permission ({name})");
            *permission_requested_for = Some(name);
        }

        return Ok(AttemptOutcome::WaitingForPermission);
    }

    let connection = env
        .call_method(
            usb_manager,
            "openDevice",
            "(Landroid/hardware/usb/UsbDevice;)Landroid/hardware/usb/UsbDeviceConnection;",
            &[(&device).into()],
        )?
        .l()?;
    if connection.is_null() {
        return Ok(AttemptOutcome::OpenFailed("openDevice failed".into()));
    }

    let fd = env
        .call_method(&connection, "getFileDescriptor", "()I", &[])?
        .i()?;
    let descriptors: JByteArray = env
        .call_method(&connection, "getRawDescriptors", "()[B", &[])?
        .l()?
        .into();
    let descriptors = env.convert_byte_array(&descriptors)?;

    let outcome = stream_from_device(fd, &descriptors, core_ctx, config, running);

    env.call_method(&connection, "close", "()V", &[])?;

    Ok(outcome)
}

// Reassembles frames from UVC payloads
struct FrameAssembler {
    frame: Vec<u8>,
    corrupted: bool,
    last_fid: Option<u8>,
}

impl FrameAssembler {
    fn new(capacity: usize) -> Self {
        Self {
            frame: Vec::with_capacity(capacity),
            corrupted: false,
            last_fid: None,
        }
    }

    // Returns true when `frame` holds a complete and valid frame. It must then be cleared with
    // `take_frame()` before the next payload.
    fn push(&mut self, payload: &[u8]) -> bool {
        if payload.len() < 2 {
            return false;
        }
        let header_len = payload[0] as usize;
        let header_info = payload[1];
        if header_len < 2 || header_len > payload.len() {
            return false;
        }

        let fid = header_info & UVC_PAYLOAD_HEADER_FID;
        if self.last_fid.is_some_and(|last| last != fid) && !self.frame.is_empty() {
            // A new frame started without an end-of-frame marker
            self.discard_frame();
        }
        self.last_fid = Some(fid);

        if header_info & UVC_PAYLOAD_HEADER_ERROR != 0 {
            self.corrupted = true;
        }
        self.frame.extend_from_slice(&payload[header_len..]);

        if header_info & UVC_PAYLOAD_HEADER_EOF == 0 {
            return false;
        }

        let complete = !self.corrupted && self.frame.len() == FRAME_BYTES;
        if !complete {
            self.discard_frame();
        }

        complete
    }

    // Hands out the completed frame, continuing with `replacement` as the buffer
    fn take_frame(&mut self, replacement: Vec<u8>) -> Vec<u8> {
        self.corrupted = false;

        mem::replace(&mut self.frame, replacement)
    }

    fn discard_frame(&mut self) {
        self.frame.clear();
        self.corrupted = false;
    }
}

// Splits the frames into the two eye images, encodes them and sends them to the streamer. This
// runs on its own thread so that the transfer thread never stalls.
fn encoder_loop(
    frames: Receiver<(Duration, Vec<u8>)>,
    recycled_buffers: Sender<Vec<u8>>,
    core_ctx: Arc<ClientCoreContext>,
    jpeg_quality: u32,
) {
    let mut left = vec![0u8; EYE_IMAGE_WIDTH * EYE_IMAGE_HEIGHT];
    let mut right = vec![0u8; EYE_IMAGE_WIDTH * EYE_IMAGE_HEIGHT];
    let mut left_jpeg = Vec::new();
    let mut right_jpeg = Vec::new();

    while let Ok((timestamp, mut frame)) = frames.recv() {
        for row in 0..EYE_IMAGE_HEIGHT {
            let source = &frame[row * FRAME_ROW_BYTES..(row + 1) * FRAME_ROW_BYTES];
            right[row * EYE_IMAGE_WIDTH..(row + 1) * EYE_IMAGE_WIDTH]
                .copy_from_slice(&source[..EYE_IMAGE_WIDTH]);
            left[row * EYE_IMAGE_WIDTH..(row + 1) * EYE_IMAGE_WIDTH]
                .copy_from_slice(&source[EYE_IMAGE_WIDTH..]);
        }
        frame.clear();
        recycled_buffers.send(frame).ok();

        let encoded = encode_jpeg(&left, jpeg_quality, &mut left_jpeg)
            && encode_jpeg(&right, jpeg_quality, &mut right_jpeg);
        if encoded {
            core_ctx.send_eye_camera_frame(timestamp, &left_jpeg, &right_jpeg);
        }
    }
}

// Reads frames until stopped or until the camera fails
fn stream_frames(
    camera: &Camera,
    core_ctx: &Arc<ClientCoreContext>,
    config: &EyeCamerasConfig,
    running: &RelaxedAtomic,
) {
    let frame_interval = select_frame_interval(&camera.interface, config.fps);
    let max_payload_size = match camera.commit_stream(frame_interval) {
        Ok(size) => size,
        Err(e) => {
            warn!("Eye cameras: stream negotiation failed: {e}");
            return;
        }
    };
    let camera_fps = 10_000_000.0 / frame_interval as f64;
    let queued_transfers = (TRANSFER_QUEUE_DURATION.as_secs_f64() * camera_fps * FRAME_BYTES as f64
        / max_payload_size as f64)
        .ceil() as usize;
    let queued_transfers = queued_transfers.clamp(MIN_QUEUED_TRANSFERS, MAX_QUEUED_TRANSFERS);
    let mut transfers = match TransferQueue::new(camera, max_payload_size, queued_transfers) {
        Ok(transfers) => transfers,
        Err(e) => {
            warn!("Eye cameras: {e}");
            return;
        }
    };
    info!(
        "Eye cameras: camera streaming at {camera_fps:.0}fps, sending {}fps, payload \
         {max_payload_size}B x{queued_transfers}",
        config.fps
    );

    // The encoder gets at most one frame ahead; frames are dropped while it is busy
    let (frame_sender, frame_receiver) = mpsc::sync_channel(1);
    let (recycled_sender, recycled_receiver) = mpsc::channel();
    let encoder_thread = thread::spawn({
        let core_ctx = Arc::clone(core_ctx);
        let jpeg_quality = config.jpeg_quality;
        move || encoder_loop(frame_receiver, recycled_sender, core_ctx, jpeg_quality)
    });

    let send_interval = Duration::from_secs_f64(1.0 / config.fps.max(1) as f64);
    let mut next_send_time = Instant::now();
    let frame_capacity = FRAME_BYTES + max_payload_size;
    let mut assembler = FrameAssembler::new(frame_capacity);
    let mut consecutive_timeouts = 0;

    while running.value() {
        let (index, status) = match transfers.reap(TRANSFER_POLL_TIMEOUT) {
            Ok(Some(transfer)) => {
                consecutive_timeouts = 0;
                transfer
            }
            Ok(None) => {
                consecutive_timeouts += 1;
                if consecutive_timeouts >= MAX_TRANSFER_TIMEOUTS {
                    warn!("Eye cameras: the camera stopped sending frames, reopening it");
                    break;
                }
                continue;
            }
            Err(e) => {
                warn!("Eye cameras: {e}, reopening the camera");
                break;
            }
        };

        if let TransferStatus::Completed(size) = status {
            let payload = &transfers.buffers[index][..size];
            if assembler.push(payload) {
                let now = Instant::now();
                if now >= next_send_time {
                    next_send_time = (next_send_time + send_interval).max(now - send_interval);

                    let timestamp = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default();
                    let replacement = recycled_receiver
                        .try_recv()
                        .unwrap_or_else(|_| Vec::with_capacity(frame_capacity));
                    let frame = assembler.take_frame(replacement);
                    if let Err(
                        TrySendError::Full((_, mut frame))
                        | TrySendError::Disconnected((_, mut frame)),
                    ) = frame_sender.try_send((timestamp, frame))
                    {
                        // The encoder is still busy with the previous frame: drop this one and
                        // keep its buffer instead of the replacement
                        frame.clear();
                        drop(assembler.take_frame(frame));
                    }
                } else {
                    assembler.discard_frame();
                }
            }
        }

        if let Err(e) = transfers.submit(index) {
            warn!("Eye cameras: {e}, reopening the camera");
            break;
        }
    }

    drop(transfers);
    drop(frame_sender);
    encoder_thread.join().ok();
}

fn encode_jpeg(image: &[u8], quality: u32, output: &mut Vec<u8>) -> bool {
    output.clear();
    let encoder = JpegEncoder::new_with_quality(&mut *output, quality.clamp(1, 100) as u8);

    match encoder.write_image(
        image,
        EYE_IMAGE_WIDTH as u32,
        EYE_IMAGE_HEIGHT as u32,
        ExtendedColorType::L8,
    ) {
        Ok(()) => true,
        Err(e) => {
            error!("Eye cameras: JPEG encoding failed: {e}");
            false
        }
    }
}

fn streaming_loop(
    core_ctx: Arc<ClientCoreContext>,
    config: EyeCamerasConfig,
    running: Arc<RelaxedAtomic>,
) {
    let vm = alvr_system_info::android::vm();
    let Ok(mut env) = vm.attach_current_thread() else {
        error!("Eye cameras: cannot attach the JVM");
        return;
    };

    // Android requires the camera permission before granting access to USB video devices
    if !has_runtime_permission(&mut env, CAMERA_PERMISSION) {
        alvr_system_info::try_get_permission(CAMERA_PERMISSION);
        info!("Eye cameras: waiting for the camera permission");
        while running.value() && !has_runtime_permission(&mut env, CAMERA_PERMISSION) {
            thread::sleep(RETRY_INTERVAL);
        }
    }

    let usb_manager = match get_usb_manager(&mut env) {
        Ok(manager) => manager,
        Err(e) => {
            error!("Eye cameras: USB host API not available: {e}");
            return;
        }
    };

    let mut permission_requested_for = None;
    let mut logged_not_found = false;
    let mut logged_busy = false;
    while running.value() {
        // A local reference frame per attempt keeps the JNI reference table bounded
        let outcome = env.with_local_frame(32, |env| {
            attempt_streaming(
                env,
                usb_manager.as_obj(),
                &mut permission_requested_for,
                &core_ctx,
                &config,
                &running,
            )
        });

        match outcome {
            Ok(AttemptOutcome::CameraNotFound) => {
                if !logged_not_found {
                    warn!("Eye cameras: camera not found, is the headset eye tracking enabled?");
                    logged_not_found = true;
                }
            }
            Ok(AttemptOutcome::WaitingForPermission) => (),
            Ok(AttemptOutcome::OpenFailed(reason)) => warn!("Eye cameras: {reason}"),
            Ok(AttemptOutcome::CameraBusy) => {
                if !logged_busy {
                    warn!(
                        "Eye cameras: the camera is held by the headset eye tracking service. Run once: adb shell am force-stop com.htc.vr.device.eye"
                    );
                    logged_busy = true;
                }
            }
            Ok(AttemptOutcome::Streamed) => {
                logged_not_found = false;
                logged_busy = false;
            }
            Err(e) => {
                error!("Eye cameras: JNI error: {e}");
                env.exception_clear().ok();
            }
        }

        if running.value() {
            thread::sleep(RETRY_INTERVAL);
        }
    }
}

fn get_usb_manager(env: &mut JNIEnv) -> Result<GlobalRef, JniError> {
    let service = env.new_string("usb")?;
    let manager = env
        .call_method(
            jni_context(),
            "getSystemService",
            "(Ljava/lang/String;)Ljava/lang/Object;",
            &[(&service).into()],
        )?
        .l()?;
    if manager.is_null() {
        return Err(JniError::NullPtr("UsbManager"));
    }

    env.new_global_ref(manager)
}

pub struct EyeCamerasStreamer {
    running: Arc<RelaxedAtomic>,
    thread: Option<JoinHandle<()>>,
}

impl EyeCamerasStreamer {
    pub fn new(core_ctx: Arc<ClientCoreContext>, config: EyeCamerasConfig) -> Self {
        let running = Arc::new(RelaxedAtomic::new(true));

        let thread = thread::spawn({
            let running = Arc::clone(&running);
            move || streaming_loop(core_ctx, config, running)
        });

        Self {
            running,
            thread: Some(thread),
        }
    }
}

impl Drop for EyeCamerasStreamer {
    fn drop(&mut self) {
        self.running.set(false);
        if let Some(thread) = self.thread.take() {
            thread.join().ok();
        }
    }
}
