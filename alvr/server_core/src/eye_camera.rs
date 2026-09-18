//! Serves the eye camera frames received from the client as MJPEG streams over HTTP, for PC-side
//! eye tracking software (http://127.0.0.1:<port>/left.mjpg, /right.mjpg, /left.jpg, /right.jpg)
use alvr_common::{
    anyhow::Result,
    info,
    parking_lot::{Condvar, Mutex},
    warn,
};
use std::{
    io::{BufRead, BufReader, Write},
    net::{Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

const MULTIPART_BOUNDARY: &str = "alvr-eye-camera-frame";
// How often a streaming connection checks for shutdown while no frame arrives
const FRAME_WAIT_TIMEOUT: Duration = Duration::from_secs(1);
const REQUEST_READ_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy)]
enum Eye {
    Left,
    Right,
}

#[derive(Default)]
struct Frames {
    sequence: u64,
    left_jpeg: Arc<Vec<u8>>,
    right_jpeg: Arc<Vec<u8>>,
}

struct Shared {
    frames: Mutex<Frames>,
    frame_available: Condvar,
    running: AtomicBool,
}

impl Shared {
    // Wait for a frame newer than `last_sequence`. Returns None when the server is shutting down or
    // when the connection was closed by the peer while waiting.
    fn wait_frame(
        &self,
        stream: &TcpStream,
        last_sequence: u64,
        eye: Eye,
    ) -> Option<(u64, Arc<Vec<u8>>)> {
        let mut frames = self.frames.lock();
        while frames.sequence == last_sequence {
            if !self.running.load(Ordering::Relaxed) {
                return None;
            }
            let timed_out = self
                .frame_available
                .wait_for(&mut frames, FRAME_WAIT_TIMEOUT)
                .timed_out();
            if timed_out && peer_closed(stream) {
                return None;
            }
        }

        let jpeg = match eye {
            Eye::Left => Arc::clone(&frames.left_jpeg),
            Eye::Right => Arc::clone(&frames.right_jpeg),
        };

        Some((frames.sequence, jpeg))
    }
}

// Detects a connection closed by the peer without blocking, so that handlers of consumers that
// went away are not kept alive until the next frame
fn peer_closed(stream: &TcpStream) -> bool {
    if stream.set_nonblocking(true).is_err() {
        return false;
    }
    let closed = match stream.peek(&mut [0u8]) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) => e.kind() != std::io::ErrorKind::WouldBlock,
    };
    stream.set_nonblocking(false).ok();

    closed
}

pub struct EyeCameraServer {
    shared: Arc<Shared>,
    port: u16,
    accept_thread: Option<JoinHandle<()>>,
}

impl EyeCameraServer {
    pub fn new(port: u16) -> Result<Self> {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, port)))?;
        info!("Eye cameras: MJPEG server listening on http://127.0.0.1:{port}/left.mjpg and /right.mjpg");

        let shared = Arc::new(Shared {
            frames: Mutex::new(Frames::default()),
            frame_available: Condvar::new(),
            running: AtomicBool::new(true),
        });

        let accept_thread = thread::spawn({
            let shared = Arc::clone(&shared);
            move || {
                for stream in listener.incoming() {
                    if !shared.running.load(Ordering::Relaxed) {
                        break;
                    }
                    if let Ok(stream) = stream {
                        let shared = Arc::clone(&shared);
                        thread::spawn(move || handle_client(stream, &shared));
                    }
                }
            }
        });

        Ok(Self {
            shared,
            port,
            accept_thread: Some(accept_thread),
        })
    }

    pub fn push_frame(&self, left_jpeg: Vec<u8>, right_jpeg: Vec<u8>) {
        let mut frames = self.shared.frames.lock();
        frames.sequence += 1;
        frames.left_jpeg = Arc::new(left_jpeg);
        frames.right_jpeg = Arc::new(right_jpeg);
        self.shared.frame_available.notify_all();
    }
}

impl Drop for EyeCameraServer {
    fn drop(&mut self) {
        self.shared.running.store(false, Ordering::Relaxed);
        self.shared.frame_available.notify_all();

        // Unblock the accept loop
        TcpStream::connect_timeout(
            &SocketAddr::from((Ipv4Addr::LOCALHOST, self.port)),
            Duration::from_millis(500),
        )
        .ok();
        if let Some(thread) = self.accept_thread.take() {
            thread.join().ok();
        }
    }
}

fn handle_client(mut stream: TcpStream, shared: &Shared) {
    stream.set_read_timeout(Some(REQUEST_READ_TIMEOUT)).ok();
    stream.set_nodelay(true).ok();

    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return;
    }
    // Discard the headers
    let mut line = String::new();
    while reader.read_line(&mut line).is_ok_and(|n| n > 0) && !line.trim().is_empty() {
        line.clear();
    }

    let path = request_line
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_lowercase();

    let result = match path.as_str() {
        "/left.mjpg" | "/left.mjpeg" => serve_multipart(&mut stream, shared, Eye::Left),
        "/right.mjpg" | "/right.mjpeg" => serve_multipart(&mut stream, shared, Eye::Right),
        "/left.jpg" | "/left.jpeg" => serve_single(&mut stream, shared, Eye::Left),
        "/right.jpg" | "/right.jpeg" => serve_single(&mut stream, shared, Eye::Right),
        "/" => stream.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nConnection: close\r\n\r\nALVR eye cameras\n/left.mjpg\n/right.mjpg\n/left.jpg\n/right.jpg\n",
        ),
        _ => stream.write_all(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n"),
    };
    if let Err(e) = result {
        if e.kind() != std::io::ErrorKind::BrokenPipe
            && e.kind() != std::io::ErrorKind::ConnectionReset
            && e.kind() != std::io::ErrorKind::ConnectionAborted
        {
            warn!("Eye cameras: HTTP client error: {e}");
        }
    }

    stream.shutdown(Shutdown::Both).ok();
}

fn serve_multipart(stream: &mut TcpStream, shared: &Shared, eye: Eye) -> std::io::Result<()> {
    stream.write_all(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: multipart/x-mixed-replace; boundary={MULTIPART_BOUNDARY}\r\nCache-Control: no-cache, no-store\r\nPragma: no-cache\r\nConnection: close\r\n\r\n"
        )
        .as_bytes(),
    )?;

    let mut last_sequence = 0;
    while let Some((sequence, jpeg)) = shared.wait_frame(stream, last_sequence, eye) {
        last_sequence = sequence;
        stream.write_all(
            format!(
                "--{MULTIPART_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                jpeg.len()
            )
            .as_bytes(),
        )?;
        stream.write_all(&jpeg)?;
        stream.write_all(b"\r\n")?;
        stream.flush()?;
    }

    Ok(())
}

fn serve_single(stream: &mut TcpStream, shared: &Shared, eye: Eye) -> std::io::Result<()> {
    let Some((_, jpeg)) = shared.wait_frame(stream, 0, eye) else {
        return stream.write_all(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\n\r\n");
    };

    stream.write_all(
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\nCache-Control: no-cache, no-store\r\nConnection: close\r\n\r\n",
            jpeg.len()
        )
        .as_bytes(),
    )?;
    stream.write_all(&jpeg)?;
    stream.flush()
}
