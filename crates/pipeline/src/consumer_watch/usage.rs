//! v4l2loopback's client-usage event: whether a reader holds the capture stream.
//! It exists since v4l2loopback 0.13 and fires on capture `STREAMON` / `STREAMOFF`,
//! including the implicit ones from `read()` I/O and from `close()`.

use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::fs::{Mode, OFlags};
use rustix::io::Errno;
use rustix::ioctl::{Getter, Opcode, Setter, ioctl, opcode};
use std::io;

/// `V4L2_EVENT_PRI_CLIENT_USAGE`: `V4L2_EVENT_PRIVATE_START + 0x08E00000 + 1`.
const CLIENT_USAGE: u32 = 0x0800_0000 + 0x08E0_0000 + 1;

/// `V4L2_EVENT_SUB_FL_SEND_INITIAL`: queue the current state on subscribe.
const SEND_INITIAL: u32 = 1;

/// `VIDIOC_SUBSCRIBE_EVENT`.
const SUBSCRIBE_EVENT: Opcode = opcode::write::<EventSubscription>(b'V', 90);

/// `VIDIOC_DQEVENT`.
const DQEVENT: Opcode = opcode::read::<Event>(b'V', 89);

/// A client-usage subscription on the sink device. Its fd polls `POLLPRI`
/// while an event is queued.
pub(super) struct UsageEvents {
    fd: OwnedFd,
}

impl UsageEvents {
    /// Open `device` and subscribe; the current state is queued at once.
    /// Fails with `EINVAL` before v4l2loopback 0.13, `ENOTTY` on a non-V4L2 node.
    pub(super) fn subscribe(device: &str) -> io::Result<Self> {
        let flags = OFlags::RDONLY | OFlags::NONBLOCK | OFlags::CLOEXEC;
        let fd = rustix::fs::open(device, flags, Mode::empty())?;
        subscribe_event(
            &fd,
            EventSubscription {
                type_: CLIENT_USAGE,
                id: 0,
                flags: SEND_INITIAL,
                reserved: [0; 5],
            },
        )?;
        Ok(Self { fd })
    }

    /// Dequeue every queued event: `Some(streaming)` from the latest
    /// client-usage one, `None` when nothing was queued.
    pub(super) fn drain(&self) -> io::Result<Option<bool>> {
        let mut latest = None;
        loop {
            match dequeue_event(&self.fd) {
                Ok(event) if event.type_ == CLIENT_USAGE => latest = Some(event.count() != 0),
                Ok(_) => {}
                Err(Errno::NOENT) => return Ok(latest),
                Err(e) => return Err(e.into()),
            }
        }
    }
}

impl AsFd for UsageEvents {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

#[allow(unsafe_code)]
fn subscribe_event(fd: &OwnedFd, sub: EventSubscription) -> rustix::io::Result<()> {
    // SAFETY: the opcode is `_IOW('V', 90, …)` sized from `EventSubscription`, its exact mirror.
    unsafe { ioctl(fd, Setter::<SUBSCRIBE_EVENT, EventSubscription>::new(sub)) }
}

#[allow(unsafe_code)]
fn dequeue_event(fd: &OwnedFd) -> rustix::io::Result<Event> {
    // SAFETY: `_IOR('V', 89, …)` sized from `Event`, all integers, so any bytes are valid.
    unsafe { ioctl(fd, Getter::<DQEVENT, Event>::new()) }
}

/// `struct v4l2_event_subscription`.
#[repr(C)]
struct EventSubscription {
    type_: u32,
    id: u32,
    flags: u32,
    reserved: [u32; 5],
}

/// `struct v4l2_event` on 64-bit targets. The union holds an `__s64`, so it
/// is 8-byte aligned.
#[repr(C)]
struct Event {
    type_: u32,
    u: [u64; 8],
    pending: u32,
    sequence: u32,
    timestamp: [i64; 2],
    id: u32,
    reserved: [u32; 8],
}

impl Event {
    /// `struct v4l2_event_client_usage { __u32 count; }` at the union's start.
    fn count(&self) -> u32 {
        let b = self.u[0].to_ne_bytes();
        u32::from_ne_bytes([b[0], b[1], b[2], b[3]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::offset_of;

    #[test]
    fn abi_matches_kernel_headers() {
        assert_eq!(size_of::<EventSubscription>(), 32);
        assert_eq!(size_of::<Event>(), 136);
        assert_eq!(offset_of!(Event, u), 8);
        assert_eq!(SUBSCRIBE_EVENT as u64, 0x4020_565a);
        assert_eq!(DQEVENT as u64, 0x8088_5659);
        assert_eq!(CLIENT_USAGE, 0x10E0_0001);
    }

    #[test]
    fn subscribe_fails_on_a_non_v4l2_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("video");
        std::fs::write(&path, b"").unwrap();
        let Err(err) = UsageEvents::subscribe(path.to_str().unwrap()) else {
            panic!("subscribed on a regular file");
        };
        // ENOTTY, not an open error: the ioctl was reached and refused.
        assert_eq!(err.raw_os_error(), Some(Errno::NOTTY.raw_os_error()));
    }
}
