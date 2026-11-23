//! Whether the window is on the user's screen, for the GUI-preview
//! heartbeat.
//!
//! The toolkit can't tell us: winit's `is_minimized` is always `None` on
//! Wayland, and on X11 an unmapped (`Visible(false)`) window doesn't read
//! as minimized. So the heartbeat follows our own show/hide requests
//! instead, plus one compositor signal: on Wayland a hidden window stays
//! in the taskbar and can be restored from there without any request
//! from us, which shows up as the window regaining keyboard focus.

pub(super) struct Presence {
    hidden: bool,
    /// A focus gain un-hides only after a user hide. A headless launch
    /// starts hidden and may get a spurious focus while the compositor
    /// maps the surface; that must not light the camera.
    focus_unhides: bool,
    focused: bool,
}

impl Presence {
    pub(super) fn new(start_hidden: bool) -> Self {
        Self {
            hidden: start_hidden,
            focus_unhides: false,
            focused: false,
        }
    }

    pub(super) fn show(&mut self) {
        self.hidden = false;
    }

    pub(super) fn hide(&mut self) {
        self.hidden = true;
        self.focus_unhides = true;
    }

    /// Feed this frame's focus state; returns whether the window is shown.
    pub(super) fn observe(&mut self, focused: bool) -> bool {
        if focused && !self.focused && self.focus_unhides {
            self.hidden = false;
        }
        self.focused = focused;
        !self.hidden
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// X11: the window is unmapped, never reads as minimized and loses
    /// focus. It must stay hidden until the tray's Show.
    #[test]
    fn hide_holds_without_any_toolkit_signal() {
        let mut p = Presence::new(false);
        assert!(p.observe(true));
        p.hide();
        assert!(!p.observe(true), "hidden in the same frame, still focused");
        for _ in 0..10 {
            assert!(!p.observe(false));
        }
        p.show();
        assert!(p.observe(false));
    }

    /// Wayland: restoring from the taskbar never calls `show()`; the focus
    /// gain that comes with it does.
    #[test]
    fn focus_gain_after_user_hide_shows_again() {
        let mut p = Presence::new(false);
        assert!(p.observe(true));
        p.hide();
        assert!(!p.observe(true));
        assert!(!p.observe(false));
        assert!(p.observe(true));
    }

    /// Headless start: a focus blip while the compositor maps the surface
    /// must not un-hide (that would light the camera at login).
    #[test]
    fn headless_start_ignores_focus_until_shown() {
        let mut p = Presence::new(true);
        assert!(!p.observe(false));
        assert!(!p.observe(true));
        assert!(!p.observe(false));
        p.show();
        assert!(p.observe(true));
    }
}
