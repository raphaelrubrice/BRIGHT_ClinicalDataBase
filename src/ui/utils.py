import platform
import subprocess
import ctypes
import ctypes.util

class SleepInhibitor:
    """
    Prevents system sleep due to user inactivity while active.

    Windows  : SetThreadExecutionState
    macOS    : IOPMAssertionCreateWithName
    Linux    : systemd-inhibit (if available)
    """

    def __init__(self, reason: str = "ClinicalDatabase processing"):
        self.reason = reason
        self.system = platform.system()
        self._active = False

        # Windows
        self._kernel32 = None

        # macOS
        self._assertion_id = ctypes.c_uint32(0)

        # Linux
        self._linux_proc = None

    # -------------------------
    # Public API
    # -------------------------
    def enable(self):
        if self._active:
            return

        try:
            if self.system == "Windows":
                self._enable_windows()
            elif self.system == "Darwin":
                self._enable_macos()
            elif self.system == "Linux":
                self._enable_linux()
        except Exception:
            # Never fail hard because of power management
            pass

        self._active = True

    def disable(self):
        if not self._active:
            return

        try:
            if self.system == "Windows":
                self._disable_windows()
            elif self.system == "Darwin":
                self._disable_macos()
            elif self.system == "Linux":
                self._disable_linux()
        except Exception:
            pass

        self._active = False

    # -------------------------
    # Windows
    # -------------------------
    def _enable_windows(self):
        self._kernel32 = ctypes.windll.kernel32
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        self._kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )

    def _disable_windows(self):
        ES_CONTINUOUS = 0x80000000
        self._kernel32.SetThreadExecutionState(ES_CONTINUOUS)

    # -------------------------
    # macOS
    # -------------------------
    def _enable_macos(self):
        iokit = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("IOKit")
        )
        cf = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("CoreFoundation")
        )

        kIOPMAssertionTypeNoIdleSleep = cf.CFStringCreateWithCString(
            None,
            b"NoIdleSleepAssertion",
            0x08000100,  # kCFStringEncodingUTF8
        )

        reason = cf.CFStringCreateWithCString(
            None,
            self.reason.encode("utf-8"),
            0x08000100,
        )

        iokit.IOPMAssertionCreateWithName(
            kIOPMAssertionTypeNoIdleSleep,
            255,  # kIOPMAssertionLevelOn
            reason,
            ctypes.byref(self._assertion_id),
        )

    def _disable_macos(self):
        iokit = ctypes.cdll.LoadLibrary(
            ctypes.util.find_library("IOKit")
        )
        iokit.IOPMAssertionRelease(self._assertion_id)

    # -------------------------
    # Linux
    # -------------------------
    def _enable_linux(self):
        # Use systemd-inhibit if available
        if subprocess.call(
            ["which", "systemd-inhibit"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ) != 0:
            return

        self._linux_proc = subprocess.Popen(
            [
                "systemd-inhibit",
                "--what=sleep",
                "--why=" + self.reason,
                "--mode=block",
                "sleep",
                "infinity",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def _disable_linux(self):
        if self._linux_proc:
            self._linux_proc.terminate()
            self._linux_proc = None
