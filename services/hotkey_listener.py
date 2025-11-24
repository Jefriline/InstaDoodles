import threading
from typing import Optional

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False


class GlobalHotkeyListener:
    
    def __init__(self, cancel_event: "threading.Event", hotkey: str = "f9") -> None:
        if not PYNPUT_AVAILABLE:
            raise ImportError(
                "pynput no está disponible. Instálalo con: pip install pynput"
            )
        
        self.cancel_event = cancel_event
        self.hotkey = hotkey.lower()
        self.listener: Optional["keyboard.Listener"] = None
        self.listener_thread: Optional["threading.Thread"] = None
        self._is_listening = False

    def _build_global_hotkey(self):
        try:
            hotkey_pattern = self.hotkey
            handler = lambda: self.cancel_event.set()
            return keyboard.GlobalHotKeys({hotkey_pattern: handler})
        except Exception:
            return None

    def _on_key_press(self, key) -> None:
        try:
            key_name = None
            if hasattr(key, "name"):
                key_name = key.name.lower()
            elif hasattr(key, "char") and key.char:
                key_name = key.char.lower()
            else:
                return
            if key_name == self.hotkey.lower():
                self.cancel_event.set()
        except Exception:
            pass
    
    def start_listening(self) -> None:
        if self._is_listening:
            return
        
        self._is_listening = True
        ghk = self._build_global_hotkey()
        if ghk is not None:
            self.listener = ghk
            self.listener.start()
        else:
            self.listener = keyboard.Listener(on_press=self._on_key_press)
            self.listener.start()
    
    def stop_listening(self) -> None:
        if not self._is_listening:
            return
        
        self._is_listening = False
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
