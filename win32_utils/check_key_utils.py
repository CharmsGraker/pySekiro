import win32api,win32con
class Win32KeySignal:
    def __init__(self, keyCode):
        self.keyCode = keyCode
        self.state = win32api.GetKeyState(self.keyCode)
        self.is_press = False

    def _getNewState(self):
        return win32api.GetKeyState(self.keyCode)

    def check(self):
        """
        this will update state
        :return:
        """
        new_state = self._getNewState()
        old_state = self.state
        if old_state != new_state:
            self.state = new_state
            self.is_press = True if new_state < 0 else False
            return True
        return False

    def isPress(self):
        return self.is_press

    def isRelease(self):
        return not self.is_press

    def checkAndIsPress(self):
        if self.check():
            return self.isPress()
        return False


state_lButton = Win32KeySignal(win32con.VK_LBUTTON)
state_rButton = Win32KeySignal(win32con.VK_RBUTTON)
state_space = Win32KeySignal(win32con.VK_SPACE)
state_leftAlt = Win32KeySignal(win32con.VK_LMENU) # left alt
state_esc = Win32KeySignal(win32con.VK_ESCAPE)
state_capsLock = Win32KeySignal(win32con.VK_CAPITAL)
state_W = Win32KeySignal(ord("W"))
state_A = Win32KeySignal(ord("A"))
state_S = Win32KeySignal(ord("S"))
state_D = Win32KeySignal(ord("D"))

state_R = Win32KeySignal(ord("R"))
state_Q = Win32KeySignal(ord('Q'))



