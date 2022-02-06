import time

import win32api
import win32con

from win32_utils.check_key_utils import Win32KeySignal, state_capsLock
from win32_utils.keyboard_simulation import Simulate_Keyboard
l_button = Win32KeySignal(win32con.VK_LBUTTON)
state_right = win32api.GetKeyState(win32con.VK_RBUTTON)
state_w = win32api.GetKeyState(ord("W"))
LALT_KEY = win32con.VK_LMENU
RALT_KEY = win32con.VK_RMENU

state_lalt = win32api.GetKeyState(LALT_KEY)
state_ralt = win32api.GetKeyState(LALT_KEY)
Simulate_Keyboard.press_key("caps_lock")

if __name__ == '__main__':
    while True:
        r_click = win32api.GetKeyState(win32con.VK_RBUTTON)
        w_button = win32api.GetKeyState(ord("W"))
        lalt_button = win32api.GetKeyState(LALT_KEY)
        ralt_button = win32api.GetKeyState(RALT_KEY)

        if l_button.check():
            if l_button.isPress():
                print("left pressed")

        if r_click != state_right:
            state_right = r_click
            if r_click:
                print("right click press")
            else:
                print("right click release")

        if w_button != state_w:
            state_w = w_button
            print("W")

        if lalt_button != state_lalt:
            state_lalt = lalt_button
            print("Left alt")
        if ralt_button != state_ralt:
            if ralt_button < 0:
                print("right alt press")
            else:
                pass
            state_ralt = ralt_button

        if state_capsLock.checkAndIsPress():
            print("caps Lock open")
        time.sleep(0.00001)
