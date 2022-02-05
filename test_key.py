import time

from win32_utils.directkeys import PressKey, W, LSHIFT

# PressKey(W)
from win32_utils.keyboard_simulation import Simulate_Keyboard

Num9 = 0x49
Num8 = 0x48
F1 = 0xBB


# for i in range(4):
#     print(i+1)
#     time.sleep(1)

line = 0x2B # \

Simulate_Keyboard.press_key('spacebar')
