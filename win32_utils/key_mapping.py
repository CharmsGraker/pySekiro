import time

from win32_utils.directkeys import *
from win32_utils.keyboard_simulation import Simulate_Keyboard


def defense():
    Simulate_Keyboard.press_key('l')
    time.sleep(0.05)
    Simulate_Keyboard.release_key('l')
    # time.sleep(0.1)


def attack():
    PressKey(J)
    time.sleep(0.05)
    ReleaseKey(J)
    # time.sleep(0.1)


def go_forward():
    PressKey(W)
    time.sleep(0.4)
    ReleaseKey(W)


def go_back():
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)


def go_left():
    PressKey(A)
    time.sleep(0.4)
    ReleaseKey(A)


def go_right():
    PressKey(D)
    time.sleep(0.4)
    ReleaseKey(D)


def jump():
    Simulate_Keyboard.press_key('spacebar')
    time.sleep(0.1)
    Simulate_Keyboard.release_key('spacebar')
    # time.sleep(0.1)


def dodge():  # 闪避
    # PressKey(Q)
    Simulate_Keyboard.press_key('alt')

    time.sleep(0.1)
    Simulate_Keyboard.release_key('alt')
    # time.sleep(0.1)


def lock_vision():
    PressKey(V)
    time.sleep(0.3)
    ReleaseKey(V)
    time.sleep(0.1)


def go_forward_QL(t):
    PressKey(W)
    time.sleep(t)
    ReleaseKey(W)


def turn_left(t):
    PressKey(left)
    time.sleep(t)
    ReleaseKey(left)


def turn_up(t):
    PressKey(up)
    time.sleep(t)
    ReleaseKey(up)


def turn_right(t):
    PressKey(right)
    time.sleep(t)
    ReleaseKey(right)


def F_go():
    PressKey(F)
    time.sleep(0.5)
    ReleaseKey(F)


def forward_jump(t):
    PressKey(W)
    time.sleep(t)
    Simulate_Keyboard.press_key('spacebar')
    Simulate_Keyboard.release_key('spacebar')
    ReleaseKey(K)


def press_esc():
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)


# def dead():
#     PressKey(M)
#     time.sleep(0.5)
#     ReleaseKey(M)


