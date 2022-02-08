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


def dodge_right():  # 闪避
    # PressKey(Q)
    Simulate_Keyboard.press_key('alt')
    Simulate_Keyboard.press_key('d')

    time.sleep(0.1)
    Simulate_Keyboard.release_key('d')

    Simulate_Keyboard.release_key('alt')
    # time.sleep(0.1)


def dodge_left():  # 闪避
    # PressKey(Q)
    Simulate_Keyboard.press_key('alt')
    Simulate_Keyboard.press_key('a')

    time.sleep(0.1)
    Simulate_Keyboard.release_key('a')

    Simulate_Keyboard.release_key('alt')
    # time.sleep(0.1)


def dodge_back():  # 闪避
    # PressKey(Q)
    Simulate_Keyboard.press_key('alt')
    Simulate_Keyboard.release_key('s')

    time.sleep(0.1)
    Simulate_Keyboard.release_key('s')
    Simulate_Keyboard.release_key('alt')
    # time.sleep(0.1)


def dodge_forward():  # 闪避
    # PressKey(Q)
    Simulate_Keyboard.press_key('alt')
    Simulate_Keyboard.release_key('w')

    time.sleep(0.1)
    Simulate_Keyboard.release_key('w')
    Simulate_Keyboard.release_key('alt')
    # time.sleep(0.1)


def lock_vision():
    PressKey(V)
    time.sleep(0.3)
    ReleaseKey(V)
    time.sleep(0.1)


def recover():
    Simulate_Keyboard.press_key("r")
    time.sleep(0.3)
    Simulate_Keyboard.release_key("r")


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


# comb
def jump_forward(t):
    PressKey(W)
    time.sleep(t)
    Simulate_Keyboard.press_key('spacebar')
    Simulate_Keyboard.release_key('spacebar')
    ReleaseKey(W)


def jump_back(t):
    PressKey(S)
    time.sleep(t)
    Simulate_Keyboard.press_key('spacebar')
    Simulate_Keyboard.release_key('spacebar')
    ReleaseKey(S)


def jump_left(t):
    PressKey(A)
    time.sleep(t)
    Simulate_Keyboard.press_key('spacebar')
    Simulate_Keyboard.release_key('spacebar')
    ReleaseKey(A)


def jump_right(t):
    PressKey(D)
    time.sleep(t)
    Simulate_Keyboard.press_key('spacebar')
    Simulate_Keyboard.release_key('spacebar')
    ReleaseKey(D)


def press_esc():
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)


def nonAction():
    pass


from win32_utils.keyboard_simulation import Simulate_Keyboard


def interrupt_game():
    Simulate_Keyboard.press_key('esc')
    return 1


def continue_game(interrupt_state):
    if interrupt_state != 0:
        Simulate_Keyboard.release_key('esc')
        return 0
    return interrupt_state

# def dead():
#     PressKey(M)
#     time.sleep(0.5)
#     ReleaseKey(M)
