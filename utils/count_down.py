import time

from settings import pause_key
from win32_utils.getkeys import key_check


def CountDown(tik):
    for i in range(tik)[::-1]:
        print(i+1)
        time.sleep(1)


def pause(sec=0.2):
    time.sleep(0.05)
    while True:
        if pause_key in key_check():
            break
        time.sleep(sec)

    time.sleep(0.05)
