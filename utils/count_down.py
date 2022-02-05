import time


def CountDown(tik):
    for i in range(tik)[::-1]:
        print(i+1)
        time.sleep(1)
