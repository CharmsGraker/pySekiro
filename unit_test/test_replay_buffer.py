import cv2

from framework.replay_memory import ReplayMemory
from framework.environment import SekiroEnv
import numpy as np


def ReplayBufferTest(batch_size=32):
    env = SekiroEnv()
    obs = env.reset()
    buffer = ReplayMemory(max_size=100)
    for i in range(32):
        action = np.random.random(5)
        next_obs, reward, done = env.step(action)
        buffer.append((obs, action, reward, next_obs, done))

    all_batch = buffer.sample(batch_size)
    print(all_batch[0].shape)


if __name__ == '__main__':
    ReplayBufferTest()
    cv2.destroyAllWindows()