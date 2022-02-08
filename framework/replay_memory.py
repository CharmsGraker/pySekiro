#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modified from https://github.com/seungeunrho/minimalRL/blob/master/dqn.py

import random
import collections
import numpy as np
from tqdm import tqdm

from settings import logger


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = None
        if isinstance(self.buffer, collections.deque):
            mini_batch = random.sample(self.buffer, batch_size)
        elif isinstance(self.buffer, np.ndarray):
            rand_idx = random.sample(range(self.buffer.shape[0]), batch_size)
            mini_batch = self.buffer[rand_idx]

        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for idx, experience in enumerate(mini_batch):
            s, a, r, s_p, done = experience
            if idx == 0:
                obs_batch = s[np.newaxis, :, :, :]
            else:
                obs_batch = np.concatenate([obs_batch, s[np.newaxis, :, :, :]], axis=0)
                # obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch), \
               np.array(action_batch), np.array(reward_batch), \
               np.array(next_obs_batch), np.array(done_batch)

    def __len__(self):
        return len(self.buffer)

    def loadFrom(self, npy_path):
        try:
            self.buffer = np.load(npy_path, allow_pickle=True)
            state_imgs = np.array(list(self.buffer[:, 0]))
            global VGG_MEAN
            VGG_MEAN = get_mean(state_imgs)
            logger.info("success load history")
            self.use_history = True
        except Exception as e:
            print(e)
            self.use_history = False


def separate_channel(img):
    """
    matplotlib 色彩空间是RGB
    存储数据时是BGR
    :param img:
    :return:
    """
    assert len(img.shape) == 3
    B = img[..., 0].mean()
    G = img[..., 1].mean()
    R = img[..., 2].mean()
    return B, G, R


def get_mean(img_arrays):
    """
        请进入tools目录下运行
    :param dataset_name:
    :return:
    """
    # 如果是以进入tools目录运行 则os.getcwd() 为 D:\Gary\Program\GitHubProject\AnimeGAN\tools
    # print('os.pardir: ', os.pardir) # 受启动方式有很大影响

    # 用于查找
    image_num = len(img_arrays)
    print('image_num:', image_num)

    B_total = 0
    G_total = 0
    R_total = 0
    for f in tqdm(img_arrays):
        bgr = separate_channel(f)
        B_total += bgr[0]
        G_total += bgr[1]
        R_total += bgr[2]

    B_mean, G_mean, R_mean = B_total / image_num, G_total / image_num, R_total / image_num
    mean = (B_mean + G_mean + R_mean) / 3

    return mean - B_mean, mean - G_mean, mean - R_mean
