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
            logger.info("success load history")
            self.use_history = True
        except Exception as e:
            print(e)
            self.use_history = False
