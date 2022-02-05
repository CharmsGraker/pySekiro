# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 19:56:38 2021

@author: pang
"""

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
import random
from collections import deque
import numpy as np
import os

# experiences replay buffer size
REPLAY_SIZE = 2000
# memory size 1000
# size of minibatch
small_BATCH_SIZE = 16
big_BATCH_SIZE = 128
BATCH_SIZE_door = 1000

# these are the hyper Parameters for DQN
# discount factor for target Q to caculate the TD aim value
GAMMA = 0.9
# the start value of epsilon E
INITIAL_EPSILON = 0.5
# the final value of epsilon
FINAL_EPSILON = 0.01


class DQNall():
    def __init__(self, algorithm, observation_width, observation_height, action_space, model_file, log_file):
        # the state is the input vector of network, in this env, it has four dimensions
        self.algorithm = algorithm

        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        # the action is the output vector and it has two dimensions
        # self.action_dim = action_space
        # init experience replay, the deque is a list that first-in & first-out
        # self.replay_buffer = deque()
        # you can create the network by the two parameters
        # self.model = create_Q_network()
        # after create the network, we can define the training methods
        # self.create_updating_method()
        # set the value in choose_action
        self.model_path = model_file + "/save_model.ckpt"
        self.model_file = model_file
        self.log_file = log_file
        # 因为保存的模型名字不太一样，只能检查路径是否存在
        # Init session
        self.session = tf.InteractiveSession()
        if os.path.exists(self.model_file):
            print("model exists , load model\n")
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, self.model_path)
        else:
            print("model don't exists , create new one\n")
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
        # init
        # 只有把框图保存到文件中，才能加载到浏览器中观看
        self.writer = tf.summary.FileWriter(self.log_file, self.session.graph)

        ####### 路径中不要有中文字符，否则加载不进来 ###########
        # tensorboard --logdir=logs_gpu --host=127.0.0.1
        self.merged = tf.summary.merge_all()
        # 把所有summary合并在一起，就是把所有loss,w,b这些的数据打包
        # 注意merged也需要被sess.run才能发挥作用



    # the function to create the network
    # there are two networks, the one is action_value and the other is target_action_value
    # these two networks has same architecture

    # this the function that define the method to update the current_net's parameters
    # def create_updating_method(self):

    # # this is the function that use the network output the action
    # def Choose_Action(self, state):
    #     # the output is a tensor, so the [0] is to get the output as a list
    #     Q_value = self.Q_value.eval(feed_dict={
    #         self.state_input: [state]
    #     })[0]
    #     # use epsilon greedy to get the action
    #     if random.random() <= self.epsilon:
    #         # if lower than epsilon, give a random value
    #         self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
    #         return random.randint(0, self.action_dim - 1)
    #     else:
    #         # if bigger than epsilon, give the argmax value
    #         self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
    #         return np.argmax(Q_value)

    # this the function that store the data in replay memory
    def Store_Data(self, state, action, reward, next_state, done):
        # generate a list with all 0,and set the action is 1
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        # store all the elements
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        # if the length of replay_buffer is bigger than REPLAY_SIZE
        # delete the left value, make the len is stable
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
            # update replay_buffer

    # train the network, update the parameters of Q_value
    def Train_Network(self, BATCH_SIZE, num_step):
        # Step 1: obtain random minibatch from replay memory
        # 从记忆库中采样BATCH_SIZE
        # state_batch, action_batch, reward_batch, next_state_batch,done_batch = replay_buffer.sample(BATCH_SIZE)

        # Step 2: calculate TD aim value
        y_batch = []
        # give the next_state_batch flow to target_Q_value and caculate the next state's Q_value
        Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # caculate the TD aim value by the formulate
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            # see if the station is the final station
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        # step 3: update the network
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            # y即为更新后的Q值,与Q_action构成损失函数更新网络
            self.action_input: action_batch,
            self.state_input: state_batch
        })

        if num_step % 100 == 0:
            # save loss graph
            result = self.session.run(self.merged, feed_dict={
                self.y_input: y_batch,
                self.action_input: action_batch,
                self.state_input: state_batch
            })
            # 把merged的数据放进writer中才能画图
            self.writer.add_summary(result, num_step)

    # def Update_Target_Network(self):
    #     # update target Q netowrk
    #

    # use for test
    def action(self, state):
        return np.argmax(self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0])

    def save_model(self):
        self.save_path = self.saver.save(self.session, self.model_path)
        print("Save to path:", self.save_path)
