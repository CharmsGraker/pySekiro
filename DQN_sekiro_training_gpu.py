# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 21:10:06 2021

@author: pang
"""

import numpy as np

from win32keys.grabscreen import grab_screen
import cv2
import time
from win32keys.getkeys import key_check
from framework.algorithm import DQN
from restart import restart


def pause_game(paused):
    keys = key_check()
    if 'T' in keys:
        if paused:
            paused = False
            print('start game')
            time.sleep(1)
        else:
            paused = True
            print('pause game')
            time.sleep(1)
    if paused:
        print('paused')
        while True:
            keys = key_check()
            # pauses game and can get annoying.
            if 'T' in keys:
                if paused:
                    paused = False
                    print('start game')
                    time.sleep(1)
                    break
                else:
                    paused = True
                    time.sleep(1)
    return paused




# def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood, stop, emergence_break):
#     # get action reward
#     # emergence_break is used to break down training
#     # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
#     if next_self_blood < 3:     # self dead
#         if emergence_break < 2:
#             reward = -10
#             done = 1
#             stop = 0
#             emergence_break += 1
#             return reward, done, stop, emergence_break
#         else:
#             reward = -10
#             done = 1
#             stop = 0
#             emergence_break = 100
#             return reward, done, stop, emergence_break
#     elif next_boss_blood - boss_blood > 15:   #boss dead
#         if emergence_break < 2:
#             reward = 20
#             done = 0
#             stop = 0
#             emergence_break += 1
#             return reward, done, stop, emergence_break
#         else:
#             reward = 20
#             done = 0
#             stop = 0
#             emergence_break = 100
#             return reward, done, stop, emergence_break
#     else:
#         self_blood_reward = 0
#         boss_blood_reward = 0
#         # print(next_self_blood - self_blood)
#         # print(next_boss_blood - boss_blood)
#         if next_self_blood - self_blood < -7:
#             if stop == 0:
#                 self_blood_reward = -6
#                 stop = 1
#                 # 防止连续取帧时一直计算掉血
#         else:
#             stop = 0
#         if next_boss_blood - boss_blood <= -3:
#             boss_blood_reward = 4
#         # print("self_blood_reward:    ",self_blood_reward)
#         # print("boss_blood_reward:    ",boss_blood_reward)
#         reward = self_blood_reward + boss_blood_reward
#         done = 0
#         emergence_break = 0
#         return reward, done, stop, emergence_break
#


if __name__ == '__main__':
    model = Model.create_Q_network(sess, WIDTH, HEIGHT)
    algorithm = DQN(sess, model)
    agent = DQNall(algorithm, action_size, DQN_model_path, DQN_log_path)
    # DQN init
    paused = pause_game(paused)
    # paused at the begin
    emergence_break = 0     
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    for episode in range(EPISODES):
        screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)
        # collect station gray graph
        blood_window_gray = cv2.cvtColor(grab_screen(blood_window),cv2.COLOR_BGR2GRAY)
        # collect blood gray graph for count self and boss blood
        station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
        # change graph to WIDTH * HEIGHT for station input
        boss_blood = boss_blood_count(blood_window_gray)
        self_blood = self_blood_count(blood_window_gray)
        # count init blood
        target_step = 0
        # used to update target Q network
        done = 0
        total_reward = 0
        stop = 0    
        # 用于防止连续帧重复计算reward
        last_time = time.time()
        while True:
            station = np.array(station).reshape(-1,HEIGHT,WIDTH,1)[0]
            # reshape station for tf input placeholder
            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            target_step += 1
            # get the action by state
            action = agent.Choose_Action(station)
            take_action(action)
            # take station then the station change
            screen_gray = cv2.cvtColor(grab_screen(window_size),cv2.COLOR_BGR2GRAY)
            # collect station gray graph
            blood_window_gray = cv2.cvtColor(grab_screen(blood_window),cv2.COLOR_BGR2GRAY)
            # collect blood gray graph for count self and boss blood
            next_station = cv2.resize(screen_gray,(WIDTH,HEIGHT))
            next_station = np.array(next_station).reshape(-1,HEIGHT,WIDTH,1)[0]
            next_boss_blood = boss_blood_count(blood_window_gray)
            next_self_blood = self_blood_count(blood_window_gray)
            reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
                                                               self_blood, next_self_blood,
                                                               stop, emergence_break)
            # get action reward
            if emergence_break == 100:
                # emergence break , save model and paused
                # 遇到紧急情况，保存数据，并且暂停
                print("emergence_break")
                agent.save_model()
                paused = True
            agent.Store_Data(station, action, reward, next_station, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                # save loss graph
                # print('train')
                agent.Train_Network(big_BATCH_SIZE, num_step)
            if target_step % UPDATE_STEP == 0:
                agent.Update_Target_Network()
                # update target Q network
            station = next_station
            self_blood = next_self_blood
            boss_blood = next_boss_blood
            total_reward += reward
            paused = pause_game(paused)
            if done == 1:
                break
        if episode % 10 == 0:
            agent.save_model()
            # save model
        print('episode: ', episode, 'Evaluation Average Reward:', total_reward/target_step)
        restart()
