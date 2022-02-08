from utils.config import Config

DQN_model_path = "model_gpu"
DQN_log_path = "logs_gpu/"
WIDTH = 96
HEIGHT = 88

obs_dim = Config({"state_w": 488,
                  "state_h": 385,
                  "state_d": 3})

act_dim = 18

# the game window border
border_offset = 30
border_starter = 25

window_size = (320, border_starter + border_offset, 704, 542)  # 384,352  192,176 96,88 48,44 24,22
# station window_size
# game_window = (0, 0, 1024, 576)


window_mid = (window_size[0] + window_size[2]) // 2
endurance_len = 200

endurance_window = (window_mid - endurance_len,
                    window_size[1],
                    window_mid + endurance_len,
                    int(2 * border_starter) + border_offset)

# used to get boss and self blood

action_size = 5
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

EPISODES = 3000
big_BATCH_SIZE = 16
UPDATE_STEP = 50
# times that evaluate the network
num_step = 0
# used to save log graph
target_step = 0
# used to update target Q network
paused = True
# used to stop training


# training settings
LEARN_FREQ = 5  # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 1800  # 2000  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
BATCH_SIZE = 80  # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.0005  # 学习率
GAMMA = 0.99  # reward 的衰减因子，一般取 0.9 到 0.999 不等

pause_key = 'T'

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sample_data_save_root = r'E:\GrakerDatasets\Sekiro_RL_dataset'
