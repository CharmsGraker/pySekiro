from framework.environment import SekiroEnv
from utils.count_down import CountDown

if __name__ == '__main__':
    env = SekiroEnv(disable_resize=True)
    obs = env.reset()
    action = 0
    CountDown(3)
    while True:
        next_obs, reward, done = env.step(action, log_info=True)
