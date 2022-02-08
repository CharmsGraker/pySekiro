import os.path
from matplotlib import  pyplot as plt
import numpy as np
from settings import sample_data_save_root
filename = 'episode_2022_02_08_22_54-total_step-539.npy'
sample_data_path = os.path.join(sample_data_save_root,filename)
data = np.load(sample_data_path, allow_pickle=True)


