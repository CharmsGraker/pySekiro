import numpy as np

sampled_boss_weak = np.array([(7, 122, 157),
                              (6, 122, 157)])

sampled_boss_mid = np.array([
    [17, 122, 152],
    [15, 122, 153],
    [12, 122, 155],
    [13, 122, 154]])

sampled_boss_little_high = np.array([[18, 119, 147],
                                     [18, 122, 152]])

sampled_boss_high = np.array([[23, 122, 150]])
# blooded strong also be this
sampled_boss_little_strong = np.array([(0, 116, 162),
                                       (0, 116, 167),
                                       (0, 116, 164),
                                       (0, 116, 161)])

sampled_boss_very_strong = np.array([
    [67, 152, 240],
    [62, 152, 242],
    [62, 152, 242],
    [60, 152, 243]])
# [ 29 122 148]

# when boss not bleeding, but accumulate endurance so much
sampled_boss_tough = np.array([[74, 152, 237],
                               [72, 152, 238]])
sampled_boss_bleed_strong = np.array([[0, 106, 176],
                                      [0, 106, 175]])

# when boss bleed
sampled_boss_bleed_very_strong = np.array([
    [30, 146, 249],
    [30, 146, 250],
    [30, 146, 254],
    [30, 146, 252]])

sampled_boss_red = np.array([[30, 146, 252], [30, 136, 255]])

sampled_boss_shinobi = np.array([[30, 125, 255]])
# bgr
boss_endurance_toler = {
    "weak": np.mean(sampled_boss_weak, axis=0),
    'mid': np.mean(sampled_boss_mid, axis=0),
    'little_high': np.mean(sampled_boss_little_high, axis=0),
    'high': np.mean(sampled_boss_high, axis=0),
    'little_strong': np.mean(sampled_boss_little_strong, axis=0),
    "very_strong": np.mean(sampled_boss_very_strong, axis=0),
    'tough': np.mean(sampled_boss_tough, axis=0),
    'bleed_strong': np.mean(sampled_boss_bleed_strong, axis=0),
    'bleed_very_strong': np.mean(sampled_boss_bleed_very_strong, axis=0),
    'red': np.mean(sampled_boss_red, axis=0),
    'shinobi': np.mean(sampled_boss_shinobi, axis=0)

}