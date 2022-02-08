endu_reward_idc = {
    'weak': 0.3,
    'mid': 1,
    'little_high': 1.5,
    'high': 2.5,
    'little_strong': 4,
    'very_strong': 6,
    'tough': 10,
    'bleed_strong': 9,
    'bleed_very_strong': 10,
    'red': 15,
    'almost_shinobi': 15,
    "shinobi": 30
}


def rewardOfBossEndurance(key):
    try:
        r = endu_reward_idc[key]
    except Exception as e:
        r = 0
    return r
