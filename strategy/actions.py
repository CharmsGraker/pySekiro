from win32_utils.key_mapping import attack, jump, defense, dodge


def take_action(action):
    if action == 0:  # no_choose
        pass
    elif action == 1:  # j
        attack()
    elif action == 2:  # k
        jump()
    elif action == 3:  # m
        defense()
    elif action == 4:  # r
        dodge()
