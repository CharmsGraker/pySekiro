from win32_utils.key_mapping import *

action_table = [
    nonAction,  # no action
    attack,
    defense,

    go_forward,
    go_back,
    go_left,
    go_right,

    recover,
    roll_up,  # equal to dodge
    jump,

    roll_forward,
    roll_back,
    roll_left,
    roll_right,

    jump_forward,
    jump_back,
    jump_left,
    jump_right
]

# action2label = {
#     "nonAction": 0,  # no action
#     "attack": 1,
#     "defense": 2,
#     "jump": 3,
#     "roll_up": 4,  # equal to dodge
#     "go_forward": 5,
#     "go_back": 6,
#     "go_left": 7,
#     "go_right": 8,
#     "roll_left": 9,
#     "roll_right": 10,
#     "roll_back": 11,
#     "roll_forward": 12
# }


def take_action(action):
    """
    :param action: a index
    :return:
    """
    action_table[action]()
