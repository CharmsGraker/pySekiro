使用强化学习玩只狼。<br>
花了两天学习了下PaddlePaddle的强化学习课程， 按照PARL框架（Model,Algorithm,Agent,Environment）对强化学习代码进行重构，尽量将代码解耦。
### Platform
DL框架选择tensorflow<br>（因为对PARL还不熟）
目前代码应该是可以跑通的。因为只是为了测试有无bug，
### Backbone
目前backbone实现的非常简单。强化学习算法采用的是DQN，按键通过pywin32模拟按键实现。
### reward
关于reward的计分规则，参考了Up主蓝魔的实现，以下是他的项目链接：https://github.com/analoganddigital/DQN_play_sekiro/blob/main/README.md
# At Last
欢迎感兴趣的小伙伴，可以选择自己擅长的模块进行创作吧！