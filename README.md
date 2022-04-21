# option_critic_4atari

Implement option critic for atari in pytorch.

There are **two** kinds of files.
**for one** 
* _main.py_
* _option_critic.py_

**for two**
* _main_debug.py_
* _option_critic_debug.py_

Some differences:(e.g.)
* optim
* critic_loss_fn
* actor_loss_fn
* net structure of option_critic

requirements
* pytorch >= 1.6
* atari_py = 1.2.2
* gym = 0.21.0
* ale-py = 0.7.3
* tensorboard = 2.6.0
* tqdm = 4.62.3
* opencv-python = 4.5.4.60

you can just:
> python main.py or main_debug.py

you can also select to change some default params(e.g.)
> python main_debug.py --num-options 8 --actor_lr 0.0001 --critic_lr 0.0000625