from cv2 import moments
import numpy as np
import argparse
import torch
import os
from copy import deepcopy

from option_critic import OptionCriticFeatures, OptionCriticConv
from option_critic import critic_loss as critic_loss_fn
from option_critic import actor_loss as actor_loss_fn

from experience_replay import ReplayBuffer
from wrappers import *
from logger import Logger

import time

parser = argparse.ArgumentParser(description="Option Critic PyTorch")
parser.add_argument('--env', default='BreakoutNoFrameskip-v4', help='ROM to run') # 'Pong-v0'  'BreakoutNoFrameskip-v4'
parser.add_argument('--render', default=False, help='render by using gym.Monitor or not')
parser.add_argument('--actor_lr',type=float, default=.00025, help='Actor Learning rate')
parser.add_argument('--critic_lr',type=float, default=.0000625, help='Critic Learning rate')
parser.add_argument('--termination_lr',type=float, default=.00025, help='Termination Learning rate')
parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
parser.add_argument('--epsilon-start',  type=float, default=1.0, help=('Starting value for epsilon.'))
parser.add_argument('--epsilon-min', type=float, default=.01, help='Minimum epsilon.')
parser.add_argument('--epsilon-decay', type=float, default=1000000, help=('Number of steps to minimum epsilon.'))
parser.add_argument('--optimal-eps',  type=float, default=0.01, help=('Testing epsilon for optimal.'))
parser.add_argument('--max-history', type=int, default=100000, help=('Maximum number of steps stored in replay'))
parser.add_argument('--batch-size', type=int, default=32, help='Batch size.')
parser.add_argument('--freeze-interval', type=int, default=10000, help=('Interval between target freezes.'))
parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
parser.add_argument('--termination-reg', type=float, default=0.01, help=('Regularization to decrease termination prob.'))
parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
parser.add_argument('--num-options', type=int, default=4, help=('Number of options to create.'))
parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
parser.add_argument('--max_steps_total', type=int, default=int(5e6), help='number of maximum steps to take.') # bout 5 million
parser.add_argument('--random_steps', type=int, default=5e4, help='number of random steps to take before training.') # bout 50k
parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
parser.add_argument('--seed', type=int, default=42, help='Random seed for numpy, torch, random.')
parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 1k eps in fourrooms')

def run(args):


    env, is_atari = make_atari(args.env)
    # wrapper
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)

    option_critic = OptionCriticConv if is_atari else OptionCriticFeatures
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    option_critic = option_critic(
        in_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_options=args.num_options,
        temperature=args.temp,
        eps_start=args.epsilon_start,
        eps_min=args.epsilon_min,
        eps_decay=args.epsilon_decay,
        eps_test=args.optimal_eps,
        device=device
    )
    # initialze_weight
    option_critic.apply(option_critic.weight_init)
    # Create a prime network for more stable Q values
    option_critic_prime = deepcopy(option_critic)
  
    # optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.critic_lr)
    optim = torch.optim.Adam(option_critic.parameters(), lr=args.critic_lr, eps=1.5e-4)


    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env.seed(args.seed)

    buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
    logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}")
    save_path =  args.logdir + '/' + f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}-{time.ctime()}"+ '/'
    save_id = 0

    steps = 0 ;
    if args.switch_goal: print(f"Current goal {env.goal}")
    while steps < args.max_steps_total:

        rewards = 0 ; option_lengths = {opt:[] for opt in range(args.num_options)}
        obs = env.reset() # 4*84*84 after wrapper
        state = option_critic.get_state(to_tensor(obs)) # state means after cnn embedding
        greedy_option  = option_critic.greedy_option(state)
        current_option = np.random.choice(args.num_options)


        done = False ; ep_steps = 0 ; option_termination = True ; curr_op_len = 0
        # random action to collect data
        while steps < args.random_steps:
            
            state = option_critic.get_state(to_tensor(obs)) # state means after cnn embedding   
            greedy_option  = option_critic.greedy_option(state)
            current_option = np.random.choice(args.num_options)
            action, logp, entropy = option_critic.get_action(state, current_option)
            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)

            steps += 1
            obs = next_obs
            if done:
                break

        while not done and ep_steps < args.max_steps_ep:

            epsilon = option_critic.epsilon

            if option_termination:
                option_lengths[current_option].append(curr_op_len)
                current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                curr_op_len = 0


            action, logp, entropy = option_critic.get_action(state, current_option)

            next_obs, reward, done, _ = env.step(action)
            buffer.push(obs, current_option, reward, next_obs, done)

            # old_state = state
            state = option_critic.get_state(to_tensor(next_obs)).detach()

            option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)
            rewards += reward

            actor_loss, critic_loss = None, None
            if len(buffer) > args.batch_size:
                actor_loss = actor_loss_fn(obs, current_option, reward, done,\
                            next_obs, option_critic, option_critic_prime, args)
                total_loss = actor_loss

                if steps % args.update_frequency == 0:
                    data_batch = buffer.sample(args.batch_size)
                    critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                    total_loss += critic_loss
                    
                optim.zero_grad()
                total_loss.backward()
                optim.step()

                if steps % args.freeze_interval == 0:
                    option_critic_prime.load_state_dict(option_critic.state_dict())

            # update global steps etc
            steps += 1
            ep_steps += 1
            curr_op_len += 1
            obs = next_obs

            logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)
        if steps >= args.random_steps:
            logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)
        
        if ((steps-args.random_steps)//10000) - save_id > 0:
            save_id += 1
            torch.save(option_critic.state_dict(),save_path+'train_env_%s_steps_%dx10k_lr_%s.pth' %(args.env,save_id,str(args.critic_lr)))

if __name__=="__main__":
    args = parser.parse_args()
    run(args)
