import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli

from math import exp
import numpy as np

from wrappers import to_tensor


class OptionCriticConv(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                rng = np.random.seed(0),
                device='cpu',
                testing=False):

        super(OptionCriticConv, self).__init__()

        self.in_channels = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.magic_number = 7 * 7 * 64
        self.rng = rng
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.modules.Flatten(),
            nn.Linear(self.magic_number, 512),
            nn.ReLU()
        )


        self.Q            = nn.Linear(512, num_options)                 # Policy-Over-Options
        self.terminations = nn.Linear(512, num_options)                 # Option-Termination
        self.options_W = nn.Parameter(torch.Tensor(num_options, 512, num_actions).uniform_(-1,1))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))
        # self.option_layer = nn.ModuleList([nn.Linear(512, num_actions) for _ in range(num_options)])

        self.to(device)
        self.train(not testing)

    def weight_init(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            # nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)
        elif classname.find('Linear') != -1:
            # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.xavier_normal_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.1)
            # nn.init.xavier_norrmal_(m.weight)
            # nn.init.constant_(m.bias, 0.0)

    def get_state(self, obs, grid_update=False):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        if grid_update:
            state = self.features(obs)
        else:
            with torch.no_grad():
                state = self.features(obs)
        return state

    def get_Q(self, state, grid_update=False):
        if grid_update:
            Q = self.Q(state)
        else:
            with torch.no_grad():
                Q = self.Q(state)
        return Q
    
    def predict_option_termination(self, state, current_option, grid_update=False):
        if grid_update:
            termination = self.terminations(state)[:, current_option].sigmoid()
        else:
            with torch.no_grad():
                termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        return bool(option_termination.item())
    
    def get_terminations(self, state):
        termination = self.terminations(state).sigmoid() 
        return termination

    def get_action(self, state, option, epsilon):
        if self.testing or self.rng.random() > epsilon:
            with torch.no_grad():
                logits = state @ self.options_W[option] + self.options_b[option]
                action = logits.max(1).indices.item()
                return action
        return self.rng.randint(0, self.num_actions-1)

    def get_action_(self, state, option):
        logits = state @ self.options_W[option] + self.options_b[option]
        prob = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(prob)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()
        return action.item(), logp, entropy

    
    def greedy_option(self, state, grid_update=False):
        if grid_update:
            Q = self.get_Q(state)
        else:
            with torch.no_grad():
                Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


class OptionCriticFeatures(nn.Module):
    def __init__(self,
                in_features,
                num_actions,
                num_options,
                temperature=1.0,
                eps_start=1.0,
                eps_min=0.1,
                eps_decay=int(1e6),
                eps_test=0.05,
                device='cpu',
                testing=False):

        super(OptionCriticFeatures, self).__init__()

        self.in_features = in_features
        self.num_actions = num_actions
        self.num_options = num_options
        self.device = device
        self.testing = testing

        self.temperature = temperature
        self.eps_min   = eps_min
        self.eps_start = eps_start
        self.eps_decay = eps_decay
        self.eps_test  = eps_test
        self.num_steps = 0
        
        self.features = nn.Sequential(
            nn.Linear(in_features, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.Q            = nn.Linear(64, num_options)                 # Policy-Over-Options
        self.terminations = nn.Linear(64, num_options)                 # Option-Termination
        self.options_W = nn.Parameter(torch.zeros(num_options, 64, num_actions))
        self.options_b = nn.Parameter(torch.zeros(num_options, num_actions))

        self.to(device)
        self.train(not testing)

    def get_state(self, obs):
        if obs.ndim < 4:
            obs = obs.unsqueeze(0)
        obs = obs.to(self.device)
        state = self.features(obs)
        return state

    def get_Q(self, state):
        return self.Q(state)
    
    def predict_option_termination(self, state, current_option):
        termination = self.terminations(state)[:, current_option].sigmoid()
        option_termination = Bernoulli(termination).sample()
        Q = self.get_Q(state)
        next_option = Q.argmax(dim=-1)
        return bool(option_termination.item()), next_option.item()
    
    def get_terminations(self, state):
        return self.terminations(state).sigmoid() 

    def get_action(self, state, option):
        logits = state @ self.options_W[option] + self.options_b[option]
        action_dist = (logits / self.temperature).softmax(dim=-1)
        action_dist = Categorical(action_dist)

        action = action_dist.sample()
        logp = action_dist.log_prob(action)
        entropy = action_dist.entropy()

        return action.item(), logp, entropy
    
    def greedy_option(self, state):
        Q = self.get_Q(state)
        return Q.argmax(dim=-1).item()

    @property
    def epsilon(self):
        if not self.testing:
            eps = self.eps_min + (self.eps_start - self.eps_min) * exp(-self.num_steps / self.eps_decay)
            self.num_steps += 1
        else:
            eps = self.eps_test
        return eps


def critic_loss(model, model_prime, data_batch, args):
    obs, options, rewards, next_obs, dones = data_batch
    batch_idx = torch.arange(len(options)).long()
    obs       = to_tensor(obs, model.device)
    next_obs  = to_tensor(next_obs, model.device)
    options   = torch.LongTensor(options).to(model.device)
    rewards   = torch.FloatTensor(rewards).to(model.device)
    masks     = 1 - torch.FloatTensor(dones).to(model.device)

    # The loss is the TD loss of Q and the update target, so we need to calculate Q
    states = model.get_state(obs).squeeze(0)
    Q      = model.get_Q(states)
    
    # the update target contains Q_next, but for stable learning we use prime network for this
    next_states_prime = model_prime.get_state(next_obs).squeeze(0)
    next_Q_prime      = model_prime.get_Q(next_states_prime) # detach?

    # Additionally, we need the beta probabilities of the next state
    next_states            = model.get_state(next_obs).squeeze(0)
    next_termination_probs = model.get_terminations(next_states)
    next_options_term_prob = next_termination_probs[batch_idx, options]

    # Now we can calculate the update target gt
    gt = rewards + masks * args.gamma * \
        ((1 - next_options_term_prob) * next_Q_prime[batch_idx, options] + next_options_term_prob  * next_Q_prime.max(dim=-1)[0])

    # to update Q we want to use the actual network, not the prime
    # clip delta
    td_errors = gt.detach() - Q[batch_idx, options]
    quadratic_part = torch.minimum(abs(td_errors),torch.tensor([1]).to(model.device))
    linear_part = abs(td_errors) - quadratic_part
    td_cost = (0.5*quadratic_part.pow(2) + linear_part).mean()
    return td_cost

def actor_loss(obs, option, reward, done, next_obs, model, model_prime, args):
    state = model.get_state(obs).detach()
    next_state = model.get_state(next_obs).detach()
    next_state_prime = model_prime.get_state(next_obs).detach()

    option_term_prob = model.get_terminations(state)[:, option]
    next_option_term_prob = model.get_terminations(next_state)[:, option]

    # calculate for advantage
    Q = model.get_Q(state).detach().squeeze()
    advantage = Q[option] - Q.max(dim=-1)[0]
    
    next_Q_prime = model_prime.get_Q(next_state_prime).detach().squeeze()

    action, logp, entropy = model.get_action_(state, option)

    # Target update gt
    gt = reward + (1 - done) * args.gamma * \
        ((1 - next_option_term_prob) * next_Q_prime[option] + next_option_term_prob  * next_Q_prime.max(dim=-1)[0])

    # The termination loss
    termination_loss = option_term_prob * (advantage + args.termination_reg) * (1 - done)
    
    # actor-critic policy gradient with entropy regularization
    policy_loss = -logp * (gt.detach() - Q[option]) - args.entropy_reg * entropy
    actor_loss = termination_loss + policy_loss
    return actor_loss
