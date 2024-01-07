import numpy as np
import torch
import torch.nn as nn


def weight_orthogonal_init(layer, gain=1.0):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        nn.init.constant_(layer.bias, 0)


class DiagonalGaussian(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc_mean = nn.Linear(in_dim, out_dim)
        # weight_orthogonal_init(self.fc_mean, 0.01)
        self.b_logstd = nn.Parameter(torch.zeros(1, out_dim))

    def forward(self, x):
        mean = self.fc_mean(x)
        logstd = torch.zeros_like(mean) + self.b_logstd
        return torch.distributions.Normal(mean, logstd.exp())


class GeneralNet(nn.Module):
    def __init__(self, input_shape):
        super(GeneralNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.AvgPool2d(2, 2)
            # nn.MaxPool2d(2, 2)
        )
        # self.conv.apply(weight_orthogonal_init)

        with torch.no_grad():
            conv_out = np.prod(self.conv(torch.zeros(1, *input_shape)).size()).item()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # self.fc.apply(weight_orthogonal_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


class Actor(nn.Module):
    def __init__(self, input_shape, action_num):
        super(Actor, self).__init__()
        self.general_net = GeneralNet(input_shape)
        self.actor_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            DiagonalGaussian(64, action_num)
        )
        # self.actor_head.apply(weight_orthogonal_init)

    def forward(self, x):
        x = self.general_net(x)
        x = self.actor_head(x)
        return x


class Critic(nn.Module):
    def __init__(self, input_shape):
        super(Critic, self).__init__()
        self.general_net = GeneralNet(input_shape)
        self.critic_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # self.critic_head.apply(weight_orthogonal_init)

    def forward(self, x):
        x = self.general_net(x)
        x = self.critic_head(x)
        return x


class ActorCritic(nn.Module):
    # For seperate version
    # def __init__(self, input_shape, action_num, device):
    #     super(ActorCritic, self).__init__()
    #     self.actor_head = Actor(input_shape, action_num).to(device)
    #     self.critic_head = Critic(input_shape).to(device)

    def __init__(self, input_shape, action_num, device):
        super(ActorCritic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            # nn.BatchNorm2d(64)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            # nn.BatchNorm2d(64)
            nn.ReLU(),
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.BatchNorm2d(128),
            # nn.ReLU(),
            # nn.AvgPool2d(2, 2)
            # nn.MaxPool2d(2, 2)
        )
        # self.conv.apply(weight_orthogonal_init)

        with torch.no_grad():
            conv_out = np.prod(self.conv(torch.zeros(1, *input_shape)).size()).item()

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )
        # self.fc.apply(weight_orthogonal_init)

        self.actor_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            DiagonalGaussian(64, action_num),
        )
        # self.actor_head.apply(weight_orthogonal_init)

        self.critic_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        # self.actor_head.apply(weight_orthogonal_init)

        self.to(device)

    # def forward(self, state):
    #     dist = self.actor_head(state)
    #     action = dist.sample()
    #     action_prob = dist.log_prob(action).sum(-1)
    #     state_value = self.critic_head(state)

    #     action = action.detach().cpu().numpy()[0]
    #     action_prob = action_prob.detach().cpu().numpy()
    #     state_value = state_value.detach().cpu().numpy()

    #     return action, action_prob, state_value[:, 0]
    
    def forward(self, state):
        state = self.conv(state)
        state = self.fc(state)
        dist = self.actor_head(state)
        action = dist.mean
        action_prob = dist.log_prob(action).sum(-1)
        state_value = self.critic_head(state)

        action = action.detach().cpu().numpy()[0]
        action_prob = action_prob.detach().cpu().numpy()
        state_value = state_value.detach().cpu().numpy()

        return action, action_prob, state_value[:, 0]
    
    # def evaluate(self, state, action):
    #     dist = self.actor_head(state)
    #     action_prob = dist.log_prob(action).sum(-1)
    #     entropy = dist.entropy().sum(-1)
    #     state_value = self.critic_head(state)
    #     return action_prob, entropy, state_value[:, 0]

    def evaluate(self, state, action):
        state = self.conv(state)
        state = self.fc(state)
        dist = self.actor_head(state)
        action_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        state_value = self.critic_head(state)
        return action_prob, entropy, state_value[:, 0]
    