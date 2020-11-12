from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))
init_head = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class Actor(nn.Module):
    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        # obs = obs / 255.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class CNNCategoricalActor(Actor):
    def __init__(self, obs_shape, act_dim, params):
        super().__init__()
        # import pdb; pdb.set_trace()
        self.input_shape = obs_shape
        if params["render_mode"]=="rgb_array":
            self.preprocess = nn.Sequential(init_relu_(nn.Conv2d(3, 32, 16, stride= 8)),
                                            nn.ReLU(),
                                            init_relu_(nn.Conv2d(32, 64, 4, stride = 2)), nn.ReLU(),
                                            nn.ReLU()).to(device)
        else:

            self.preprocess = nn.Sequential(init_relu_(nn.Conv2d(3, 64, 3)),
                                            nn.ReLU()).to(device)

        self.convnet = nn.Sequential(init_relu_(nn.Conv2d(64, 64, 3)), nn.ReLU(), Flatten()).to(device)

        nets = [
        init_relu_(nn.Linear(self.feature_size(), 512)), nn.ReLU()]
        # for _ in range(params["nb_hidden"]):
        #     nets.append(init_relu_(nn.Linear(params["hidden_dim"], params["hidden_dim"])))
        #     nets.append(nn.ReLU())
        nets.append(init_head(nn.Linear(512, act_dim)))

        self.logits_net = nn.Sequential(*nets).to(device)
        # .to(
        #     params["device"])
        # self.features = nn.Sequential(
        #     nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     nn.ReLU()
        # )
        #
        # self.logits_net = nn.Sequential(
        #     nn.Linear(self.feature_size(), 512),
        #     nn.ReLU(),
        #     nn.Linear(512, act_dim)
        # )

    def _distribution(self, obs):
        obs=obs/255.

        obs=self.preprocess(obs)
        obs = self.convnet(obs)

        # import pdb; pdb.set_trace()
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _probs(self, obs):
        obs = obs / 255.

        obs = self.preprocess(obs)
        obs = self.convnet(obs)

        # import pdb; pdb.set_trace()
        logits = self.logits_net(obs)
        pi = Categorical(logits=logits)
        return pi.probs

    def feature_sizeold(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)

    def feature_size(self):
        # import pdb; pdb.set_trace()
        return self.convnet(self.preprocess(torch.zeros(1, *self.input_shape).to(device).permute(0,3,1,2))).size(1)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act)

class CNNCritic(nn.Module):
    def __init__(self, obs_dim,params):
        super().__init__()

        self.input_shape = obs_dim
        if params["render_mode"]=="rgb_array":
            self.preprocess = nn.Sequential(init_relu_(nn.Conv2d(3, 32, 16, stride= 8)),
                                            nn.ReLU(),
                                            init_relu_(nn.Conv2d(32, 64, 4, stride = 2)), nn.ReLU(),
                                            nn.ReLU()).to(device)
        else:

            self.preprocess = nn.Sequential(init_relu_(nn.Conv2d(3, 64, 3)),
                                            nn.ReLU()).to(device)

        self.convnet = nn.Sequential(init_relu_(nn.Conv2d(64, 64, 3)), nn.ReLU(), Flatten()).to(device)

        nets = [
        init_relu_(nn.Linear(self.feature_size(), 512)), nn.ReLU()]
        # for _ in range(params["nb_hidden"]):
        #     nets.append(init_relu_(nn.Linear(params["hidden_dim"], params["hidden_dim"])))
        #     nets.append(nn.ReLU())
        nets.append(init_head(nn.Linear(512, 1)))

        self.v_net = nn.Sequential(*nets).to(device)

    def feature_size(self):
        return self.convnet(self.preprocess(torch.zeros(1, *self.input_shape).to(device).permute(0,3,1,2))).size(1)

    def forward(self, obs):
        obs=obs/255.

        obs = self.preprocess(obs)
        obs = self.convnet(obs)

        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, ac_kwargs):
        super().__init__()
        if ac_kwargs["render_mode"]=="rgb_array":
            obs_dim = observation_space.shape
        else:
            obs_dim = (10, 10, 3)

        self.pi = CNNCategoricalActor(obs_dim, action_space.n, ac_kwargs).to(device)

        # build value function
        self.v = CNNCritic(obs_dim, ac_kwargs).to(device)

    def step(self, obs):
        # obs=obs/255.
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def get_prob(self, obs):
        with torch.no_grad():
            return self.pi._probs(obs)

    def act(self, obs):
        # obs=obs/255.
        return self.step(obs)[0]
