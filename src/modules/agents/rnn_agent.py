import torch.nn as nn
import torch.nn.functional as F
import torch as th
from . import REGISTRY
from collections import OrderedDict

class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        if isinstance(input_shape, tuple):
            assert len(input_shape) == 1, "Input shape has unsupported dimensionality: {}".format(input_shape)
            input_shape = input_shape[0]
        elif isinstance(input_shape, (dict, OrderedDict)):  # assemble all 1d input regions
            #input_shape = sum([v[0] for v in input_shape.values() if len(v) == 1])
            input_shape = input_shape["1d"][0]

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if isinstance(inputs, OrderedDict):
            # inputs = th.cat(list(inputs.values()), -1)
            inputs = inputs["1d"]

        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class RNNInputActionAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNInputActionAgent, self).__init__()
        self.args = args
        self.input_shape = input_shape
        if isinstance(input_shape, tuple):
            assert len(input_shape) == 1, "Input shape has unsupported dimensionality: {}".format(input_shape)
            input_shape = input_shape[0]
        elif isinstance(input_shape, (dict, OrderedDict)):  # assemble all 1d input regions
            # input_shape = sum([v[0] for v in input_shape.values() if len(v) == 1])
            input_shape = input_shape["1d"][0]
        self.fc1 = nn.Linear(input_shape+self.args.n_actions, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if isinstance(inputs, OrderedDict):
            # inputs = th.cat(list(inputs.values()), -1)
            inputs = inputs["1d"]
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

# From: https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_bodies.py
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class RNNConvDDPGAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNConvDDPGAgent, self).__init__()
        self.args = args

        # self.conv1 = EncoderResNet(Bottleneck, [3, 4, 6, 3])

        in_channels, n_dim_x, n_dim_y = input_shape["2d"]

        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

        # find output dim
        dummy_th = th.zeros(1, *input_shape["2d"]).to(next(self.parameters()).device)
        out = self.conv2(self.conv1(dummy_th))

        self.fc1 = nn.Linear(input_shape["1d"][0] + out.shape[-3] * out.shape[-2] * out.shape[-1], args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        y = F.elu(self.conv1(inputs["2d"]))
        y = F.elu(self.conv2(y))
        x = F.relu(self.fc1(th.cat([inputs["1d"], y.view(y.shape[0], -1)],
                                   dim=1)))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h

class RNNConvDDPGInputGridAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNConvDDPGInputGridAgent, self).__init__()
        self.args = args

        # self.conv1 = EncoderResNet(Bottleneck, [3, 4, 6, 3])

        in_channels, n_dim_x, n_dim_y = input_shape["2d"]
        in_channels += 2 # action channels

        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

        # find output dim
        dummy_th = th.zeros(1, in_channels, *input_shape["2d"][1:]).to(next(self.parameters()).device)
        out = self.conv2(self.conv1(dummy_th))

        self.fc1 = nn.Linear(input_shape["1d"][0] + out.shape[-3] * out.shape[-2] * out.shape[-1], args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        y = F.elu(self.conv1(th.cat([inputs["2d"], inputs["actions_2d"]], dim=-3)))
        y = F.elu(self.conv2(y))
        x = F.relu(self.fc1(th.cat([inputs["1d"], y.view(y.shape[0], -1)],
                                   dim=1)))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        q = self.fc2(h)
        return q, h

#Needs action_input_representation=InputFlat, obs_input_representation=Grid
class MathiasAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MathiasAgent, self).__init__()
        self.args = args

        # self.conv1 = EncoderResNet(Bottleneck, [3, 4, 6, 3])

        in_channels, n_dim_x, n_dim_y = input_shape["2d"]

        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

        # find output dim
        dummy_th = th.zeros(1, in_channels, *input_shape["2d"][1:]).to(next(self.parameters()).device)
        out = self.conv2(self.conv1(dummy_th))

        self.fc1 = nn.Linear(input_shape["1d"][0] + self.args.n_actions + out.shape[-3] * out.shape[-2] * out.shape[-1], args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        y = F.elu(self.conv1(inputs["2d"]))
        y = F.elu(self.conv2(y))
        inp = th.cat([inputs["1d"], y.view(y.shape[0], -1)],
               dim=1)
        x = F.relu(self.fc1(inp))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        q = self.fc2(h)
        return q, h




class RNNConvDDPGInputGridNoIDAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNConvDDPGInputGridAgent, self).__init__()
        self.args = args

        # self.conv1 = EncoderResNet(Bottleneck, [3, 4, 6, 3])

        in_channels, n_dim_x, n_dim_y = input_shape["2d"]
        in_channels += 2 # action channels

        self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=2))
        self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))

        # find output dim
        dummy_th = th.zeros(1, in_channels, *input_shape["2d"][1:]).to(next(self.parameters()).device)
        out = self.conv2(self.conv1(dummy_th))

        self.fc1 = nn.Linear(out.shape[-3] * out.shape[-2] * out.shape[-1], args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        y = F.elu(self.conv1(th.cat([inputs["2d"], inputs["actions_2d"]], dim=-3)))
        y = F.elu(self.conv2(y))
        x = F.relu(self.fc1(y.view(y.shape[0], -1)))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)

        q = self.fc2(h)
        return q, h


#Gudrun = RNNAgent
#Hannelore = RNNInputAction
#Julia = DDPG1
#Cameron = DDPGInputGrid
#Mathias = MathiasAgent

