import torch.nn as nn
import torch.nn.functional as F
import torch as th

# From: https://github.com/ShangtongZhang/DeepRL/blob/master/deep_rl/network/network_bodies.py
def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer

class ConvDDPGInputGridDeepAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ConvDDPGInputGridDeepAgent, self).__init__()
        self.args = args

        # self.conv1 = EncoderResNet(Bottleneck, [3, 4, 6, 3])

        in_channels, n_dim_x, n_dim_y = input_shape["2d"]
        in_channels += 2 # action channels
        if(int(self.args.env_args["obs_grid_shape"].split("x")[0]) < 12):
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=2, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=2))
        else:
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))
        # find output dim
        dummy_th = th.zeros(1, in_channels, *input_shape["2d"][1:]).to(next(self.parameters()).device)
        out = self.conv2(self.conv1(dummy_th))

        self.fc1 = nn.Linear(input_shape["1d"][0] + out.shape[-3] * out.shape[-2] * out.shape[-1], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def forward(self, inputs, hidden_state):
        y = th.cat([inputs["2d"], inputs["actions_2d"]], dim=-3)
        y = F.elu(self.conv1(y))
        y = F.elu(self.conv2(y))
        y = th.cat([inputs["1d"], y.view(y.shape[0], -1)], dim=1)
        y = F.relu(self.fc1(y))
        h = F.relu(self.fc2(y))
        y = self.fc3(h)
        th.cuda.empty_cache()
        return y, h

class ConvDDPGInputGridShallowAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ConvDDPGInputGridShallowAgent, self).__init__()
        self.args = args

        # self.conv1 = EncoderResNet(Bottleneck, [3, 4, 6, 3])

        in_channels, n_dim_x, n_dim_y = input_shape["2d"]
        in_channels += 2 # action channels
        if(int(self.args.env_args["obs_grid_shape"].split("x")[0]) < 12):
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=2, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=2))
        else:
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))
        # find output dim
        dummy_th = th.zeros(1, in_channels, *input_shape["2d"][1:]).to(next(self.parameters()).device)
        out = self.conv2(self.conv1(dummy_th))

        self.fc1 = nn.Linear(input_shape["1d"][0] + out.shape[-3] * out.shape[-2] * out.shape[-1], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def forward(self, inputs, hidden_state):
        y = th.cat([inputs["2d"], inputs["actions_2d"]], dim=-3)
        y = F.elu(self.conv1(y))
        y = F.elu(self.conv2(y))
        y = th.cat([inputs["1d"], y.view(y.shape[0], -1)], dim=1)
        h = F.relu(self.fc1(y))
        y = self.fc2(h)
        th.cuda.empty_cache()
        return y, h

class ConvDDPGInputGridDeepNoIDAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ConvDDPGInputGridDeepNoIDAgent, self).__init__()
        self.args = args

        # self.conv1 = EncoderResNet(Bottleneck, [3, 4, 6, 3])

        in_channels, n_dim_x, n_dim_y = input_shape["2d"]
        in_channels += 2 # action channels
        if(int(self.args.env_args["obs_grid_shape"].split("x")[0]) < 12):
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=2, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=2))
        else:
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))
        # find output dim
        dummy_th = th.zeros(1, in_channels, *input_shape["2d"][1:]).to(next(self.parameters()).device)
        out = self.conv2(self.conv1(dummy_th))

        self.fc1 = nn.Linear(input_shape["1d"][0] + out.shape[-3] * out.shape[-2] * out.shape[-1], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def forward(self, inputs, hidden_state):
        y = th.cat([inputs["2d"], inputs["actions_2d"]], dim=-3)
        y = F.elu(self.conv1(y))
        y = F.elu(self.conv2(y))
        y = y.view(y.shape[0], -1)
        y = F.relu(self.fc1(y))
        h = F.relu(self.fc2(y))
        y = self.fc3(h)
        th.cuda.empty_cache()
        return y, h

class ConvDDPGInputGridShallowNoIDAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ConvDDPGInputGridShallowNoIDAgent, self).__init__()
        self.args = args

        # self.conv1 = EncoderResNet(Bottleneck, [3, 4, 6, 3])

        in_channels, n_dim_x, n_dim_y = input_shape["2d"]
        in_channels += 2 # action channels
        if(int(self.args.env_args["obs_grid_shape"].split("x")[0]) < 12):
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=2, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=2))
        else:
            self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=3, stride=3))
            self.conv2 = layer_init(nn.Conv2d(32, 32, kernel_size=3))
        # find output dim
        dummy_th = th.zeros(1, in_channels, *input_shape["2d"][1:]).to(next(self.parameters()).device)
        out = self.conv2(self.conv1(dummy_th))

        self.fc1 = nn.Linear(input_shape["1d"][0] + out.shape[-3] * out.shape[-2] * out.shape[-1], args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()


    def forward(self, inputs, hidden_state):
        y = th.cat([inputs["2d"], inputs["actions_2d"]], dim=-3)
        y = F.elu(self.conv1(y))
        y = F.elu(self.conv2(y))
        y = y.view(y.shape[0], -1)
        h = F.relu(self.fc1(y))
        y = self.fc2(h)
        th.cuda.empty_cache()
        return y, h