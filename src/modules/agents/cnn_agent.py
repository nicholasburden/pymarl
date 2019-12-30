import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math


def _get_padding(size, kernel_size, stride, dilation):
    padding = ((size - 1) * (stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding

class CNNOutputGridLinearBeforeAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNOutputGridLinearBeforeAgent, self).__init__()
        self.args = args

        channels, n_dim_x, n_dim_y = input_shape["2d"][0]
        channels += 1 #for inputs 1d to form another channel
        self.cnn_kernel_size = 3
        self.cnn_stride = 1
        self.n_cnn_layers = 3
        # Construct CNN layers
        self.cnn_layers = []

        self.fc = nn.Linear(input_shape["1d"][0], input_shape["2d"][0][1] * input_shape["2d"][0][2])

        for i in range(self.n_cnn_layers - 1):

            # Create layer
            self.cnn_layers.append(
                nn.Conv2d(channels, channels, self.cnn_kernel_size, stride=self.cnn_stride, padding = _get_padding(n_dim_x, self.cnn_kernel_size, self.cnn_stride, 1), dilation=1))
            # Register parameters (not done automatically for lists)
            for (name, par) in self.cnn_layers[-1].named_parameters():
                self.register_parameter('CNN%u-' % i + name, par)

        self.cnn_layers.append(nn.Conv2d(channels, 2, self.cnn_kernel_size, stride=self.cnn_stride, padding = _get_padding(n_dim_x, self.cnn_kernel_size, self.cnn_stride, 1), dilation=1))
        if args.use_cuda:
            self.cnn_layers = [l.cuda() for l in self.cnn_layers]

    def init_hidden(self):
        # make hidden states on same device as model
        return self.cnn_layers[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = inputs['2d'][0]
        input1d_grid = F.relu(self.fc(inputs["1d"][0])).view(-1, 1, 5, 5)
        x = F.cat((x, input1d_grid), 1)
        # Apply CNN layers
        for i in range(self.cnn_layers.size() - 1):
            x = F.relu(self.cnn_layers[i](x))
        x = self.cnn_layers[-1](x)
        actions = inputs['actions_2d'].bool()
        q = x[actions].unsqueeze(1)
        return q, self.cnn_layers[0].weight.new(q.shape[0], self.args.rnn_hidden_dim).zero_()


class CNNOutputGridLinearAfterAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CNNOutputGridLinearAfterAgent, self).__init__()
        self.args = args

        channels, n_dim_x, n_dim_y = input_shape["2d"][0]
        self.cnn_kernel_size = 3
        self.cnn_stride = 1
        self.n_cnn_layers = 3
        # Construct CNN layers
        self.cnn_layers = []
        in_features = 1 + input_shape["1d"][0] #q value and id

        self.fc = nn.Linear(in_features, 1)
        for i in range(self.n_cnn_layers - 1):

            # Create layer
            self.cnn_layers.append(
                nn.Conv2d(channels, channels, self.cnn_kernel_size, stride=self.cnn_stride, padding = _get_padding(n_dim_x, self.cnn_kernel_size, self.cnn_stride, 1), dilation=1))
            # Register parameters (not done automatically for lists)
            for (name, par) in self.cnn_layers[-1].named_parameters():
                self.register_parameter('CNN%u-' % i + name, par)

        self.cnn_layers.append(nn.Conv2d(channels, 2, self.cnn_kernel_size, stride=self.cnn_stride, padding = _get_padding(n_dim_x, self.cnn_kernel_size, self.cnn_stride, 1), dilation=1))
        if args.use_cuda:
            self.cnn_layers = [l.cuda() for l in self.cnn_layers]

    def init_hidden(self):
        # make hidden states on same device as model
        return self.cnn_layers[0].weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        if(th.cuda.is_available()):
            x = inputs['2d'][0].cuda()
        else:
            x = inputs['2d'][0]
        # Apply CNN layers
        for f in self.cnn_layers:
            x = F.relu(f(x))
        actions = inputs['actions_2d'].bool()
        q = x[actions].unsqueeze(1)
        q = self.fc(th.cat((q, inputs["1d"][0]), 1))
        return q, self.cnn_layers[0].weight.new(q.shape[0], self.args.rnn_hidden_dim).zero_()





