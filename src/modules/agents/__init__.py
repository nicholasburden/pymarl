REGISTRY = {}

from .rnn_agent import RNNAgent, RNNInputActionAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_input_action"] = RNNInputActionAgent