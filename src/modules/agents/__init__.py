REGISTRY = {}

from .rnn_agent import RNNAgent, RNNInputActionAgent, RNNConvDDPGAgent, RNNConvDDPGInputGridAgent, RNNConvDDPGInputFlatAgent, RNNConvDDPGInputGridNoIDAgent
from .no_rnn_agent import ConvDDPGInputGridDeepAgent, ConvDDPGInputGridShallowAgent

#RNN
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_input_action"] = RNNInputActionAgent
REGISTRY["rnn_convddpg"] = RNNConvDDPGAgent
REGISTRY["rnn_convddpg_input_grid"] = RNNConvDDPGInputGridAgent
REGISTRY["rnn_convddpg_input_grid_no_id"] = RNNConvDDPGInputGridNoIDAgent
REGISTRY["rnn_convddpg_input_flat"] = RNNConvDDPGInputFlatAgent

#no_rnn
REGISTRY["convddpg_input_grid_deep"] = ConvDDPGInputGridDeepAgent
REGISTRY["convddpg_input_grid_shallow"] = ConvDDPGInputGridShallowAgent