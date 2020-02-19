REGISTRY = {}

from .rnn_agent import RNNAgent, RNNInputActionAgent, RNNConvDDPGAgent, RNNConvDDPGInputGridAgent, RNNConvDDPGInputFlatAgent, RNNConvDDPGInputGridNoIDAgent, RNNConvDDPGNoIDAgent
from .no_rnn_agent import ConvDDPGInputGridDeepAgent, ConvDDPGInputGridShallowAgent, ConvDDPGInputGridDeepNoIDAgent, ConvDDPGInputGridShallowNoIDAgent, ConvDDPGNoIDAgent

#RNN
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_input_action"] = RNNInputActionAgent
REGISTRY["rnn_convddpg"] = RNNConvDDPGAgent
REGISTRY["rnn_convddpg_no_id"] = RNNConvDDPGNoIDAgent
REGISTRY["rnn_convddpg_input_grid"] = RNNConvDDPGInputGridAgent
REGISTRY["rnn_convddpg_input_grid_no_id"] = RNNConvDDPGInputGridNoIDAgent
REGISTRY["rnn_convddpg_input_flat"] = RNNConvDDPGInputFlatAgent


#no_rnn
REGISTRY["convddpg_input_grid_deep"] = ConvDDPGInputGridDeepAgent
REGISTRY["convddpg_input_grid_shallow"] = ConvDDPGInputGridShallowAgent
REGISTRY["convddpg_input_grid_deep_no_id"] = ConvDDPGInputGridDeepNoIDAgent
REGISTRY["convddpg_input_grid_shallow_no_id"] = ConvDDPGInputGridShallowNoIDAgent
REGISTRY["convddpg_no_id"] = ConvDDPGNoIDAgent
