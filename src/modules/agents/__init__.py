REGISTRY = {}

from .rnn_agent import RNNAgent, RNNInputActionAgent, RNNConvDDPGAgent, RNNConvDDPGInputGridAgent, MathiasAgent, RNNConvDDPGInputGridNoIDAgent
from .cnn_agent import CNNOutputGridLinearBeforeAgent, CNNOutputGridLinearAfterAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_input_action"] = RNNInputActionAgent
REGISTRY["rnn_convddpg"] = RNNConvDDPGAgent
REGISTRY["rnn_convddpg_input_grid"] = RNNConvDDPGInputGridAgent
REGISTRY["rnn_mathias"] = MathiasAgent
REGISTRY["cnn_output_grid_linear_before"] = CNNOutputGridLinearBeforeAgent
REGISTRY["cnn_output_grid_linear_after"] = CNNOutputGridLinearAfterAgent