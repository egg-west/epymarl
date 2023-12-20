REGISTRY = {}

from .rnn_agent import RNNAgent
from .task_encoder_rnn_agent import TaskEncoderRNNAgent
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
REGISTRY["rnn"] = RNNAgent
REGISTRY["task_encoder_rnn"] = TaskEncoderRNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
