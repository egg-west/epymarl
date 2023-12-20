REGISTRY = {}

from .basic_controller import BasicMAC
from .non_shared_controller import NonSharedMAC
from .maddpg_controller import MADDPGMAC
from .task_encoder_controller import TaskEncoderMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["task_encoder_mac"] = TaskEncoderMAC
REGISTRY["non_shared_mac"] = NonSharedMAC
REGISTRY["maddpg_mac"] = MADDPGMAC