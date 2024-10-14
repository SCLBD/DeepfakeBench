import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)


from .I2G_dataset import I2GDataset
from .iid_dataset import IIDDataset
from .abstract_dataset import DeepfakeAbstractBaseDataset
from .ff_blend import FFBlendDataset
from .fwa_blend import FWABlendDataset
from .lrl_dataset import LRLDataset
from .pair_dataset import pairDataset
from .sbi_dataset import SBIDataset
from .lsda_dataset import LSDADataset
from .tall_dataset import TALLDataset
