import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

from metrics.registry import BACKBONE

from .xception import Xception
from .mesonet import Meso4, MesoInception4
from .resnet34 import ResNet34
from .efficientnetb4 import EfficientNetB4
from .xception_sladd import Xception_SLADD
