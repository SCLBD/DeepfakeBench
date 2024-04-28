import abc
import torch
from typing import Union

class AbstractBackbone(abc.ABC):
    """
    All backbones for detectors should subclass this class.
    """
    def __init__(self, config, load_param: Union[bool, str] = False):
        """
        config:   (dict)
            configurations for the model
        load_param:  (False | True | Path(str))
            False Do not read; True Read the default path; Path Read the required path
        """
        pass
    
    @abc.abstractmethod
    def features(self, data_dict: dict) -> torch.tensor:
        """
        """
        
    @abc.abstractmethod
    def classifier(self, features: torch.tensor) -> torch.tensor:
        """
        """
        
    def init_weights(self, pretrained_path: Union[bool, str]):
        """
        This method can be optionally implemented by subclasses.
        """
        pass