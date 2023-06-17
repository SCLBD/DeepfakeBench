import datetime
from copy import deepcopy
from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """
    """

    def __init__(
        self, 
        config, 
        model, 
        optimizer, 
        scheduler,
        writer,
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or scheduler is None or writer is None:
            raise NotImplementedError("config, model, optimizier, scheduler, and tensorboard writer must be implemented")
        
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = writer

    @abstractmethod
    def speed_up(self):
        pass

    @abstractmethod
    def setTrain(self):
        pass

    @abstractmethod
    def setEval(self):
        pass

    @abstractmethod
    def load_ckpt(self, model_path):
        pass

    @abstractmethod
    def save_ckpt(self, dataset, epoch, iters, best=False):
        pass

    @abstractmethod
    def inference(self, data_dict):
        pass
