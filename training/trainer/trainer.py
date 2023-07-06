# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30
# description: trainer

import os
import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

from utils.plot_utils import plot_FaceMask
from utils.gradcam import GradCam
from metrics.base_metrics_class import Recorder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self, 
        config, 
        model, 
        optimizer, 
        scheduler,
        logger,
        metric_scoring='auc',
        ):
        # check if all the necessary components are implemented
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizier, logger, and tensorboard writer must be implemented")
        
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writers = {}  # dict to maintain different tensorboard writers for each dataset and metric
        self.logger = logger
        self.metric_scoring = metric_scoring
        # maintain the best metric of all epochs
        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(lambda: float('-inf') 
            if self.metric_scoring != 'eer' else float('inf'))
        ) 
        self.speed_up()  # move model to GPU

        # get current time
        self.timenow = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        # create directory path
        self.log_dir = os.path.join(
            self.config['log_dir'], 
            self.config['model_name'] + '_' + self.timenow
        )
        os.makedirs(self.log_dir, exist_ok=True)
    
    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            # update directory path
            writer_path = os.path.join(
                self.log_dir,
                phase, 
                dataset_key,
                metric_key
            )
            os.makedirs(writer_path, exist_ok=True)
            # update writers dictionary
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    def speed_up(self):
        if self.config['ngpu'] > 1:
            self.model = DataParallel(self.model)
        self.model.to(device)
    
    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError(
                "=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = f"ckpt_best.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        if self.config['ngpu'] > 1:
            torch.save(self.model.module.state_dict(), save_path)
        else:
            torch.save(self.model.state_dict(), save_path)
        self.logger.info(f"Checkpoint saved to {save_path}")

    def save_feat(self, phase, pred_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        features = pred_dict['feat']
        feat_name = f"feat_best.npy"
        save_path = os.path.join(save_dir, feat_name)
        np.save(save_path, features.cpu().numpy())
        self.logger.info(f"Feature saved to {save_path}")
    
    def save_data_dict(self, phase, data_dict, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'data_dict_{phase}.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        save_dir = os.path.join(self.log_dir, phase, dataset_key)
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, 'metric_dict_best.pickle')
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    def train_epoch(
        self, 
        epoch, 
        train_data_loader, 
        test_data_loaders=None,
        ):

        self.logger.info("===> Epoch[{}] start!".format(epoch))
        test_step = len(train_data_loader) // 10    # test 10 times per epoch
        step_cnt = epoch * len(train_data_loader)

        # save the training data_dict
        data_dict = train_data_loader.dataset.data_dict
        self.save_data_dict('train', data_dict, ','.join(self.config['train_dataset']))
        
        # define training recorder
        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        for iteration, data_dict in tqdm(enumerate(train_data_loader)):
            self.setTrain()

            # get data
            data, label, mask, landmark = \
                data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
            if 'label_spe' in data_dict:
                label_spe = data_dict['label_spe']
                data_dict['label_spe'] = label_spe.to(device)
            
            # move data to GPU
            data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
            if mask is not None:
                data_dict['mask'] = mask.to(device)
            if landmark is not None:
                data_dict['landmark'] = landmark.to(device)

            # zero grad
            self.optimizer.zero_grad()
            
            # model forward
            predictions = self.model(data_dict)
            
            # compute all losses for each batch data
            losses = self.model.get_losses(data_dict, predictions)
            
            # gradient backpropagation
            losses['overall'].backward()

            # update model weights
            self.optimizer.step()

            # update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
                
            # compute training metric for each batch data
            batch_metrics = self.model.get_train_metrics(data_dict, predictions)
            
            # store data by recorder
            ## store metric
            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            ## store loss
            for name, value in losses.items():
                train_recorder_loss[name].update(value)
            
            # run tensorboard to visualize the training process
            if iteration % 300 == 0:
                
                # info for loss
                loss_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_loss.items():
                    loss_str += f"training-loss, {k}: {v.average()}    "
                self.logger.info(loss_str)
                # info for metric
                metric_str = f"Iter: {step_cnt}    "
                for k, v in train_recorder_metric.items():
                    metric_str += f"training-metric, {k}: {v.average()}    "
                self.logger.info(metric_str)

                # tensorboard-1. loss
                for k, v in train_recorder_loss.items():
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_loss/{k}', v.average(), global_step=step_cnt)
                # tensorboard-2. metric
                for k, v in train_recorder_metric.items():
                    writer = self.get_writer('train', ','.join(self.config['train_dataset']), k)
                    writer.add_scalar(f'train_metric/{k}', v.average(), global_step=step_cnt)
                
                # clear recorder. 
                # Note we only consider the current 300 samples for computing batch-level loss/metric
                for name, recorder in train_recorder_loss.items():  # clear loss recorder
                    recorder.clear()
                for name, recorder in train_recorder_metric.items():  # clear metric recorder
                    recorder.clear()

            # run test
            if (step_cnt+1) % test_step == 0:
                if test_data_loaders is not None:
                    self.logger.info("===> Test start!")
                    test_best_metric = self.test_epoch(
                        epoch, 
                        iteration,
                        test_data_loaders, 
                        step_cnt,
                    )
            step_cnt += 1
        return test_best_metric
    
    def test_one_dataset(self, data_loader):
        # define test recorder
        test_recorder_loss = defaultdict(Recorder)
        for i, data_dict in tqdm(enumerate(data_loader)):
            # get data
            data, label, mask, landmark = \
            data_dict['image'], data_dict['label'], data_dict['mask'], data_dict['landmark']
            # FIXME: do not consider the specific label when testing
            label = torch.where(data_dict['label']!=0, 1, 0)  # fix the label to 0 and 1 only
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')  # remove the specific label
        
            # move data to GPU
            data_dict['image'], data_dict['label'] = data.to(device), label.to(device)
            if mask is not None:
                data_dict['mask'] = mask.to(device)
            if landmark is not None:
                data_dict['landmark'] = landmark.to(device)

            # model forward without considering gradient computation
            predictions = self.inference(data_dict)
            
            # compute all losses for each batch data
            losses = self.model.get_losses(data_dict, predictions)

            # store data by recorder
            for name, value in losses.items():
                test_recorder_loss[name].update(value)

        return test_recorder_loss, predictions
    
    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        # set model to eval mode
        self.setEval()

        # define test recorder
        losses_all_datasets = {}
        metrics_all_datasets = {}
        best_metrics_per_dataset = defaultdict(dict)  # best metric for each dataset, for each metric

        # testing for all test data
        keys = test_data_loaders.keys()
        for key in keys:
            # save the testing data_dict
            data_dict = test_data_loaders[key].dataset.data_dict
            self.save_data_dict('test', data_dict, key)

            # compute loss for each dataset
            losses_one_dataset_recorder, predictions_dict = self.test_one_dataset(test_data_loaders[key])
            losses_all_datasets[key] = losses_one_dataset_recorder
            
            # compute metric for each dataset
            metric_one_dataset = self.model.get_test_metrics()
            metrics_all_datasets[key] = metric_one_dataset
            
            # FIXME: ugly, need to be modified
            pred_tmp = deepcopy(metric_one_dataset['pred'])
            label_tmp = deepcopy(metric_one_dataset['label'])
            del metric_one_dataset['pred']
            del metric_one_dataset['label']
            
            # maintain the best metric and save the feat, ckpt, and pred&label in whole test dataset
            best_metric = self.best_metrics_all_time[key].get(self.metric_scoring, float('-inf') if self.metric_scoring != 'eer' else float('inf'))
            # Check if the current score is an improvement
            improved = (metric_one_dataset[self.metric_scoring] > best_metric) if self.metric_scoring != 'eer' else (metric_one_dataset[self.metric_scoring] < best_metric)
            if improved:
                # Update the best metric
                self.best_metrics_all_time[key][self.metric_scoring] = metric_one_dataset[self.metric_scoring]
                # Save checkpoint, feature, and metrics if specified in config
                if self.config['save_ckpt']:
                    self.save_ckpt('test', key)
                if self.config['save_feat']:
                    self.save_feat('test', predictions_dict, key)
                # FIXME: ugly, need to be modified
                metric_one_dataset['pred'] = pred_tmp
                metric_one_dataset['label'] = label_tmp
                self.save_metrics('test', metric_one_dataset, key)
                del metric_one_dataset['pred']
                del metric_one_dataset['label']
        
            # info for each dataset
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                loss_str += f"testing-loss, {k}: {v.average()}    "
            self.logger.info(loss_str)
            tqdm.write(loss_str)
            metric_str = f"dataset: {key}    step: {step}    "
            for k, v in metric_one_dataset.items():
                metric_str += f"testing-metric, {k}: {v}    "
            self.logger.info(metric_str)
            tqdm.write(metric_str)

            # tensorboard-1. loss
            for k, v in losses_one_dataset_recorder.items():
                writer = self.get_writer('test', key, k)
                writer.add_scalar(f'test_losses/{k}', v.average(), global_step=step)
            # tensorboard-2. metric
            for k, v in metric_one_dataset.items():
                writer = self.get_writer('test', key, k)
                writer.add_scalar(f'test_metrics/{k}', v, global_step=step)

        self.logger.info('===> Test Done!')
        return best_metrics_per_dataset  # return all types of mean metrics for determining the best ckpt

    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions
