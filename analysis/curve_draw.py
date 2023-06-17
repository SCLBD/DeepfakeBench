import os
import glob
import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score

# collect path for each detector
available_list = ['srm', 'xception', 'f3net', 'ucf', 'cnn_aug', 'spsl', 'efficientnetb4', 'capsule', 'meso4', 'recce', 'meso4Inception', 'ffd', 'core', 'fwa', 'facexray']
data_list = ['/mntcephfs/lab_data/zhiyuanyan/benchmark_results/auc_draw', '/mntcephfs/lab_data/yuanxinhang/benchmark_results/auc_draw', '/mntcephfs/lab_data/tianshuoge/benchmark_results/auc_draw']
detector_list = []
for data in data_list:
    detector = os.listdir(data)
    for d in detector:
        if d in available_list:
            detector_list.append(os.path.join(data, d))

# Let's suppose we have N test datasets. Modify this according to your needs.
N = len(detector_list)

# A dict to hold data for each test dataset.
dataset_dict = defaultdict(list)

for i, detector in enumerate(detector_list):
    print('------------------------------------------')
    print(f'evaluate on {detector}...')
    
    # collect pred, label, prob for each detector
    test_datasets = glob.glob(os.path.join(detector, '*', 'test'))

    detector = detector.split('/')[-1]

    for test_data_path in test_datasets:
        test_data = os.listdir(test_data_path)
        train_dataset = os.listdir(test_data_path.replace('test', 'train'))[0]
        print(f'train on {train_dataset}, test on {test_data}')

        for one_test_data in test_data:
            print(f'evaluate on test data: {one_test_data}...')
            test_metric_dict_path = os.path.join(test_data_path, one_test_data, 'metric_dict_best.pickle')
            with open(test_metric_dict_path, 'rb') as f:
                test_metric_dict = pickle.load(f)
            prob = test_metric_dict['pred']
            label = test_metric_dict['label']
            
            if train_dataset not in ['FF-DF', 'FF-F2F', 'FF-NT', 'FF-FS']:
                # Calculate AUC
                fpr, tpr, _ = roc_curve(label, prob)
                roc_auc = auc(fpr, tpr)
                
                # Calculate Precision-Recall
                precision, recall, _ = precision_recall_curve(label, prob)
                average_precision = average_precision_score(label, prob)

                # Store the metrics for each dataset
                dataset_dict[one_test_data].append((fpr, tpr, roc_auc, precision, recall, average_precision, detector))

# Now we have the metrics for all detectors on each dataset, we can plot them.
for dataset, metrics_list in tqdm(dataset_dict.items()):
    fig1, ax1 = plt.subplots(figsize=(10, 7)) # ROC-AUC
    fig2, ax2 = plt.subplots(figsize=(10, 7)) # Precision-Recall

    for fpr, tpr, roc_auc, precision, recall, average_precision, detector in metrics_list:
        # Plot ROC curve
        ax1.plot(fpr, tpr, lw=1, label=f'{detector} (AUC = %0.4f)' % roc_auc)

        # Plot Precision-Recall line
        ax2.step(recall, precision, where='post', label=f'{detector} (AP = %0.4f)' % average_precision)

    # Plot settings for ROC curve
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate (TPR)', fontdict={'size': 15})
    ax1.set_ylabel('True Positive Rate (FPR)', fontdict={'size': 15})
    ax1.set_title(f'Receiver Operating Characteristic (ROC) - {dataset}', fontdict={'size': 18})
    ax1.legend(loc="lower right")

    # Plot settings for Precision-Recall curve
    ax2.set_xlabel('Recall', fontdict={'size': 15})
    ax2.set_ylabel('Precision', fontdict={'size': 15})
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])
    ax2.set_title(f'Precision-Recall - {dataset}', fontdict={'size': 18})
    ax2.legend(loc="upper right")

    plt.tight_layout()

    # Save plots separately
    os.makedirs('curve', exist_ok=True)
    fig1.savefig(f'curve/roc_curve_{dataset}.png')
    fig2.savefig(f'curve/pr_curve_{dataset}.png')
