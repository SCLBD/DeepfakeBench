import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 定义所有的检测器和数据集名称
detectors = ['fwa', 'facexray']
# detectors = ['srm', 'xception', 'f3net', 'ucf', 'cnn_aug', 'spsl', 'efficientnetb4', 'capsule', 'meso4', 'recce', 'meso4Inception', 'ffd', 'core',] # 'fwa', 'facexray'
dataset_names = ['Celeb-DF-v1', 'Celeb-DF-v2', 'DeeperForensics-1.0', 'FaceShifter', 'DeepFakeDetection', 'DFDC', 'DFDCP', 'FaceForensics++', 'FaceForensics++_c40', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT', 'UADFV']

# 设置图形参数
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

metrics = ["auc", "ap", "eer"]
os.makedirs('test_all', exist_ok=True)

# 遍历所有的数据集和检测器，为每个 metric 生成一张图
for dataset_name in dataset_names:
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for detector in detectors:
            # 生成文件路径
            file_path = os.path.join("all_test_results", detector, metric, f"test_{dataset_name}_{metric}.csv")
            # 检查文件是否存在
            if os.path.exists(file_path):
                # 读取 csv 文件
                data = pd.read_csv(file_path)
                # 绘制图像，设置线宽为 1
                plt.plot(data['Step'][:72], data['Value'][:72], label=detector, linewidth=1)
            else:
                print(f"No data found for {detector} on {dataset_name} for {metric} metric.")

        plt.title(f"Evaluation on {dataset_name} dataset for {metric.upper()} metric", fontdict={'fontsize': 18})
        plt.xlabel('Step', fontdict={'fontsize': 16})
        plt.ylabel('Value', fontdict={'fontsize': 16})
        plt.legend()
        plt.grid(True)
        plt.savefig('test_all/' + dataset_name + '_' + metric + '.png')
