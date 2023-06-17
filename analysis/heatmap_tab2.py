import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 创建一个新的颜色映射
cmap = sns.color_palette("Reds")
cmap = ListedColormap([rgb + (0.6,) for rgb in cmap])

# 定义行列标签
row_labels = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]
col_labels = ["FF-DF", "FF-F2F", "FF-FS", "FF-NT"]

# 创建数据
data_CORE = np.array([
    [0.9946, 0.7284, 0.7008, 0.7568],
    [0.6803, 0.9871, 0.6364, 0.6828],
    [0.5586, 0.6205, 0.9941, 0.5615],
    [0.6369, 0.6004, 0.5477, 0.9554],
])
df_CORE = pd.DataFrame(data_CORE, index=row_labels, columns=col_labels)

data_EfficientB4 = np.array([
    [0.9951, 0.7279, 0.7057, 0.7710],
    [0.6613, 0.9824, 0.6289, 0.6821],
    [0.5546, 0.5958, 0.9919, 0.5532],
    [0.6486, 0.5841, 0.5252, 0.9559],
])
df_EfficientB4 = pd.DataFrame(data_EfficientB4, index=row_labels, columns=col_labels)

data_DSP_FWA = np.array([
    [0.8990, 0.9127, 0.9201, 0.9014],
    [0.8727, 0.8936, 0.9049, 0.8833],
    [0.8748, 0.8992, 0.8917, 0.8718],
    [0.7916, 0.8058, 0.8014, 0.7982],
])
df_DSP_FWA = pd.DataFrame(data_DSP_FWA, index=row_labels, columns=col_labels)

data_FaceXray = np.array([
    [0.9275, 0.9421, 0.9256, 0.8446],
    [0.9615, 0.9706, 0.9468, 0.8983],
    [0.9711, 0.9672, 0.9506, 0.8754],
    [0.9453, 0.9578, 0.9286, 0.8624]
])
df_FaceXray = pd.DataFrame(data_FaceXray, index=row_labels, columns=col_labels)

data_SPSL = np.array([
    [0.9953, 0.7548, 0.6686, 0.7803],
    [0.7114, 0.9857, 0.6624, 0.6779],
    [0.5380, 0.6072, 0.9924, 0.5548],
    [0.6634, 0.5961, 0.5457, 0.9547],
])
df_SPSL = pd.DataFrame(data_SPSL, index=row_labels, columns=col_labels)

data_SRM = np.array([
    [0.9954, 0.7237, 0.6838, 0.7623],
    [0.6699, 0.9848, 0.6216, 0.6800],
    [0.5502, 0.6070, 0.9931, 0.5505],
    [0.6492, 0.5781, 0.5301, 0.9546],
])
df_SRM = pd.DataFrame(data_SRM, index=row_labels, columns=col_labels)

data_Xception = np.array([
    [0.9936, 0.7225, 0.7027, 0.7575],
    [0.6737, 0.9842, 0.6326, 0.6791],
    [0.5518, 0.6093, 0.9927, 0.5508],
    [0.6352, 0.5845, 0.5353, 0.9550],
])
df_Xception = pd.DataFrame(data_Xception, index=row_labels, columns=col_labels)

data_ResNet = np.array([
    [0.9762,	0.6366,	0.7344,	0.6947],
    [0.6011,	0.9620,	0.5829,	0.6559],
    [0.6432,	0.5824,	0.9748,	0.5236],
    [0.5798,	0.5620,	0.5229,	0.8456],
])
df_ResNet = pd.DataFrame(data_ResNet, index=row_labels, columns=col_labels)

data_F3Net = np.array([
    [0.9762,	0.6366,	0.7344,	0.6947],
    [0.6011,	0.9620,	0.5829,	0.6559],
    [0.6432,	0.5824,	0.9748,	0.5236],
    [0.5798,	0.5620,	0.5229,	0.8456],
])
df_F3Net = pd.DataFrame(data_F3Net, index=row_labels, columns=col_labels)

data_FFD = np.array([
    [0.9953,	0.7548,	0.6686,	0.7803],
    [0.7114,	0.9857,	0.6624,	0.6779],
    [0.5380,	0.6072,	0.9924,	0.5548],
    [0.6634,	0.5961,	0.5457,	0.9547],
])
df_FFD = pd.DataFrame(data_FFD, index=row_labels, columns=col_labels)

# 使用subplot绘制多个热力图
fig, axs = plt.subplots(1, 10, figsize=(55, 5))
# 设置每个子图中x和y轴标签的字体大小
tick_fontsize = 15

vmin = min(np.min(data_CORE), np.min(data_EfficientB4), np.min(data_DSP_FWA), np.min(data_SPSL), np.min(data_SRM), np.min(data_Xception), np.min(data_F3Net), np.min(data_FFD), np.min(data_ResNet), np.min(data_FaceXray))
vmax = max(np.max(data_CORE), np.max(data_EfficientB4), np.max(data_DSP_FWA), np.max(data_SPSL), np.max(data_SRM), np.max(data_Xception), np.max(data_F3Net), np.max(data_FFD), np.max(data_ResNet), np.max(data_FaceXray))

sns.heatmap(df_CORE, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[0])
axs[0].set_title('CORE', fontsize=20)
axs[0].tick_params(axis='x', labelsize=tick_fontsize)
axs[0].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_EfficientB4, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[1])
axs[1].set_title('EfficientB4', fontsize=20)
axs[1].tick_params(axis='x', labelsize=tick_fontsize)
axs[1].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_DSP_FWA, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[2])
axs[2].set_title('DSP-FWA', fontsize=20)
axs[2].tick_params(axis='x', labelsize=tick_fontsize)
axs[2].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_FaceXray, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[3])
axs[3].set_title('Face X-ray', fontsize=20)
axs[3].tick_params(axis='x', labelsize=tick_fontsize)
axs[3].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_SPSL, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[4])
axs[4].set_title('SPSL', fontsize=20)
axs[4].tick_params(axis='x', labelsize=tick_fontsize)
axs[4].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_SRM, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[5])
axs[5].set_title('SRM', fontsize=20)
axs[5].tick_params(axis='x', labelsize=tick_fontsize)
axs[5].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_Xception, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[6])
axs[6].set_title('Xception', fontsize=20)
axs[6].tick_params(axis='x', labelsize=tick_fontsize)
axs[6].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_ResNet, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[7])
axs[7].set_title('ResNet34', fontsize=20)
axs[7].tick_params(axis='x', labelsize=tick_fontsize)
axs[7].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_F3Net, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[8])
axs[8].set_title('F3Net', fontsize=20)
axs[8].tick_params(axis='x', labelsize=tick_fontsize)
axs[8].tick_params(axis='y', labelsize=tick_fontsize)

sns.heatmap(df_FFD, annot=True, cmap=cmap, cbar=True,
            annot_kws={"size": 15, "color": "black"},
            vmin=vmin, vmax=vmax, linewidths=0.5, ax=axs[9])
axs[9].set_title('FFD', fontsize=20)
axs[9].tick_params(axis='x', labelsize=tick_fontsize)
axs[9].tick_params(axis='y', labelsize=tick_fontsize)

plt.savefig('heatmap_tab2.png')
