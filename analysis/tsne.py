import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pickle

color_map = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
label_dict = {
        0: 'FF_Real',    1: 'Deepfakes', 2: 'Face2Face', 3: 'FaceSwap', 4: 'NeuralTextures', 
    }

def tsne_draw(x_transformed, numerical_labels, ax, epoch=0, log='', detector_name=None):
    labels = [label_dict[label] for label in numerical_labels]

    tsne_df = pd.DataFrame(x_transformed, columns=['X', 'Y'])
    tsne_df["Targets"] = labels
    tsne_df["NumericTargets"] = numerical_labels
    tsne_df.sort_values(by="NumericTargets", inplace=True)
    
    marker_list = ['*' if label == 0 else 'o' for label in tsne_df["NumericTargets"]]

    for _x, _y, _c, _m in zip(tsne_df['X'], tsne_df['Y'], [color_map[i] for i in tsne_df["NumericTargets"]], marker_list):
        ax.scatter(_x, _y, color=_c, s=30, alpha=0.7, marker=_m)

    print(f'epoch{epoch} ' + log)
    ax.axis('off')


detector_name_list = [
    '/mntcephfs/lab_data/zhiyuanyan/benchmark_results/tsne/tsne_dict_xception_0_270.pkl',
    '/mntcephfs/lab_data/zhiyuanyan/benchmark_results/tsne/tsne_dict_xception_0.pkl',
]

tsne = TSNE(n_components=2, perplexity=20, random_state=1024, learning_rate=250)
fig, axs = plt.subplots(1, 2, figsize=(20,10))

for i, tsne_dict in enumerate(detector_name_list):
    print(f'Processing {tsne_dict}...')
    name = str(tsne_dict.split('/')[-1].split('.')[0].split('_')[-1])
    with open(tsne_dict, 'rb') as f:
        tsne_dict = pickle.load(f)
    
    feat = tsne_dict['feat'].reshape((tsne_dict['feat'].shape[0], -1))
    label_spe = tsne_dict['label_spe']

    label_0_indices = np.where(label_spe == 0)[0][:2500]
    other_label_indices = np.where(label_spe != 0)[0]
    num_samples = len(label_0_indices)
    other_label_indices_sampled = np.random.choice(other_label_indices, size=num_samples, replace=False)
    sampled_indices = np.concatenate((label_0_indices, other_label_indices_sampled))
    np.random.shuffle(sampled_indices)

    feat = feat[sampled_indices]
    label_spe = label_spe[sampled_indices]
    feat_transformed = tsne.fit_transform(feat)
    scatter = tsne_draw(feat_transformed, label_spe, ax=axs[i], epoch=0, log='share_in_specific', detector_name='xception')

    # # give a title to the subplot
    # axs[i].set_title(f'Xception with {name} frames')  

# create a legend for the whole figure after the loop
handles = [plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=color_map[i], markersize=10) if i == 0 else plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i], markersize=10) for i in range(5)]
labels = [label_dict[i] for i in range(5)]
# fig.legend(handles, labels, title="Classes", loc="upper right", fontsize=14)

plt.tight_layout()
plt.savefig('xcep_4vs270_3.png')