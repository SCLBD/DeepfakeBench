import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import os

def get_top3_avg_auc(csv_path):
    df = pd.read_csv(csv_path)
    return df["Value"].nlargest(3).mean()

def plot_radar_chart(df, architectures):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

    for ax, arch in zip(axs, architectures):
        # Extract data for the given architecture
        df_arch = df[df['Architecture'] == arch]

        # Set up radar chart
        num_vars = len(df_arch['Dataset'].unique())
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Ensure radar chart is a complete circle
        angles += angles[:1]

        ax.set_title(f'Average Top-3 AUC for {arch}', fontsize=15, fontweight='bold')

        # For each pretrain status, plot a separate line on the radar chart
        for pretrain in df_arch['Pretrain'].unique():
            df_pretrain = df_arch[df_arch['Pretrain'] == pretrain].copy()  # Create a copy of the DataFrame slice

            # Order the dataset according to the specified order
            dataset_order = ['DFDCP', 'Celeb-DF-v2', 'FaceForensics++', 'FaceForensics++_c40', 'DeepFakeDetection']
            df_pretrain['Dataset'] = pd.Categorical(df_pretrain['Dataset'], categories=dataset_order, ordered=True)
            df_pretrain.sort_values('Dataset', inplace=True)

            values = df_pretrain['Top-3 AUC'].tolist()
            values += values[:1]  # ensure the plot is a complete circle

            line, = ax.plot(angles, values, label=pretrain, linewidth=1)  # reduce line width
            ax.fill(angles, values, color=line.get_color(), alpha=0.1)  # fill the area of the plot with lowered opacity

        # Add legend and labels
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(df_arch['Dataset'].unique().tolist())
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # Add percentage on radius

        # Add a grid
        ax.grid(True)

    fig.suptitle('Average Top-3 AUC Across Different Pretraining States', size=15, color='black', y=1.05)
    plt.tight_layout()
    plt.savefig('radar_chart_pretrained.png')

# Read and process the AUC values
base_dir = 'pretrained_explore'
archs = ['Xception', 'ResNet34', 'EfficientNetB4']
pretrains = ['wpre', 'wopre']
datasets = ['DFDCP', 'Celeb-DF-v2', 'FaceForensics++', 'FaceForensics++_c40', 'DeepFakeDetection']

results = []

for arch in archs:
    for pretrain in pretrains:
        folder_path = os.path.join(base_dir, arch, f'{arch}_{pretrain}')
        for dataset in datasets:
            csv_file = f'test_{dataset}_auc.csv'
            if csv_file in os.listdir(folder_path):
                avg_auc = get_top3_avg_auc(os.path.join(folder_path, csv_file))
                results.append({
                    'Architecture': arch,
                    'Pretrain': 'With Pretrain' if pretrain == 'wpre' else 'Without Pretrain',
                    'Dataset': dataset,
                    'Top-3 AUC': avg_auc
                })

df_results = pd.DataFrame(results)

# Plot a radar chart for Xception, ResNet34, and EfficientNetB4
plot_radar_chart(df_results, ['Xception', 'ResNet34', 'EfficientNetB4'])
