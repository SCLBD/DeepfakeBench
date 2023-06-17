import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import os

import matplotlib as mpl

# 设置全局字体大小
# mpl.rcParams['font.size'] = 15  # 调整字体大小为16（根据需要进行调整）

def get_top3_avg_auc(csv_path):
    df = pd.read_csv(csv_path)
    return df["Value"].nlargest(3).mean()

# def plot_radar_chart(df, methods):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))

#     # Flatten the axes to easily iterate over it
#     axs = axs.flatten()

#     for ax, method in zip(axs, methods):
#         # Extract data for the given method
#         df_method = df[df['Method'] == method]

#         # Set up radar chart
#         num_vars = len(df_method['Dataset'].unique())
#         angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

#         # Ensure radar chart is a complete circle
#         angles += angles[:1]

#         ax.set_title(f'Average Top-3 AUC for {method.upper()}', fontsize=16, fontweight='bold')

#         # For each architecture, plot a separate line on the radar chart
#         for arch in df_method['Architecture'].unique():
#             df_arch = df_method[df_method['Architecture'] == arch].copy()  # Create a copy of the DataFrame slice

#             # Order the dataset according to the specified order
#             dataset_order = ['DFDCP', 'Celeb-DF-v2', 'FaceForensics++', 'FaceForensics++_c40', 'DeepFakeDetection']
#             df_arch['Dataset'] = pd.Categorical(df_arch['Dataset'], categories=dataset_order, ordered=True)
#             df_arch.sort_values('Dataset', inplace=True)

#             values = df_arch['Top-3 AUC'].tolist()
#             values += values[:1]  # ensure the plot is a complete circle

#             if arch == 'EfficientNet':
#                 color = 'tab:blue'
#                 ax.plot(angles, values, label=arch, color=color, linewidth=1.2)  # reduce line width
#             elif arch == 'ResNet':
#                 color = 'tab:orange'
#                 ax.plot(angles, values, label=arch, color=color, linewidth=1.2)  # reduce line width
#                 ax.fill(angles, values, color=color, alpha=0.05)  # fill with color
#             else:
#                 color = 'tab:green'
#                 ax.plot(angles, values, label=arch, color=color, linewidth=1.2)  # reduce line width

#         # Add legend and labels
#         ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
#         ax.set_theta_offset(np.pi / 2)
#         ax.set_theta_direction(-1)
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(df_arch['Dataset'].tolist())
#         ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#         ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # Add percentage on radius

#         # Add a grid
#         ax.grid(True)

#     fig.suptitle('Average Top-3 AUC Across Different Methods and Architectures', size=15, color='black', y=1.05)
#     plt.subplots_adjust(wspace=5, hspace=1)  # Adjust subplot spacing
#     plt.tight_layout()
#     plt.savefig('radar_chart_model_archi_2.png')

def plot_radar_chart(df, methods):
    fig, axs = plt.subplots(1, len(methods), figsize=(6 * len(methods), 6), subplot_kw=dict(polar=True))

    for ax, method in zip(axs, methods):
        # Extract data for the given method
        df_method = df[df['Method'] == method]

        # Set up radar chart
        num_vars = len(df_method['Dataset'].unique())
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Ensure radar chart is a complete circle
        angles += angles[:1]

        ax.set_title(f'Average Top-3 AUC for {method.upper()}', fontsize=16, fontweight='bold')

        # For each architecture, plot a separate line on the radar chart
        for arch in df_method['Architecture'].unique():
            df_arch = df_method[df_method['Architecture'] == arch].copy()  # Create a copy of the DataFrame slice

            # Order the dataset according to the specified order
            dataset_order = ['DFDCP', 'Celeb-DF-v2', 'FaceForensics++', 'FaceForensics++_c40', 'DeepFakeDetection']
            df_arch['Dataset'] = pd.Categorical(df_arch['Dataset'], categories=dataset_order, ordered=True)
            df_arch.sort_values('Dataset', inplace=True)

            values = df_arch['Top-3 AUC'].tolist()
            values += values[:1]  # ensure the plot is a complete circle

            if arch == 'EfficientNet':
                color = 'tab:blue'
                ax.plot(angles, values, label=arch, color=color, linewidth=1.2)  # reduce line width
            elif arch == 'ResNet':
                color = 'tab:orange'
                ax.plot(angles, values, label=arch, color=color, linewidth=1.2)  # reduce line width
                ax.fill(angles, values, color=color, alpha=0.05)  # fill with color
            else:
                color = 'tab:green'
                ax.plot(angles, values, label=arch, color=color, linewidth=1.2)  # reduce line width

        # Add legend and labels
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1))
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(df_arch['Dataset'].tolist())
        ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))  # Add percentage on radius

        # Add a grid
        ax.grid(True)

    fig.suptitle('Average Top-3 AUC Across Different Methods and Architectures', size=15, color='black', y=1.05)
    plt.tight_layout()
    plt.savefig('radar_chart_model_archi_2.png')

# Read and process the AUC values
base_dir = 'architecture_explore'
methods = ['core', 'spsl', 'ucf', 'facexray']
arch_mapping = {
    'core_eff': 'EfficientNet', 'core_res': 'ResNet', 'core_xcep': 'Xception',
    'spsl_eff': 'EfficientNet', 'spsl_res': 'ResNet', 'spsl_xcep': 'Xception',
    'ucf_eff': 'EfficientNet', 'ucf_res': 'ResNet', 'ucf_xcep': 'Xception',
    'facexray_eff': 'EfficientNet', 'facexray_res': 'ResNet', 'facexray_xcep': 'Xception'
}
datasets = ['DFDCP', 'Celeb-DF-v2', 'FaceForensics++', 'FaceForensics++_c40', 'DeepFakeDetection']

results = []

for method in methods:
    method_dir = os.path.join(base_dir, method)
    for arch_dir in [d for d in os.listdir(method_dir) if d != '.DS_Store']:
        folder_path = os.path.join(method_dir, arch_dir)
        arch = arch_mapping[arch_dir]
        for dataset in datasets:
            csv_file = f'test_{dataset}_auc.csv'
            if csv_file in os.listdir(folder_path):
                avg_auc = get_top3_avg_auc(os.path.join(folder_path, csv_file))
                results.append({
                    'Method': method,
                    'Architecture': arch,
                    'Dataset': dataset,
                    'Top-3 AUC': avg_auc
                })

df_results = pd.DataFrame(results)

# Plot a radar chart for each method
plot_radar_chart(df_results, methods)
