import matplotlib.pyplot as plt
import pandas as pd
import os

# Set up colors for different datasets
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

# Read and process the AUC values
base_dir = 'number_frames_explore'
models = ['xception', 'spsl', 'facexray']
datasets = ['FaceForensics++', 'FaceForensics++_c40', 'DFDCP', 'Celeb-DF-v2', 'DeepFakeDetection']

fig, axs = plt.subplots(1, len(datasets), figsize=(30, 5), sharey=True)

for ax, dataset, color in zip(axs, datasets, colors):
    ax.set_title(dataset, fontsize=16, fontweight='bold')
    ax.set_xlabel('Number of Frames', fontsize=12)
    ax.set_ylabel('AUC', fontsize=13)
    ax.tick_params(axis='both', which='major', labelsize=10)

    for model, linestyle in zip(models, ['-', '--', ':']):
        folder_path = os.path.join(base_dir, model)

        auc_values = []
        frame_numbers = []

        for frame_folder in os.listdir(folder_path):
            frame_path = os.path.join(folder_path, frame_folder)

            if os.path.isdir(frame_path):
                csv_file = f'test_{dataset}_auc.csv'

                if csv_file in os.listdir(frame_path):
                    frame_number = int(frame_folder.split('_')[1].replace('frames', ''))
                    frame_numbers.append(frame_number)

                    df = pd.read_csv(os.path.join(frame_path, csv_file))
                    auc_values.append(df['Value'].mean())

        # Sort frame numbers and corresponding AUC values
        sorted_values = [x for _, x in sorted(zip(frame_numbers, auc_values))]
        sorted_frame_numbers = sorted(frame_numbers)

        ax.plot(sorted_frame_numbers, sorted_values, label=model, color=color, linestyle=linestyle, linewidth=2.0)

    # Add legend and grid
    ax.legend(loc='center right', fontsize=14)
    ax.grid(True, linestyle='--')

# plt.suptitle('Effect of Number of Training Frames on AUC Performance', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('line_chart_number_frames_2.png')
