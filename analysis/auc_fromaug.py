import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def replace_name(csv_name: str):
    if csv_name == 'test_FaceForensics++_auc':
        return_name = 'FF++_c23'
    elif csv_name == 'test_FaceForensics++_c40_auc':
        return_name = 'FF++_c40'
    elif csv_name == 'test_Celeb-DF-v2_auc':
        return_name = 'CelebDF-v2'
    elif csv_name == 'test_DeepFakeDetection_auc':
        return_name = 'DFD'
    elif csv_name == 'test_DFDCP_auc':
        return_name = 'DFDCP'
    else:
        raise ValueError(f'Unknown csv name: {csv_name}')
    return return_name

detectors = glob.glob(os.path.join('aug_exp/*'))
results = []

for detector in detectors:
    for train_data in glob.glob(f'{detector}/*'):
        train_data_name = os.path.basename(train_data)
        csv_files = glob.glob(f'{train_data}/*.csv')
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            top_3_auc = df['Value'].nlargest(3).mean() 
            test_data = os.path.basename(csv_file).replace('.csv','') 
            model_name = os.path.basename(os.path.dirname(detector))
            results.append({
                'Model': model_name,
                'Detector': detector.replace('aug_exp/', ''),
                'Augmentation Methods': train_data_name,
                'Test Data': replace_name(test_data),
                'Top-3 AUC': top_3_auc
            })

import matplotlib as mpl

# 设置全局字体大小
mpl.rcParams['font.size'] = 15  # 调整字体大小为16（根据需要进行调整）

df_results = pd.DataFrame(results)
final_df = df_results.pivot_table(index=['Model', 'Detector', 'Augmentation Methods'], columns='Test Data', values='Top-3 AUC')


arr1 = np.tile(final_df.loc[(slice(None), slice(None), 'w_All'), :].values[0], (final_df.shape[0]//2, 1))
arr2 = np.tile(final_df.loc[(slice(None), slice(None), 'w_All'), :].values[1], (final_df.shape[0]//2, 1))
final_df = final_df.loc[(slice(None), slice(None)), :] - np.concatenate((arr1, arr2), 0)

final_df.reset_index(inplace=True)


# Set the number of rows and columns for the subplots
n_rows = 2
n_cols = 5

# Create a figure with the specified number of rows and columns
fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 10))

# Flatten the axes array to iterate through subplots
axes = axes.flatten()

# Define the test datasets
test_datasets = ['FF++_c23', 'FF++_c40', 'CelebDF-v2', 'DFD', 'DFDCP']

# Define the augmentation methods
aug_methods = ['wo_All', 'wo_ColorJitter', 'wo_Compression', 'wo_GaussianBlur', 'wo_HorizontalFlip', 'wo_IsotropicResize', 'wo_Rotate']

# Define the colors for bars
colors = ['lightcoral', 'lightsalmon', 'sandybrown', 'darkorange', 'gold', 'yellowgreen', 'lightgreen']

# Iterate through each detector
for i, detector in enumerate(['SPSL', 'Xception']):
    # Filter the dataframe for the current detector
    detector_df = final_df[final_df['Detector'] == detector]
    
    # Iterate through each test dataset
    for j, test_data in enumerate(test_datasets):
        # Filter the dataframe for the current test dataset and detector
        test_data_df = detector_df[test_data]
        
        # Sort the dataframe by the Difference column in descending order
        test_data_df = test_data_df.sort_values(ascending=False, inplace=False)
        
        # Calculate the number of augmentation methods for the current test dataset
        num_methods = len(test_data_df)
        
        # Generate the y-ticks positions
        y_ticks = np.arange(num_methods)
        
        # Generate the bar plot
        bars = axes[i * n_cols + j].barh(y_ticks, test_data_df, color=colors[:num_methods])
        
        # Add values on the bars
        for bar in bars:
            width = bar.get_width()
            axes[i * n_cols + j].annotate(f'{width:.3f}', xy=(width, bar.get_y() + bar.get_height() / 2),
                                            xytext=(3, -7), textcoords='offset points', ha='left', va='center')
        
        # Set the y-ticks and labels
        axes[i * n_cols + j].set_yticks(y_ticks)
        axes[i * n_cols + j].set_yticklabels(detector_df['Augmentation Methods'].reindex(test_data_df.index))
        
        # Add baseline line
        axes[i * n_cols + j].axvline(0, color='black', linestyle='--')
        
        # Set the x-axis label
        axes[i * n_cols + j].set_xlabel('Difference (%) from Baseline (w_All)')
        
        # Set the title
        axes[i * n_cols + j].set_title(f'Detector: {detector} | Test Data: {test_data}')
        axes[i * n_cols + j].title.set_fontsize(16)  # 调整标题字体大小为16（根据需要进行调整）
    
# Adjust the spacing between subplots
fig.tight_layout()

# Save the plot
plt.savefig(f"all_results_aug_bar_plot_2.png")
