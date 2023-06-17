import pandas as pd
import glob
import os


def replace_name(csv_name: str):
    if csv_name == 'test_Celeb-DF-v1_auc':
        return_name = 'CelebDF-v1'
    elif csv_name == 'test_Celeb-DF-v2_auc':
        return_name = 'CelebDF-v2'
    elif csv_name == 'test_DeeperForensics-1.0_auc':
        return_name = 'DF-1.0'
    elif csv_name == 'test_FaceShifter_auc':
        return_name = 'FaceShifter'
    elif csv_name == 'test_DeepFakeDetection_auc':
        return_name = 'DFD'
    elif csv_name == 'test_DFDC_auc':
        return_name = 'DFDC'
    elif csv_name == 'test_DFDCP_auc':
        return_name = 'DFDCP'
    elif csv_name == 'test_FaceForensics++_auc':
        return_name = 'FF++_c23'
    elif csv_name == 'test_FaceForensics++_c40_auc':
        return_name = 'FF++_c40'
    elif csv_name == 'test_FF-DF_auc':
        return_name = 'FF-DF'
    elif csv_name == 'test_FF-F2F_auc':
        return_name = 'FF-F2F'
    elif csv_name == 'test_FF-FS_auc':
        return_name = 'FF-FS'
    elif csv_name == 'test_FF-NT_auc':
        return_name = 'FF-NT'
    elif csv_name == 'test_UADFV_auc':
        return_name = 'UADFV'
    else:
        raise ValueError(f'Unknown csv name: {csv_name}')
    return return_name

detectors = glob.glob(os.path.join('exp1_record/*'))  # Assuming the script is running in the parent directory
results = []

for detector in detectors:
    csv_files = glob.glob(f'{detector}/*.csv')

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        top_3_auc = df['Value'].nlargest(3).mean()  # Get mean of top 3 AUC
        
        test_data = os.path.basename(csv_file).replace('.csv','')  # Assuming test_data is the file name
        results.append({'Detector': detector.replace('exp1_record/', ''), 'Test Data': replace_name(test_data), 'Top-3 AUC': top_3_auc})

# Convert list of dicts to DataFrame
df_results = pd.DataFrame(results)

# Pivot the dataframe to have detectors as rows and test_data as columns
final_df = df_results.pivot(index='Detector', columns='Test Data', values='Top-3 AUC')

# Add the 'avg' column as the mean of other columns
final_df['Avg.'] = final_df.mean(axis=1)

print(final_df)

# Save the dataframe to excel
final_df.to_excel('auc_table1_fromrecord.xlsx')