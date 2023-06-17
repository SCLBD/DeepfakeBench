import pandas as pd
import glob
import os


def replace_name(csv_name: str):
    if csv_name == 'test_FaceForensics++_auc':
        return_name = 'FF++_c23'
    elif csv_name == 'test_FF-DF_auc':
        return_name = 'FF-DF'
    elif csv_name == 'test_FF-F2F_auc':
        return_name = 'FF-F2F'
    elif csv_name == 'test_FF-FS_auc':
        return_name = 'FF-FS'
    elif csv_name == 'test_FF-NT_auc':
        return_name = 'FF-NT'
    else:
        raise ValueError(f'Unknown csv name: {csv_name}')
    return return_name

detectors = glob.glob(os.path.join('exp2_record/*'))
results = []

for detector in detectors:
    for train_data in glob.glob(f'{detector}/*'):
        train_data_name = os.path.basename(train_data)  # Assuming train_data is the directory name
        csv_files = glob.glob(f'{train_data}/*.csv')
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            top_3_auc = df['Value'].nlargest(3).mean() 
            
            test_data = os.path.basename(csv_file).replace('.csv','') 
            results.append({
                'Detector': detector.replace('exp2_record/', ''),
                'Training Dataset': train_data_name,
                'Test Data': replace_name(test_data),
                'Top-3 AUC': top_3_auc
            })

df_results = pd.DataFrame(results)

final_df = df_results.pivot_table(index=['Detector', 'Training Dataset'], columns='Test Data', values='Top-3 AUC')

final_df['Avg.'] = final_df.mean(axis=1)

print(final_df)
final_df.transpose().to_excel('auc_table2_fromrecord_transpose.xlsx')

final_df.to_excel('auc_table2_fromrecord.xlsx')