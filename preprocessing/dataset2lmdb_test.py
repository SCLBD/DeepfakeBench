import os
import json
import cv2
import lmdb
import yaml
from PIL import Image
import io
import numpy as np
def file_to_binary(file_path):
    """将图片转换为二进制数据"""
    if file_path.endswith('.npy'):
        data = np.load(file_path)
        file_binary = data.tobytes()
    else:
        with open(file_path, 'rb') as f:
            file_binary = f.read()
    return file_binary


def create_lmdb_dataset(source_folder, lmdb_path, dataset_name, map_size):
    """创建LMDB数据集"""
    # 打开LMDB文件，创建数据库
    db = lmdb.open(lmdb_path, map_size=map_size)
    with db.begin(write=True) as txn:
        for root, dirs, files in os.walk(source_folder,followlinks=True):
            print(root)
            if 'video' in root:
                continue
            for file in files:
                print(file)
                image_path = os.path.join(root, file)
                # 生成相对路径键
                relative_path = f"{dataset_name}\\" + os.path.relpath(image_path, source_folder)
                key = relative_path.encode('utf-8')
                # txn.delete(key)
                # relative_path = f"{dataset_name}\\original_sequences" + os.path.relpath(image_path, source_folder)
                # key = relative_path.encode('utf-8')
                value = file_to_binary(image_path)

                # 写入数据库
                txn.put(key, value)

    db.close()
 

def read_lmdb(lmdb_dir_path):
    # validate the key and value in the generated LMDB
    env = lmdb.open(lmdb_dir_path)

    idx = '%09d' % 5
    with env.begin(write=False) as txn:
        # key for validation
        key='npy_test\\000_003\\000.npy'
        binary = txn.get(key.encode())
        data = np.frombuffer(binary, dtype=np.uint32).reshape((81, 2))

        # image_buf = np.frombuffer(image_bin, dtype=np.uint8)
        # img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
        # image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# 使用示例
import argparse
# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process some inputs.')

# 添加 --name 参数
parser.add_argument('--dataset_size', type=int, default=25, required=True,
                    help='lmdb requires pre-specifying the total dataset size (GB)')

# 解析参数
args = parser.parse_args()

if __name__ == '__main__':
    # from config.yaml load parameters
    yaml_path = './preprocessing/config.yaml'
    # open the yaml file
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.parser.ParserError as e:
        print("YAML file parsing error:", e)

    config=config['to_lmdb']
    dataset_name = config['dataset_name']['default']
    dataset_size = args.dataset_size
    dataset_root_path = config['dataset_root_path']['default']
    output_lmdb_dir =config['output_lmdb_dir']['default']
    os.makedirs(output_lmdb_dir,exist_ok=True)
    dataset_dir_path = f"{dataset_root_path}/{dataset_name}"
    lmdb_path=f"{output_lmdb_dir}/{dataset_name}_lmdb"
    create_lmdb_dataset(dataset_dir_path, lmdb_path, dataset_name,map_size=int(dataset_size) * 1024 * 1024 * 1024)
    #read_lmdb(lmdb_path)
