'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2023-03-30

The code is specifically designed for generating nearest sample pairs for Face X-ray. 
Alternatively, you can utilize the pre-generated pkl files available in our GitHub repository. Please refer to the "Releases" section on our repository for accessing these files.
'''

import os
import json
import pickle
import numpy as np
import heapq
import random
from tqdm import tqdm
from scipy.spatial import KDTree


def load_landmark(file_path):
    """
    Load 2D facial landmarks from a file path.

    Args:
        file_path: A string indicating the path to the landmark file.

    Returns:
        A numpy array containing the loaded landmarks.

    Raises:
        None.
    """
    if file_path is None:
        return np.zeros((81, 2))
    if os.path.exists(file_path):
        landmark = np.load(file_path)
        return np.float32(landmark)
    else:
        return np.zeros((81, 2))


def get_landmark_dict(dataset_folder):
    # Check if the dictionary has already been created
    if os.path.exists('landmark_dict_ff.pkl'):
        with open('landmark_dict_ff.pkl', 'rb') as f:
            return pickle.load(f)
    # Open the metadata file for the current folder
    metadata_path = os.path.join(dataset_folder, "FaceForensics++.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    # Iterate over the metadata entries and add the landmark paths to the list
    ff_real_data = metadata['FaceForensics++']['FF-real']
    # Using dictionary comprehension to generate the landmark_dict
    landmark_dict = {
        frame_path.replace('frames', 'landmarks').replace(".png", ".npy"): load_landmark(
            frame_path.replace('frames', 'landmarks').replace(".png", ".npy")
        )
        for mode, value in ff_real_data.items()
        for video_name, video_info in tqdm(value['c23'].items())
        for frame_path in video_info['frames']
    }
    # Save the dictionary to a pickle file
    with open('landmark_dict_ffall.pkl', 'wb') as f:
        pickle.dump(landmark_dict, f)
    return landmark_dict


def get_nearest_faces_fixed_pair(landmark_info, num_neighbors):
    '''
    Using KDTree to find the nearest faces for each image (Much faster!!)
    '''
    random.seed(1024)  # Fix the random seed for reproducibility

    # Check if the dictionary has already been created
    if os.path.exists('nearest_face_info.pkl'):
        with open('nearest_face_info.pkl', 'rb') as f:
            return pickle.load(f)

    landmarks_array = np.array([lmk.flatten() for lmk in landmark_info.values()])
    landmark_ids = list(landmark_info.keys())

    # Build a KDTree using the flattened landmarks
    tree = KDTree(landmarks_array)

    nearest_faces = {}
    for idx, this_lmk in tqdm(enumerate(landmarks_array), total=len(landmarks_array)):
        # Query the KDTree for the nearest neighbors (excluding itself)
        dists, indices = tree.query(this_lmk, k=num_neighbors + 1)
        # Randomly pick one from the nearest N neighbors (excluding itself)
        picked_idx = random.choice(indices[1:])
        nearest_faces[landmark_ids[idx]] = landmark_ids[picked_idx]

    # Save the dictionary to a pickle file
    with open('nearest_face_info.pkl', 'wb') as f:
        pickle.dump(nearest_faces, f)

    return nearest_faces


def get_nearest_faces(landmark_info, num_neighbors):
    '''
    Using KDTree to find the nearest faces for each image (Much faster!!)
    '''
    random.seed(1024)  # Fix the random seed for reproducibility

    # Check if the dictionary has already been created
    if os.path.exists('nearest_face_info.pkl'):
        with open('nearest_face_info.pkl', 'rb') as f:
            return pickle.load(f)

    landmarks_array = np.array([lmk.flatten() for lmk in landmark_info.values()])
    landmark_ids = list(landmark_info.keys())

    # Build a KDTree using the flattened landmarks
    tree = KDTree(landmarks_array)

    nearest_faces = {}
    for idx, this_lmk in tqdm(enumerate(landmarks_array), total=len(landmarks_array)):
        # Query the KDTree for the nearest neighbors (excluding itself)
        dists, indices = tree.query(this_lmk, k=num_neighbors + 1)
        # Store the nearest N neighbors (excluding itself)
        nearest_faces[landmark_ids[idx]] = [landmark_ids[i] for i in indices[1:]]

    # Save the dictionary to a pickle file
    with open('nearest_face_info.pkl', 'wb') as f:
        pickle.dump(nearest_faces, f)

    return nearest_faces

# Load the landmark dictionary and obtain the landmark dict
dataset_folder = "/home/zhiyuanyan/disfin/deepfake_benchmark/preprocessing/dataset_json/"
landmark_info = get_landmark_dict(dataset_folder)

# Get the nearest faces for each image (in landmark_dict)
num_neighbors = 100
nearest_faces_info = get_nearest_faces(landmark_info, num_neighbors)  # running time: about 20 mins
