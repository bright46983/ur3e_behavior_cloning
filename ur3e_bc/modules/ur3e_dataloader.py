import os
import zipfile
import torch
import pickle as pkl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import re
import resource
import sys


class UR3EDataset(Dataset):
    def __init__(self, zip_dir):
        """
        Args:
            zip_dir (str): Path to the directory containing ZIP files.
        """
        # param
        self.frame_hist = 3 # number of previous frames
        self.frame_skipped = 2 # how far apart of each frame in timestep

        self.zip_dir = zip_dir
        self.zip_files = sorted([f for f in os.listdir(zip_dir) if f.endswith(".zip")])
        self.metadata_files = sorted([f for f in os.listdir(zip_dir) if f.endswith(".pkl")])
        self.episode_dict = self._group_episodes()
        
        # print(self.episode_dict)

        # Collect valid (t-2, t-1, t) sequences
        self.valid_sequences = self._create_valid_sequences()
        self.metadata_dict = self._create_metadata()

    def _group_episodes(self):
        """Groups files inside each ZIP by episode."""
        episode_dict = {}

        for zip_filename in self.zip_files:
            zip_path = os.path.join(self.zip_dir, zip_filename)
            with zipfile.ZipFile(zip_path, "r") as zipf:
                pkl_files = sorted([f for f in zipf.namelist() if f.endswith(".pkl")])
                sorted_files = sorted(pkl_files, key=self.extract_step_number)
            
            episode_dict[zip_filename] = sorted_files  # Store files in order

        return episode_dict

    def _create_valid_sequences(self):
        """Finds valid (t-2, t-1, t) sequences in each episode."""
        valid_sequences = []
        for zip_filename, files in self.episode_dict.items():
            for i in range(len(files)):
                data = [zip_filename]
                for j in range(0,self.frame_hist*self.frame_skipped, self.frame_skipped):
                    if i-j < 0:
                        data.append(files[0])
                    else:
                        data.append(files[i-j])
                
                valid_sequences.append(tuple(data))
                        
            # for i in range((self.frame_hist-1)*self.frame_skipped,len(files)):  # Ensure (t-2, t-1, t) exist
            #     valid_sequences.append((zip_filename, files[i-self.frame_skipped*2], files[i-self.frame_skipped], files[i]))
        # print(valid_sequences)
        return valid_sequences
    
    def _create_metadata(self):

        metadata = {}

        for meta_filename in self.metadata_files:
            pkl_path = os.path.join(self.zip_dir, meta_filename)
            print(pkl_path)
            with open(pkl_path, 'rb') as f:
                data = pkl.load(f)
            identifier = meta_filename.replace("metadata_", "").replace(".pkl", ".zip")
            metadata[identifier] = data
            
        return metadata

    def __len__(self):
        return len(self.valid_sequences)

    def _load_pkl_from_zip(self, zip_filename, pkl_filename):
        """Loads a pickle file from inside a ZIP archive."""
        zip_path = os.path.join(self.zip_dir, zip_filename)
        with zipfile.ZipFile(zip_path, "r") as zipf:
            with zipf.open(pkl_filename, "r") as f:
                return pkl.load(f)
            
    def extract_step_number(self,filename):
        # Extract the last number (step number) from the filename
        match = re.search(r"_(\d+)\.pkl$", filename)
        # print(int(match.group(1)) if match else -1)
        return int(match.group(1)) if match else -1  # Return -1 if no number is found
    
    def _get_episode(self,zip_filename):
        trajectories = []
        for step in self.episode_dict[zip_filename]:
            data = self._load_pkl_from_zip(zip_filename,step)
            trajectories.append(data)
        return trajectories


    def __getitem__(self, idx):
        """Loads a sequence of (t-2, t-1, t) as input and output from t."""
        zip_filename,t_file, t_minus_1_file, t_minus_2_file   = self.valid_sequences[idx]
        print(zip_filename,t_file, t_minus_1_file, t_minus_2_file)

        # Load all 3 frames
        data_t_minus_2 = self._load_pkl_from_zip(zip_filename, t_minus_2_file)
        data_t_minus_1 = self._load_pkl_from_zip(zip_filename, t_minus_1_file)
        data_t = self._load_pkl_from_zip(zip_filename, t_file)  # Output from @t

        # Extract input images (Normalize 0-1)
        front_cam = torch.tensor(np.stack([
            data_t_minus_2["frame1"], data_t_minus_1["frame1"], data_t["frame1"]
        ]), dtype=torch.float32) / 255.0  # (3, 180,240,3)

        side_cam = torch.tensor(np.stack([
            data_t_minus_2["frame2"], data_t_minus_1["frame2"], data_t["frame2"]
        ]), dtype=torch.float32) / 255.0

        hand_cam = torch.tensor(np.stack([
            data_t_minus_2["frame3"], data_t_minus_1["frame3"], data_t["frame3"]
        ]), dtype=torch.float32) / 255.0

        # Extract input EE pose
        ee_pose = torch.tensor(np.stack([
            data_t_minus_2["ee_pose"].reshape(7),
            data_t_minus_1["ee_pose"].reshape(7),
            data_t["ee_pose"].reshape(7)
        ]), dtype=torch.float32)  # (3,7)

        # Extract outputs from @t
        ee_velocity = torch.tensor(data_t["ee_vel"].reshape(6), dtype=torch.float32)  # (6)
        hole_pose = torch.tensor(data_t["hole_pose"].reshape(7), dtype=torch.float32)  # (7)
        state = torch.tensor(data_t["state"], dtype=torch.int64)  # (1)

        return {
            "front_cam": front_cam,  # (3, 240, 180)
            "side_cam": side_cam,    # (3, 240, 180)
            "hand_cam": hand_cam,    # (3, 240, 180)
            "ee_pose": ee_pose,      # (3, 7)
            "ee_velocity": ee_velocity,  # (6)
            "hole_pose": hole_pose,  # (7)
            "state": state           # (1)
        }

# # Example usage
# zip_dir = "/home/tanakrit-ubuntu/ur3e_mujoco_tasks/scripts/data"
# batch_size = 16

# dataset = UR3EDataset(zip_dir)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
# print(dataset.zip_files)
# print(dataset._get_episode("20250407_122419_0.zip"))
# print(dataset.metadata_dict["20250407_122419_0.zip"])

# # Example: Fetch one batch
# for batch in dataloader:
#     print(batch)
    
