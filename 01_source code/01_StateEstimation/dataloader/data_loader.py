"""
created by
"""
import os
import torch
import pickle
import random
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


class DataLoaderModule:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tactile_data_curr_norm_th, self.tactile_data_curr_norm_ff, self.tactile_data_curr_norm_mf = [], [], []
        self.chamber_press_force_th, self.chamber_press_force_ff, self.chamber_press_force_mf          = [], [], []
        self.fing_press_norm_value                                                                     = []
        self.marker_poses_offset2palm_norm_pts_th = []
        self.marker_poses_offset2palm_norm_pts_ff = []
        self.marker_poses_offset2palm_norm_pts_mf = []
        self.filename = []

        filename_list = os.listdir(self.data_dir)
        random.shuffle(filename_list)

        for filename in filename_list:
            if filename.endswith('.pickle'):
                file_path = os.path.join(self.data_dir, filename)

                with open(file=file_path, mode='rb') as file:
                    self.file_data = pickle.load(file)
                    self.data = self.file_data

                    # input
                    # tactile
                    self.tactile_data_curr_norm_th.append(self.data['tactile_data_curr']['norm']['th'][:8, :])
                    self.tactile_data_curr_norm_ff.append(self.data['tactile_data_curr']['norm']['ff'][:8, :])
                    self.tactile_data_curr_norm_mf.append(self.data['tactile_data_curr']['norm']['mf'][:8, :])
                    # chamber
                    self.chamber_press_force_th.append(self.data['tactile_data_curr']['norm']['th'][-1:, :])
                    self.chamber_press_force_ff.append(self.data['tactile_data_curr']['norm']['ff'][-1:, :])
                    self.chamber_press_force_mf.append(self.data['tactile_data_curr']['norm']['mf'][-1:, :])
                    # pressure
                    self.fing_press_norm_value.append(self.data['fing_press']['norm']['value'])

                    # output
                    self.marker_poses_offset2palm_norm_pts_th.append(self.data['marker_poses_offset2palm']['norm']['pts_th'])
                    self.marker_poses_offset2palm_norm_pts_ff.append(self.data['marker_poses_offset2palm']['norm']['pts_ff'])
                    self.marker_poses_offset2palm_norm_pts_mf.append(self.data['marker_poses_offset2palm']['norm']['pts_mf'])

                    # filename
                    self.filename.append(filename)

    def __len__(self):
        lists = [self.tactile_data_curr_norm_th, self.tactile_data_curr_norm_ff, self.tactile_data_curr_norm_mf,
                 self.chamber_press_force_th, self.chamber_press_force_ff, self.chamber_press_force_mf,
                 self.fing_press_norm_value,
                 self.marker_poses_offset2palm_norm_pts_th,
                 self.marker_poses_offset2palm_norm_pts_ff,
                 self.marker_poses_offset2palm_norm_pts_mf]
        if all(len(lst) == len(lists[0]) for lst in lists):
            return len(lists[0])
        else:
            raise ValueError("Data length and format are abnormal!")

    def __getitem__(self):
        return (self.tactile_data_curr_norm_th, self.tactile_data_curr_norm_ff, self.tactile_data_curr_norm_mf,
                self.chamber_press_force_th, self.chamber_press_force_ff, self.chamber_press_force_mf,
                self.fing_press_norm_value,
                self.marker_poses_offset2palm_norm_pts_th,
                self.marker_poses_offset2palm_norm_pts_ff,
                self.marker_poses_offset2palm_norm_pts_mf)


def load_data(data_dir, batch_size=8, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, train_flag=True):
    data_pickle = DataLoaderModule(data_dir=data_dir)
    data_size = data_pickle.__len__()
    if (train_ratio + val_ratio + test_ratio) != 1:
        raise ValueError("Invalid dataset ratio!")
    else:
        train_val = int(data_size * train_ratio)
        val_test = int(data_size * (1 - test_ratio))

    if train_flag:
        train_set = TensorDataset(torch.tensor(np.array(data_pickle.tactile_data_curr_norm_th[:train_val])),
                                  torch.tensor(np.array(data_pickle.tactile_data_curr_norm_ff[:train_val])),
                                  torch.tensor(np.array(data_pickle.tactile_data_curr_norm_mf[:train_val])),
                                  torch.tensor(np.array(data_pickle.chamber_press_force_th[:train_val])),
                                  torch.tensor(np.array(data_pickle.chamber_press_force_ff[:train_val])),
                                  torch.tensor(np.array(data_pickle.chamber_press_force_mf[:train_val])),
                                  torch.tensor(np.array(data_pickle.fing_press_norm_value[:train_val])),
                                  torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_th[:train_val])),
                                  torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_ff[:train_val])),
                                  torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_mf[:train_val])))
        val_set = TensorDataset(torch.tensor(np.array(data_pickle.tactile_data_curr_norm_th[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.tactile_data_curr_norm_ff[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.tactile_data_curr_norm_mf[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.chamber_press_force_th[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.chamber_press_force_ff[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.chamber_press_force_mf[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.fing_press_norm_value[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_th[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_ff[train_val:val_test])),
                                torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_mf[train_val:val_test])),)
        test_set = TensorDataset(torch.tensor(np.array(data_pickle.tactile_data_curr_norm_th[val_test:])),
                                 torch.tensor(np.array(data_pickle.tactile_data_curr_norm_ff[val_test:])),
                                 torch.tensor(np.array(data_pickle.tactile_data_curr_norm_mf[val_test:])),
                                 torch.tensor(np.array(data_pickle.chamber_press_force_th[val_test:])),
                                 torch.tensor(np.array(data_pickle.chamber_press_force_ff[val_test:])),
                                 torch.tensor(np.array(data_pickle.chamber_press_force_mf[val_test:])),
                                 torch.tensor(np.array(data_pickle.fing_press_norm_value[val_test:])),
                                 torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_th[val_test:])),
                                 torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_ff[val_test:])),
                                 torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_mf[val_test:])),)
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_set, batch_size=1, shuffle=False)
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
        return train_loader, val_loader, test_loader

    else:
        test_set = TensorDataset(torch.tensor(np.array(data_pickle.tactile_data_curr_norm_th[:])),
                                 torch.tensor(np.array(data_pickle.tactile_data_curr_norm_ff[:])),
                                 torch.tensor(np.array(data_pickle.tactile_data_curr_norm_mf[:])),
                                 torch.tensor(np.array(data_pickle.chamber_press_force_th[:])),
                                 torch.tensor(np.array(data_pickle.chamber_press_force_ff[:])),
                                 torch.tensor(np.array(data_pickle.chamber_press_force_mf[:])),
                                 torch.tensor(np.array(data_pickle.fing_press_norm_value[:])),
                                 torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_th[:])),
                                 torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_ff[:])),
                                 torch.tensor(np.array(data_pickle.marker_poses_offset2palm_norm_pts_mf[:])))
        test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)
        return test_loader, data_pickle.filename
