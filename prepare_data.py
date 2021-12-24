# -*- coding: utf-8 -*-
from io import open
import os.path
from os import path
import random
import numpy as np
import pickle
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from glob import glob

from sklearn.model_selection import train_test_split

file_path = np.array(glob('data/train/kaai.csv'))

class TrajectoryDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, length=40, predict_length=30, file_path=file_path):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """

        self.X_frames_trajectory = []
        self.Y_frames_trajectory = []

        self.length = length
        self.predict_length = predict_length
        for csv_file in file_path:
            self.csv_file = csv_file
            self.load_data()
        self.normalize_data()

    def __len__(self):
        return len(self.X_frames_trajectory)

    def __getitem__(self, idx):
        single_trajectory_data = self.X_frames_trajectory[idx]
        single_trajectory_label = self.Y_frames_trajectory[idx]
        return (single_trajectory_data, single_trajectory_label)

    def load_data(self):
        dataS = pd.read_csv(self.csv_file)
        count_ = []
        for vid in dataS.ID.unique():  # 
            frame_ori = dataS[dataS.ID == vid]  # Access all vehicle data

             # Of all the data the vehicle has, only the column below is imported into the column of the frame
            frame = dataS[
                    ['Position X', 'Position Z', 'car speed(m/s)', 'Yaw Angle (degree)', 
                     'LB Position X','LB Position Z', 'LB speed',
                     'LF Position X', 'LF Position Z','LF speed', 
                     'B Position X', 'B Position Z','B speed', 
                     'F Position X', 'F Position Z','F speed',
                     'RB Position X', 'RB Position Z','RB speed', 
                     'RF Position X', 'RF Position Z','RF speed', 
                     'LC',
                    ]]
           # The number of input features : 23
            
            frame = np.asarray(frame) # Change pandas DataFrame to numpy
            
            dis = frame[1:, :2] - frame[:-1, :2] # Calculate the difference in distance. 
            dis = dis.astype(np.float64) 
            dis = np.sqrt(np.power(dis[:, 0], 2) + np.power(dis[:, 1], 2)) # Find the distance between dots.

            # Visualize trajectory
            plt.plot(frame[:,0],frame[:,1],c='r',label='trajectory')
            plt.plot(frame[0,0],frame[0,1],c='k',marker='o') # Starting point :  black
            plt.plot(frame[-1,0],frame[-1,1],c='b',marker='o') # Arrival point : blue
            plt.xlabel('Position X')
            plt.ylabel('Position Z')
            plt.legend()
            plt.show()
            

            frame[:, 0:2] = scipy.signal.savgol_filter(frame[:, 0:2], window_length=21, polyorder=3, axis=0)

            All_vels = []
            for i in range(1):
                # Calculate the speed in the x direction.
                x_vel = (frame[1:, 0 + i * 5] - frame[:-1, 0 + i * 5]) / 0.1; # Speed = distance/time
                v_avg = (x_vel[1:] + x_vel[:-1]) / 2.0;
                v_begin = [2.0 * x_vel[0] - v_avg[0]];
                v_end = [2.0 * x_vel[-1] - v_avg[-1]];
                velx = (v_begin + v_avg.tolist() + v_end)
                velx = np.array(velx)
                
                # Calculate the speed in the y direction.
                y_vel = (frame[1:, 1 + i * 5] - frame[:-1, 1 + i * 5]) / 0.1;  # Speed = distance/time
                vy_avg = (y_vel[1:] + y_vel[:-1]) / 2.0;
                vy1 = [2.0 * y_vel[0] - vy_avg[0]];
                vy_end = [2.0 * y_vel[-1] - vy_avg[-1]];
                vely = (vy1 + vy_avg.tolist() + vy_end)
                vely = np.array(vely)

                if isinstance(All_vels, (list)):
                    All_vels = np.vstack((velx, vely))
                else:
                    All_vels = np.vstack((All_vels, velx.reshape(1, -1)))
                    All_vels = np.vstack((All_vels, vely.reshape(1, -1)))
                    
            All_vels = np.transpose(All_vels)
            total_frame_data = np.concatenate((All_vels[:, :2], frame), axis=1)

            X = total_frame_data[:-self.predict_length, :] 
            Y = total_frame_data[self.predict_length:, :4]
         
            count = 0  
            
            for i in range(X.shape[0] - self.length): 
                self.X_frames_trajectory = self.X_frames_trajectory + [X[i:i + self.length, :]]              
                self.Y_frames_trajectory = self.Y_frames_trajectory + [Y[i:i + self.length, :]]

                count = count + 1
            count_.append(count)
            
            print('Sum Trajectory:',np.sum(count_),'Average Trajectory:', np.mean(count_))

    def normalize_data(self):  # Standardize input data for each vehicle.
        A = [list(x) for x in zip(*(self.X_frames_trajectory))]
        
        A = np.array(A).astype(np.float64)
        A = torch.from_numpy(A)
        
        A = A.view(-1, A.shape[2])
        
        if self.csv_file.split("/")[1] == 'train':
            # Store mean, standard deviation and range for train dataset
            self.mn = torch.mean(A, dim=0)
            self.range = (torch.max(A, dim=0).values - torch.min(A, dim=0).values) / 2.0
            self.range = torch.ones(self.range.shape, dtype=torch.double)
            self.std = torch.std(A, dim=0)
            std = self.std.numpy()    
            mn = self.mn.numpy() 
            rg = self.range.numpy()    
           
            np.savetxt("std.txt", std)
            np.savetxt("mean.txt", mn)
            np.savetxt("rg.txt", rg)
            print('std, mean, rgs textfile saved!!!')
        else:
            mn= torch.from_numpy(np.loadtxt('mean.txt'))
            std = torch.from_numpy(np.loadtxt('std.txt'))
            rg = torch.from_numpy(np.loadtxt('rg.txt'))
            self.mn = mn
            self.range = rg
            self.std = std
        

        self.X_frames_trajectory = np.array(self.X_frames_trajectory)
        self.Y_frames_trajectory = np.array(self.Y_frames_trajectory)

        self.X_frames_trajectory = (self.X_frames_trajectory - mn) / (std * rg)
        self.Y_frames_trajectory = (self.Y_frames_trajectory - mn[:4]) / (std[:4] * rg[:4])



def get_dataloader(BatchSize=64, length=40, predict_length=30,file_path = np.array(glob('data/train/kaai.csv')),daset = 'train'):
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''

    dataset = TrajectoryDataset(length, predict_length,file_path) 
    
    # split dataset into train test and validation 8:1:1
    length_traj = dataset.__len__()
    print('length_traj : ', length_traj) 
    num_train_traj = (int)(length_traj * 0.8)
    num_test_traj = (int)(length_traj * 0.9)
  
    # ------------ train set ------------ # 
    train_x = dataset.X_frames_trajectory[:num_train_traj]
    train_y = dataset.Y_frames_trajectory[:num_train_traj]

    train_x = torch.tensor(train_x, dtype=torch.double)
    train_y = torch.tensor(train_y, dtype=torch.double)
    
    # --------- validation set ---------- #
    validation_x = dataset.X_frames_trajectory[num_train_traj:num_test_traj]
    validation_y = dataset.Y_frames_trajectory[num_train_traj:num_test_traj]
    
    validation_x = torch.tensor(validation_x, dtype=torch.double)
    validation_y = torch.tensor(validation_y, dtype=torch.double)
    
    # ------------- test set ------------ #
    test_x = dataset.X_frames_trajectory[num_test_traj:]
    test_y = dataset.Y_frames_trajectory[num_test_traj:]
    
    test_x = torch.tensor(test_x, dtype=torch.double)
    test_y = torch.tensor(test_y, dtype=torch.double)
    
    # ----------Tensor Dataset ----------- #
    train_traj = TensorDataset(train_x, train_y)
    validation_traj = TensorDataset(validation_x, validation_y)
    test_traj = TensorDataset(test_x, test_y)

    # ----------- Data Loader ------------ #
    # Since it is time series data, shuffle is set as False.
    train_loader_traj = DataLoader(train_traj, batch_size=BatchSize, shuffle=False) 
    test_loader_traj = DataLoader(test_traj, batch_size=BatchSize, shuffle=False) 
    validation_loader_traj = DataLoader(validation_traj, batch_size=BatchSize, shuffle=False)
    
    print('train_loader_traj : ',len(train_loader_traj))
    return (train_loader_traj,test_loader_traj,validation_loader_traj,dataset)

