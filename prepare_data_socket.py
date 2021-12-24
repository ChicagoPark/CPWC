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
from torch.utils.data import TensorDataset, DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from glob import glob

class TrajectoryDataset():
    """Face Landmarks dataset."""

    def __init__(self, DATA):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """

        self.X_frames_trajectory = []
        self.Y_frames_trajectory = []

        self.DATA = DATA
        self.load_data_sy()
        self.normalize_data()

    def __len__(self):
        return len(self.X_frames_trajectory)

    def __getitem__(self, idx):
        single_trajectory_data = self.X_frames_trajectory[idx]
        single_trajectory_label = self.Y_frames_trajectory[idx]
        return (single_trajectory_data, single_trajectory_label)

    def load_data_sy(self):
        
        dataS = pd.DataFrame(self.DATA) 
        frame = np.asarray(dataS) # Change pandas DataFrame -> numpy.ndarray


        dis = frame[1:, :2] - frame[:-1, :2] # Calculate the difference in distance. 
        dis = dis.astype(np.float64)
        dis = np.sqrt(np.power(dis[:, 0], 2) + np.power(dis[:, 1], 2)) # Calculate the difference in distance. 

        frame[:, 0:2] = scipy.signal.savgol_filter(frame[:, 0:2], window_length=21, polyorder=3, axis=0)
        
        All_vels = [] 
        for i in range(1):
            x_vel = (frame[1:, 0 + i * 5] - frame[:-1, 0 + i * 5]) / 0.1; # Speed = distance/time
            v_avg = (x_vel[1:] + x_vel[:-1]) / 2.0;  
            v_begin = [2.0 * x_vel[0] - v_avg[0]];   
            v_end = [2.0 * x_vel[-1] - v_avg[-1]];   
            velx = (v_begin + v_avg.tolist() + v_end)
            velx = np.array(velx)                    

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

        X = total_frame_data[:, :]  
        Y = total_frame_data[:, :4]
                
        self.X_frames_trajectory = X
        self.Y_frames_trajectory = Y

    def normalize_data(self):
        mn= np.loadtxt('mean.txt')
        std = np.loadtxt('std.txt')
        rg = np.loadtxt('rg.txt')
        self.mn = mn
        self.range = rg
        self.std = std
        
        # Change variable's type from list to numpy.ndarray
        self.X_frames_trajectory = np.array(self.X_frames_trajectory)
        self.Y_frames_trajectory = np.array(self.Y_frames_trajectory)
        
        # Normalization using mean, standard
        self.X_frames_trajectory = (self.X_frames_trajectory - mn) / (std * rg)
        self.Y_frames_trajectory = (self.Y_frames_trajectory - mn[:4]) / (std[:4] * rg[:4])
        
        print('DATA normalize successed!!')

def calcu_XY(predY):
    '''
    deltaY = v0*delta_t + 0.5* a *delta_t^2
    a = (v - v0)/delta_t
    vo
    '''
    predict_length = 30
    vels = predY[:,:,0:2] 
    rst_xy = np.zeros(predY[:,:,:4].shape)
    rst_xy[:,:-predict_length,:] = predY[:,:-predict_length,:4]
    # index 0,1 of predY : speed 
    # index 2,3 of predY : location  
    
    delta_t = 0.1 # Data were collected at intervals of 0.1 second.
    
    for i in range(predict_length):
        a = (vels[:,-(predict_length-i),:] - vels[:,-(predict_length+1-i),:])/delta_t
        delta_xy = vels[:,-(predict_length-i),:]*vels[:,-(predict_length-i),:]-vels[:,-(predict_length+1-i),:]*vels[:,-(predict_length+1-i),:]
        delta_xy = delta_xy/(2*a)
        # Optimize to increase accuracy.
        rst_xy[:,-(predict_length-i),2:4] = rst_xy[:,-(predict_length+1-i),2:4] + delta_xy
    
    return rst_xy        
        
def get_dataloader(DATA):
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    dataset = TrajectoryDataset(DATA)

    length_traj = dataset.__len__()
    print('length_traj : ', length_traj)
        
    return dataset

