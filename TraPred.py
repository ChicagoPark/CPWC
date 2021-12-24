# Trajectory Prediction Model Using BILSTM
# 1. Import Module
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import unicodedata
import string
import re
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import torch.distributions as dist

class TraPred(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,batch_size, dropout=0.8):
        super(TraPred, self).__init__()
        
        self.batch_size = batch_size # Batch size
        self.hidden_size = hidden_size # The number of nodes in the LSTM hidden layer. 
        self.num_layers = 2 # The number of LSTM hidden layers.

        self.in2lstm = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=self.num_layers,bidirectional=False,batch_first=True,dropout =dropout)
        self.in2bilstm = nn.Linear(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size//2,num_layers=self.num_layers,bidirectional=True,batch_first=True,dropout =dropout)
    
        self.fc0 = nn.Linear(hidden_size,hidden_size*2)
        self.fc1 = nn.Linear(hidden_size*2,int(hidden_size/2))
        self.in2out = nn.Linear(input_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2) ,output_size)
        self.tanh = nn.Tanh()
        
    def forward(self, input):
        bilstm_out,_= self.bilstm(self.in2bilstm(input))
        lstm_out,_= self.lstm(self.in2lstm(input))
        out = self.tanh(self.fc0(lstm_out+bilstm_out))
        out = self.tanh(self.fc1(out))
        out =  out + self.in2out(input)
        output = self.fc2(out)# range [0 -> 1]
        return output

