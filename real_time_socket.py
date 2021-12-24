import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import visdom
import socket

from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from LSTM_FCN import *
from TraPred import *
from prepare_data_socket import *

time_step=30
global inputdata

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
    
print('Using Pytorch Version: ',torch.__version__,'Device: ',DEVICE)


# Load Lane Change Maneuver Model
model=LSTMFCN(30, 20) # Set Model
# Get the parameters of the model saved through training.
model.load_state_dict(torch.load('model/train_0927.pth'))
model = model.to(DEVICE) 
model.eval() # Model evaluation mode.

# Load Trajectory Prediction Model
Prednet = TraPred(25, 4, 256, 128)
Prednet.load_state_dict(torch.load('model/trajectory_predict_UB-LSTM.pt'))
# Get the parameters of the model saved through training.
Prednet = Prednet.double() 
Prednet = Prednet.to(DEVICE) 
Prednet.eval() # Model evaluation mode.

def load_data(inputdata):
    # Get data through socket communication
    data = client_socket_1.recv(1024)
    logdata = repr(data.decode('utf-8'))
    
    logdata = str(logdata) # Change logdata to string format
    logdata = np.fromstring(logdata.strip("''"), dtype=float, sep=',')
    logdata = logdata[0:]
    logdata = np.reshape(logdata, ((1,)+logdata.shape))


    inputdata = np.vstack([inputdata, logdata]) # Add logdata to inputdata
    return inputdata

def prepare_data(behav_data): # Prepare 30 frames data for Lane Change Maneuver Model
    while True:
        behav_data = load_data(behav_data)
        if len(behav_data) == 30: # If the frame is piled up by 30,
            behav_data = np.delete(behav_data, 0, 0) # Delete the row you used to initialize.
            break      
    return behav_data # return 29 frames 

HOST = '10.3.70.105' # UC-Win/Road server IPV4 address

# Port number specified by the server.
PORT_1 = 1000 #LogDisplay
PORT_2 = 1500 #SendData

behav_data = np.zeros((1, 22)) #Initialize inputdata  (1rows, 22columns)
traj_data = np.zeros((1, 23))

# Create a clinet socket object. 
# It uses TCP as an address family, IPv4, and socket type.
client_socket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# Access the server using the specified HOST and PORT. 
client_socket_1.connect((HOST, PORT_1))
client_socket_2.connect((HOST, PORT_2))
behav_data = prepare_data(behav_data) # Initial: 29 frames are included.

while True:
    behav_data = load_data(behav_data) # 30 frames
    
    # BEHAVIOR PREDICTION 
    # Pre-processeing inputdata
    test_x = np.reshape(behav_data, ((1,) + behav_data.shape))
    test_x = torch.tensor(test_x[:, :, 2:], dtype=torch.float32)
    test_x = test_x.to(DEVICE)

    output = model(test_x)
    prediction = output.max(1, keepdim = True)[1]
    prediction=prediction[0].cpu().numpy()
#     print("log: ", behav_data[-1]) # error (not global variables)
    print("maneuver: ",prediction[0])
    pre=np.hstack((behav_data[-1],prediction))
    #print("with pre", pre.shape)
    pre = np.reshape(pre, (1,23))
    traj_data = np.vstack([traj_data, pre]) # Prepare data for trajectory prediction
    behav_data = np.delete(behav_data, 0, 0) # Delete the first rows for the next loop.

    # TRAJECTORY PREDICTION
    if (len(traj_data) == 31): 
        if (traj_data[-1,-1] == 1 or traj_data[-1,-1] == 2): 
            # In case of LANE CHANGE and to avoid overlapping, run only when the STACK is full,
            # For route optimization, we receive data in line 31 through INPUT of Trajectory prediction model.
           
            DATASET = get_dataloader(traj_data)
            
            # Bring back mean, standard, range values saved during training
            std = DATASET.std[:4]
            mn = DATASET.mn[:4]
            rg = DATASET.range[:4]
            
            inputs = DATASET.X_frames_trajectory # numpy.ndarray
            inputs = inputs.reshape(1,inputs.shape[0],inputs.shape[1]) # Adding dimensions to put into the model.
            inputs = torch.tensor(inputs).to(DEVICE)
            outputs = Prednet(inputs)

          
            
            outputs = outputs.detach().cpu()
            outputs = outputs.numpy() # outputs.shape :  (1, 31, 4)
            outputs = (outputs*(rg*std)+mn) # Reverse normalization

            rst_xy = calcu_XY(outputs) # rst_xy.shape : (1, 31, 4)
            rst_xy = rst_xy.reshape(rst_xy.shape[1], rst_xy.shape[2]) # rst_xy.shape :  (31, 4)
            
            outputs = rst_xy[:,2:4] # [2:4] contains the predictions X and Y of data.

            print(' ------------ Lane Change Success!! --------------- ')
            try:
                TRAJECTORT_PREDICT = ''
                for output in outputs:
                    print(output)
                    TRAJECTORT_PREDICT += str(output[0])+','+str(output[1])+','
                    
                TRAJECTORT_PREDICT = TRAJECTORT_PREDICT[:-1] # remove ',' at the end     
                client_socket_2.send(TRAJECTORT_PREDICT.encode())

                traj_data = np.zeros((1, 23)) # Initialize input data
                continue
            except socket.error as e:
                print(e)
        else:
            traj_data = np.delete(traj_data, 0, 0)