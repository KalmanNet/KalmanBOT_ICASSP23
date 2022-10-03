# Kalman Filter Mean Reversion Strategy
import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.tsa.stattools as ts
from myUtils import *
# from Linear_KF_hedge import KalmanFilter
# from Linear_sysmdl import SystemModel
import datetime
from sklearn.linear_model import LinearRegression
from KalmanNet_nn import KalmanNetNN
from KalmanNet_sysmdl import SystemModel
# from Pipeline_Offline_hedge import Pipeline_Offline
from Pipeline_EKF import Pipeline_EKF

os.environ['KMP_DUPLICATE_LIB_OK']='True'

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")

dataFolderName = 'HedgeRatio' + '/'
if not os.path.exists(dataFolderName):
    os.makedirs(dataFolderName)


train_size = 2000 ########################
traj_length = 500 ########################
Chan = 0 # Change data source

if Chan==0:
    df=pd.read_csv(dataFolderName+'forex1421'+'.csv', index_col='time')
    df_train = df.iloc[:train_size]
    df = df.iloc[train_size:]

    x=df['chf']
    y=df['eur'].values

    q2, r2, R_0, beta_0 = prepare_forex(df_train)

else:
    df=pd.read_csv(dataFolderName+'EWA_EWC'+'.csv')
    df['Date']=pd.to_datetime(df['Date'], format='%Y%m%d').dt.date # remove HH:MM:SS
    df.set_index('Date', inplace=True)
    x=df['EWA']
    y=df['EWC'].values
    delta=0.0001 # delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).
    q2=delta/(1-delta)
    r2=0.001
    R_0 = torch.zeros((2,2))
    beta_0=torch.tensor([[0.], [0.]])

training_set = train_set(df_train, traj_length, train_size)

F = torch.eye(2)
H = torch.tensor([[1., 1.]])
T = T_test = traj_length ###### short traj length

def f(x):
    return torch.matmul(F,x)
def h(x):
    return torch.matmul(H,x)

# KNet_unsuper_train(training_set, q2, r2, R_0, beta_0, F, f, H, h, T, T_test, train_size, traj_length, dataFolderName)

e, Q, beta = KNet_unsuper_test(df, dataFolderName, dev, beta_0, len(df))

y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
x = np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.

H = torch.tensor(x, dtype=torch.float32)
T = T_test = x.shape[0]

e_1, Q_1, beta_1 = KF_result(x, y, q2, r2, R_0, beta_0, F, H, T, T_test)

const_e = 100
pnl_1, cum_pnl_1 = PnL_new(e*const_e, Q, df, beta)
pnl_2, cum_pnl_2 = PnL_new(e_1, Q_1, df, beta_1)

np.save(dataFolderName+'cum_KF_bollinger', cum_pnl_2)
np.save(dataFolderName+'cum_KNet_unsupervised_bollinger', cum_pnl_1)
