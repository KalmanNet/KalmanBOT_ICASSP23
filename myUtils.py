import torch
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as ts
from Linear_KF_hedge import KalmanFilter
from Linear_KF_flow import KF_flow
from Linear_sysmdl import System_Model
import datetime
from sklearn.linear_model import LinearRegression
from KalmanNet_nn import KalmanNetNN
from KalmanNet_sysmdl import SystemModel
# from Pipeline_Offline_hedge import Pipeline_Offline
from Pipeline_EKF import Pipeline_EKF
from Pipeline_trading import Pipeline_trading
from position_MLP import position_MLP
from KNet_delta import KNet_delta
from position_new_bollinger import position_new_bollinger
from position_learn_bollinger import position_learn_bollinger
from Pipeline_kf_train import Pipeline_kf_train
from KNet_delta_s import KNet_delta_s
import sys

def pnl(positions, train_yx, beta):
    positions = positions[0,:]

    pnl = torch.zeros_like(positions)
    tmp = torch.transpose(-beta[0,0:1,:], 1,0)
    tmp = torch.hstack([tmp, torch.ones(tmp.shape[0],1)])
    position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
    asset_price = torch.transpose(train_yx[0], 1, 0)[:,:-1][:, [1,0]]
    # asset_price = np.sum(asset_price, axis = 1)
    asset_price_diff = torch.diff(asset_price.T).T
    pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)

    cum_pnl = pnl
    cum_pnl[0] = 0
    cum_pnl = torch.cumsum(cum_pnl, dim=0)
    print('cum_pnl:', cum_pnl[-1])
    return cum_pnl.cpu().numpy()

def pnl_limit(positions, train_yx, beta, ret = 0):
    if ret==0:
        positions = positions[0,:]

        pnl = torch.zeros_like(positions)
        tmp = torch.transpose(-beta[0,0:1,:], 1,0)
        tmp = torch.hstack([-torch.ones(tmp.shape[0],1), torch.ones(tmp.shape[0],1)])
        position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
        asset_price = torch.transpose(train_yx[0], 1, 0)[:,:-1][:, [1,0]]
        # asset_price = np.sum(asset_price, axis = 1)
        asset_price_diff = torch.diff(asset_price.T).T
        pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)

        cum_pnl = pnl
        cum_pnl[0] = 0
        cum_pnl = torch.cumsum(cum_pnl, dim=0)
        print('cum_pnl:', cum_pnl[-1])
        return cum_pnl.cpu().numpy()
    else:
        positions = positions[0,:]

        pnl = torch.zeros_like(positions)
        tmp = torch.transpose(-beta[0,0:1,:], 1,0)
        tmp = torch.hstack([-torch.ones(tmp.shape[0],1), torch.ones(tmp.shape[0],1)])
        position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
        asset_price = torch.transpose(train_yx[0], 1, 0)[:,:-1][:, [1,0]]
        # asset_price = np.sum(asset_price, axis = 1)
        asset_price_diff = torch.diff(asset_price.T).T
        pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)

        cum_pnl = pnl
        cum_pnl[0] = 0
        cum_pnl = torch.cumsum(cum_pnl, dim=0)
        print('cum_pnl:', cum_pnl[-1])
        return cum_pnl.cpu().numpy()


def position_Bollinger(e, Q):
    longsEntry = e < -np.sqrt(Q)
    longsExit = e >= 0

    shortsEntry = e > np.sqrt(Q)
    shortsExit = e <= 0

    numUnitsLong=np.zeros(longsEntry.shape)
    numUnitsLong[:]=np.nan

    numUnitsShort=np.zeros(shortsEntry.shape)
    numUnitsShort[:]=np.nan

    numUnitsLong[0]=0
    numUnitsLong[longsEntry]=1
    numUnitsLong[longsExit]=0
    numUnitsLong=pd.DataFrame(numUnitsLong)
    numUnitsLong.fillna(method='ffill', inplace=True)

    numUnitsShort[0]=0
    numUnitsShort[shortsEntry]=-1
    numUnitsShort[shortsExit]=0
    numUnitsShort=pd.DataFrame(numUnitsShort)
    numUnitsShort.fillna(method='ffill', inplace=True)

    numUnits=numUnitsLong+numUnitsShort
    return numUnits

# def position_linear_no_thres(e,Q):
#   numUnits=e/Q
#   return numUnits

# def position_linear_with_thres(e,Q):
#     numUnits=e/Q
#     numUnits[numUnits>1] = 1
#     numUnits[numUnits<-1] = 1
#     return numUnits

# def PnL_old(e, Q, df, beta):

#   numUnits = position_Bollinger(e, Q)

#   positions=pd.DataFrame(np.tile(numUnits.values, [1, 2]) * ts.add_constant(-beta[0,:].T)[:, [1,0]] *df.values) #  [hedgeRatio -ones(size(hedgeRatio))] is the shares allocation, [hedgeRatio -ones(size(hedgeRatio))].*y2 is the dollar capital allocation, while positions is the dollar capital in each ETF.
#   pnl=np.sum((positions.shift().values)*(df.pct_change().values), axis=1) # daily P&L of the strategy

#   cum_pnl = pnl
#   cum_pnl[0] = 0
#   cum_pnl = np.cumsum(cum_pnl)
#   print('cum_pnl:', cum_pnl[-1])  # 67.2747825163527 for Chan
#   return pnl, cum_pnl

def PnL_new(e, Q, df, beta):

    numUnits = position_Bollinger(e, Q)

    pnl = np.zeros_like(e)
    position = ts.add_constant(-beta[0,:].T)[:, [1,0]] * np.tile(numUnits.values, [1, 2])
    asset_price = df.values
    # asset_price = np.sum(asset_price, axis = 1)
    asset_price_diff = np.diff(asset_price.T).T
    pnl[1:] = np.sum(asset_price_diff * position[:-1], axis = 1)

    cum_pnl = pnl
    cum_pnl[0] = 0
    cum_pnl = np.cumsum(cum_pnl)
    print('cum_pnl:', cum_pnl[-1])
    return pnl, cum_pnl


# def PnL_new_linear_position(e, Q, df, beta):

#     numUnits = position_linear_no_thres(e, Q)

#     pnl = np.zeros_like(e)
#     position = ts.add_constant(-beta[0,:].T)[:, [1,0]] * np.tile(np.expand_dims(numUnits, axis=0).T, [1, 2])
#     asset_price = df.values
#     # asset_price = np.sum(asset_price, axis = 1)
#     asset_price_diff = np.diff(asset_price.T).T
#     pnl[1:] = np.sum(asset_price_diff * position[:-1], axis = 1)

#     cum_pnl = pnl
#     cum_pnl[0] = 0
#     cum_pnl = np.cumsum(cum_pnl)
#     print('cum_pnl:', cum_pnl[-1])
#     return pnl, cum_pnl

# def PnL_new_linear_position_with_thres(e, Q, df, beta):

#     numUnits = position_linear_with_thres(e, Q)

#     pnl = np.zeros_like(e)
#     position = ts.add_constant(-beta[0,:].T)[:, [1,0]] * np.tile(np.expand_dims(numUnits, axis=0).T, [1, 2])
#     asset_price = df.values
#     # asset_price = np.sum(asset_price, axis = 1)
#     asset_price_diff = np.diff(asset_price.T).T
#     pnl[1:] = np.sum(asset_price_diff * position[:-1], axis = 1)

#     cum_pnl = pnl
#     cum_pnl[0] = 0
#     cum_pnl = np.cumsum(cum_pnl)
#     print('cum_pnl:', cum_pnl[-1])
#     return pnl, cum_pnl

def prepare_forex(df_train):
    modelRegL = LinearRegression()
    modelRegL.fit(df_train['chf'].values.reshape(-1,1), df_train['eur'].values.reshape(-1,1))
    hedge_ratio = modelRegL.coef_.item() 
    intercept = modelRegL.intercept_.item()

    epsilon = df_train['eur'].values.reshape(-1,1)-modelRegL.predict(df_train['chf'].values.reshape(-1,1))
    delta=0.00001 # delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).
    q2=delta/(1-delta)
    # r2=epsilon.var() ###############
    r2=epsilon.var()*0.0001
    
    R_0 = torch.zeros((2,2))
    beta_0=torch.tensor([[hedge_ratio], [intercept]])
    # beta_0=torch.tensor([[0.], [0.]])
    return q2, r2, R_0, beta_0
    # return R_0, beta_0

def prepare_rival(df_train):
    modelRegL = LinearRegression()
    modelRegL.fit(df_train['ADBE'].values.reshape(-1,1), df_train['RHT'].values.reshape(-1,1))
    hedge_ratio = modelRegL.coef_.item() 
    intercept = modelRegL.intercept_.item()

    epsilon = df_train['RHT'].values.reshape(-1,1)-modelRegL.predict(df_train['ADBE'].values.reshape(-1,1))
    delta=0.00001 # delta=1 gives fastest change in beta, delta=0.000....1 allows no change (like traditional linear regression).
    q2=delta/(1-delta)
    # r2=epsilon.var() ###############
    r2=epsilon.var()*0.0001
    
    R_0 = torch.zeros((2,2))
    beta_0=torch.tensor([[hedge_ratio], [intercept]])
    # beta_0=torch.tensor([[0.], [0.]])
    return q2, r2, R_0, beta_0
    # return R_0, beta_0

def KF_train_grid(df_train):
    x=df_train['chf']
    y=df_train['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x = np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.
    
    R_0 = torch.zeros((2,2))
    beta_0=torch.tensor([[0.], [0.]]) ##########

    F = torch.eye(2)
    H = torch.tensor(x, dtype=torch.float32)
    T = T_test = x.shape[0]
    # grid search
    delta_ = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    r2_ = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    max_pnl = 0
    max_delta, max_r2 = 0, 0
    for delta in delta_:
        for r2 in r2_:
            q2=delta/(1-delta)
            e, Q, beta = KF_result(x, y, q2, r2, R_0, beta_0, F, H, T, T_test)
            pnl, cum_pnl = PnL_new(e, Q, df_train, beta)
            if cum_pnl[-1]>max_pnl:
                max_pnl = cum_pnl[-1]
                max_delta, max_r2 = delta, r2
                # print('max:',q2,r2)
    print('KF Grid Search Result:', max_delta, max_r2, max_pnl)
    return max_delta, max_r2

def KF_train(training_set, df_train, train_size, traj_length, dataFolderName, rival = 0):
    # end-to-end KF QR opt

    R_0 = torch.zeros((2,2))
    beta_0=torch.tensor([[0.], [0.]]) ##########

    F = torch.eye(2)
    H = torch.tensor([[1., 1.]])
    T = T_test = traj_length

    # SysModel = System_Model(F, q, H, r, T, T_test, hedge = 1)
    # SysModel.InitSequence(beta_0, R_0)
    model = KF_flow()
    model.init_SS(F, H, T, ratio = 1)
    model.InitSequence(beta_0, R_0)
    position_model = position_new_bollinger(0.01)
    if rival==0:
        Pipeline = Pipeline_kf_train(dataFolderName, "KF_train_qr", 'approx_bollinger')
    else:
        Pipeline = Pipeline_kf_train(dataFolderName, "KF_train_qr_rival", 'approx_bollinger_rival')        
    Pipeline.setMoment(beta_0, R_0)
    Pipeline.setModel(model)
    Pipeline.set_positionModel(position_model)
    Pipeline.setTrainingParams(n_Epochs= 11, n_Batch=1, learningRate=1e-4, weightDecay=1e-6)
    Pipeline.NNTrain(n_Examples=int(train_size/traj_length), training_dataset= training_set, n_CV= 10, cv_dataset= None)
    print('q:',Pipeline.model.q.item(), ' r:', Pipeline.model.r.item())
    return Pipeline.model.q.item(), Pipeline.model.r.item()

def KF_bollinger_train(training_set, df_train, train_size, traj_length, dataFolderName, q, r):
    # end-to-end learnable bollinger train

    R_0 = torch.zeros((2,2))
    beta_0=torch.tensor([[0.], [0.]]) ##########

    F = torch.eye(2)
    H = torch.tensor([[1., 1.]])
    T = T_test = traj_length

    # SysModel = System_Model(F, q, H, r, T, T_test, hedge = 1)
    # SysModel.InitSequence(beta_0, R_0)
    model = KF_flow()
    model.init_SS(F, H, T, ratio = 1)
    model.set_qr(q, r)
    model.InitSequence(beta_0, R_0)
    position_model = position_learn_bollinger(0.01, 1.)
    
    Pipeline = Pipeline_kf_train(dataFolderName, "KF_learnable_bollinger", 'learnable_bollinger_position')
    Pipeline.setMoment(beta_0, R_0)
    Pipeline.setModel(model)
    Pipeline.set_positionModel(position_model)
    Pipeline.setTrainingParams(n_Epochs= 4, n_Batch=1, learningRate=1e-4, weightDecay=1e-6, choice=1)
    Pipeline.NNTrain(n_Examples=int(train_size/traj_length), training_dataset= training_set, n_CV= 10, cv_dataset= None)
    return

def KF_test(df, delta, r2, R_0, beta_0):
    x=df['chf']
    y=df['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x = np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.

    F = torch.eye(2)
    H = torch.tensor(x, dtype=torch.float32)
    T = T_test = x.shape[0]

    R_0 = R_0
    beta_0 = beta_0 ######

    q2=delta/(1-delta)
    e, Q, beta = KF_result(x, y, q2, r2, R_0, beta_0, F, H, T, T_test)

    pnl, cum_pnl = PnL_new(e, Q, df, beta)    

    return cum_pnl

def KF_test_with_approx_bollinger(df, delta, r2, R_0, beta_0, rival = 0):
    if rival==1:
        x=df['ADBE']
        y=df['RHT'].values
    else:
        x=df['chf']
        y=df['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x = np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.

    F = torch.eye(2)
    H = torch.tensor(x, dtype=torch.float32)
    T = T_test = x.shape[0]

    R_0 = R_0
    beta_0 = beta_0 ######

    q2=delta/(1-delta)
    e, Q, beta = KF_result_torch(x, y, q2, r2, R_0, beta_0, F, H, T, T_test)
    # print(e.shape)
    # print(Q.shape)
    positions = torch.zeros((1, T_test))
    if e[0] < -torch.sqrt(Q[0]):
        positions[0, 0] = 1.
    elif e[0] > torch.sqrt(Q[0]):
        positions[0, 0] = -1.
    pos_model = position_new_bollinger(0.01)
    for i in range(1, T_test):
        positions[:, i] = pos_model(e[i], Q[i], positions[0, i-1])
    # return positions
    x = torch.tensor(x, dtype=torch.float32)
    data = torch.hstack([y.T, x])
    data = data.T
    if rival==0:
        return pnl_limit(positions, data.unsqueeze(dim=0), beta.unsqueeze(dim=0))
        
    positions = positions[0,:]
    pnl = torch.zeros_like(positions)
    tmp = torch.transpose(-beta[0:1,:], 1,0)

    tmp = torch.hstack([-torch.ones(tmp.shape[0],1), torch.ones(tmp.shape[0],1)])

    position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
    asset_price = torch.transpose(data, 1, 0)[:,:-1][:, [1,0]]
    # print(position.shape)
    # print(asset_price.shape)
    asset_price_abs = torch.abs(torch.sum(asset_price*position, axis = 1))
    # print(torch.where(asset_price_abs<0.001))
    asset_price_abs[torch.where(asset_price_abs<0.001)[0]] = asset_price_abs[torch.where(asset_price_abs<0.001)[0]-1]
    asset_price_abs[torch.where(asset_price_abs<0.001)[0]] = asset_price_abs[torch.where(asset_price_abs<0.001)[0]-1]

    asset_price_diff = torch.diff(asset_price.T).T
    pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)
    # print(pnl)
    # print(asset_price_abs)
    ret = pnl / asset_price_abs
    ret[0] = 0

    cum_ret = torch.cumprod(1+ret, dim=0)
    # print(cum_ret)
    print('cum_ret:', cum_ret[-1]-1)
    # return pnl
    cum_pnl = pnl
    cum_pnl[0] = 0
    cum_pnl = torch.cumsum(cum_pnl, dim=0)
    print('cum_pnl:', cum_pnl[-1])
    return cum_pnl, cum_ret-1

def KF_test_with_learnable_bollinger(df, delta, r2, R_0, beta_0):
    x=df['chf']
    y=df['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x = np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.

    F = torch.eye(2)
    H = torch.tensor(x, dtype=torch.float32)
    T = T_test = x.shape[0]

    R_0 = R_0
    beta_0 = beta_0 ######

    q2=delta/(1-delta)
    e, Q, beta = KF_result_torch(x, y, q2, r2, R_0, beta_0, F, H, T, T_test)
    # print(e.shape)
    # print(Q.shape)
    positions = torch.zeros((1, T_test))
    if e[0] < -torch.sqrt(Q[0]):
        positions[0, 0] = 1.
    elif e[0] > torch.sqrt(Q[0]):
        positions[0, 0] = -1.
    pos_model = torch.load('HedgeRatio/'+'learnable_bollinger_position.pt')
    with torch.no_grad():
        pos_model.eval()
        for i in range(1, T_test):
            positions[:, i] = pos_model(e[i], Q[i], positions[0, i-1])
        # return positions
        x = torch.tensor(x, dtype=torch.float32)
        data = torch.hstack([y.T, x])
        data = data.T
        return pnl(positions, data.unsqueeze(dim=0), beta.unsqueeze(dim=0))

def KF_result(x, y, q2, r2, R_0, beta_0, F, H, T, T_test):

    # x=df['chf']
    # y=df['eur'].values

    # y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    # x = np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.


    SysModel = System_Model(F, np.sqrt(q2), H, np.sqrt(r2), T, T_test, hedge = 1)
    SysModel.InitSequence(beta_0, R_0)
    KF = KalmanFilter(SysModel, 1)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KF.GenerateSequence(y, T_test)

    e = KF.innovations.squeeze().cpu().numpy()
    Q = KF.y_vars.squeeze().cpu().numpy()
    beta = KF.x.cpu().numpy()
    return e, Q, beta

def KF_result_torch(x, y, q2, r2, R_0, beta_0, F, H, T, T_test):

    SysModel = System_Model(F, np.sqrt(q2), H, np.sqrt(r2), T, T_test, hedge = 1)
    SysModel.InitSequence(beta_0, R_0)
    KF = KalmanFilter(SysModel, 1)
    KF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)

    KF.GenerateSequence(y, T_test)

    e = KF.innovations.squeeze()#.cpu().numpy()
    Q = KF.y_vars.squeeze()#.cpu().numpy()
    beta = KF.x#.cpu().numpy()
    return e, Q, beta

def train_set(df_train, traj_length, train_size):
    x=df_train['chf']
    y=df_train['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.
    x = torch.tensor(x, dtype=torch.float32)

    data = torch.hstack([y.T, x])
    data = data.T

    data_split = data[:,:traj_length]
    for i in range(1, int(train_size/traj_length)):
        data_split = torch.vstack([data_split, data[:,i*traj_length:(i*traj_length+traj_length)]])
    training_set = data_split.reshape(int(train_size/traj_length),3,traj_length)
    return training_set

def train_set_rival(df_train, traj_length, train_size):
    x=df_train['ADBE']
    y=df_train['RHT'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.
    x = torch.tensor(x, dtype=torch.float32)

    data = torch.hstack([y.T, x])
    data = data.T

    data_split = data[:,:traj_length]
    for i in range(1, int(train_size/traj_length)):
        data_split = torch.vstack([data_split, data[:,i*traj_length:(i*traj_length+traj_length)]])
    training_set = data_split.reshape(int(train_size/traj_length),3,traj_length)
    return training_set

def KNet_unsuper_train(training_set, q2, r2, R_0, beta_0, F, f, H, h, T, T_test, train_size, traj_length, dataFolderName):
    ssModel = SystemModel(F, f, np.sqrt(q2), H, h, np.sqrt(r2), T, hedge = 1)
    ssModel.InitSequence(beta_0, R_0)

    # Pipeline = Pipeline_Offline(dataFolderName, 'KNet')
    model = KalmanNetNN()
    model.Build(ssModel)
    Pipeline = Pipeline_EKF(dataFolderName, "KNet_unsupervised")
    Pipeline.setssModel(ssModel)
    Pipeline.setModel(model)
    Pipeline.setTrainingParams(n_Epochs= 10, n_Batch= 1, learningRate=1e-4, weightDecay=1e-6, unsupervised= True)
    Pipeline.NNTrain(n_Examples=int(train_size/traj_length), training_dataset= training_set, n_CV= 10, cv_dataset= None)
    return 

def KNet_mlp_train(training_set, q2, r2, R_0, beta_0, F, f, H, h, T, T_test, train_size, traj_length, dataFolderName):
    # end-to-end KNet+MLPposition
    
    ssModel = SystemModel(F, f, np.sqrt(q2), H, h, np.sqrt(r2), T, hedge = 1)
    ssModel.InitSequence(beta_0, R_0)

    model = KNet_delta()
    position_model = position_MLP()
    model.Build(ssModel)
    Pipeline = Pipeline_trading(dataFolderName, "KNet_end_to_end", 'mlp_position_end_to_end')
    Pipeline.setssModel(ssModel)
    Pipeline.setModel(model)
    Pipeline.set_positionModel(position_model)
    Pipeline.setTrainingParams(n_Epochs= 10, n_Batch= 1, learningRate=1e-3, weightDecay=1e-6)
    Pipeline.NNTrain(n_Examples=int(train_size/traj_length), training_dataset= training_set, n_CV= 10, cv_dataset= None)
    return 

def KNet_approx_bollinger_train(training_set, q2, r2, R_0, beta_0, F, f, H, h, T, T_test, train_size, traj_length, dataFolderName):
    # end-to-end KNet+approx_bollinger_position
    
    ssModel = SystemModel(F, f, np.sqrt(q2), H, h, np.sqrt(r2), T, hedge = 1)
    ssModel.InitSequence(beta_0, R_0)

    model = KNet_delta_s()
    position_model = position_new_bollinger(0.01)
    model.Build(ssModel)
    Pipeline = Pipeline_trading(dataFolderName, "KNet_approx_bollinger_end_to_end", 'approx_bollinger_end_to_end')
    Pipeline.setssModel(ssModel)
    Pipeline.setModel(model)
    Pipeline.set_positionModel(position_model)
    Pipeline.setTrainingParams(n_Epochs= 10, n_Batch= 1, learningRate=5e-5, weightDecay=1e-6)
    Pipeline.NNTrain(n_Examples=int(train_size/traj_length), training_dataset= training_set, n_CV= 10, cv_dataset= None)
    return 

def KNet_learnable_bollinger_train(training_set, q2, r2, R_0, beta_0, F, f, H, h, T, T_test, train_size, traj_length, dataFolderName):
    # end-to-end KNet+learnable_bollinger_position end-to-end
    
    ssModel = SystemModel(F, f, np.sqrt(q2), H, h, np.sqrt(r2), T, hedge = 1)
    ssModel.InitSequence(beta_0, R_0)

    model = KNet_delta_s()
    position_model = position_learn_bollinger(0.01, 50.)
    model.Build(ssModel)
    Pipeline = Pipeline_trading(dataFolderName, "KNet_learnable_bollinger_end_to_end", 'learnable_bollinger_end_to_end')
    Pipeline.setssModel(ssModel)
    Pipeline.setModel(model)
    Pipeline.set_positionModel(position_model)
    Pipeline.setTrainingParams(n_Epochs= 12, n_Batch= 1, learningRate=5e-5, weightDecay=1e-6, learnable_pos=1)
    Pipeline.NNTrain(n_Examples=int(train_size/traj_length), training_dataset= training_set, n_CV= 10, cv_dataset= None)
    return 

def KNet_learnable_bollinger_train_rival(training_set, q2, r2, R_0, beta_0, F, f, H, h, T, T_test, train_size, traj_length, dataFolderName):
    # end-to-end KNet+learnable_bollinger_position end-to-end
    
    ssModel = SystemModel(F, f, np.sqrt(q2), H, h, np.sqrt(r2), T, hedge = 1)
    ssModel.InitSequence(beta_0, R_0)

    model = KNet_delta_s()
    position_model = position_learn_bollinger(0.01, 50.)
    model.Build(ssModel)
    Pipeline = Pipeline_trading(dataFolderName, "KNet_learnable_bollinger_end_to_end_rival", 'learnable_bollinger_end_to_end_rival')
    Pipeline.setssModel(ssModel)
    Pipeline.setModel(model)
    Pipeline.set_positionModel(position_model)
    Pipeline.setTrainingParams(n_Epochs= 12, n_Batch= 1, learningRate=5e-5, weightDecay=1e-6, learnable_pos=1)
    Pipeline.NNTrain(n_Examples=int(train_size/traj_length), training_dataset= training_set, n_CV= 10, cv_dataset= None)
    return 

def KNet_unsuper_test(df, dataFolderName, dev, beta_0, T_test):
    ## constructing testing set
    x=df['chf']
    y=df['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.
    x = torch.tensor(x, dtype=torch.float32)

    data = torch.hstack([y.T, x])
    data = data.T

    testing_set = data.reshape(1,3,-1)
    test_len = data.shape[1]
    ###################testing
    model = torch.load(dataFolderName+'KNet_unsupervised.pt')
    model.InitSequence(beta_0, T_test)
    innov_array = torch.zeros((1, test_len)).to(dev, non_blocking=True)
    s_array = torch.zeros((1, test_len)).to(dev, non_blocking=True)
    beta = torch.zeros((2, 1, test_len)).to(dev, non_blocking=True)
    with torch.no_grad():
        model.eval()
        for i in range(test_len):
            beta[:,:,i] = model(testing_set[:,:,i]).unsqueeze(0).T
            innov_array[:, i] = model.dy.item()
            s_array[:, i] = model.S_t.item()

    e = innov_array.squeeze().cpu().numpy()
    Q = torch.abs(s_array).squeeze().cpu().numpy()

    beta = beta.cpu().numpy()

    return e, Q, beta

def KNet_mlp_test(df, dataFolderName, dev, beta_0, T_test):
    # end-to-end KNet+MLPposition
    x=df['chf']
    y=df['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.
    x = torch.tensor(x, dtype=torch.float32)

    data = torch.hstack([y.T, x])
    data = data.T

    testing_set = data.reshape(1,3,-1)
    test_len = data.shape[1]
    ###################testing
    model = torch.load(dataFolderName+'KNet_end_to_end.pt')
    model.InitSequence(beta_0, T_test)
    position_model = torch.load(dataFolderName+'mlp_position_end_to_end.pt')
    positions = torch.empty(1, 1, T_test).to(dev, non_blocking=True)
    x_out_training = torch.empty(1, 2, T_test).to(dev, non_blocking=True)
    # beta = torch.zeros((2, 1, test_len)).to(dev, non_blocking=True)
    with torch.no_grad():
        model.eval()
        for i in range(test_len):
            # beta[:,:,i] = model(testing_set[:,:,i]).unsqueeze(0).T
            dy, x_out = model(data.unsqueeze(dim=0)[:,:,i])
            positions[:,:,i] = position_model(dy)
            x_out_training[:,:,i] = x_out.T

    positions = positions[0,0,:]

    pnl = torch.zeros_like(positions)
    tmp = torch.transpose(-x_out_training[0,0:1,:], 1,0)
    tmp = torch.hstack([tmp, torch.ones(tmp.shape[0],1)])
    position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
    asset_price = torch.transpose(data, 1, 0)[:,:-1][:, [1,0]]
    # asset_price = np.sum(asset_price, axis = 1)
    asset_price_diff = torch.diff(asset_price.T).T
    pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)

    cum_pnl = pnl
    cum_pnl[0] = 0
    cum_pnl = torch.cumsum(cum_pnl, dim=0)
    print('cum_pnl:', cum_pnl[-1])
    return cum_pnl

def KNet_approx_bollinger_test(df, dataFolderName, dev, beta_0, T_test):
    # end-to-end KNet+approx_bollinger
    x=df['chf']
    y=df['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.
    x = torch.tensor(x, dtype=torch.float32)

    data = torch.hstack([y.T, x])
    data = data.T

    testing_set = data.reshape(1,3,-1)
    test_len = data.shape[1]
    ###################testing
    model = torch.load(dataFolderName+'KNet_approx_bollinger_end_to_end.pt')
    model.InitSequence(beta_0, T_test)
    position_model = torch.load(dataFolderName+'approx_bollinger_end_to_end.pt')
    positions = torch.empty(1, 1, T_test).to(dev, non_blocking=True)
    x_out_training = torch.empty(1, 2, T_test).to(dev, non_blocking=True)
    with torch.no_grad():
        model.eval()
        for i in range(test_len):
            dy, x_out, S = model(data.unsqueeze(dim=0)[:,:,i])
            dy = 100 * dy
            # S = torch.abs(S)
            if i>0:
                positions[:,:,i] = position_model(dy, S, positions[:,:,i-1])
            x_out_training[:,:,i] = x_out.T
            # dy, x_out = model(data.unsqueeze(dim=0)[:,:,i])
            # positions[:,:,i] = position_model(dy)
            # x_out_training[:,:,i] = x_out.T

    positions = positions[0,0,:]
    # return positions##
    pnl = torch.zeros_like(positions)
    tmp = torch.transpose(-x_out_training[0,0:1,:], 1,0)
    # return tmp
    tmp = torch.hstack([tmp, torch.ones(tmp.shape[0],1)])
    position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
    asset_price = torch.transpose(data, 1, 0)[:,:-1][:, [1,0]]
    # asset_price = np.sum(asset_price, axis = 1)
    asset_price_diff = torch.diff(asset_price.T).T
    pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)
    # return pnl
    cum_pnl = pnl
    cum_pnl[0] = 0
    cum_pnl = torch.cumsum(cum_pnl, dim=0)
    print('cum_pnl:', cum_pnl[-1])
    return cum_pnl

def KNet_learnable_bollinger_test(df, dataFolderName, dev, beta_0, T_test):
    # end-to-end KNet+learnable_bollinger end-to-end
    x=df['chf']
    y=df['eur'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.
    x = torch.tensor(x, dtype=torch.float32)

    data = torch.hstack([y.T, x])
    data = data.T

    testing_set = data.reshape(1,3,-1)
    test_len = data.shape[1]
    ###################testing 
    model = torch.load(dataFolderName+'KNet_learnable_bollinger_end_to_end.pt')
    model.InitSequence(beta_0, T_test)
    position_model = torch.load(dataFolderName+'learnable_bollinger_end_to_end.pt')
    positions = torch.empty(1, 1, T_test).to(dev, non_blocking=True)
    x_out_training = torch.empty(1, 2, T_test).to(dev, non_blocking=True)
    with torch.no_grad():
        model.eval()
        for i in range(test_len):
            dy, x_out, S = model(data.unsqueeze(dim=0)[:,:,i])
            # S = torch.abs(S)
            if i>0:
                positions[:,:,i] = position_model(dy, S, positions[:,:,i-1])
            x_out_training[:,:,i] = x_out.T
            # dy, x_out = model(data.unsqueeze(dim=0)[:,:,i])
            # positions[:,:,i] = position_model(dy)
            # x_out_training[:,:,i] = x_out.T

    positions = positions[0,0,:]
    # return positions##
    pnl = torch.zeros_like(positions)
    tmp = torch.transpose(-x_out_training[0,0:1,:], 1,0)
    # return tmp
    tmp = torch.hstack([tmp, torch.ones(tmp.shape[0],1)])
    position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
    asset_price = torch.transpose(data, 1, 0)[:,:-1][:, [1,0]]
    # asset_price = np.sum(asset_price, axis = 1)
    asset_price_diff = torch.diff(asset_price.T).T
    pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)
    # return pnl
    cum_pnl = pnl
    cum_pnl[0] = 0
    cum_pnl = torch.cumsum(cum_pnl, dim=0)
    print('cum_pnl:', cum_pnl[-1])
    return cum_pnl

def KNet_learnable_bollinger_test_rival(df, dataFolderName, dev, beta_0, T_test):
    # end-to-end KNet+learnable_bollinger end-to-end
    x=df['ADBE']
    y=df['RHT'].values
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(dim=0)
    x=np.array(ts.add_constant(x))[:, [1,0]] # Augment x with ones to  accomodate possible offset in the regression between y vs x.
    x = torch.tensor(x, dtype=torch.float32)

    data = torch.hstack([y.T, x])
    data = data.T

    testing_set = data.reshape(1,3,-1)
    test_len = data.shape[1]
    ###################testing 
    model = torch.load(dataFolderName+'KNet_learnable_bollinger_end_to_end_rival.pt')
    model.InitSequence(beta_0, T_test)
    position_model = torch.load(dataFolderName+'learnable_bollinger_end_to_end_rival.pt')
    positions = torch.empty(1, 1, T_test).to(dev, non_blocking=True)
    x_out_training = torch.empty(1, 2, T_test).to(dev, non_blocking=True)
    with torch.no_grad():
        model.eval()
        for i in range(test_len):
            # print(i)
            dy, x_out, S = model(data.unsqueeze(dim=0)[:,:,i])
            if i>0:
                positions[:,:,i] = position_model(dy, S, positions[:,:,i-1])
            x_out_training[:,:,i] = x_out.T


    positions = positions[0,0,:]
    pnl = torch.zeros_like(positions)
    tmp = torch.transpose(-x_out_training[0,0:1,:], 1,0)

    tmp = torch.hstack([-torch.ones(tmp.shape[0],1), torch.ones(tmp.shape[0],1)])

    position = tmp * torch.tile(positions.unsqueeze(dim=0).T, [1, 2])
    asset_price = torch.transpose(data, 1, 0)[:,:-1][:, [1,0]]
    # print(position.shape)
    # print(asset_price.shape)
    asset_price_abs = torch.abs(torch.sum(asset_price*position, axis = 1)) ###### 2 ways
    asset_price_abs[torch.where(asset_price_abs<0.001)[0][1:]] = asset_price_abs[torch.where(asset_price_abs<0.001)[0][1:]-1]

    asset_price_diff = torch.diff(asset_price.T).T
    pnl[1:] = torch.sum(asset_price_diff * position[:-1], axis = 1)
    # print(pnl)
    # print(asset_price_abs)
    ret = pnl / asset_price_abs
    ret[0] = 0

    cum_ret = torch.cumprod(1+ret, dim=0)
    # print(cum_ret)
    print('cum_ret:', cum_ret[-1]-1)
    # return pnl
    cum_pnl = pnl
    cum_pnl[0] = 0
    cum_pnl = torch.cumsum(cum_pnl, dim=0)
    print('cum_pnl:', cum_pnl[-1])
    return cum_pnl, cum_ret-1














