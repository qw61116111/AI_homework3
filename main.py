import time
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv    
import random
import argparse
import re
import datetime

data_fold=['generation','consumption']

#%%
num_day=168
num_pred_day=1
num_train=3600

label_fold=['generation','consumption']


train_mean=[0.959,1.596]
train_std=[1.414,1.228]



#%%
def input_data_add(input_data,out):

    out-=np.array(train_mean)
    out/=np.array(train_std)
    for i in range(len(input_data[0])):
        for j in range(len(data_fold)):
            if i<len(input_data[0])-2:

                input_data[0][i][j]=input_data[0][i+1][j]

            else:

                input_data[0][i][j]=out[0][j]

    return torch.from_numpy(input_data)
def data_csv_nor(data_csv):
    data=[]
    label=[]
    temp=[]

    for i in range(len(data_csv)):

        for j in range(len(data_fold)):
            temp.append(data_csv[data_fold[j]][i])
        data.append(temp)
        temp=[]
    z=np.array(data).T



    for i in range(len(data_fold)):
            for j in range(len(data_csv)):
                z[i][j]-=train_mean[i]
                z[i][j]/=train_std[i]
    processed_data=z.T[np.newaxis,:]

    return  torch.from_numpy(processed_data)
#%%
epochs = 1200

batch_size = 32
val_size=200
hidden_size = 128
num_layers = 2
num_feature=len(data_fold)
#
class LSTM(nn.Module):
    def __init__(self,num_feature,hidden_size, num_layers):
        super().__init__()
        self.lstm=nn.LSTM(
            input_size=num_feature,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
            )
    
        self.fc=nn.Linear(hidden_size,64)
        self.fc1=nn.Linear(64,2)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,inputs):
        out,(h_n,c_n)=self.lstm(inputs, None)
        outputs=self.relu(self.fc(h_n[1]))
        outputs=self.fc1(outputs)
        return  outputs



#%%


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--consumption',
                       default='consumption.csv',
                       )
    parser.add_argument('--generation',
                        default='generation.csv',
                        )
    parser.add_argument('--bidresult',
                        default='bidresult.csv',
                        )
    parser.add_argument('--output',
                        )
    args = parser.parse_args()


    consumption_csv=pd.read_csv(args.consumption)
    generation_csv=pd.read_csv(args.generation)
    data_csv = pd.concat([generation_csv, consumption_csv['consumption']],axis=1) 
 
    input_data=data_csv_nor(data_csv)
    
map_location=torch.device('cpu')
net = torch.load('net.hdf5',map_location)



#%%
pred_list=[]
input_data=input_data.float()

pred_list=np.zeros([24,2])
with torch.no_grad():
    for i in range(24):
        net.eval()
        out_z=net(input_data)


        pred_list[i]=out_z.numpy()
        
        input_data=input_data_add(input_data.numpy(),out_z.numpy())

with open(args.output, 'w',newline='') as csvfile:
    
    index=consumption_csv['time'][167].find(':')-1
    


    writer = csv.writer(csvfile)
    writer.writerow(['Time','action','target_price','target_volume'])
    
    z = pd.read_csv(args.consumption, index_col = 0, parse_dates = True)
    
    time1=(z.index[-1])

    for i in range(24):
        time1+=datetime.timedelta(hours=1)
        c=time1.strftime('%Y-%m-%d %H:%M:%S')
        if pred_list[i][0]>pred_list[i][1]:
            writer.writerow([c,'sell','%.2f'%(random.randrange(100, 200, 1)/100),'%.2f' %(pred_list[i][0]-pred_list[i][1])])
        else:
            writer.writerow([c,'buy','%.2f'%(pred_list[i][1]-pred_list[i][0]),'%.2f'%(random.randrange(50, 150, 1)/100)])
       