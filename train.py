import argparse
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import numpy as np
import csv    
from torchvision import transforms
import random
import argparse


data_fold=['generation','consumption']





#%%
num_day=168
num_pred_day=1


label_fold=['generation','consumption']



train_mean=[0.7808,1.4447]
train_std=[1.1422,1.1166]

class dataset(torch.utils.data.Dataset):

    def __init__(self,is_train=True):
        self.data=[]
        self.label=[]
        self.temp=[]
        if is_train:
            for i in range(len(data_csv)):
                for j in range(len(data_fold)):
                    self.temp.append(data_csv[data_fold[j]][i])
                self.data.append(self.temp)
                self.temp=[]
            z=np.array(self.data).T

            for i in range(len(data_fold)):
                for j in range(len(data_csv)):

                    z[i][j]-=train_mean[i]
                    z[i][j]/=train_std[i]
            self.data=[]
            self.data=z.T

            for i in range(len(data_csv)):
                for j in range(len(label_fold)):
                    self.temp.append(data_csv[label_fold[j]][i])
                self.label.append(self.temp)
                self.temp=[]
            self.label=np.array(self.label)
        '''
        if is_train:
            for i in range(len(data_csv[:num_train])):
                for j in range(len(data_fold)):
                    self.temp.append(data_csv[data_fold[j]][i])
                self.data.append(self.temp)
                self.temp=[]
            z=np.array(self.data).T

            for i in range(len(data_fold)):
                for j in range(len(data_csv[:num_train])):

                    z[i][j]-=train_mean[i]
                    z[i][j]/=train_std[i]
            self.data=[]
            self.data=z.T

            for i in range(len(data_csv[:num_train])):
                for j in range(len(label_fold)):
                    self.temp.append(data_csv[label_fold[j]][i])
                self.label.append(self.temp)
                self.temp=[]
            self.label=np.array(self.label)
        '''
        '''
        else:
            for i in range(len(data_csv[num_train:])):
                for j in range(len(data_fold)):
                    self.temp.append(data_csv[data_fold[j]][i+num_train])
                self.data.append(self.temp)
                self.temp=[]
            z=np.array(self.data).T

            for i in range(len(data_fold)):
                for j in range(len(data_csv[num_train:])):

                    z[i][j]-=train_mean[i]
                    z[i][j]/=train_std[i]
            self.data=[]
            self.data=z.T

            for i in range(len(data_csv[num_train:])):
                for j in range(len(label_fold)):
                    self.temp.append(data_csv[label_fold[j]][i+num_train])
                self.label.append(self.temp)
                self.temp=[]
            self.label=np.array(self.label)
        '''

    def __len__(self):
        
        return len(self.data)-num_day-num_pred_day+1
    
    def __getitem__(self, index):
        
        a=self.data[index:index+num_day]
        b=self.label[index+num_day:index+num_day+num_pred_day]

        return a,np.squeeze(b)




#%%
epochs = 10

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


def MSE(y_pred,y_true): 

    return  torch.sqrt(torch.mean(((y_pred-y_true))**2))
#%%

'''
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')
    parser.add_argument('--testing',
                        default='testing_data.csv',
                        help='input testing data file name')
    parser.add_argument('--output',
                        default='output.csv',
                        help='output file name')
    args = parser.parse_args()

    data_csv=pd.read_csv(args.training, names = column_names)
    test_data_csv=pd.read_csv(args.testing, names = column_names)
    
    num_train=len(data_csv)
    test_data,test_z=test_nor(test_data_csv)
'''
    
net=LSTM(num_feature ,hidden_size,num_layers)

#trainloader=DataLoader(dataset(is_train=True),batch_size=batch_size,shuffle=False)



net.cuda()

#%%
optimizer = torch.optim.Adam(net.parameters(), lr=0.001,weight_decay=0.001)

z=0
gain=[]
pred=[]
for q in range(100):
    for k in range(50):
        raw_data="C://Users/Q56091087/Desktop/download/training_data/target%d.csv"%k
        data_csv = pd.read_csv(raw_data)
        trainloader=DataLoader(dataset(is_train=True),batch_size=batch_size,shuffle=False)
        #for i in range(epochs):
        z=0
        for num_batch,data in enumerate(trainloader,0):
            net.train()
            
            inputs,label=data
    
            inputs,label=(inputs).float().cuda(),(label).float().cuda()
            out=net(inputs)
    
            loss=MSE(torch.squeeze(out),label)
            optimizer.zero_grad()
            loss.backward()
    
            optimizer.step()
            #print(loss.item())
            z+=loss.item()
            
        print('train_loss= %.2f,  EpochLeft=%d target%d '%((z/(num_batch+1)),100-1-q,k))
    '''
    if i%10==0:
        with torch.no_grad():
            for i,test_data in enumerate(testloader,0):
                net.eval()
                test_input,test_label=test_data
                test_input,test_label=test_input.float().cuda(),test_label.float().cuda()
                test_out=net(test_input)
                test_loss=MSE(torch.squeeze(test_out),test_label)
                print(test_loss)
    '''
    
#%%

torch.save(net, 'net.hdf5')
#net = torch.load('net.zxcvzxcv')