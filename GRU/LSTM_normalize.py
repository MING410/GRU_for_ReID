import torch
#import os
import numpy as np
import torch.nn as nn
from datetime import datetime
import csv_read
import matplotlib

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report




def one_hot(x, class_count):
	# 第一构造一个[class_count, class_count]的对角线为1的向量
	# 第二保留label对应的行并返回
	return torch.eye(class_count)[x,:]


def GetBatch(data,label,sampleNum,frames,batchnum=4):
    for i in range(0,sampleNum//frames):
        low=i*frames
        x=data[low:low+frames]
        x=x.reshape(1,28,18)
        #print(x.shape)
        y=label[low]
        y=y.reshape(1,1)
        yield x,y
    
def zscore(x, axis):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore 

class LSTM(nn.Module):
    def __init__(self, hidden=6,lr=0.001):
        super().__init__()
        self.features=18
        self.layers=2
        self.output=8
        self.frames=28
        #self.cnn=tv.models.resnet18(pretrained=True)
        #self.cnn.eval()
        #self.final_pool=torch.nn.MaxPool2d(3,2)

        self.LSTM=torch.nn.LSTM(self.features,hidden,self.layers,batch_first=True)
        self.Linear=torch.nn.Linear(hidden,self.output)
        self.criteria=torch.nn.CrossEntropyLoss()
        self.opt=torch.optim.Adam([{'params':self.LSTM.parameters()},
                                   {'params':self.Linear.parameters()}],lr)
        #self.train_data = torch.FloatTensor(csv_read.train_data.astype(np.float64))
        self.train_label = torch.LongTensor(csv_read.train_label)
        #self.test_data  = torch.FloatTensor(csv_read.test_data.astype(np.float64))
        self.test_label = torch.LongTensor(csv_read.test_label)
        self.train_num = 0
        self.test_num = 0
        self.train_data = csv_read.train_data.astype(np.float64)
        self.test_data = csv_read.test_data.astype(np.float64)

        self.train_data = zscore(self.train_data,axis = 1)
        self.test_data = zscore(self.test_data,axis = 1)

        self.train_data  = torch.FloatTensor(self.train_data) 
        self.test_data  = torch.FloatTensor(self.test_data) 


    def train(self,epochNum=65,batchNum=4,finalLoss=1e-5):
        
        train_input = self.train_data
        train_output=self.train_label
        test_input = self.test_data
        test_output = self.test_label
        self.train_num = len(train_input)
        self.test_num = len(test_input)
        self.LSTM.train()
        self.Linear.train()
        self.accs=[]
        self.test_loss_list = []
        self.train_loss_list = []
        self.epochs=[]
        self.output_list=[]
        self.label_list=[]
        print(self.train_num)
        print(batchNum)
        print('train')
        print(self.test_num)
        for epoch in range(epochNum):
            train_loss=0
            test_loss=0
            for x,y in GetBatch(train_input,train_output,self.train_num,self.frames,batchnum=4):
                y=y.squeeze(0)
                #print("********",x.size())
                self.opt.zero_grad()
                
                # _,out=self.LSTM(x)
                # #最後のセルの隠れ情報しか使わない
                # out_last=out[0]
                out,_ = self.LSTM(x)
                out_last=out[:,-1,:]
                #print("fssfsfsfs_outlast",out_last.size())
                pred=self.Linear(out_last)
                #print("fssfsfsfs",pred.size())
                #print("fssfsfsfs",y.size())
                loss=self.criteria(pred,y)
                loss.backward()
                self.opt.step()

                train_loss+=loss.item()
            #/= は　a = a / b に同じ
            train_loss=train_loss/(self.train_num//batchNum)

            #test loss
            loss=[]
            with torch.no_grad():
                #print("$$$$$$$$$$$",test_input.size())
                test_input=test_input.reshape(-1,28,18)
                out,_=self.LSTM(test_input)
                out_last=out[:,-1,:]
                #print("out_last size",out_last.size())
                #out_last=out_last.reshape(26,28,18)
                pred=self.Linear(out_last)
                #print("pred size",pred.size())
                #print("test_output size",test_output.type())
                test_loss=self.criteria(pred,test_output)
                y=np.argmax(pred,axis=1)
                #print("y size",y.type())
                acc=torch.sum(y==test_output).float()/len(test_input)
                self.output_list=y.numpy()
                #print(len(self.output_list))
                #print(self.output_list)
                self.label_list.append(test_output.numpy().tolist())
                #print('labellist',self.label_list[0])
                #print(type(self.label_list))
                self.test_loss_list.append(test_loss.item())
                self.train_loss_list.append(train_loss)
                self.accs.append(acc.item())
                #print(self.accs)
                #print(type(self.accs))
                self.epochs.append(epoch)
                #print(type(self.epochs))
            print('epoch:{},train_loss:{},test_loss:{},accurancy:{}'.format(epoch,train_loss,test_loss,acc))



            if (epoch%5==0)or(test_loss<finalLoss):
                state = {'net1':self.LSTM.state_dict(),
                        'net2':self.Linear.state_dict(),
                        'optimizer':self.opt.state_dict()}
                saveName='{}.pth'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
                torch.save(state,'./'+saveName)

                if test_loss<finalLoss:
                    break
      

n=LSTM()
n.train()
#cm=confusion_matrix(n.label_list[0],n.output_list)
print(n.accs)
#print(classification_report(n.label_list[0],n.output_list))
# print('precision:',precision_score(n.label_list[0],n.output_list))
# print('recall:',recall_score(n.label_list[0],n.output_list))
# print('f1:',f1_score(n.label_list[0],n.output_list))
#print(cm)
x=np.array(n.epochs)
y1=np.array(n.accs)
y2=np.array(n.test_loss_list)
y3=np.array(n.train_loss_list)
fig = plt.figure()
fig.patch.set_facecolor('white')
plt.xlabel('epoch')
plt.ylim(0, 3)
plt.plot(x,y1, label='test_accuracy')
plt.plot(x,y2,linestyle="--",label='test_loss')
plt.plot(x,y3,linestyle="dotted",label='train_loss')
plt.legend()
# # Visualize
plt.savefig('acc.png')





