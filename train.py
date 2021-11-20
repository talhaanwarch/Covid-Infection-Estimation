#!/usr/bin/env python
# coding: utf-8
'''
img_size,efficientnetv2_rw_s
'''


path='/media/talha/data/image/classification/2D/Covid/data5/'
# In[1]:

import logging 
import numpy as np
from pytorch_lightning import seed_everything, LightningModule, Trainer
from sklearn.utils import class_weight
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau,CosineAnnealingWarmRestarts,OneCycleLR,CosineAnnealingLR
import torchvision
from sklearn.metrics import classification_report,f1_score,accuracy_score
import matplotlib.pyplot as plt
from pytorch_lightning.loggers import TensorBoardLogger
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import cv2
import os

# In[2]:


import albumentations as A

from albumentations.pytorch import ToTensorV2

def augmentation(img_size=288):
    aug= A.Compose([
                A.Resize(img_size,img_size),
                A.CenterCrop(img_size,img_size),
                A.HorizontalFlip(0.5),
                #A.VerticalFlip(0.5),
                A.ShiftScaleRotate(rotate_limit=10),
                A.Blur(),
                A.Sharpen(), 
                A.CoarseDropout(max_holes=8, max_height=16, max_width=16),
                A.CLAHE(0.5),
                A.Normalize(),
                ToTensorV2(p=1.0),
            ], p=1.0)
    return aug


# In[3]:


from time import time
class DataReader(torch.utils.data.Dataset):
    #Characterizes a dataset for PyTorch'
    def __init__(self, df,aug=False,test=False):
        'Initialization'
        self.df=df
        self.transform=aug
        self.test=test
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.df)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        file = self.df['image'][index]
        image=cv2.imread(file)
        image=np.array(image)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image=self.transform(image=image)['image']
        if self.test is False:
            label=self.df['label'][index]
            return image,label.flatten().astype('float32')
        else:
            return image

df=pd.read_csv(os.path.join(path,'Train.csv'),header=None)
df.columns=['image','label','subject']
df.image=path+'Train/'+df.image
print(df.head())




from sklearn.model_selection import train_test_split
val_path=sorted(glob(os.path.join(path,'Val','*.png')))
test_split=pd.DataFrame(val_path,columns=['image'])
images_name=[i.split('/')[-1] for i in val_path]
# # import time
# # from atpbar import atpbar
# # df=df.iloc[0:10000,:]
# # for num_workers in range(2, 12, 2):  
# #     train_reader=DataReader(df,aug);    
# #     train_loader = DataLoader(train_reader,shuffle=False,num_workers=num_workers,batch_size=32,pin_memory=True)
# #     start = time.time()
# #     for epoch in range(1, 2):
# #         for i, data in enumerate(train_loader, 0):
# #             pass
# #     end = time.time()
# #     print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
# # #13.9    


# # In[9]:


# train_reader=DataReader(train_split,aug);    
# train_loader = DataLoader(train_reader,shuffle=False,num_workers=0,batch_size=16,)
# batch=next(iter(train_loader));
# print(batch[0].shape,batch[1])



# # In[10]:


# # import matplotlib.pyplot as plt
# # plt.figure(figsize=(16,16))
# # grid_img=torchvision.utils.make_grid(batch[0],4,2)
# # plt.imshow(grid_img.permute(1, 2, 0))




# # In[13]:
from time import sleep

import timm
import torchmetrics
import torchvision.models as models
class OurModel(LightningModule):
    def __init__(self,train_split,val_split,fold,scheduler,batch_size):
        super(OurModel,self).__init__()
        #architecute
        #lambda resnet
        
        self.train_split=train_split
        self.val_split=val_split
        self.fold=fold
        self.scheduler=scheduler
        #########TIMM#################
        self.model =  timm.create_model(model_name,pretrained=True)
       
	    
        self.fc1=nn.Linear(1000,500)
        self.relu=nn.ReLU()
        self.fc2= nn.Linear(500,250)
        self.fc3= nn.Linear(250,1)
        #parameters
        self.lr=1e-3
        self.batch_size=batch_size
        self.numworker=4
        self.criterion=nn.SmoothL1Loss()
        self.metrics=torchmetrics.MeanAbsoluteError()
        self.trainloss,self.valloss,self.dfs=[],[],[]
    def forward(self,x):
        x= self.model(x)
        x=self.fc1(x)
        x=self.relu(x)
        x=self.fc2(x)
        x=self.relu(x)
        x=self.fc3(x)
        return x

    def configure_optimizers(self):
        opt=torch.optim.AdamW(params=self.parameters(),lr=self.lr )
        if self.scheduler=='cosine':
            scheduler=CosineAnnealingLR(opt,T_max=10,  eta_min=1e-6, last_epoch=-1)
            return {'optimizer': opt,'lr_scheduler':scheduler}
        elif self.scheduler=='reduce':
            scheduler=ReduceLROnPlateau(opt,mode='min', factor=0.5, patience=5)
            return {'optimizer': opt,'lr_scheduler':scheduler,'monitor':'val_loss'}
        elif self.scheduler=='warm':
            scheduler=CosineAnnealingWarmRestarts(opt,T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
            return {'optimizer': opt,'lr_scheduler':scheduler}
        elif self.scheduler=='constant':
            return opt
        
    def train_dataloader(self):
        return DataLoader(DataReader(self.train_split,aug), batch_size = self.batch_size, 
            #sampler=self.datasampler,
                          num_workers=self.numworker,pin_memory=True,shuffle=True)

    def training_step(self,batch,batch_idx):
        image,label=batch
        out = self(image)
        loss=self.criterion(out,label)
        return {'loss':loss}

    def training_epoch_end(self, outputs):
        loss=torch.stack([x["loss"] for x in outputs]).mean().detach().cpu().numpy().round(2)
        self.trainloss.append(loss)
        self.log('train_loss', loss)
        
    def val_dataloader(self):
        ds=DataLoader(DataReader(self.val_split,aug), batch_size = self.batch_size,
                      num_workers=self.numworker,pin_memory=True, shuffle=False)
        return ds

    def validation_step(self,batch,batch_idx):
        image,label=batch
        out=self(image)
        loss=self.criterion(out,label)

        mae=self.metrics(out,label)
        return {'loss':loss,'mae':mae}

    def validation_epoch_end(self, outputs):
        loss=torch.stack([x["loss"] for x in outputs]).mean().detach().cpu().numpy().round(2)
        mae=torch.stack([x["mae"] for x in outputs]).mean().detach().cpu().numpy().round(2)
        self.valloss.append(loss)
        self.log('val_loss', loss)


    def test_dataloader(self):
        return DataLoader(DataReader(test_split,aug,True), batch_size = self.batch_size,
                          num_workers=self.numworker,pin_memory=True,shuffle=False)

    def test_step(self,batch,batch_idx):
        image=batch
        out=self(image)
        return {'pred':out}
    def test_epoch_end(self, outputs):
        pred=torch.cat([x["pred"] for x in outputs]).detach().cpu().numpy().ravel()
        pred=np.where(pred<0,0,pred)
        df=pd.DataFrame(zip(images_name,pred))
        df.to_csv('predictions/predictions_{}.csv'.format(self.fold),index=False,header=None)
        #self.dfs.append(df)
# In[14]:
from sklearn.model_selection import KFold,GroupKFold
import csv
logfile='log.txt'

model_dict={'ecaresnet50t':{'batchsize':72,'img_size':224},'seresnext50_32x4d':{'batchsize':72,'img_size':224},
'resnest50d':{'batchsize':72,'img_size':224},'skresnext50_32x4d':{'batchsize':48,'img_size':224},'resnet200d':{'batchsize':24,'img_size':256}}

if __name__ == "__main__":

    seed_everything(0)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',dirpath='checkpoints',
                                        filename='file',save_last=True)
    train_loss,val_loss,val_score=[],[],[]
    kfold = GroupKFold(n_splits=5)

    for model_name in model_dict.keys():
        batch_size=model_dict[model_name]['batchsize']
        img_size=model_dict[model_name]['img_size']
        aug=augmentation(img_size)

        print('model name',model_name,batch_size,img_size)
        for scheduler in ['cosine']:
            fold_score=[]
            model_path=os.path.join('models',model_name)
            model_path_sch=os.path.join(model_path,scheduler)
            
            csv_path=os.path.join('average_predictions',model_name)
            csv_path_sch=os.path.join(csv_path,scheduler)

            if os.path.isdir(model_path) is False:
                os.mkdir(model_path)
            if os.path.isdir(model_path_sch) is False:
                os.mkdir(model_path_sch)
            if os.path.isdir(csv_path) is False:
                os.mkdir(csv_path)
            if os.path.isdir(csv_path_sch) is False:
                os.mkdir(csv_path_sch)

            fold_completed=len(glob(model_path_sch+'/*.pth'))
            print('fold_completed',model_name, scheduler,fold_completed)
            if fold_completed<5:#== mean fold completed, < not completed
                for fold,(train_idx,val_idx) in enumerate(kfold.split(df,groups=df.subject)):
                    if fold+1>fold_completed:#if num folds greater than fold completed
                        print('{}-----------fold no---------{}---------------{}-------'.format(scheduler,fold,model_name))
                        train_split=df.loc[train_idx].reset_index(drop=True)
                        val_split=df.loc[val_idx].reset_index(drop=True)

                        model=OurModel(train_split,val_split,fold,scheduler,batch_size)

                        trainer = Trainer(max_epochs=50, auto_lr_find=False, auto_scale_batch_size=False,
                                        deterministic=True,
                                        gpus=-1,precision=16,
                                        accumulate_grad_batches=4,
                                        stochastic_weight_avg=False,
                                        enable_progress_bar = False,
                                        #log_every_n_steps=10,
                                        num_sanity_val_steps=0,
                                        #limit_train_batches=20,
                                        #limit_val_batches=5,
                                        callbacks=[lr_monitor,checkpoint_callback],
                                        #resume_from_checkpoint=path+'code/checkpoints/lambdalast.ckpt',
                                        #logger=logger
                                        )

                        trainer.fit(model)
                        #model.load_state_dict(torch.load(os.path.join(path,'code',model_path_sch,'model_{}.pth'.format(fold))))
                        #train_loss.append(model.trainloss)
                        #val_loss.append(model.valloss)
                        
                        res=trainer.validate(model)
                        fold_score.append(res[0]['val_loss'])

                        trainer.test(model)
                        
                        torch.save(model.state_dict(), os.path.join(path,'code',model_path_sch,'model_{}.pth'.format(fold)))
                
                        #open(logfile, 'a').write("fold {}  of model {} with scheduler {} is completed \n".format(fold,model_name,scheduler))
                        sleep(60)
                avg=np.round(np.mean(fold_score),3)
                # open(logfile, 'a').write("all folds of model {} with scheduler {} are completed \n".format(model_name,scheduler))
                dfs=[]
                for i in range(5):
                    dfs.append(pd.read_csv('predictions/predictions_{}.csv'.format(i),header=None))

                #mean_pred=np.average([i.values[:,1].astype('float') for i in dfs],0,weights=fold_score)
                mean_pred=np.mean([i.values[:,1].astype('float') for i in dfs],0)
                dfs[0].iloc[:,1]=mean_pred
                dfs[0].to_csv('predictions/predictions.csv',index=False,header=None)
                dfs[0].to_csv(os.path.join(csv_path_sch,'predictions.csv'),index=False,header=None)


                with open('result.csv', 'a', newline='') as csvfile:
                    fieldnames = ['model_name','scheduler','batchsize','valscore','img_size']
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writerow({'model_name':model_name, 'scheduler':scheduler,'batchsize':model.batch_size,'valscore':avg,'img_size':img_size})
                # # open(logfile, 'a').write("============= model completed ================= \n")
                # # sleep(60*5)