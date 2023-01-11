from custom_dataset import Custom_dataset
import torchvision
import torchvision.transforms as transform
import torch
from skimage import  io
from torch.utils.data import DataLoader
import  csv
import cv2
from PIL import Image
from matplotlib import pyplot as plt

cvs_file='C:/Users/farib/Desktop/data_lable.csv'

dir='C:/Users/farib/Desktop/Dataset/'
m_transform=transform.Compose([transform.ToPILImage(),transform.Resize((256,256)),transform.RandomHorizontalFlip()
                              ,transform.RandomRotation(1),transform.ToTensor(),transform.Normalize(mean=[0.0,0.0,0.0],std=[1.0,1.0,1.0])])
# image = io.imread(dir)
dataset_=Custom_dataset(dir=dir,cvs_file=cvs_file,transform=m_transform)
print(len(dataset_))
train_,test_=torch.utils.data.random_split(dataset_,[4,4])
train_loader=DataLoader(dataset=train_,batch_size=1,shuffle=True)
test_loader=DataLoader(dataset=test_,batch_size=1,shuffle=True)
for _ , (data,target) in enumerate(train_loader):
    data=data.squeeze(0)
    data=data.permute(1, 2, 0)
    im=data.numpy()
    plt.imshow(data)
    plt.show()
