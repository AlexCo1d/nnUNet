import os
from pathlib import Path
import shutil

raw_data_path=Path("/home/yangjiaqi/data/nnUNet/raw_Data/")   #所有图片的父地址
image_path=os.path.join(raw_data_path,"image")  #图像
label_path=os.path.join(raw_data_path,"label")  #标签

data_path=Path("/home/yangjiaqi/data/nnUNet/Data/nnUNet_raw/nnUNet_raw_data/Task666_CervicalTumor") #存储父地址
imagesTr=os.path.join(data_path,"imagesTr")
imagesTs=os.path.join(data_path,"imagesTs")
labelsTr=os.path.join(data_path,"labelsTr")

random_num=100 #设置随机选测试图像数目

#将图片统一编号
image_list=os.listdir(image_path)
label_list=os.listdir(label_path)

for index,file_name in enumerate(image_list):
    shutil.copy()