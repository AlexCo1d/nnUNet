import os
import random
from pathlib import Path

import imageio
from PIL import Image
import shutil
import SimpleITK as sitk
import numpy as np
import json
from nnunet.dataset_conversion.utils import generate_dataset_json

raw_data_path = Path("/home/yangjiaqi/data/nnUNet/raw_Data/")  # 所有图片的父地址
image_path = os.path.join(raw_data_path, "image")  # 图像
label_path = os.path.join(raw_data_path, "label")  # 标签

#为满足nnUnet的要求，label必须连续，将所有label的128转化为1
for file in os.listdir(label_path):
    image=np.array(Image.open(os.path.join(label_path,file)))
    image[image==128]=1
    Image.fromarray(image).save(os.path.join(label_path,file))

data_path = Path("/home/yangjiaqi/data/nnUNet/Data/nnUNet_raw/nnUNet_raw_data/Task066_CervicalTumor")  # 存储父地址
imagesTr = os.path.join(data_path, "imagesTr")
imagesTs = os.path.join(data_path, "imagesTs")
labelsTr = os.path.join(data_path, "labelsTr")
labelsTs = os.path.join(data_path, "labelsTs")
random_num = 0  # 设置随机选测试图像数目

image_list = os.listdir(image_path)
label_list = os.listdir(label_path)

# 随机采样分离出测试图片
test_list = random.sample(image_list, random_num)
# save test image to imagesTs和labelsTs
for index, file_name in enumerate(test_list):
    # shutil.copy(os.path.join(image_path, file_name),os.path.join(imagesTs,'Test_Cervical_'+str(index)+'.jpg'))
    cur_image = np.asarray(np.array(Image.open(os.path.join(image_path, file_name)).convert('L')))
    w, h = cur_image.shape
    cur_image = cur_image.reshape(1, w, h)
    cur_image_nii = sitk.GetImageFromArray(cur_image)
    cur_image_name = 'Test_Cervical_' + str(index) + '_0000.nii.gz'
    sitk.WriteImage(cur_image_nii, os.path.join(imagesTs, cur_image_name))

    shutil.copy(os.path.join(label_path, file_name.replace(".jpg", ".png")),
                os.path.join(labelsTs, 'Test_Cervical_' + str(index) + '.png'))

# 分理出需要训练的图片集
train_list = list(set(image_list) - set(test_list))  # end with jpg

for index, file_name in enumerate(train_list):
    # 读取文件并将其转为array
    cur_image = np.asarray(np.array(Image.open(os.path.join(image_path, file_name)).convert('L')))
    cur_label = np.asarray(np.array(Image.open(os.path.join(label_path, file_name.replace(".jpg", '.png')))))
    w, h = cur_image.shape
    cur_image = cur_image.reshape(1, w, h)
    cur_label = cur_label.reshape(1, w, h)
    cur_image_name = 'Cervical_' + str(index) + '_0000.nii.gz'
    cur_label_name = 'Cervical_' + str(index) + '.nii.gz'

    # 转化为ntfi格式存储进各自需要的路径
    cur_image_nii = sitk.GetImageFromArray(cur_image)
    cur_label_nii = sitk.GetImageFromArray(cur_label)

    sitk.WriteImage(cur_image_nii, os.path.join(imagesTr, cur_image_name))
    sitk.WriteImage(cur_label_nii, os.path.join(labelsTr, cur_label_name))

# generate json file
generate_dataset_json(output_file=os.path.join(data_path, "dataset.json", ), imagesTr_dir=imagesTr,
                      imagesTs_dir=imagesTs, modalities=('gray',)
                      , labels={0: 'background', 1: 'tumor'}, dataset_name='Task666_CervicalTumor',
                      license='hands_off')
