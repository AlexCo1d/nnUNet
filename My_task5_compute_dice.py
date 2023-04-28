#  标签图像文件夹
import os
from pathlib import Path
import cv2
import numpy as np
import SimpleITK as sitk


classNum=2
data_path = Path("/home/yangjiaqi/data/nnUNet/Data/nnUNet_trained_models/nnUNet/2d/Task067_Cervical2D"
                 "/nnUNetTrainerV2__nnUNetPlansv2.1/")  # 存储父地址

# from

LabelPath = os.path.join(data_path,'gt_niftis') # total 1002 501+501
#  预测图像文件夹
PredictPath = os.path.join(data_path,'fold_3','validation_raw_postprocessed') # total 201

def ConfusionMatrix(numClass, imgPredict, Label):
    #  返回混淆矩阵
    mask = (Label >= 0) & (Label < numClass)
    label = numClass * Label[mask] + imgPredict[mask]
    count = np.bincount(label, minlength=numClass ** 2)
    confusionMatrix = count.reshape(numClass, numClass)
    return confusionMatrix


def dice(confusionMatrix):
    #  返回交并比IoU
    intersection = 2 * np.diag(confusionMatrix)
    union = np.sum(confusionMatrix, axis=1) + np.sum(confusionMatrix, axis=0)
    dice = intersection / union
    dice = np.nanmean(dice)
    return dice


label_all=[]
predict_all=[]

for nii_file in os.listdir(PredictPath):
    if nii_file.endswith('.nii.gz'):

        cur_addr = os.path.join(LabelPath, nii_file)
        nii_label = sitk.ReadImage(cur_addr)
        label = sitk.GetArrayFromImage(nii_label).squeeze()

        cur_addr = os.path.join(PredictPath, nii_file)
        nii_predict=sitk.ReadImage(cur_addr)
        predict=sitk.GetArrayFromImage(nii_predict).squeeze()

        label_all.append(label)
        predict_all.append(predict)

label_all = np.concatenate([array.flatten() for array in label_all])
predict_all = np.concatenate([array.flatten() for array in predict_all])

confusionMatrix = ConfusionMatrix(classNum, predict_all, label_all)
dice = dice(confusionMatrix)
print("混淆矩阵:")
print(confusionMatrix)
print('dice: \n',dice)