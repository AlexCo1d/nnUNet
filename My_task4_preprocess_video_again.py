import os
import random
from pathlib import Path
from PIL import Image
import shutil
import SimpleITK as sitk
import numpy as np
import cv2

'''
处理.avi文件和对应label .nii.gz文件的裁剪工作。
'''
def main():
    father_path=''
    file_name=''
    output_fpath=

    input_video = os.path.join(father_path,file_name)  # single video'
    input_label=''

    output_video=''
    output_label=''

    x_start,y_start,x_end,y_end = ,,,
    process_video(input_video,output_video, x_start,y_start,x_end,y_end)
    process_label(input_label,output_label, x_start,y_start,x_end,y_end)


def crop_and_convert_to_grayscale(frame, x_start, y_start, x_end, y_end):
    """裁剪并将给定图像帧转换为灰度图像。

    Args:
        y_end(int): 裁剪区域的结束横坐标
        x_end(int): 裁剪区域的结束纵坐标
        y_start(int): 裁剪区域的起始纵坐标
        x_start(int): 裁剪区域的起始横坐标。
        frame (numpy.ndarray): 输入的 BGR 图像帧。

    Returns:
        numpy.ndarray: 裁剪并转换为灰度的图像帧。
    """
    cropped_frame = frame[y_start:y_end, x_start:x_end]
    grayscale_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    return grayscale_frame


def process_video(input_file, output_file, x_start,y_start,x_end,y_end):
    """处理输入视频，裁剪并将其转换为灰度图像。

    Args:
        output_file:(str):  输出NIfTI文件的路径。
        input_file (str): 输入视频文件的路径。
        y_end(int): 裁剪区域的结束横坐标
        x_end(int): 裁剪区域的结束纵坐标
        y_start(int): 裁剪区域的起始纵坐标
        x_start(int): 裁剪区域的起始横坐标。


    """
    cap = cv2.VideoCapture(input_file)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        grayscale_frame = crop_and_convert_to_grayscale(frame, x_start,y_start,x_end,y_end)
        frames.append(grayscale_frame)

    cap.release()

    processed_data = np.array(frames)
    itk_image = sitk.GetImageFromArray(processed_data)
    sitk.WriteImage(itk_image, output_file)


def process_label(input_label,output_label, x_start, y_start, x_end, y_end):
    """读取 NIfTI 文件并将其转换为 NumPy 数组。

    Args:
        output_label(str): 输出 NIfTI 文件的路径。
        input_label (str): 输入 NIfTI 文件的路径。
        y_end(int): 裁剪区域的结束横坐标
        x_end(int): 裁剪区域的结束纵坐标
        y_start(int): 裁剪区域的起始纵坐标
        x_start(int): 裁剪区域的起始横坐标。
    """
    itk_image = sitk.ReadImage(input_label)
    data = sitk.GetArrayFromImage(itk_image)
    data = data[:, y_start:y_end, x_start:x_end]
    itk_image = sitk.GetImageFromArray(data)
    sitk.WriteImage(itk_image, output_label)

if __name__ == "__main__":
    main()
