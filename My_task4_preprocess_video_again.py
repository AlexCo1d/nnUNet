import os
import random
import tarfile
from pathlib import Path
from PIL import Image
import shutil
import SimpleITK as sitk
import numpy as np
import cv2

set_value = 1

'''
处理.avi文件进行相应裁剪
'''


def main():
    father_path = r'D:\learning\UNNC 科研\202210_CSD超声标注_图片及视频\CSD视频及标注_15例\20220901_国妇婴_CSD超声视频及标注_4例'
    file_name = '20180915_083831_6.avi'
    output_fpath = r'D:\learning\UNNC 科研\data\nnUNet'

    input_video = os.path.join(father_path, file_name)  # single video'

    output_video = os.path.join(output_fpath, 'video_image')
    output_label = os.path.join(output_fpath, 'video_label')
    if not os.path.exists(output_label): os.makedirs(output_label)
    if not os.path.exists(output_video): os.makedirs(output_video)

    # x_start,y_start,x_end,y_end=()
    extract_and_rename_nii_gz(file_name.replace(".avi", "_avi_Label.tar").replace(".AVI", "_AVI_Label.tar"),
                              father_path)
    resample_extract_crop_frames()
    resample_extract_crop_label()


def extract_and_rename_nii_gz(tar_file, output_dir):
    """
    从给定的 tar 文件中解压所有 .nii.gz 文件，并将文件名小写。解压后的文件将保存到指定的输出目录。

    Args:
        tar_file (str): 要处理的 tar 文件的路径。
        output_dir (str): 父路径

    Returns:
        None
    """
    with tarfile.open(os.path.join(output_dir, tar_file), "r") as tar:
        for member in tar.getmembers():
            if member.name.lower().endswith(".nii.gz"):
                member_name_lower = member.name.lower()
                extracted_path = os.path.join(output_dir, member_name_lower)

                with tar.extractfile(member) as extracted_file:
                    with open(extracted_path, "wb") as output_file:
                        shutil.copyfileobj(extracted_file, output_file)

                print(f"Extracted and renamed: {member.name} -> {member_name_lower}")


def resample_extract_crop_label(fpath, input_label, output_dir, crop_params, frame_interval=10):
    """读取 NIfTI 文件并将其转换为 NumPy 数组。

    Args:
        fpath (str): 输入 NIfTI 文件的父路径。
        frame_interval (int, optional): 抽取帧的间隔，默认为 10。
        output_label (str): 输出 NIfTI 文件的路径。
        input_label (str): 输入 NIfTI 文件的路径。
        crop_params (tuple): 一个包含裁剪参数的元组，格式为 (x1, y1, x2, y2)。
    """
    itk_image = sitk.ReadImage(os.path.join(fpath, input_label))
    data = sitk.GetArrayFromImage(itk_image)
    x_start, y_start, x_end, y_end = crop_params
    for frames in range(0, data.shape[0]):
        if frames % frame_interval == 0:
            cur_frame = data[frames:, y_start:y_end, x_start:x_end]
            cur_frame = process_img(cur_frame)
            Image.fromarray(cur_frame).save(
                os.path.join(output_dir, f'{input_label.replace("_avi_label.nii.gz", "")}_{frames:04d}.jpg'))


def resample_extract_crop_frames(fpath, video_file, output_dir, crop_params, frame_interval=10):
    """
    从给定的视频文件中每隔一定帧数抽取一帧，并按照指定的裁剪参数进行裁剪。裁剪后的帧将作为图像文件（PNG 格式）保存到指定的输出目录。

    Args:
        fpath(str): 要处理的视频文件的父路径。
        video_file (str): 要处理的视频文件的名字。
        output_dir (str): 用于保存裁剪后帧的输出目录。
        crop_params (tuple): 一个包含裁剪参数的元组，格式为 (x1, y1, x2, y2)。
        frame_interval (int, optional): 抽取帧的间隔，默认为 10。

    Returns:
        None

    Raises:
        IOError: 无法打开指定的视频文件。
    """
    cap = cv2.VideoCapture(os.path.join(fpath, video_file))

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            x1, y1, x2, y2 = crop_params
            cropped_frame = frame[y1:y2, x1:x2]
            output_file = os.path.join(output_dir, f"{video_file.lower().replace('.avi', '')}_{saved_count:04d}.jpg")
            Image.fromarray(cropped_frame).save(output_file)
            saved_count += 1

        frame_count += 1

    cap.release()


def process_img(img):
    img[img == 4] = 0
    img[img == 2] = 0
    img[img == 1] = set_value
    return img


if __name__ == "__main__":
    main()
