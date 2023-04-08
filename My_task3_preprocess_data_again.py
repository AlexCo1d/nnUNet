import os
import random
from pathlib import Path

import imageio
from PIL import Image
import shutil
import numpy as np
import json

padding = 200


def main():
    '''
    记录一下裁剪参数：
    Dual_Image\large_left_space： (0.540419, 0.1965908, 0.9296875, 0.8511363)

    '''
    Image_path = r'D:\learning\UNNC 科研\Cervical\datasets\target\√_Dual_Image\large_left_space'
    Label_path = r'D:\learning\UNNC 科研\Cervical\datasets\SegmentationClass'
    O_image_path = r'D:\learning\UNNC 科研\Cervical\datasets\target\O_image'
    O_label_path = r'D:\learning\UNNC 科研\Cervical\datasets\target\O_label'

    label_list = [s.replace('.jpg', '.png') for s in os.listdir(Image_path)]
    percentage = get_max_bounding_box_percentage(Label_path, label_list)
    # print(percentage)

    set_percentage = (0.540419, 0.1965908, 0.9296875, 0.8511363)

    # crop_image(os.path.join(Image_path, image), os.path.join(O_image_path, image), set_percentage)

    for image in os.listdir(Image_path):
        crop_image(os.path.join(Image_path, image),os.path.join(Label_path,image.replace(".jpg",'.png')), os.path.join(
            O_image_path, image), os.path.join(O_label_path,image.replace(".jpg",'.png')), set_percentage)
        # crop_image(os.path.join(Label_path,image.replace(".jpg",'.png')),os.path.join(O_label_path,image.replace(".jpg",'.png')),set_percentage)


def crop_image(image_path, label_path, o_image_path, o_label_path, crop_percentages, pad=False):

    # 打开图片
    with Image.open(image_path) as img:
        width, height = img.size

        # 根据百分比计算裁剪坐标
        left = max(0, int(width * crop_percentages[0]) - 1)
        upper = max(0, int(height * crop_percentages[1]) - 1)
        right = min(width, int(width * crop_percentages[2]) + 1)
        lower = min(height, int(height * crop_percentages[3]) + 1)

        if pad is True:
            # 向各方向扩充200像素
            left = max(0, left - padding)
            upper = max(0, upper - padding)
            right = min(width, right + padding)
            lower = min(height, lower + padding)
        # 裁剪图片
        cropped_img, zero_crop_location = crop_zero_elements(img.crop((left, upper, right, lower)))
        print((left, upper, right, lower))
        print(zero_crop_location)
        # 保存裁剪后的图片
        if cropped_img is not None:
            cropped_img.save(o_image_path)

        # 处理标签图片
        label_img = Image.open(label_path)
        width, height = label_img.size

        # 根据百分比计算裁剪坐标
        left = max(0, int(width * crop_percentages[0]) - 1)
        upper = max(0, int(height * crop_percentages[1]) - 1)
        right = min(width, int(width * crop_percentages[2]) + 1)
        lower = min(height, int(height * crop_percentages[3]) + 1)

        if pad is True:
            # 向各方向扩充200像素
            left = max(0, left - padding)
            upper = max(0, upper - padding)
            right = min(width, right + padding)
            lower = min(height, lower + padding)

        cropped_img = label_img.crop((left, upper, right, lower))
        # 保存裁剪后的图片
        cropped_img = cropped_img.crop(zero_crop_location)
        cropped_img.save(o_label_path)


'''
计算图像集中的最小左边界和上边界百分比值以及最大右边界和下边界百分比值
'''


def get_max_bounding_box_percentage(directory, label_list):
    left_min, upper_min, right_max, lower_max = None, None, None, None

    for file_name in label_list:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(os.path.join(directory, file_name)) as img:
                width, height = img.size
                img_array = np.array(img)

                # 获取非零像素的位置
                non_zero_indices = np.nonzero(img_array)
                # 全零图像则退出
                if non_zero_indices[0].size == 0 or non_zero_indices[1].size == 0:
                    continue

                left, upper, right, lower = (np.min(non_zero_indices[1]) / width, np.min(non_zero_indices[0]) / height,
                                             np.max(non_zero_indices[1]) / width, np.max(non_zero_indices[0]) / height)

                # 更新最大边框
                if left_min is None or left < left_min:
                    left_min = left
                if upper_min is None or upper < upper_min:
                    upper_min = upper
                if right_max is None or right > right_max:
                    right_max = right
                if lower_max is None or lower > lower_max:
                    lower_max = lower

    if left_min is None or upper_min is None or right_max is None or lower_max is None:
        return None

    return left_min, upper_min, right_max, lower_max


'''
裁剪掉图像边缘的零元素
'''


def crop_zero_elements(img):
    # 将图片转换为NumPy数组
    img_array = np.array(img)

    # 将像素值小于等于1的元素设为0
    img_array[img_array <= 3] = 0

    # 获取非零像素的位置
    non_zero_indices = np.nonzero(img_array)
    if non_zero_indices[0].size == 0 or non_zero_indices[1].size == 0:
        return None

    # 计算边界框
    left, upper, right, lower = (np.min(non_zero_indices[1]), np.min(non_zero_indices[0]),
                                 np.max(non_zero_indices[1]), np.max(non_zero_indices[0]))

    # 裁剪图片
    cropped_img = img.crop((left, upper, right, lower))
    return cropped_img, (left, upper, right, lower)


if __name__ == "__main__":
    main()
