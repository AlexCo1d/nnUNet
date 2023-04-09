import os
import random
from pathlib import Path

import imageio
from PIL import Image
import shutil
import numpy as np
import json

padding = 200
threshold = 10
'''
    记录一下裁剪参数：
    Dual_Image:                 
        -large_left_space:      (0.540419, 0.1965908, 0.9296875, 0.8511363)
        -left_space:            (545/1024, 127/768, 981/1024, 716/768)
        -no space:              (568 / 1136, 211 / 852, 1, 750/852)
        -other2:                (545 / 1087, 95 / 797, 1060/1087, 645/797)
        -other:                 (568 / 1136, 220 / 852, 1, 830/852)
        -both_small_space:      (576 / 1152, 130 / 872, 1097/1152, 761/872)
        -both_space:            (512 / 1024, 146/768, 1011/1024, 692/872)
        -full:                  (482 / 960, 181/720, 1, 654/720)
        -right_space:           (480 / 975, 227/735, 960/975, 666/720)
        -wide_left_space:       (460 / 800, 211/600, 780/800, 480 / 600) 
        
    Large_Image:                (336 / 1264, 215/880, 1024/1264, 714 / 880)
    
    Small_Image:                (195 / 1136, 150/768, 903/1024, 0.91)
    
    Middle_Image:
        -Normal:                (230 / 1024, 212 / 768, 896/1024, 724/768)
        -Wider_Image:           (178 / 800, 127 / 600, 727 / 800, 500 / 600)
    
    other_condition_Image:
        -Condition1_W:          (0.2, 0.4, 0.75, 0.85)
        -Condition2:            (300/960, 114/720, 640/960, 318/720)
        -Condition3_N:          (195/1024, 114/768, 850/1024, 569/768)
        -test:                   
'''


def main():
    target = ''
    Image_path = os.path.join(r'D:\learning\UNNC 科研\Cervical\datasets\target', target)
    Label_path = r'D:\learning\UNNC 科研\Cervical\datasets\SegmentationClass'
    O_image_path = os.path.join(r'D:\learning\UNNC 科研\Cervical\datasets\target\O_image', target)
    O_label_path = os.path.join(r'D:\learning\UNNC 科研\Cervical\datasets\target\O_label', target)
    final_image=r'D:\learning\UNNC 科研\Cervical\datasets\target\final_image_set'
    final_label=r'D:\learning\UNNC 科研\Cervical\datasets\target\final_label_set'
    if not os.path.exists(O_image_path):
        os.makedirs(O_image_path)
    if not os.path.exists(O_label_path):
        os.makedirs(O_label_path)

    label_list = [s.replace('.jpg', '.png') for s in os.listdir(Image_path)]
    percentage = get_max_bounding_box_percentage(Label_path, label_list)
    print(f"max label box: {percentage}")

    set_percentage = (287/800, 103/600, 644/800, 503/600)
    print(f"set box: {set_percentage}")

    for image in os.listdir(Image_path):
        crop_image(os.path.join(Image_path, image), os.path.join(Label_path, image.replace(".jpg", '.png')),
                   os.path.join(O_image_path, image), os.path.join(O_label_path, image.replace(".jpg", '.png')),
                   set_percentage)

    move_files_to_parent(O_image_path,final_image)
    move_files_to_parent(O_label_path,final_label)


def crop_image(image_path, label_path, o_image_path, o_label_path, crop_percentages, pad=False):
    """
    处理image和label，根据crop_percentages输出裁剪过后的图像。其中还调用了crop_zero_elements以使所有的image取得最简洁的边界。

    Args:
        image_path(str):    the input image path
        label_path(str):    the input label path
        o_image_path(str):   the output image path
        o_label_path(str):   the output label path
        crop_percentages (tuple[float,float,float,float]):  image's cropping percentage
        pad(bool): use padding or not
    """
    # 打开并处理image图片
    img = Image.open(image_path)
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
    img = img.crop((left, upper, right, lower))
    cropped_img, zero_crop_location = crop_zero_elements(img)
    print((left, upper, right, lower))
    print(zero_crop_location)
    # 保存裁剪后的图片
    cropped_img.save(o_image_path)

    # 处理并处理label图片
    label_img = Image.open(label_path)
    width, height = label_img.size

    # 根据百分比计算裁剪坐标
    left = max(0, int(width * crop_percentages[0]) - 1)
    upper = max(0, int(height * crop_percentages[1]) - 1)
    right = min(width, int(width * crop_percentages[2]) + 1)
    lower = min(height, int(height * crop_percentages[3]) + 1)

    if pad is True:
        # 向各方向扩充padding像素
        left = max(0, left - padding)
        upper = max(0, upper - padding)
        right = min(width, right + padding)
        lower = min(height, lower + padding)

    cropped_label = label_img.crop((left, upper, right, lower))
    # 保存裁剪后的图片
    cropped_label = cropped_label.crop(zero_crop_location)
    cropped_label.save(o_label_path)


'''
计算图像集中的最小左边界和上边界百分比值以及最大右边界和下边界百分比值
'''


def get_max_bounding_box_percentage(directory, label_list):
    """
        得到label最大的百分比裁剪框。
    Args:
        directory:  目标文件夹
        label_list: image对应的label组成的list

    Returns:
        (float,float,float,float): a tuple of cropped box
    """
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


def crop_zero_elements(img):
    """
    裁剪掉图像边缘的零（threshold）元素
    Args:
        img(PIL.Image): Image格式的图像

    Returns:
        (PIL.Image): Image格式的图像

    """
    # 将图片转换为NumPy数组
    img_array = np.array(img.convert('L'))

    # 将像素值小于等于1的元素设为0
    img_array[img_array <= threshold] = 0

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


def move_files_to_parent(parent_folder,target_path):
    for root, dirs, files in os.walk(parent_folder):
        for file in files:
            source_path = os.path.join(root, file)
            destination_path = os.path.join(target_path, file)

            if source_path != destination_path:
                shutil.move(source_path, destination_path)


if __name__ == "__main__":
    main()
