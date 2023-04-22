import json
import math
import os
import shutil

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

# with open("summary.json", "r") as file:
#     data = json.load(file)
#
# dice_values = []
#
# for entry in data["results"]["all"]:
#     dice_value = entry["1"]["Dice"]
#     dice_values.append(dice_value)
# count=0
# count1=0
# sum=0
# for i in dice_values:
#     if not math.isnan(i):
#         sum+=i
#         count+=1
#         if i==0:
#             count1+=1
# t=sum/(count-count1)
# #95个非nan值。95/221
# #363/728
# import seaborn as sns
# filtered_data = [x for x in dice_values if not math.isnan(x)]
#
# # # 创建直方图
# sns.histplot(filtered_data, kde=False, bins=50)  # 您可以根据需要调整bw_adjust参数以改变曲线的平滑程度
#
# # 设置图像标题和坐标轴标签
# plt.title('Data Distribution')
# plt.xlabel('Value')
# plt.ylabel('Count')
#
# # 显示图像
# plt.show()


# path=r'D:\learning\UNNC 科研\data\nnUNet\exclude_image'
# o_path=r'D:\learning\UNNC 科研\data\nnUNet\exclude_label'
# image_list=os.listdir(path)
# for image in image_list:
#     shutil.move(os.path.join(r'D:\learning\UNNC 科研\data\nnUNet\final_label_set',image.replace(".jpg",".png")),o_path)
# output_fpath = r'D:\learning\UNNC 科研\data\nnUNet\final_image_plus_video'
#
# # input_video = os.path.join(father_path, file_name)  # single video'
#
# output_video = os.path.join(output_fpath, 'image')
# output_label = os.path.join(output_fpath, 'label')
# image = os.listdir(output_video)
# label = os.listdir(output_label)
count = 0
# for im in image:
#     old_path_v = os.path.join(output_video, im)
#     new_path_v = os.path.join(output_fpath, 'image1', f'{count}.jpg')
#     old_path_l = os.path.join(output_label, im.replace(".jpg", '.png'))
#     new_path_l = os.path.join(output_fpath, 'label1', f'{count}.png')
#
#     # 如果新路径与旧路径不同，移动文件以实现重命名
#     if old_path_v != new_path_v:
#         shutil.move(old_path_v, new_path_v)
#         shutil.move(old_path_l, new_path_l)
#     count+=1
#
# print(count)


from skimage import io
fpath=r'D:\learning\UNNC 科研\data\nnUNet\bal_label'
for image in os.listdir(fpath):
    img = Image.open(os.path.join(fpath,image))
    img=np.array(img)
    # print(io.imread(os.path.join(fpath,image)).shape)
    img[img!=0]=1
    # 将图像转换为单通道（灰度）
    gray_image = Image.fromarray(img).convert('L')
    gray_image.save(os.path.join(fpath,image))
