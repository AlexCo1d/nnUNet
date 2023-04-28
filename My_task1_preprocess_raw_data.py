import os
import random
from pathlib import Path

from PIL import Image
import shutil
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunet.dataset_conversion.utils import generate_dataset_json
from nnunet.paths import nnUNet_raw_data
from nnunet.utilities.file_conversions import convert_2d_image_to_nifti

#label中所含的值
set_label=1

raw_data_path = Path("/home/yangjiaqi/data/nnUNet/raw_Data/")  # 所有图片的父地址
task_name = 'Task067_Cervical2D'
base=join(raw_data_path,'Task067')
target_base = join("/home/yangjiaqi/data/nnUNet/Data/nnUNet_raw/nnUNet_raw_data/", task_name)

image_path = os.path.join(base, "image")  # 图像
label_path = os.path.join(base, "label")  # 标签
target_imagesTr = join(target_base, "imagesTr")
target_imagesTs = join(target_base, "imagesTs")
target_labelsTs = join(target_base, "labelsTs")
target_labelsTr = join(target_base, "labelsTr")

data_path = Path("/home/yangjiaqi/data/nnUNet/Data/nnUNet_raw/nnUNet_raw_data/Task067_Cervical2D")  # 存储父地址
imagesTr = os.path.join(data_path, "imagesTr")
imagesTs = os.path.join(data_path, "imagesTs")
labelsTr = os.path.join(data_path, "labelsTr")
labelsTs = os.path.join(data_path, "labelsTs")
random_num = 0  # 设置随机选测试图像数目

image_list = os.listdir(image_path)
label_list = os.listdir(label_path)

#为满足nnUnet的要求，label必须连续，将所有label的128转化为1
# for file in os.listdir(label_path):
#     image=np.array(Image.open(os.path.join(label_path,file)))
#     image[image==128]=1
#     Image.fromarray(image).save(os.path.join(label_path,file))

def main():
    process1()
    #process2()


def process1():
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
        print(np.shape(cur_image))
        spac = (999, 1, 1)
        # 转化为ntfi格式存储进各自需要的路径
        cur_image_nii = sitk.GetImageFromArray(cur_image)
        cur_label_nii = sitk.GetImageFromArray(cur_label)
        cur_label_nii.SetSpacing(list(spac)[::-1])
        cur_image_nii.SetSpacing(list(spac)[::-1])
        sitk.WriteImage(cur_image_nii, os.path.join(imagesTr, cur_image_name))
        sitk.WriteImage(cur_label_nii, os.path.join(labelsTr, cur_label_name))

    # generate json file
    generate_dataset_json(output_file=os.path.join(data_path, "dataset.json", ), imagesTr_dir=imagesTr,
                          imagesTs_dir=imagesTs, modalities=('gray',)
                          , labels={0: 'background', 1: 'tumor'}, dataset_name='Task067_CervicalTumor',
                          license='hands_off')


def check_shape():
    for file in os.listdir(imagesTr):
        itkimage1=sitk.ReadImage(os.path.join(imagesTr,file))
        itkimage2 = sitk.ReadImage(os.path.join(labelsTr, file.replace('_0000','')))
        array=sitk.GetArrayFromImage(itkimage1).shape
        array2=sitk.GetArrayFromImage(itkimage2).shape
        if array != array2:
            print('!!!')


def process2():
    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    # convert the training examples. Not all training images have labels, so we just take the cases for which there are
    # labels
    labels_dir_tr = join(base, 'label')
    images_dir_tr = join(base, 'image')
    training_cases = subfiles(labels_dir_tr, suffix='.png', join=False)
    for t in training_cases:
        unique_name = t[:-4]  # just the filename with the extension cropped away, so img-2.png becomes img-2 as unique_name
        input_segmentation_file = join(labels_dir_tr, t)
        input_image_file = join(images_dir_tr, t)

        output_image_file = join(target_imagesTr,
                                 unique_name)  # do not specify a file ending! This will be done for you
        output_seg_file = join(target_labelsTr, unique_name)  # do not specify a file ending! This will be done for you

        # this utility will convert 2d images that can be read by skimage.io.imread to nifti. You don't need to do anything.
        # if this throws an error for your images, please just look at the code for this function and adapt it to your needs
        convert_2d_image_to_nifti(input_image_file.replace(".png",".jpg"), output_image_file, is_seg=False)

        # the labels are stored as 0: background, 255: road. We need to convert the 255 to 1 because nnU-Net expects
        # the labels to be consecutive integers. This can be achieved with setting a transform
        print(input_segmentation_file)
        convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
                                  transform=lambda x: (x == set_label).astype(int))

    # now do the same for the test set
    # labels_dir_ts = join(base, 'testing', 'output')
    # images_dir_ts = join(base, 'testing', 'input')
    # testing_cases = subfiles(labels_dir_ts, suffix='.png', join=False)
    # for ts in testing_cases:
    #     unique_name = ts[:-4]
    #     input_segmentation_file = join(labels_dir_ts, ts)
    #     input_image_file = join(images_dir_ts, ts)
    #
    #     output_image_file = join(target_imagesTs, unique_name)
    #     output_seg_file = join(target_labelsTs, unique_name)
    #
    #     convert_2d_image_to_nifti(input_image_file, output_image_file, is_seg=False)
    #     convert_2d_image_to_nifti(input_segmentation_file, output_seg_file, is_seg=True,
    #                               transform=lambda x: (x == 255).astype(int))

    # finally we can call the utility for generating a dataset.json
    generate_dataset_json(join(target_base, 'dataset.json'), target_imagesTr, target_imagesTs, ('Red', 'Green', 'Blue'),
                          labels={0: 'background', 1: 'tumor'}, dataset_name=task_name, license='hands off!')


if __name__=='__main__':
    main()