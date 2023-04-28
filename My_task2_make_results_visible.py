import os
from pathlib import Path
import SimpleITK as sitk
import cv2
import numpy as np
from PIL import Image


data_path = Path("/home/yangjiaqi/data/nnUNet/Data/nnUNet_trained_models/nnUNet/2d/Task066_CervicalTumor/nnUNetTrainerV2__nnUNetPlansv2.1/")  # 存储父地址
raw_data_path = Path("/home/yangjiaqi/data/nnUNet/Data/nnUNet_raw/nnUNet_raw_data/Task066_CervicalTumor/")

# from
imagesTr_path = os.path.join(raw_data_path, "imagesTr")
labelsTr_path = os.path.join(raw_data_path, "labelsTr")
imageTr_pp_path = os.path.join(data_path, "cv_niftis_postprocessed")  # postprocessed image whole, 3644
labelTr_pp_path = os.path.join(os.path.join(data_path, "fold_4"),
                               'validation_raw_postprocessed')  # postprocessed label for test, 728
labelTr_path = os.path.join(os.path.join(data_path, "fold_4"),
                            'validation_raw')  # raw label for test, 728, contain other stuff

# to
to_label_path = os.path.join(raw_data_path, "o_inferslabel")  # label, for test
to_image_pp_path = os.path.join(raw_data_path, "o_infersimage_pp")
to_label_pp_path = os.path.join(raw_data_path, "o_inferslabel_pp")
to_raw_labelsTr_path = os.path.join(raw_data_path, "o_labelsTr")
to_raw_imagesTr_path = os.path.join(raw_data_path, "o_imagesTr")
to_blend_gt_path=os.path.join(raw_data_path,'blend_gt')
to_blend_predict_path=os.path.join(raw_data_path,'blend_predict')

color_map = [(0, 0, 0), (0, 255, 0), (0, 128, 0), (128, 128, 0), (128, 0, 128), (0, 128, 128),
               (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128),
               (192, 0, 128),
               (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0),
               (0, 64, 128), (128, 64, 12)]
alpha=0.5

def main():
    # 调整此处参数即可展现所有的图像
    # show_all_image(imagesTr_path,to_raw_imagesTr_path)
    # show_all_image(labelsTr_path,to_raw_labelsTr_path)
    # show_all_image(imageTr_pp_path,to_image_pp_path)
    # show_all_image(labelTr_path,to_label_path)
    # show_all_image(labelTr_pp_path,to_label_pp_path)
    #show_label(r'D:\learning\UNNC 科研\data\nnUNet\bal_label',r'D:\learning\UNNC 科研\data\nnUNet\test_label')

    # for image in os.listdir(to_raw_imagesTr_path):
    #     new_image=image.replace('_0000','')
    #     os.rename(os.path.join(to_raw_imagesTr_path,image),os.path.join(to_raw_imagesTr_path,new_image))
    #
    blend_raw_images(to_raw_labelsTr_path,to_raw_imagesTr_path,to_blend_gt_path,color_map=color_map)
    blend_raw_images(to_label_pp_path,to_raw_imagesTr_path,to_blend_predict_path,color_map=color_map)


def show_label(src: str, dst: str):
    '''
    从01label可视化
    Args:
        src:
        dst:

    Returns:

    '''
    if not os.path.exists(dst):
        os.makedirs(dst)
    for label in os.listdir(src):
        im=Image.open(os.path.join(src,label))
        im=np.array(im)
        im[im!=0]=128
        Image.fromarray(im).save(os.path.join(dst,label))


def show_all_image(src: str, dst: str):
    '''
    从nii文件可视化
    Args:
        src:
        dst:

    Returns:

    '''
    if not os.path.exists(dst):
        os.makedirs(dst)

    src_list = os.listdir(src)

    for nii_file in src_list:
        if nii_file.endswith('.nii.gz'):
            cur_addr = os.path.join(src, nii_file)
            nii = sitk.ReadImage(cur_addr)
            img = sitk.GetArrayFromImage(nii).squeeze()
            # img = (img - img.min()) / (img.max() - img.min())
            # img *= 255
            Image.fromarray(img).save(os.path.join(dst, nii_file.replace('.nii.gz', '.png')))
        #cv2.imwrite(os.path.join(dst, nii_file.replace('.nii.gz', '.jpg')), img)

def blend_raw_images(label_path,image_path,output_path,color_map,alpha=0.5):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # 将2个文件夹的图像融合输出进新文件夹
    for label_name in os.listdir(label_path):
        label=np.array(Image.open(os.path.join(label_path,label_name)))
        # t=np.unique(label)
        label=convert_to_rgb(label,colormap=color_map)
        # t = np.unique(label)
        # label[label!=0]=1
        # t=np.unique(label)
        image=Image.open(os.path.join(image_path,label_name).replace('.png','.jpg'))
        image=image.convert('RGB')
        blend_image=blend_images(Image.fromarray(label),image,alpha)
        blend_image.save(os.path.join(output_path,label_name).replace('.png','.jpg'))

def blend_images(image1: Image.Image, image2: Image.Image, alpha: float) -> Image.Image:
    """
    融合两幅RGB图像，设定指定的alpha，对于第一张图像零像素位置，完全使用第二张图像，其余位置按alpha进行融合。

    Args:
        image1 (Image.Image): 第一张RGB图像,label
        image2 (Image.Image): 第二张RGB图像,image
        alpha (float): 融合时的权重，范围为 0.0 到 1.0。

    Returns:
        Image.Image: 融合后的图像。
    """

    # 将PIL.Image转换为numpy数组
    image1_np = np.array(image1)
    image2_np = np.array(image2)

    # 创建一个空白的输出图像（与输入图像大小相同）
    blended_np = np.zeros_like(image1_np)

    # 找到第一张图像中非零像素的位置
    non_zero_indices = np.any(image1_np != 0, axis=-1)

    # 对于第一张图像零像素位置，完全使用第二张图像
    blended_np[~non_zero_indices] = image2_np[~non_zero_indices]

    # 按alpha进行融合的其余位置
    blended_np[non_zero_indices] = (1 - alpha) * image1_np[non_zero_indices] + alpha * image2_np[non_zero_indices]

    # 将numpy数组转换回PIL.Image
    blended_image = Image.fromarray(blended_np.astype(np.uint8))

    return blended_image

def convert_to_rgb(label_img, colormap):
    rgb_img = np.zeros((label_img.shape[0], label_img.shape[1], 3), dtype=np.uint8)

    for label in range(len(colormap)):
        color = colormap[label]
        rgb_img[np.where(label_img == label)] = color

    return rgb_img

if __name__ == '__main__':
    main()
