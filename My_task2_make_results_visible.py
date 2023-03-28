import os
from pathlib import Path
import SimpleITK as sitk
import cv2


data_path = Path("/home/yangjiaqi/data/nnUNet/Data/nnUNet_trained_model/nnUNet/2d/Task066_CervicalTumor"
                 "/nnUNetTrainerV2__nnUNetPlansv2.1")  # 存储父地址
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


def main():
    # 调整此处参数即可展现所有的图像
    show_all_image(imagesTr_path,to_raw_imagesTr_path)
    show_all_image(labelsTr_path,to_raw_labelsTr_path)
    show_all_image(imageTr_pp_path,to_image_pp_path)
    show_all_image(labelTr_path,to_label_path)
    show_all_image(labelTr_pp_path,to_label_pp_path)


def show_all_image(src: str, dst: str):
    src_list = os.listdir(src)

    for nii_file in src_list:
        cur_addr = os.path.join(src, nii_file)
        nii = stik.ReadImage(cur_addr)
        img = sitk.GetArrayFromImage(nii)
        img = (img - img.min()) / (img.max() - img.min())
        img *= 255
        cv2.imwrite(os.path.join(dst, nii_file.replace('.nii.gz', '.png')), img)


if __name__ == '__main__':
    main()
