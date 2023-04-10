"""
将视频数据转化为图片类型，并且解压所有nii文件进行对应。
"""
import cv2
import os
import tarfile
import nibabel as nib
import numpy as np

# 设定图片中癌变细胞像素的数值大小
set_value = 128


# 提取所有的视频文件，并调用函数
def main():
    ####################
    # 设置读取路径与输出路径##
    ####################
    videoPath = r"D:\learning\UNNC 科研\202210_CSD超声标注_图片及视频\CSD视频及标注_15例"  # 读取视频路径
    imgPath = r"D:\learning\UNNC 科研\202210_CSD超声标注_图片及视频\video_image"  # 保存图片路径
    imgPath_img = os.path.join(imgPath, "img")
    imgPath_mask = os.path.join(imgPath, "mask")
    if not os.path.exists(imgPath):
        os.mkdir(imgPath)
    if not os.path.exists(imgPath_img):
        os.mkdir(imgPath_img)
    if not os.path.exists(imgPath_mask):
        os.mkdir(imgPath_mask)

    for root, dirs, files in os.walk(videoPath):
        for d in dirs:
            cur_dir = os.path.join(root, d)
            for cur_file in os.listdir(cur_dir):
                if cur_file.lower().endswith(".avi"):
                    process_video(cur_dir, cur_file, imgPath_img, imgPath_mask)


'''
总函数
cur_dir: 目前avi文件的父目录
avi_video: 目前的avi文件名
imgPath_img,imgPath_mask: 输出的文件
'''


def process_video(cur_dir, avi_video, imgPath_img, imgPath_mask):
    #extract_avi_frame(cur_dir, avi_video, imgPath_img)
    extract_tar(cur_dir, avi_video, imgPath_mask)


'''
提取avi文件的每一帧，并输出至imgPath
'''


def extract_avi_frame(cur_dir, avi_video, imgPath_img):
    video_path = os.path.join(cur_dir, avi_video)
    cap = cv2.VideoCapture(video_path)
    # fps = cap.get(cv2.CAP_PROP_FPS)  # 获取帧率
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取宽度
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取高度
    suc = cap.isOpened()  # 是否成功打开
    frame_count = 1
    while True:
        suc, frame = cap.read()
        if suc:
            save_path = os.path.join(imgPath_img,
                                     avi_video.lower().replace(".avi", "") + "_%04d.jpg" % frame_count)
            cv2.imencode('.jpg', frame)[1].tofile(save_path)
            frame_count += 1
        else:
            break
    cap.release()


'''
提取对应的tar文件中的nii文件，并与img文件以相同的名字命名
'''


def extract_tar(cur_dir, avi_video, imgPath_mask):
    tar_path = os.path.join(cur_dir, avi_video.replace(".avi", "_avi_Label.tar").replace(".AVI", "_AVI_Label.tar"))
    nii = os.path.join(cur_dir, avi_video.replace(".avi", "_avi_Label.nii.gz").replace(".AVI", "_AVI_Label.nii.gz"))
    with tarfile.open(tar_path, 'r') as t:
        t.extractall(path=cur_dir)
        slice_nii(nii, avi_video, imgPath_mask)
        t.close()
    # 删除提取的文件
    os.remove(nii)
    os.remove(nii.replace(".nii.gz", ".json"))


"""
提取nii文件的每一帧
"""


def slice_nii(nii, avi_video, imgPath_mask):
    niifile = nib.load(nii)
    img = niifile.get_fdata()
    print(np.unique(img))
    img = process_img(img)

    for i in range(img.shape[2]):
        save_path = os.path.join(imgPath_mask, avi_video.lower().replace(".avi", "") + "_%04d.png" % (i + 1)).replace("\\", "/")
        cv2.imencode('.png', img[:, :, i].T)[1].tofile(save_path)
        #print(np.unique(img[:,:,i]))


"""
处理img文件中的信息
"""


def process_img(img):
    img[img == 4] = 0
    img[img == 2] = 0
    img[img == 1] = set_value
    return img

if __name__ == '__main__':
    main()
