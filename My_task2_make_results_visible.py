import os
from pathlib import Path
import nibabel as nib

data_path = Path("/home/yangjiaqi/data/nnUNet/Data/nnUNet_trained_model/nnUNet/2d/Task066_CervicalTumor"
                 "/nnUNetTrainerV2__nnUNetPlansv2.1")  # 存储父地址

imageTr_pp_path=os.path.join(data_path,"cv_niftis_postprocessed") # postprocessed image whole, 3644
labelTr_pp_path=os.path.join(os.path.join(data_path,"fold_4"),'validation_raw_postprocessed') # postprocessed label for test, 728
labelTr_path=os.path.join(os.path.join(data_path,"fold_4"),'validation_raw') # raw label for test, 728, contain other stuff