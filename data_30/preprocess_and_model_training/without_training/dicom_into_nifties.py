# 把.dcm改成.nii.gz的檔案
import glob
import os
import dicom2nifti

in_path_images = 'D:/EC-202403+202405/18/201908-RTCT'
out_path_images = 'D:/dicom_into_nifties'

list_images = glob.glob(in_path_images)

for image in list_images:
    patient_name = os.path.basename(os.path.normpath(image))
    dicom2nifti.dicom_series_to_nifti(image, os.path.join(out_path_images,'18_'+ patient_name + '.nii.gz'))


