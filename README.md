# OriCon3D: Effective 3D Object Detection using Orientation and Confidence

Paper Link: https://doi.org/10.48550/arXiv.2304.14484

Here:\
Any file with name or accrediation
- exp_6 means VGG19 based data
- exp_7 means MOBILENETv2  based data
- exp_8 means EFFICIENTNETv2_s   based data




Results are divided into:
1. BASELINE_images (REPRODUCTION RESULTS AMD REDEVELOPMENT RESULTS)
2. OUR METHOD_images (EXTENSION RESULTS)






In USER ----> Kitti_datasets_and_results ----> training (you will need 3 files)   -----> calib
                                                                                  ----> label_2
                                                                                  ----> image_2


Similarly for, USER ----> Kitti_datasets_and_results ----> validation (u will need 3 files) -----> calib
                                                                                            ----> label_2
                                                                                            ----> image_2


Here we don't have official labels of KITTI test dataset from KITTI 3D object detection benchmark, we split our train dataset into 5992 datapoints (train) and 1488 (validation) i.e. approx. 80:20 split the train dataset can be downloaded from here: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

In,  USER ----> Kitti_datasets_and_results ----> training/validation -----> pred_images ----> BASELINE_images/OUR METHOD_images -----> exp6/exp7/exp8 -----> {each image is divided into 3 sections, the top section is 2D YOLO output,; the middle section is Ground truth 3D Bounding box and the third section has our predictions for 3D Bounding box predictions} and for logistic convenience we have uploaded limited number of images in results, possibly covering all 3 classes.


Results (OriCon3D):

VGG19

MOBILENETv2

EFFICIENTNETv2_s


