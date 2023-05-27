# OriCon3D: Effective 3D Object Detection using Orientation and Confidence

**Paper Link: https://doi.org/10.48550/arXiv.2304.14484**

## General Information
Here:\
Any file with name or accrediation
- exp_6 means VGG19 based data
- exp_7 means MOBILENETv2  based data
- exp_8 means EFFICIENTNETv2_s   based data




Results are divided into:
1. BASELINE_images (REPRODUCTION RESULTS AMD REDEVELOPMENT RESULTS)
2. OUR METHOD_images (EXTENSION RESULTS)






- In USER ----> Kitti_datasets_and_results ----> training (you will need 3 files)   -----> calib
                                                                                  ----> label_2
                                                                                  ----> image_2


- Similarly for, USER ----> Kitti_datasets_and_results ----> validation (u will need 3 files) -----> calib
                                                                                            ----> label_2
                                                                                            ----> image_2


- Here we don't have official labels of KITTI test dataset from KITTI 3D object detection benchmark, we split our train dataset into 5992 datapoints (train) and 1488 (validation) i.e. approx. 80:20 split the train dataset can be downloaded from here: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d

- In,  USER ----> Kitti_datasets_and_results ----> training/validation -----> pred_images ----> BASELINE_images/OUR METHOD_images -----> exp6/exp7/exp8 -----> {each image is divided into 3 sections, the top section is 2D YOLO output,; the middle section is Ground truth 3D Bounding box and the third section has our predictions for 3D Bounding box predictions} and for logistic convenience we have uploaded limited number of images in results, possibly covering all 3 classes.

- We develop correlations between the oreintation and confidence aspects of our multibin architecture for making the algorithm light weight:

<img width="600" alt="image" src="https://github.com/DhyeyR-007/OriCon3D/assets/86003669/24df1b30-cf20-42b5-a8d1-ae1ac3b3c360">


# Results (OriCon3D):

## VGG19

<img width="847" alt="image" src="https://github.com/DhyeyR-007/OriCon3D/assets/86003669/5e0fbe29-6744-48ab-a2bc-c32eb651781f">

![](https://github.com/DhyeyR-007/OriCon3D/blob/main/Gifs%20and%20videos/VGG_gif.gif)

------------------------------------------------------------------------------------------------------------------------------------------------


## MOBILENETv2

<img width="847" alt="image" src="https://github.com/DhyeyR-007/OriCon3D/assets/86003669/b686ec3c-92a0-4135-91c8-53103d8a2895">

![](https://github.com/DhyeyR-007/OriCon3D/blob/main/Gifs%20and%20videos/MOBILE_gif.gif)


------------------------------------------------------------------------------------------------------------------------------------------------


## EFFICIENTNETv2_s

<img width="847" alt="image" src="https://github.com/DhyeyR-007/OriCon3D/assets/86003669/4cd424be-e5e7-4c0d-92d9-c678a1d720ba">

![](https://github.com/DhyeyR-007/OriCon3D/blob/main/Gifs%20and%20videos/EFF_gif.gif)


