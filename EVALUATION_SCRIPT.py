## this code here is used to evaluate the results of the trained weights

import os
import cv2
import errno

import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import argparse
import torch

from lib.DataUtils import *
from lib.Utils import *
from tqdm import tqdm

from lib import Model, ClassAverages









def main():

    exp_no = 6   # <------#'{you can change this to the number to which your models weights are accredieted to}'

    print ("Beginning with evaluation")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")



    print('Accessing the path of stored weights')
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/exp_' + str(exp_no) + '/'
    ## for multiple weights(for various epochs) we make a weight list, this completely optional but provides a more arranged manner of loading weights
    weight_list = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]





    # Create output folder for predictedlabels and predicted images

    # Output Validation folder
        # ----> pred-labels
                # -----> exp_6
                        #  ----> ######.txt 
                        # .........

        # ----> pred-imgs
                # -----> exp_6
                        #  ----> ######.png 
                        # .........



    for x in range(len(weight_list)):

        C_M_direc('Kitti/results/validation/labels/exp_' + str(exp_no) +"/epoch_%s/" % str(x+1))  #CYCLIST, CARS, PEDESTRIAN
    C_M_direc('Kitti/results/validation/pred_imgs/exp_' + str(exp_no))   #CYCLIST, CARS, PEDESTRIAN






    if len(weight_list) == 0:
        print('No weights')
        exit()
    
    for model_weight in weight_list:
        epoch_no = model_weight.split(".")[0].split('_')[-1]
        print ("Evaluating for Epoch: ",epoch_no)

        print ('Loading model with %s'%model_weight)



        ### THE MODEL
        ## when u select vgg19:  go to Models.py and make sure layer shape just after backbone is 512x7x7
        my_vgg = models.vgg19_bn(pretrained=True)
        ## when u select mobilenetv2 or efficientnetv2_s:  go to Models.py and make sure layer shape just after backbone is 1280x7x7
        # my_vgg = models.efficientnet_v2_s(pretrained=True)
        # my_vgg = models.mobilenet_v2(pretrained=True)


        model = Model.Model(features=my_vgg.features, bins=2)

        ## Checkpointing, as described in the report, we tried on the small cuda environment of GoogleColab
        if use_cuda: 
            checkpoint = torch.load(weights_path + model_weight)
        else: 
            checkpoint = torch.load(weights_path + model_weight)
        model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        model.eval()



## Input validation folder

    # Input Validation folder
        # ----> labels_2
                #  -----> ####.txt
                #  -----> ####.txt
                #  -----> ####.txt
                # ........

        # ----> image_2
                #  -----> ####.png
                #  -----> ####.png
                #  -----> ####.png
                # ........

        # ----> calib
                #  -----> ####.txt
                #  -----> ####.txt
                #  -----> ####.txt
                # ........



        # Load Test Images from eval folder
        dataset = Dataset('/home/drajani/Downloads/3D-Bounding-Boxes-From-Monocular-Images/Kitti/validation')
        # dataset = Dataset('/home/drajani/Downloads/3D-Bounding-Boxes-From-Monocular-Images/Kitti/validation/PEDESTRIAN')  #CYCLIST, CARS, PEDESTRIAN

        all_images = dataset.all_objects()
        print ("Length of eval data",len(all_images))
        averages = ClassAverages.ClassAverages()

        all_images = dataset.all_objects()
        print ("Model is commencing predictions.....")
        for key in tqdm(sorted(all_images.keys())):

            data = all_images[key]
            truth_img = data['Image']
            img = np.copy(truth_img)
            imgGT = np.copy(truth_img)
            objects = data['Objects']
            cam_to_img = data['Calib']

            filename =  "Kitti/results/validation/labels/exp_" +str(exp_no) + '/epoch_' + str(epoch_no) + "/" +str(key)+".txt"
            # filename =  "Kitti/results/validation/labels/exp_" +str(exp_no)+ '_PEDESTRIAN' + '/epoch_' + str(epoch_no) + "/" +str(key)+".txt"    #CYCLIST, CARS, PEDESTRIAN

            C_M_direc(filename)   
            file = open(filename,"w")

            for object in objects:

                label = object.label
                theta_ray = object.theta_ray
                input_img = object.img

                input_tensor = torch.zeros([1,3,224,224])
                input_tensor[0,:,:,:] = input_img
                input_tensor.cuda()


                ### SEPARATING THE MODEL INTO OREINT, CONF AND DIM
                ## NOW THE AFOREMNTIONED SPLIT IS ACCORDING TO THE AUTOR'S METHOD. sINCE IN OPUR METHOD WE SPLIT THE MULTIBIN ARCHITEC TURE INTO
                ## JUST ORIENT AND CONFIDENC SCORE,WE CAN EITHER SPLIT THIS AS TWO PARTS OR AS THREE PARTS EITHER WAY IS FINE HERE, SINCE WE HAVE TRAINED ONLY ON 
                ## ORIENTATION AND CONFIDENCE SCORE HENCE WE WILL GET ANSWERS JUST ALONG THOSE LINE EVEN IF WE TAG IN THE UNEEDED DIMENSION


                [orient, conf, dim] = model(input_tensor)  # OR [orient, conf, _] = model(input_tensor) WITH SUITABLE ADJUSTMENTS
                orient = orient.cpu().data.numpy()[0, :, :]
                conf = conf.cpu().data.numpy()[0, :]
                dim = dim.cpu().data.numpy()[0, :]
                dim += averages.get_item(label['Class'])


                ## DEFINING THE ORIENTATION(SIN+COS) AND ALPHA COMPOENTS FROM THE MODEL.PY FILE
                argmax = np.argmax(conf)
                orient = orient[argmax, :]
                cos = orient[0]
                sin = orient[1]
                alpha = np.arctan2(sin, cos)
                alpha += dataset.angle_bins[argmax]
                alpha -= np.pi
                ### 2D AND 3D BOX REGRESSION
                location = Regresssed3Dbox_2(img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)
                locationGT = Regresssed3Dbox_2(imgGT, truth_img, cam_to_img, label['Box_2D'], label['Dimensions'], label['Alpha'], theta_ray)

                file.write( \
                    #  Class label
                    str(label['Class']) + " -1 -1 " + \
                    # Alpha
                    str(round(alpha,2)) + " " + \
                    # 2D Bounding box coordinates
                    str(label['Box_2D'][0][0]) + " " + str(label['Box_2D'][0][1]) + " " + \
                        str(label['Box_2D'][1][0]) + " " + str(label['Box_2D'][1][1]) + " " + \
                    # 3D Box Dimensions
                    str(' '.join(str(round(e,2)) for e in dim)) + " " + \
                    # 3D Box Location
                    str(' '.join(str(round(e,2)) for e in location)) + " 0.0 " + \
                    # Ry
                    str(round(theta_ray + alpha ,2)) + " " + \
                    # Confidence
                    str( round(max(softmax(conf)),2) ) + "\n" 
                )

                # print('Estimated pose: %s'%location)
                # print('Truth pose: %s'%label['Location'])
                # print('-------------')



            file.close()
            
            
            numpy_vertical = np.concatenate((truth_img,imgGT, img), axis=0)
            image_name = 'Kitti/results/validation/pred_imgs/exp_' + str(exp_no)+ '/' + "/epoch_" + epoch_no + '_' + str(key) + '.jpg'  #CYCLIST, CARS, PEDESTRIAN
            C_M_direc(image_name)    
            cv2.imwrite(image_name, numpy_vertical)

        print ("Finished.")


# DRIVER CODE
if __name__ == '__main__':


    main()