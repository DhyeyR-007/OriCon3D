### THIS IS THE TRAINING SCRIPT FOR TRAINING THE FCN (in Model.py file) BY LEVERAGING IMAGE CROPS FROM LABELS OF TRAINING DATA
# Here (you can pick any experiment number)
# exp6 is  vgg19 
# exp7 is  mobilenetv2
# exp8 is  efficientnetv2_s
# custom function : means our function


### IMPORT ###
import os
import torch
import cv2
import argparse
import torch.nn as nn

## All custom functions
from lib.DataUtils import *
from lib.Utils import *
from lib.Model import *
from lib import ClassAverages

from torch.utils import data as torch_data
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


from torch.autograd import Variable
import torchvision.models as models
from torchsummary import summary
from tqdm import tqdm






def main():
    print ("Initializing....")


    ### HYPERPARAMETER INITALIZATION ###
    epochs = 12
    momentum = 0.99

    alpha = 0.6     # Dimension variable
    w = 0.7         # Orientation variable

    batch_size = 8  # batch size
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 6}
    lr = 0.0001     # learning rate

    exp_no = 6 #.......{can be anything you want, this is just for segregation}

  




    print ("Beginning Training...........for {0} epochs and {1} batch size".format(epochs,batch_size))

    ### DEVICE SELECTION
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")


    ### DATA LOADER
    train_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/training'
    dataset = Dataset(train_path)
    generator = torch_data.DataLoader(dataset, **params)
    print('Loaded Training Dataset')


    ### THE MODEL
    ## when u select vgg19:  go to Models.py and make sure layer shape just after backbone is 512x7x7
    my_vgg = models.vgg19_bn(pretrained=True)
    ## when u select mobilenetv2 or efficientnetv2_s:  go to Models.py and make sure layer shape just after backbone is 1280x7x7
    # my_vgg = models.efficientnet_v2_s(pretrained=True)
    # my_vgg = models.mobilenet_v2(pretrained=True)
    model = Model(features=my_vgg.features).cuda()


    ### WEIGHT STORAGE PATH
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights/'
    first_epoch = 0


    ### DEFINING OPTIMIZER AND LOSS FUNCTIONS
    Optimizersgd = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)


    ### all kinds of losses we will be using 
    orientation_score_cal_func = OrientationLoss
    confidence_score_cal_func = nn.CrossEntropyLoss().cuda()
    dimension_score_cal_func = nn.MSELoss().cuda()
    


    total_num_batches = int(len(dataset) / batch_size)
    losses=[]
    epoch_losses=[]
    dim_lossess=[]
    theta_lossess=[]
    orient_lossess=[]
    
    for epoch in range(first_epoch+1, epochs+1):
        curr_batch = 0
        passes = 0
        for local_batch, local_labels in generator:

            Ground_orientation = local_labels['Orientation'].float().cuda()
            Ground_confidence = local_labels['Confidence'].long().cuda()
            Ground_dimension = local_labels['Dimensions'].float().cuda()

            local_batch=local_batch.float().cuda()
            [orient, conf, dim] = model(local_batch)

            orient_loss = orientation_score_cal_func(orient, Ground_orientation, Ground_confidence)
            dim_loss = dimension_score_cal_func(dim, Ground_dimension)

            Ground_confidence = torch.max(Ground_confidence, dim=1)[1]
            conf_loss = confidence_score_cal_func(conf, Ground_confidence)




        ### this the loss for authors multibin architecture
            # loss_theta = conf_loss + w * orient_loss
            # loss = alpha * dim_loss + loss_theta

        ### this the loss for out proposed multibin architecture
            loss_theta = w * conf_loss + w * orient_loss
            loss = loss_theta







            ### BACKPROP
            Optimizersgd.zero_grad()
            loss.backward()
            Optimizersgd.step()

            ### LETS OBSERVE THE MODEL'S BEHAVIOUS AFTER EVERY 50 ITERATIONS IN AN EPOCH
            if passes % 50 == 0:
                print("--- epoch %s | batch %s/%s --- [loss: %s]" %(epoch, curr_batch, total_num_batches, loss.item()))
                passes = 0

            orient_lossess.append(orient_loss.item())
            dim_lossess.append(dim_loss.item())
            theta_lossess.append(loss_theta.item())
            losses.append(loss.item())
            passes += 1
            curr_batch += 1

       
        epoch_losses.append(loss.item())
        ### ++++++++++++++++++++++++++++++++++++++++++++
        # SAVING MODEL STATISTICS AFTER EVERY 10 EPOCHS
        if epoch % 1 == 0:
            name = weights_path + "exp_"+ str(exp_no) + "/exp_"+ str(exp_no) + '_epoch_%s.pkl' % epoch

            print ("Done with epoch %s!" % epoch)
            print ("Saving weights as %s ..." % name)
            torch.save({'epoch': epoch,'model_state_dict': model.state_dict(),'optimizer_state_dict': Optimizersgd.state_dict(),'loss': loss}, name)




        print ("Now we plot all the resultant graphs")

        if epoch == epochs:
            plt.figure(figsize=(35,15))
            plt.plot(losses)
            plt.ylabel('Overall Loss')
            plt.xlabel('Iterations')
            plt.savefig( 'Kitti/results/training/plots/exp_'+ str(exp_no) + '/' + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + '_Overall-Loss.png')
            plt.clf()
            plt.figure(figsize=(35,15))
            plt.plot(orient_lossess)
            plt.ylabel('Loss in Orientation')
            plt.xlabel('Iterations')
            plt.savefig( 'Kitti/results/training/plots/exp_'+ str(exp_no) + '/' + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + "_Orientation.png")
            plt.clf()
            plt.figure(figsize=(35,15))
            plt.plot(dim_lossess)
            plt.ylabel('Loss in Dimension')
            plt.xlabel('Iterations')
            plt.savefig( 'Kitti/results/training/plots/exp_'+ str(exp_no) + '/' + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + '_Dimension.png')
            plt.clf()
            plt.figure(figsize=(35,15))
            plt.plot(theta_lossess)
            plt.ylabel('Theta Loss')
            plt.xlabel('Iterations')
            plt.savefig( 'Kitti/results/training/plots/exp_'+ str(exp_no) + '/' + "epoch_%s/" % epoch + "exp_"+str(exp_no) +"_epoch_%s" % epoch + '_Theta.png')
            plt.clf()




### DRIVER CODE

if __name__=='__main__':

    main()