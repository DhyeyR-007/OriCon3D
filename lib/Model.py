import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable





from lib.File import *  # custom file



class Model(nn.Module):
    def __init__(self, features=None, bins=2, w = 0.4):
        super(Model, self).__init__()
        self.bins = bins
        self.w = w
        self.features = features


        ## ALGORITHMIC EXTENSION #1
        # VGG19: 512 x 7 x 7 (PAPER REPRODUCTION)
        # Mobilenetv2: 1280 x 7 x 7
        # Efficientnetv2_s: 1280 x 7 x 7


        ## ALGORITHMIC EXTENSION #2
        # OUR EXTENSION USES ONLY ORIENTATION + CONFIDENCE SCORE
        # WHEREAS BASLINE AUTHOR'S CODE USES ORIENTATION + CONFIDENCE SCORE + DIMENSION ((PAPER REPRODUCTION))

        self.orientation = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins*2) 
                )
        self.confidence = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, 256),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(256, bins),
      
                )
        


        self.dimension = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.ReLU(True),
                    nn.Dropout(),
                    nn.Linear(512, 3)
                )

    def forward(self, x):
        x = self.features(x) 
        x = x.view(-1, 512 * 7 * 7)

        orientation = self.orientation(x)
        orientation = orientation.view(-1, self.bins, 2)
        orientation = F.normalize(orientation, dim=2)

        confidence = self.confidence(x)

        dimension = self.dimension(x)

        return orientation, confidence , dimension #, center

def OrientationLoss(orient_batch, orientGT_batch, confGT_batch):

   ## hybrid novel orientation functions by the authors of our basline
    batch_size = orient_batch.size()[0]
    indexes = torch.max(confGT_batch, dim=1)[1]
    orientGT_batch = orientGT_batch[torch.arange(batch_size), indexes]
    orient_batch = orient_batch[torch.arange(batch_size), indexes]
    theta_diff = torch.atan2(orientGT_batch[:,1], orientGT_batch[:,0])
    estimated_theta_diff = torch.atan2(orient_batch[:,1], orient_batch[:,0])
    return (-1 * torch.cos(theta_diff - estimated_theta_diff).mean()) + 1





