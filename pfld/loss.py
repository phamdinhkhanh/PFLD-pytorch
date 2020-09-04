import torch
from torch import nn
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PFLDLoss(nn.Module):
    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks, train_batchsize, using_wingloss=False):
        '''
        attribute_gt: ground truth for head pose frontal, head up, head down, turn left, turn right.
        landmark_gt: ground truth land mark, 196 points.
        euler_angle_gt: euler angle 
        angle: angle
        landmarks: predicted landmark points.
        train_batchsize: train batch size.
        using_wingloss: wing loss function. True/False
        note: euler_angle_gt and angle are going to be used to calculate weight angle.
        '''
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1) # [batch_size]
        # Extract head pose
        attributes_w_n = attribute_gt[:, 1:6].float() #[batch_size, 6]
        # Distribution of head pose
        mat_ratio = torch.mean(attributes_w_n, axis=0) # [6]
        # Weight based on distribution of head pose on batch
        mat_ratio = torch.Tensor([
            1.0 / (x) if x > 0 else train_batchsize for x in mat_ratio
        ]).to(device) # [6]
        weight_attribute = torch.sum(attributes_w_n.mul(mat_ratio), axis=1) # [batch_size]
        # l2_distance between landmark ground truth and landmarks
        if using_wingloss:
            l2_distant = customed_wing_loss(landmark_gt, landmarks)
        else:
            l2_distant = torch.sum((landmark_gt - landmarks) * (landmark_gt - landmarks), axis=1) #[batch_Size]
        # Turn loss weight by angle and normal loss
        return torch.mean(weight_angle * weight_attribute * l2_distant), torch.mean(l2_distant)


def customed_wing_loss(y_true, y_pred, w=10.0, epsilon=2.0):
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    x = y_true - y_pred
    absolute_x = torch.abs(x)

    losses = torch.where(w > absolute_x, w * torch.log(1.0 + absolute_x/epsilon), absolute_x - c)
    losses = torch.sum(losses, axis=1)

    return losses # Mean wingloss for each sample in batch


def smoothL1(y_true, y_pred, beta = 1):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    mae = torch.abs(y_true - y_pred)
    loss = torch.sum(torch.where(mae>beta, mae-0.5*beta , 0.5*mae**2/beta), axis=-1)
    return torch.mean(loss)

def wing_loss(y_true, y_pred, w=10.0, epsilon=2.0, N_LANDMARK = 106):
    y_pred = y_pred.reshape(-1, N_LANDMARK, 2)
    y_true = y_true.reshape(-1, N_LANDMARK, 2) 
    
    x = y_true - y_pred
    c = w * (1.0 - math.log(1.0 + w / epsilon))
    absolute_x = torch.abs(x)
    losses = torch.where(w > absolute_x, w * torch.log(1.0 + absolute_x/epsilon), absolute_x - c)
    loss = torch.mean(torch.sum(losses, axis=[1, 2]), axis=0)
    return loss