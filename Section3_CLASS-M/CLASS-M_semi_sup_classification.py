# The CLASS-M_semi_sup_classification.py code is an example of our CLASS-M model for semi-supervised classification running on TCGA ccRCC dataset
# based on algorithms introduced in https://arxiv.org/abs/2312.06978

# Environment settings: Python 3.7.11, Pytorch 1.9.0, torchvision 0.10.0, and CUDA 10.2. The GPUs we used are NVIDIA TITAN RTX

# The hyper-parameters that can be fine-tuned are above "#######################..." line inside code. 
# The hyper-parameters in the code were already fine-tuned. 

# There are 4 program modes that can be set in "program_mode". 
# "normal_training" allows you to train CLASS-M model from scratch. 
# If the training process is unexpectedly stopped (like server shutdown...), 
# "resume_best_training" mode allows you to resume from best balanced validation epoch. 
# "resume_latest_training" mode allows you to resume from latest saved epoch. 
# We have "save_latest_epoch_frequency" to control model saving frequency.
# To just load already trained model and run validation/test again, choose "only_test" mode. 

# Input:
# INPUT_MATRIX_INFO_TXT: If LOAD_ADAPTIVE_HE_MATRIX is True, before running this code, you need to calculate stain separation matrix for each slide and load it to INPUT_MATRIX_PATH.
# If you don't want to calculate it, set LOAD_ADAPTIVE_HE_MATRIX to False, meanwhile the accuracy would drop a littile bit. 
# list_of_each_GP_txt_path: Each file inside path saves the tile list for each set (labeled/unlabeled train, val, test) and for each category (Normal tissue, Cancer, Necrosis)
# root_dir_original_images: folder that contains related training/validation/test tiles. 

# Output:
# save_models_folder: folder that will save trained models.
# save_results_folder: folder that will save experiment results on training, val, test.


# Other files/folders needed to run this program:
# Folders datasets_my_lib, pytorch_balanced_sampler should be in the same place as this python file.

# In root_dir_original_images, the relative path to each image is like: /Necrosis/IMG0281_TCGA-CJ-4923-01Z-00-DX1.ADD3D7FE-46D1-49AF-B6D4-1FD07B761EF1/
# 20230207_IMG0281_TCGA-CJ-4923-01Z-00-DX1.ADD3D7FE-46D1-49AF-B6D4-1FD07B761EF1_polygon_1_Necrosis_0.png


# In TCGA ccRCC dataset:
# Number of unlabeled training samples: 1,373,684
# Labeled training set: each class's sample number:
# [84578, 180471, 7932]
# Validation set: each class's sample number:
# [19638, 79382, 1301]
# Test set: each class's sample number:
# [15323, 62565, 6168]

# Software Version: 2023_1023_04

import os
import numpy as np
from datetime import datetime
from PIL import Image
from sklearn.metrics import roc_auc_score
from datetime import datetime
import copy
from collections import Counter
os.environ['CUDA_VISIBLE_DEVICES']='0'#set GPU index, at least 14 GB GPU memory is needed for batch size 64
import torchvision.transforms as transforms
import torchvision.models as models
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
from datasets_my_lib.sampler import RandomSampler, BatchSampler
from pytorch_balanced_sampler.sampler import SamplerFactory
from torch.optim.lr_scheduler import _LRScheduler
import warnings
warnings.filterwarnings('ignore')
program_mode='normal_training'#options: 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
initial_lr=0.0001
fixed_initial_lr=initial_lr#initial_lr could change later in 'resume_best_training','resume_latest_training' mode.
lr_strategy='DecayingWithPatience'#options: 'DecayingWithPatience', 'WarmupCosine'
if lr_strategy=='DecayingWithPatience':
    lr_decay_factor=0.9
    lr_patience=15
elif lr_strategy=='WarmupCosine':
    restart_epoch_period=100
total_epoch = 500
MANUAL_SET_num_iter_in_one_epoch=1000#-1 means we don't manual set
batchsize=64
#if batchsize==label_batchsize, we would not use unlabeled data!!!
label_batchsize=33#Set number of labeled samples in one batch!!!Should be a multiple of NUM_CLASSES
loss_contrastive_margin=37.0
loss_contrastive_weight=0.001
unlabel_batchsize=batchsize-label_batchsize
NUM_CLASSES=3
CLASSES_TYPES=['Normal_Type', 'Cancer', 'Necrosis']
USE_PROJECTION_HEAD=False
SAME_CROP_ROTATION_AUGMENTATION_FOR_H_E=False
LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING=True#if True, use root_dir_original_images in GPdataset to load input images's path, root_dir_H, root_dir_E can be 'VOID'
DIRECTLY_LOAD_VAL_IMAGES_TO_RAM=False#speed up validation process, but occupy much more RAM, could cause crash if val has too many images. "htop" to watch Mem growing and prevent Mem full.   
APPLY_MIXUP=True
ENABLE_EMA_MODEL_FOR_VAL_TEST=True#get more stable model
if APPLY_MIXUP:
    # the parameters with "mixmatch" inside means the idea was inspired from both mixup and mixmatch
    # Check https://github.com/YU1ut/MixMatch-pytorch/blob/master/train.py
    import random
    mixmatch_T=0.5
    mixmatch_alpha=0.75
    mixmatch_lambda_u=7.5#weight of unlabeled loss
    mixmatch_manual_rampup_epoch=16#-1
if ENABLE_EMA_MODEL_FOR_VAL_TEST:
    ema_decay=0.999
if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING:
    ALSO_AUGMENT_ORIGINAL_RGB_IMAGES_IN_TRAINING=True
    LOAD_ADAPTIVE_HE_MATRIX=True
    if LOAD_ADAPTIVE_HE_MATRIX:
        dict_image_index_name_to_background_color={}
        dict_image_index_name_to_RGB2HERes={}
        dict_maxH={}
        dict_maxE={}
        TOTAL_NUM_WSIS=420
        ADDITIONAL_NORM_FACTOR=2.0
        PREPROCESS_DATASET=True#!
        if PREPROCESS_DATASET:
            #we may delete some pixels less than 0.15 distance from original point
            preprocess_str='_dataset_preprocessed'
        else:
            preprocess_str=''
        INPUT_MATRIX_PATH='/your_own_path/Section2_get_stain_separation_matrices/'
        INPUT_MATRIX_INFO_TXT=INPUT_MATRIX_PATH+\
            'StepF_output_normalization_factors_based_on_maxH_maxE_H_E_exclude_portion_0.01_0.01_dataset_preprocessed.txt'
    else:
        fixed_background_RGB=[255.0, 255.0, 255.0]
        fixed_maxH=0.5
        fixed_maxE=0.5
        fixed_RGB_absorption_to_H=[1.838, 0.0341, -0.76]
        fixed_RGB_absorption_to_E=[-1.373, 0.772, 1.215]

torch.cuda.empty_cache()
num_workers=4
original_tile_size_divided_by_sqrt2=282#400/sqrt(2)=282
training_tile_size=256
my_modelname = 'CLASS_M'
bs=100#how frequent to show on terminal, show once each bs batches
save_latest_epoch_frequency=10
ADDITIONAL_NORM_FACTOR=2.0
params_model={
        "image1_channels":1,
        "image2_channels":1,
        "classification_number":NUM_CLASSES,
    }
save_models_folder='model_CLASS_M_backup/20231023_NorCanNecExp91_0'
save_results_folder='model_CLASS_M_result'

train_minor_augment_RGB = transforms.Compose([
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)#changing hue will change color a lot. 
])

train_transformerHE = transforms.Compose([
    transforms.RandomRotation(360),###Random rotation to any angle!
    transforms.CenterCrop((original_tile_size_divided_by_sqrt2,original_tile_size_divided_by_sqrt2)),#400/sqrt(2)
    transforms.RandomCrop((training_tile_size,training_tile_size)),
    transforms.RandomHorizontalFlip(),###
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.1738, 0.2507), (0.1876, 0.1546))
    #You can calculate and input your (mean_H,mean_E),(std_H, std_E) in training set(optional)
])

#the following transform is for SAME_CROP_ROTATION_AUGMENTATION_FOR_H_E==False
train_transformer_Hchannel = transforms.Compose([
    transforms.RandomRotation(360),###Random rotation to any angle!
    transforms.CenterCrop((original_tile_size_divided_by_sqrt2,original_tile_size_divided_by_sqrt2)),#400/sqrt(2)
    transforms.RandomCrop((training_tile_size,training_tile_size)),
    transforms.RandomHorizontalFlip(),###
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.1738), (0.1876))
    #You can calculate and input your (mean_H),(std_H) in training set(optional)
])

train_transformer_Echannel = transforms.Compose([
    transforms.RandomRotation(360),###Random rotation to any angle!
    transforms.CenterCrop((original_tile_size_divided_by_sqrt2,original_tile_size_divided_by_sqrt2)),#400/sqrt(2)
    transforms.RandomCrop((training_tile_size,training_tile_size)),
    transforms.RandomHorizontalFlip(),###
    transforms.RandomVerticalFlip(),
    transforms.Normalize((0.2507), (0.1546))
    #You can calculate and input your (mean_E),(std_E) in training set(optional)
])

#The output of color jittering will be set to [0,1], negative value would be set to 0.0. So your input should be [0,1]!
train_color_transformer_1channel=transforms.Compose([
    transforms.ColorJitter(brightness=0.1),
])

val_transformerHE = transforms.Compose([
    transforms.CenterCrop(training_tile_size),
    transforms.Normalize((0.1738, 0.2507), (0.1876, 0.1546))#should be same as training set
])

test_transformerHE = transforms.Compose([
    transforms.CenterCrop(training_tile_size),
    transforms.Normalize((0.1738, 0.2507), (0.1876, 0.1546))#should be same as training set
])

#####################################################################################################################################
if ENABLE_EMA_MODEL_FOR_VAL_TEST:
    class WeightEMA(object):
        def __init__(self, model, ema_model, alpha=0.999):
            self.model = model
            self.ema_model = ema_model
            self.alpha = alpha
            self.params = list(model.state_dict().values())
            self.ema_params = list(ema_model.state_dict().values())
            self.wd = 0.02 * fixed_initial_lr

            for param, ema_param in zip(self.params, self.ema_params):
                param.data.copy_(ema_param.data)

        def step(self):
            one_minus_alpha = 1.0 - self.alpha
            for param, ema_param in zip(self.params, self.ema_params):
                if ema_param.dtype==torch.float32:
                    ema_param.mul_(self.alpha)
                    ema_param.add_(param * one_minus_alpha)
                    # customized weight decay
                    param.mul_(1 - self.wd)

if APPLY_MIXUP:
    def linear_rampup(current, rampup_length=total_epoch):
        if mixmatch_manual_rampup_epoch!=-1:
            rampup_length=mixmatch_manual_rampup_epoch
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current / rampup_length, 0.0, 1.0)
            return float(current)
    class SemiLoss(object):#train_criterion
        def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
            probs_u = torch.softmax(outputs_u, dim=1)

            Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
            Lu = torch.mean((probs_u - targets_u)**2)

            return Lx, Lu, mixmatch_lambda_u * linear_rampup(epoch)

    def interleave_offsets(batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch#if not true, will produce error
        return offsets

    def interleave(xy, batch):
        nu = len(xy) - 1
        offsets = interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]

    mixmatch_train_criterion = SemiLoss()

    #generate the same random numbers each time you run the code
    manualSeed=0
    # Random seed
    if manualSeed is None:
        manualSeed = random.randint(1, 10000)
    np.random.seed(manualSeed)


class WarmupCosineLrScheduler(_LRScheduler):
    '''
    This class is from FixMatch code. (PyTorch version)
    https://github.com/valencebond/FixMatch_pytorch/blob/master/lr_scheduler.py
    It is not finally used in our model.
    '''
    def __init__(
            self,
            optimizer,
            max_iter,
            warmup_iter,
            warmup_ratio=5e-4,
            warmup='exp',
            last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup
        super(WarmupCosineLrScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        if self.last_epoch < self.warmup_iter:
            ratio = self.get_warmup_ratio()
        else:
            real_iter = self.last_epoch - self.warmup_iter
            real_max_iter = self.max_iter - self.warmup_iter
            ratio = np.cos((7 * np.pi * real_iter) / (16 * real_max_iter))
        return ratio

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio
    
if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING:
    if LOAD_ADAPTIVE_HE_MATRIX:
        input_file=open(INPUT_MATRIX_INFO_TXT, 'r')
        txt_data_ = input_file.readlines()
        input_file.close()
        for row in range(1,TOTAL_NUM_WSIS+1):
            txt_input_line_=txt_data_[row]
            txt_input_line_split_=txt_input_line_.split(';')
            txt_input_line_split_[-1]=txt_input_line_split_[-1].replace('\n','')
            image_index_name=txt_input_line_split_[1]+'_'+txt_input_line_split_[3]
            background_R_=float(txt_input_line_split_[4])
            background_G_=float(txt_input_line_split_[5])
            background_B_=float(txt_input_line_split_[6])
            background_RGB_=[background_R_,background_G_,background_B_]
            matrix_RGB2HERes_=[[float(txt_input_line_split_[36]),float(txt_input_line_split_[37]),float(txt_input_line_split_[38])],\
                            [float(txt_input_line_split_[39]),float(txt_input_line_split_[40]),float(txt_input_line_split_[41])],\
                            [float(txt_input_line_split_[42]),float(txt_input_line_split_[43]),float(txt_input_line_split_[44])]]
            maxH_=float(txt_input_line_split_[45])
            maxE_=float(txt_input_line_split_[46])
            dict_image_index_name_to_background_color[image_index_name]=background_RGB_
            dict_image_index_name_to_RGB2HERes[image_index_name]=matrix_RGB2HERes_
            dict_maxH[image_index_name]=maxH_
            dict_maxE[image_index_name]=maxE_

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class CLASS_M_Model(nn.Module):
    def __init__(self,params):
        super(CLASS_M_Model,self).__init__()
        image1_channels=params["image1_channels"]
        image2_channels=params["image2_channels"]
        net_classification_number=params["classification_number"]

        self.model1=models.resnet18(pretrained=True)

        if image1_channels==1:
            self.my_model1_conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            for filter_ind in range(64):
                for row in range(7):
                    for col in range(7):
                        self.my_model1_conv1.weight.data[filter_ind][0][row][col]=self.model1.conv1.weight.data[filter_ind][0][row][col]+\
                                                                        self.model1.conv1.weight.data[filter_ind][1][row][col]+\
                                                                        self.model1.conv1.weight.data[filter_ind][2][row][col]
                        
            self.model1.conv1 = self.my_model1_conv1
        elif image1_channels!=3:
            self.my_model1_conv1=nn.Conv2d(image1_channels, 64, kernel_size=7, stride=2, padding=3)
            self.model1.conv1 = self.my_model1_conv1

        # change the output layer
        num_ftrs = self.model1.fc.in_features
        self.model1.fc = Identity()#do nothing
        print(self.model1)

        self.model2 = models.resnet18(pretrained=True)
        #change input channel
        if image2_channels==1:
            self.my_model2_conv1=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            for filter_ind in range(64):
                for row in range(7):
                    for col in range(7):
                        self.my_model2_conv1.weight.data[filter_ind][0][row][col]=self.model2.conv1.weight.data[filter_ind][0][row][col]+\
                                                                        self.model2.conv1.weight.data[filter_ind][1][row][col]+\
                                                                        self.model2.conv1.weight.data[filter_ind][2][row][col]
                        
            self.model2.conv1 = self.my_model2_conv1
        elif image2_channels==2:#HE images
            self.my_model2_conv1=nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            for filter_ind in range(64):
                for row in range(7):
                    for col in range(7):
                        self.my_model2_conv1.weight.data[filter_ind][0][row][col]=self.model2.conv1.weight.data[filter_ind][1][row][col]+\
                                                                        self.model2.conv1.weight.data[filter_ind][2][row][col]
                        self.my_model2_conv1.weight.data[filter_ind][1][row][col]=self.model2.conv1.weight.data[filter_ind][0][row][col]
            self.model2.conv1 = self.my_model2_conv1
        elif image2_channels!=3:
            self.my_model2_conv1=nn.Conv2d(image2_channels, 64, kernel_size=7, stride=2, padding=3)
            self.model2.conv1 = self.my_model2_conv1
        # change the output layer
        self.model2.fc = Identity()#do nothing

        if USE_PROJECTION_HEAD:
            self.fc=nn.Linear(2*num_ftrs,net_classification_number)
        else:
            self.fc=nn.Linear(num_ftrs,net_classification_number)

        self.myBN_channel1=torch.nn.BatchNorm1d(num_ftrs,affine=False, track_running_stats =True)
        self.myBN_channel2=torch.nn.BatchNorm1d(num_ftrs,affine=False, track_running_stats =True)
        if USE_PROJECTION_HEAD:
            self.channel1_FC_layer1=nn.Linear(num_ftrs,num_ftrs)
            self.channel1_FC_layer2=nn.Linear(num_ftrs,num_ftrs)
            self.channel2_FC_layer1=nn.Linear(num_ftrs,num_ftrs)
            self.channel2_FC_layer2=nn.Linear(num_ftrs,num_ftrs)
            self.reLU= nn.ReLU()
            self.myBN_channel1_layer2=torch.nn.BatchNorm1d(num_ftrs,affine=False, track_running_stats =True)
            self.myBN_channel2_layer2=torch.nn.BatchNorm1d(num_ftrs,affine=False, track_running_stats =True)

    def forward(self,image1,image2):#image1 is a batch
        
        image1_features=self.model1(image1)#(batch_size,512), datatype is tensor
        image2_features=self.model2(image2)
        #in experiment, image1_features and image2_features values are usually positive, maybe because 2d batchnormaliztion has affine=Ture
        #So We want to BN again
        image1_features_BN=self.myBN_channel1(image1_features)
        image2_features_BN=self.myBN_channel2(image2_features)

        if USE_PROJECTION_HEAD:
            concat_features=torch.cat((image1_features_BN,image2_features_BN),1)#(batch_size, 1024)
            image1_features_BN_after_projection=self.channel1_FC_layer1(image1_features_BN)
            #image1_features_BN_after_projection=self.myBN_channel1_layer1(image1_features_BN_after_projection)
            image1_features_BN_after_projection=self.reLU(image1_features_BN_after_projection)
            image1_features_BN_after_projection=self.channel1_FC_layer2(image1_features_BN_after_projection)
            image1_features_BN_after_projection=self.myBN_channel1_layer2(image1_features_BN_after_projection)

            image2_features_BN_after_projection=self.channel2_FC_layer1(image2_features_BN)
            #image2_features_BN_after_projection=self.myBN_channel2_layer1(image2_features_BN_after_projection)
            image2_features_BN_after_projection=self.reLU(image2_features_BN_after_projection)
            image2_features_BN_after_projection=self.channel2_FC_layer2(image2_features_BN_after_projection)
            image2_features_BN_after_projection=self.myBN_channel2_layer2(image2_features_BN_after_projection)

            raw_result=self.fc(concat_features)
            return raw_result, image1_features_BN_after_projection, image2_features_BN_after_projection

        else:
            avg_feature=torch.add(image1_features,image2_features)#element wise addtion
            avg_feature=torch.div(avg_feature, 2)
            raw_result=self.fc(avg_feature)

            return raw_result, image1_features_BN, image2_features_BN

def read_txt_category(txt_category):
    with open(txt_category) as file:
        lines = file.readlines()
    line_data = [line.strip() for line in lines]
    return line_data

#we either use original images or H and E images. 
def get_H_E_np_images_from_image_list(img_list_original,root_dir_original,img_list_H,img_list_E,idx, dataset_type):
    if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING:
        image_original=Image.open(img_list_original[idx][0])#(height, width, channel)
        if ALSO_AUGMENT_ORIGINAL_RGB_IMAGES_IN_TRAINING:
            if dataset_type=="train_only_labeled" or dataset_type=="train_only_unlabeled":
                image_original=train_minor_augment_RGB(image_original)
        image_original=np.asarray(image_original)

        ###image_original = np.transpose(image_original, (2, 0, 1))#change to (channel, height, width)
        image_original = image_original.astype('float32')
        image_ones=np.ones(image_original.shape,dtype=np.float32)
        image_original = np.maximum(image_original,image_ones)#change all pixel intensity =0 to =1

        if LOAD_ADAPTIVE_HE_MATRIX:
            #img_list_original[idx][0] is full path, just use relative path
            one_image_relative_path=img_list_original[idx][0][len(root_dir_original):]
            relative_path_split=one_image_relative_path.split('/')
            current_image_index_name=relative_path_split[1]
            background_RGB=np.array(dict_image_index_name_to_background_color[current_image_index_name])#list
            RGB_absorption_to_H=np.array(dict_image_index_name_to_RGB2HERes[current_image_index_name][0])#numpy array
            RGB_absorption_to_E=np.array(dict_image_index_name_to_RGB2HERes[current_image_index_name][1])
            maxH=np.array(dict_maxH[current_image_index_name])
            maxE=np.array(dict_maxE[current_image_index_name])
        else:
            background_RGB=np.array(fixed_background_RGB)
            RGB_absorption_to_H=np.array(fixed_RGB_absorption_to_H)
            RGB_absorption_to_E=np.array(fixed_RGB_absorption_to_E)
            maxH=np.array(fixed_maxH)
            maxE=np.array(fixed_maxE)
        
        RGB_absorption=np.log10((image_ones/image_original)*background_RGB)
        #If a is an N-D array and b is a 1-D array, it is a sum product over the last axis of a and b.
        image_H=np.dot(RGB_absorption,RGB_absorption_to_H)
        image_H=image_H/(maxH*ADDITIONAL_NORM_FACTOR)
        image_H=np.clip(image_H,0.0,1.0)
        image_H = np.expand_dims(image_H, axis=0)

        image_E=np.dot(RGB_absorption,RGB_absorption_to_E)
        image_E=image_E/(maxE*ADDITIONAL_NORM_FACTOR)
        image_E=np.clip(image_E,0.0,1.0)
        image_E = np.expand_dims(image_E, axis=0)
    else:
        image_H=np.asarray(Image.open(img_list_H[idx][0]))
        image_H=image_H.astype(float)
        image_H=image_H/255.0
        image_H = np.expand_dims(image_H, axis=0)
        image_E=np.asarray(Image.open(img_list_E[idx][0]))
        image_E=image_E.astype(float)
        image_E=image_E/255.0
        image_E = np.expand_dims(image_E, axis=0)

    return image_H, image_E

class GPDataset(Dataset):
    def __init__(self, root_dir_original_images, root_dir_H, root_dir_E, list_of_each_GP_txt_path, txt_unlabeled, dataset_type, transform=None):
        # root_dir_H: folder that saves all H-channel raw tiles
        # root_dir_E: folder that saves all E-channel raw tiles
        # txt_benign: txt file that saves name of labeled tiles
        # txt_unlabeled: txt file that saves name of unlabeled tiles
        #https://discuss.pytorch.org/t/load-the-same-number-of-data-per-class/65198/4
        #https://github.com/khornlund/pytorch-balanced-sampler
        self.root_dir_original = root_dir_original_images
        self.root_dir_H = root_dir_H
        self.root_dir_E = root_dir_E
        self.txt_path = list_of_each_GP_txt_path #[txt_GP_favorable, txt_GP_unfavorable]
        self.classes = CLASSES_TYPES
        self.num_cls = len(self.classes)
        self.img_list_original=[]
        self.img_list_H=[]
        self.img_list_E=[]
        self.dataset_type=dataset_type
        self.transform = transform
        self.class_idxs=[]#for balanced sampler
        self.current_start_idx=0
        self.each_class_number=[]
        self.val_np_H_images=[]
        self.val_np_E_images=[]
        if self.dataset_type=="train_only_labeled" or self.dataset_type=="validation" or self.dataset_type=="test":
            for c in range(self.num_cls):#c is different classes
                if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING:
                    cls_list_original = [[os.path.join(self.root_dir_original,item), c] for item in read_txt_category(self.txt_path[c])]
                    self.img_list_original += cls_list_original

                    self.class_idxs.append(list(range(self.current_start_idx,self.current_start_idx+len(cls_list_original))))
                    self.current_start_idx+=len(cls_list_original)

                    self.each_class_number.append(len(cls_list_original))
                else:
                    cls_list_H = [[os.path.join(self.root_dir_H,item), c] for item in read_txt_category(self.txt_path[c])]
                    self.img_list_H += cls_list_H

                    cls_list_E = [[os.path.join(self.root_dir_E,item), c] for item in read_txt_category(self.txt_path[c])]
                    self.img_list_E += cls_list_E

                    self.class_idxs.append(list(range(self.current_start_idx,self.current_start_idx+len(cls_list_E))))
                    self.current_start_idx+=len(cls_list_E)

                    self.each_class_number.append(len(cls_list_E))
            
            if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING:
                self.len_labeled=len(self.img_list_original)
            else:
                self.len_labeled=len(self.img_list_H)#empty!!!!, use _H

            if DIRECTLY_LOAD_VAL_IMAGES_TO_RAM and self.dataset_type=="validation":
                print('You set DIRECTLY_LOAD_VAL_IMAGES_TO_RAM to be True, start saving validation images to RAM. ')
                print('Speed will increase in validation process, open another terminal and type "htop" to check RAM. ')
                for idx in range(self.len_labeled):
                    image_H, image_E=get_H_E_np_images_from_image_list(self.img_list_original,self.root_dir_original,\
                                                             self.img_list_H,self.img_list_E,idx,self.dataset_type)
                    self.val_np_H_images+=[image_H]
                    self.val_np_E_images+=[image_E]
                    if idx%1000==0:
                        print('Working on saving val images to RAM, process {}/{}'.format(idx, self.len_labeled))

        if self.dataset_type=="train_only_unlabeled":#if use "train", then no unlabeled data in training
            if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING:
                cls_list_original = [[os.path.join(self.root_dir_original,item), -1] for item in read_txt_category(txt_unlabeled)]
                self.img_list_original += cls_list_original
            else:
                cls_list_H = [[os.path.join(self.root_dir_H,item), -1] for item in read_txt_category(txt_unlabeled)]
                self.img_list_H += cls_list_H
                cls_list_E = [[os.path.join(self.root_dir_E,item), -1] for item in read_txt_category(txt_unlabeled)]
                self.img_list_E += cls_list_E
        
    def __len__(self):
        if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING:
            return len(self.img_list_original)
        else:
            return len(self.img_list_H)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if DIRECTLY_LOAD_VAL_IMAGES_TO_RAM and self.dataset_type=="validation":
            image_H=self.val_np_H_images[idx]
            image_E=self.val_np_E_images[idx]
        else:
            image_H, image_E = get_H_E_np_images_from_image_list(self.img_list_original,self.root_dir_original,\
                                                             self.img_list_H,self.img_list_E,idx, self.dataset_type)
            if APPLY_MIXUP and self.dataset_type=="train_only_unlabeled":
                #We need to transform twice.
                image_H_another_aug, image_E_another_aug = get_H_E_np_images_from_image_list(self.img_list_original,self.root_dir_original,\
                                                             self.img_list_H,self.img_list_E,idx, self.dataset_type)
        tensor_of_image_H=torch.from_numpy(image_H.copy()).float()#change numpy array to tensor
        tensor_of_image_E=torch.from_numpy(image_E.copy()).float()
        if APPLY_MIXUP and self.dataset_type=="train_only_unlabeled":
            tensor_of_image_H_another_aug=torch.from_numpy(image_H_another_aug.copy()).float()#change numpy array to tensor
            tensor_of_image_E_another_aug=torch.from_numpy(image_E_another_aug.copy()).float()

        if self.dataset_type=="train_only_labeled" or self.dataset_type=="train_only_unlabeled":
            tensor_of_image_H=train_color_transformer_1channel(tensor_of_image_H)#color transform: color jittering can only apply channel 3(1)
            tensor_of_image_E=train_color_transformer_1channel(tensor_of_image_E)
            if SAME_CROP_ROTATION_AUGMENTATION_FOR_H_E:
                tensor_of_image_HE=torch.cat((tensor_of_image_H, tensor_of_image_E), dim=0)
                tensor_of_image_HE=self.transform(tensor_of_image_HE)#same crop, rotation... for H and E
                tensor_of_image1, tensor_of_image2=torch.split(tensor_of_image_HE, [1,1], dim=0)#dimension will be kept
            else:
                tensor_of_image1=train_transformer_Hchannel(tensor_of_image_H)
                tensor_of_image2=train_transformer_Echannel(tensor_of_image_E)
            
            if APPLY_MIXUP and self.dataset_type=="train_only_unlabeled":
                tensor_of_image_H_another_aug=train_color_transformer_1channel(tensor_of_image_H_another_aug)#color transform: color jittering can only apply channel 3(1)
                tensor_of_image_E_another_aug=train_color_transformer_1channel(tensor_of_image_E_another_aug)
                if SAME_CROP_ROTATION_AUGMENTATION_FOR_H_E:
                    tensor_of_image_HE_another_aug=torch.cat((tensor_of_image_H_another_aug, tensor_of_image_E_another_aug), dim=0)
                    tensor_of_image_HE_another_aug=self.transform(tensor_of_image_HE_another_aug)#same crop, rotation... for H and E
                    tensor_of_image1_another_aug, tensor_of_image2_another_aug=torch.split(tensor_of_image_HE_another_aug, [1,1], dim=0)#dimension will be kept
                else:
                    tensor_of_image1_another_aug=train_transformer_Hchannel(tensor_of_image_H_another_aug)
                    tensor_of_image2_another_aug=train_transformer_Echannel(tensor_of_image_E_another_aug)

        else:
            tensor_of_image_HE=torch.cat((tensor_of_image_H, tensor_of_image_E), dim=0)
            #val/test has dif self.transform with train
            tensor_of_image_HE=self.transform(tensor_of_image_HE)#same crop, rotation... for H and E
            tensor_of_image1, tensor_of_image2=torch.split(tensor_of_image_HE, [1,1], dim=0)#dimension will be kept

        #if sample_label is -1, then it is unlabeled. 
        if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING:
            sample_label=int(self.img_list_original[idx][1])
        else:
            sample_label=int(self.img_list_H[idx][1])

        if sample_label>=0:
            sample = {'img1': tensor_of_image1,
                        'img2': tensor_of_image2,
                        'label': sample_label,
                        'labeled': 1.0}
        else:#unlabeled
            if APPLY_MIXUP:
                sample = {'img1': tensor_of_image1,
                        'img2': tensor_of_image2,
                        'img1_another_aug': tensor_of_image1_another_aug,
                        'img2_another_aug': tensor_of_image2_another_aug,
                        'label': int(0),
                        'labeled': 0.0}
            else:
                sample = {'img1': tensor_of_image1,
                        'img2': tensor_of_image2,
                        'label': int(0),
                        'labeled': 0.0}

        return sample

device = 'cuda'

def train(optimizer, epoch, num_iter_in_one_epoch, train_only_labeled_loader, train_only_unlabeled_loader=None, ema_optimizer=None):
    my_CLASS_M_model.train()
    train_loss = 0
    train_loss_ce_labeled = 0
    train_loss_ce_unlabeled = 0
    train_loss_ce_unlabeled_with_weight = 0
    train_loss_contrastive = 0
    train_loss_contrastive_with_weight = 0
    train_correct = 0
    count_labeled = 0
    avg_pos_dis=torch.tensor(0.0)
    avg_neg_dis=torch.tensor(0.0)
    train_only_labeled_iter = iter(train_only_labeled_loader)
    if batchsize>label_batchsize:
        train_only_unlabeled_iter = iter(train_only_unlabeled_loader)
    batch_index=0
    for iter_index in range(num_iter_in_one_epoch):
        batch_samples_only_labeled = train_only_labeled_iter.next()
        if batchsize>label_batchsize:
            batch_samples_only_unlabeled = train_only_unlabeled_iter.next()
        if batch_index<2:
            if batchsize>label_batchsize:
                print('batch_samples_only_unlabeled size')
                print(batch_samples_only_unlabeled['img1'].shape)
            print('Count occurence of each class in labeled portion of batch:')
            print(torch.bincount(batch_samples_only_labeled['label']))

        if APPLY_MIXUP:
            mixmatch_targets_x = torch.zeros(label_batchsize, NUM_CLASSES).scatter_(1, batch_samples_only_labeled['label'].view(-1,1).long(), 1)
            batch_samples_only_labeled_img1=batch_samples_only_labeled['img1'].to(device)
            batch_samples_only_labeled_img2=batch_samples_only_labeled['img2'].to(device)
            mixmatch_targets_x=mixmatch_targets_x.to(device)
            if batchsize>label_batchsize:
                batch_samples_only_unlabeled_img1=batch_samples_only_unlabeled['img1'].to(device)
                batch_samples_only_unlabeled_img2=batch_samples_only_unlabeled['img2'].to(device)
                batch_samples_only_unlabeled_img1_another_aug=batch_samples_only_unlabeled['img1_another_aug'].to(device)
                batch_samples_only_unlabeled_img2_another_aug=batch_samples_only_unlabeled['img2_another_aug'].to(device)
                with torch.no_grad():
                    # compute guessed labels of unlabel samples
                    mixmatch_outputs_u, _, _ = my_CLASS_M_model(batch_samples_only_unlabeled_img1,batch_samples_only_unlabeled_img2)
                    mixmatch_outputs_u2, _, _ = my_CLASS_M_model(batch_samples_only_unlabeled_img1_another_aug,batch_samples_only_unlabeled_img2_another_aug)
                    mixmatch_p = (torch.softmax(mixmatch_outputs_u, dim=1) + torch.softmax(mixmatch_outputs_u2, dim=1)) / 2
                    mixmatch_pt = mixmatch_p**(1/mixmatch_T)#sharpening by adding exponential
                    mixmatch_targets_u = mixmatch_pt / mixmatch_pt.sum(dim=1, keepdim=True)#softmax
                    mixmatch_targets_u = mixmatch_targets_u.detach()
                # mixup
                mixmatch_all_inputs_channel1 = torch.cat([batch_samples_only_labeled_img1, batch_samples_only_unlabeled_img1, batch_samples_only_unlabeled_img1_another_aug], dim=0)
                mixmatch_all_inputs_channel2 = torch.cat([batch_samples_only_labeled_img2, batch_samples_only_unlabeled_img2, batch_samples_only_unlabeled_img2_another_aug], dim=0)
                mixmatch_all_targets = torch.cat([mixmatch_targets_x, mixmatch_targets_u, mixmatch_targets_u], dim=0)
            else:
                mixmatch_all_inputs_channel1=batch_samples_only_labeled_img1
                mixmatch_all_inputs_channel2=batch_samples_only_labeled_img2
                mixmatch_all_targets=mixmatch_targets_x

            mixmatch_l = np.random.beta(mixmatch_alpha, mixmatch_alpha)

            mixmatch_l = max(mixmatch_l, 1-mixmatch_l)

            mixmatch_idx = torch.randperm(mixmatch_all_inputs_channel1.size(0))

            mixmatch_input_a_channel1, mixmatch_input_b_channel1 = mixmatch_all_inputs_channel1, mixmatch_all_inputs_channel1[mixmatch_idx]
            mixmatch_input_a_channel2, mixmatch_input_b_channel2 = mixmatch_all_inputs_channel2, mixmatch_all_inputs_channel2[mixmatch_idx]
            
            mixmatch_target_a, mixmatch_target_b = mixmatch_all_targets, mixmatch_all_targets[mixmatch_idx]

            mixed_input_channel1_original = mixmatch_l * mixmatch_input_a_channel1 + (1 - mixmatch_l) * mixmatch_input_b_channel1
            mixed_input_channel2_original = mixmatch_l * mixmatch_input_a_channel2 + (1 - mixmatch_l) * mixmatch_input_b_channel2
            mixed_target = mixmatch_l * mixmatch_target_a + (1 - mixmatch_l) * mixmatch_target_b

            if batchsize>label_batchsize:
                # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
                mixed_input_channel1 = list(torch.split(mixed_input_channel1_original, label_batchsize))
                mixed_input_channel2 = list(torch.split(mixed_input_channel2_original, label_batchsize))
                mixed_input_channel1 = interleave(mixed_input_channel1, label_batchsize)
                mixed_input_channel2 = interleave(mixed_input_channel2, label_batchsize)

                mixmatch_first_logits, _, _ = my_CLASS_M_model(mixed_input_channel1[0],mixed_input_channel2[0])
                mixmatch_logits=[mixmatch_first_logits]
                for mixmatch_i in range(1,len(mixed_input_channel1)):
                    mixmatch_another_logit, _, _=my_CLASS_M_model(mixed_input_channel1[mixmatch_i],mixed_input_channel2[mixmatch_i])
                    mixmatch_logits.append(mixmatch_another_logit)

                # put interleaved samples back
                mixmatch_logits = interleave(mixmatch_logits, label_batchsize)
                mixmatch_logits_x = mixmatch_logits[0]
                mixmatch_logits_u = torch.cat(mixmatch_logits[1:], dim=0)
                mixmatch_Lx, mixmatch_Lu, mixmatch_w = mixmatch_train_criterion(mixmatch_logits_x, mixed_target[:label_batchsize], \
                    mixmatch_logits_u, mixed_target[label_batchsize:], epoch+iter_index/num_iter_in_one_epoch)

                if mixmatch_w>0.0:
                    mixmatch_loss = mixmatch_Lx + mixmatch_w * mixmatch_Lu
                else:
                    mixmatch_loss = mixmatch_Lx
            else:
                mixmatch_output, _, _ = my_CLASS_M_model(mixed_input_channel1,mixed_input_channel2)
                mixmatch_Lx = -torch.mean(torch.sum(F.log_softmax(mixmatch_output, dim=1) * mixed_target, dim=1))#from SemiLoss
                mixmatch_Lu = 0.0
                mixmatch_loss = mixmatch_Lx


            shuffle_indexes = torch.randperm(mixed_input_channel1_original.shape[0])
            data1 = mixed_input_channel1_original[shuffle_indexes]
            data2 = mixed_input_channel2_original[shuffle_indexes]
            #just for printing
            train_loss_ce_labeled+=mixmatch_Lx
            train_loss_ce_unlabeled+=mixmatch_Lu
            train_loss_ce_unlabeled_with_weight+=mixmatch_w * mixmatch_Lu
        else:
            if batchsize>label_batchsize:
                data1=torch.cat((batch_samples_only_labeled['img1'], batch_samples_only_unlabeled['img1']), dim=0)
                data2=torch.cat((batch_samples_only_labeled['img2'], batch_samples_only_unlabeled['img2']), dim=0)
                target=torch.cat((batch_samples_only_labeled['label'], batch_samples_only_unlabeled['label']), dim=0)
                labeled=torch.cat((batch_samples_only_labeled['labeled'], batch_samples_only_unlabeled['labeled']), dim=0)
            else:
                data1=batch_samples_only_labeled['img1']
                data2=batch_samples_only_labeled['img2']
                target=batch_samples_only_labeled['label']
                labeled=batch_samples_only_labeled['labeled']
            shuffle_indexes = torch.randperm(data1.shape[0])
            #Shuffle again!!!
            data1 = data1[shuffle_indexes]
            data2 = data2[shuffle_indexes]
            target = target[shuffle_indexes]
            labeled = labeled[shuffle_indexes]
        
        if not APPLY_MIXUP:
            # move data to device
            data1, data2, target, labeled = data1.to(device), data2.to(device), target.to(device), labeled.to(device)

        if loss_contrastive_weight>0.0 or not APPLY_MIXUP:
            output, batch_image1_feature, batch_image2_feature = my_CLASS_M_model(data1, data2)
        else:
            with torch.no_grad():#In this case, no backpropagation. If you don't use no_grad, then CUDA out of memory
                output, batch_image1_feature, batch_image2_feature = my_CLASS_M_model(data1, data2)
        #batch_image feaures were normalized
        loss_internal_12 = (batch_image1_feature - batch_image2_feature).pow(2).sum(1).sqrt()#sum in dim 1
        roll_batch_image2_feature=torch.roll(batch_image2_feature, 1, 0)#roll by 1 step, index 0 goes to 1, dim=0
        loss_external_12=(batch_image1_feature - roll_batch_image2_feature).pow(2).sum(1).sqrt()
        avg_pos_dis=torch.add(avg_pos_dis,torch.sum(loss_internal_12))#just for printing
        avg_neg_dis=torch.add(avg_neg_dis,torch.sum(loss_external_12))
        if loss_contrastive_weight>0.0:
            loss_contrastive=torch.sub(loss_internal_12,loss_external_12)
            loss_contrastive=torch.add(loss_contrastive,loss_contrastive_margin)
            loss_contrastive[loss_contrastive < 0.0] = 0.0
            loss_contrastive=torch.mul(loss_contrastive,loss_contrastive_weight)
            loss_contrastive=torch.sum(loss_contrastive)
            #just for printing
            train_loss_contrastive+=loss_contrastive/loss_contrastive_weight
            train_loss_contrastive_with_weight+=loss_contrastive

        if not APPLY_MIXUP:
            loss_func = nn.CrossEntropyLoss(reduction="none")
            loss_cross_entropy_batch=loss_func(output, target.long())

            #labeled or not
            labeled_boolean_list=[True if x > 0.5 else False for x in labeled]#check if x is labeled or unlabeled
            labeled_boolean_list=torch.tensor(labeled_boolean_list)

            loss_cross_entropy_batch=loss_cross_entropy_batch[labeled_boolean_list]

            loss_cross_entropy=torch.sum(loss_cross_entropy_batch)

            if loss_contrastive_weight>0.0:
                loss=loss_cross_entropy+loss_contrastive#This is total sum loss in whole batch, but will print avg loss in each case 
            else:
                loss=loss_cross_entropy
            #just for printing
            train_loss_ce_labeled+=loss_cross_entropy
        else:
            if loss_contrastive_weight>0.0:
                loss=loss_contrastive+mixmatch_loss
            else:
                loss=mixmatch_loss


        if batch_index==0:
            if APPLY_MIXUP:
                print('Mixmatch loss(unlabeled has pseudo labels but not using cross entropy loss) in first batch=')
                print(mixmatch_loss)
            else:
                print('loss_cross_entropy in first batch=')
                print(loss_cross_entropy)
            print('loss_contrastive in first batch=')
            if loss_contrastive_weight>0.0:
                print(loss_contrastive)
            else:
                print('No contrastive loss!!!')
            print('avg pos dis in first batch=')
            print(torch.div(avg_pos_dis,float(batchsize)))
            print('avg neg dis in first batch=')
            print(torch.div(avg_neg_dis,float(batchsize)))
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()#backpropogate
        optimizer.step()#upgrade weight
        if ENABLE_EMA_MODEL_FOR_VAL_TEST:
            ema_optimizer.step()
        if lr_strategy=='WarmupCosine':
            scheduler.step()
        batch_index+=1

        if not APPLY_MIXUP:
            output_labeled=output[labeled_boolean_list]
            target_labeled=target[labeled_boolean_list]
            count_labeled+=torch.sum(labeled)
            if torch.numel(output_labeled)>0:#check number of elements
                pred = output_labeled.argmax(dim=1, keepdim=True)
                train_correct += pred.eq(target_labeled.long().view_as(pred)).sum().item()
        
        if batch_index % bs == 1:
            print('Train Epoch: {} [{}/{} ({:.4f}%)]\tTrain Loss: {:.6f}'.format(
                epoch, batch_index, num_iter_in_one_epoch,
                100.0 * (batch_index) / num_iter_in_one_epoch, loss.item()/ bs))
    print('Finished training in this epoch. ')
    avg_pos_dis=torch.div(avg_pos_dis, float(num_iter_in_one_epoch))
    avg_neg_dis=torch.div(avg_neg_dis, float(num_iter_in_one_epoch))

    train_loss = train_loss/num_iter_in_one_epoch
    train_loss_ce_labeled = train_loss_ce_labeled/num_iter_in_one_epoch
    train_loss_ce_unlabeled = train_loss_ce_unlabeled/num_iter_in_one_epoch
    train_loss_ce_unlabeled_with_weight = train_loss_ce_unlabeled_with_weight/num_iter_in_one_epoch
    train_loss_contrastive = train_loss_contrastive/num_iter_in_one_epoch
    train_loss_contrastive_with_weight = train_loss_contrastive_with_weight/num_iter_in_one_epoch
    if not APPLY_MIXUP:
        print('Train set: Average loss: {:.4f}, Accuracy in labeled data: {}/{} ({:.4f}%)\n'.format(
            train_loss, train_correct, count_labeled,
            100.0 * train_correct / count_labeled))
        f = open(save_results_folder+'/20231023_NorCanNecExp91_0_train01_{}.txt'.format(my_modelname), 'a+')
        f.write('Train set: Epoch: {} Average loss each batch: {:.4f}, ce_loss: {:.4f}, weighted_contrastive_loss: {:.4f}, contrastive_loss: {:.4f},Accuracy in labeled data: {}/{} ({:.4f}%), avg pos pair dis: {:.4f}, avg neg pair dis: {:.4f}\n'.format(epoch, 
            train_loss, train_loss_ce_labeled,train_loss_contrastive_with_weight, train_loss_contrastive, train_correct, count_labeled,
            100.0 * float(train_correct) / float(count_labeled), torch.div(avg_pos_dis,float(batchsize)), torch.div(avg_neg_dis,float(batchsize))))
        f.close()
    else:
        f = open(save_results_folder+'/20231023_NorCanNecExp91_0_train01_{}.txt'.format(my_modelname), 'a+')
        f.write('Train set: Epoch: {} Average loss each batch(unlabeled weight growing): {:.4f}, ce_loss(Mixup): {:.4f}, weighted_unlabeled_pred_loss(weight growing): {:.4f}, weighted_contrastive_loss: {:.4f}, unlabeled_pred_loss: {:.4f}, contrastive_loss: {:.4f}, avg pos pair dis: {:.4f}, avg neg pair dis: {:.4f}\n'.format(epoch, 
            train_loss, train_loss_ce_labeled,train_loss_ce_unlabeled_with_weight,train_loss_contrastive_with_weight, train_loss_ce_unlabeled, train_loss_contrastive, 
            torch.div(avg_pos_dis,float(batchsize)), torch.div(avg_neg_dis,float(batchsize))))
        f.close()


#val process is defined here

def val(my_val_model):
    
    my_val_model.eval()
    val_loss = 0.0
    correct = 0
    results = []
    
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    if program_mode =='only_test':
        confusion_matrix = [ [0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    # Don't update model
    with torch.no_grad():
        pred_list=[]
        score_list=[]
        target_list=[]
        # Predict
        for batch_index, batch_samples in enumerate(val_loader):
            data1, data2, target, labeled = batch_samples['img1'].to(device), batch_samples['img2'].to(device), \
                batch_samples['label'].to(device), batch_samples['labeled'].to(device)
            output, batch_image1_feature, batch_image2_feature = my_val_model(data1,data2)
            val_loss += loss_func(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)#Returns the indices of the maximum values along an axis.
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            target_np=target.long().cpu().numpy()
            pred_list= np.concatenate((pred_list, pred.cpu().numpy()), axis=None)
            score_list= np.concatenate((score_list, score.cpu().numpy()[:,1]), axis=None)
            target_list= np.concatenate((target_list, target_np), axis=None)
            raw_unbalanced_val_acc=100.0 * correct / len(val_loader.dataset)
        
            if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
                path_val_scores=save_results_folder+'/20231023_NorCanNecExp91_0_val_scores.txt'
                f = open(path_val_scores, 'a+')
                for row in range(score.size(dim=0)):
                    confusion_matrix[pred[row][0]][target[row]]+=1
                if batch_index%10==0:
                    print(f'working on batch index {batch_index}, total batch {valset.__len__()/batchsize}\n')
                f.close()
                if batch_index==0:
                    print("Scores, prediction, target of each sample are saved in "+path_val_scores)

        print('correct number in val is {}/{}. (we need to calculate balanced accuracy)'.format(correct, len(val_loader.dataset)))
        val_loss/=float(len(val_loader.dataset))

        if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
            path_val_scores=save_results_folder+'/20231023_NorCanNecExp91_0_val_scores.txt'
            f = open(path_val_scores, 'a+')
            f.write('confusion matrix: each row counts prediction(row index is what it is predicted to), each column counts ground truth\n')
            sum_of_a_column=[0]*NUM_CLASSES
            for row in range(NUM_CLASSES):
                for col in range(NUM_CLASSES):
                    f.write(f'{confusion_matrix[row][col]}   ')
                    sum_of_a_column[col]+=confusion_matrix[row][col]
                f.write('\n')
            f.write('\n')
            for row in range(NUM_CLASSES):
                for col in range(NUM_CLASSES):
                    f.write('{:.4f}   '.format(confusion_matrix[row][col]/sum_of_a_column[col]))
                f.write('\n')
            
            f.close()
           
    return target_list, score_list, pred_list, raw_unbalanced_val_acc, val_loss
    
#test process is defined here 
def test(my_test_model,epoch):
    
    my_test_model.eval()
    test_loss = 0.0
    correct = 0
    
    loss_func = nn.CrossEntropyLoss(reduction="sum")
    confusion_matrix = [ [0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    # Don't update model
    with torch.no_grad():
        pred_list=[]
        score_list=[]
        target_list=[]
        # Predict
        for batch_index, batch_samples in enumerate(test_loader):
            data1, data2, target, labeled = batch_samples['img1'].to(device), batch_samples['img2'].to(device), \
                batch_samples['label'].to(device), batch_samples['labeled'].to(device)
            output, batch_image1_feature, batch_image2_feature = my_test_model(data1,data2)
            
            test_loss += loss_func(output, target.long())
            score = F.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.long().view_as(pred)).sum().item()
            target_np=target.long().cpu().numpy()
            pred_list= np.concatenate((pred_list, pred.cpu().numpy()), axis=None)
            score_list= np.concatenate((score_list, score.cpu().numpy()[:,1]), axis=None)
            target_list= np.concatenate((target_list, target_np), axis=None)
            raw_unbalanced_test_acc=100.0 * correct / len(test_loader.dataset)

            if program_mode =='only_test' or epoch>=10: # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
                path_test_scores=save_results_folder+'/20231023_NorCanNecExp91_0_test_scores.txt'
                f = open(path_test_scores, 'a+')
                for row in range(score.size(dim=0)):
                    confusion_matrix[pred[row][0]][target[row]]+=1
                f.close()

        print('correct number in test is {}/{}. (we need to calculate balanced accuracy)'.format(correct, len(test_loader.dataset)))
        test_loss/=float(len(test_loader.dataset))

        if program_mode =='only_test' or epoch>=10: # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
            path_test_scores=save_results_folder+'/20231023_NorCanNecExp91_0_test_scores.txt'
            f = open(path_test_scores, 'a+')
            f.write(f'Epoch: {epoch}, confusion matrix: each row counts prediction(row index is what it is predicted to), each column counts ground truth\n')
            sum_of_a_column=[0]*NUM_CLASSES
            for row in range(NUM_CLASSES):
                for col in range(NUM_CLASSES):
                    f.write(f'{confusion_matrix[row][col]}   ')
                    sum_of_a_column[col]+=confusion_matrix[row][col]
                f.write('\n')
            f.write('\n')
            for row in range(NUM_CLASSES):
                for col in range(NUM_CLASSES):
                    f.write('{:.4f}   '.format(confusion_matrix[row][col]/sum_of_a_column[col]))
                f.write('\n')
            f.write('\n\n')
            f.close()

    return target_list, score_list, pred_list, raw_unbalanced_test_acc, test_loss

if __name__ == '__main__':
    txt_root_dir='dataset_txt_files_20230708_normal_vs_cancer_vs_necrosis_20X/'
    #if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING==True, it is OK to set root_dir_H, root_dir_E to any string, like "VOID"
    #if LOAD_ORIGINAL_IMAGES_AND_TRANSFORM_TO_HE_IN_PREPROCESSING==False, it is OK to set root_dir_original_images to any string, like "VOID"
    trainset_only_labeled = GPDataset(
                            root_dir_original_images='/your_own_path/Section1_get_tiles_from_WSIs/tiles_20230301_1_20X/',
                            root_dir_H='VOID', 
                            root_dir_E='VOID', 
                            list_of_each_GP_txt_path=[
                                txt_root_dir+'20230708_train_labeled_0_Normal_Type_tiles_list.txt',
                                txt_root_dir+'20230708_train_labeled_1_Cancer_tiles_list.txt',
                                txt_root_dir+'20230708_train_labeled_2_Necrosis_tiles_list.txt'],
                            txt_unlabeled=txt_root_dir+'unlabeled/'+'None.txt',
                            dataset_type='train_only_labeled',#"train"(only use labeled data)  
                            transform= train_transformerHE)#train_transformerHE is not used if SAME_CROP_ROTATION_AUGMENTATION_FOR_H_E is False
    if batchsize>label_batchsize:
        trainset_only_unlabeled = GPDataset(
                                root_dir_original_images='/your_own_path/Section1_get_tiles_from_WSIs/tiles_20230301_1_20X/',
                                root_dir_H='VOID', 
                                root_dir_E='VOID',
                                list_of_each_GP_txt_path=['']*NUM_CLASSES,
                                txt_unlabeled=txt_root_dir+'20230727_train_unlabeled_tiles_list_sample_step_400.txt',
                                dataset_type='train_only_unlabeled',#"train"(only use labeled data)  
                                transform= train_transformerHE)
    valset = GPDataset(
                            root_dir_original_images='/your_own_path/Section1_get_tiles_from_WSIs/tiles_20230301_1_20X/',
                            root_dir_H='VOID', 
                            root_dir_E='VOID', 
                            list_of_each_GP_txt_path=[
                                txt_root_dir+'20230708_val_0_Normal_Type_tiles_list.txt',
                                txt_root_dir+'20230708_val_1_Cancer_tiles_list.txt',
                                txt_root_dir+'20230708_val_2_Necrosis_tiles_list.txt'],
                            txt_unlabeled=txt_root_dir+'VOID',

                            dataset_type='validation',
                            transform = val_transformerHE)
    testset = GPDataset(
                            root_dir_original_images='/your_own_path/Section1_get_tiles_from_WSIs/tiles_20230301_1_20X/',
                            root_dir_H='VOID', 
                            root_dir_E='VOID', 
                            list_of_each_GP_txt_path=[
                                txt_root_dir+'20230708_test_0_Normal_Type_tiles_list.txt',
                                txt_root_dir+'20230708_test_1_Cancer_tiles_list.txt',
                                txt_root_dir+'20230708_test_2_Necrosis_tiles_list.txt'],
                            txt_unlabeled=txt_root_dir+'VOID',
                            
                            dataset_type='test',
                            transform = val_transformerHE)
    
    # Check whether the specified path exists or not
    save_models_path_Exist = os.path.exists('./'+save_models_folder)
    if not save_models_path_Exist:
        os.makedirs('./'+save_models_folder)
        print("The new directory: "+save_models_folder+ " for saving models is created!")
    save_results_path_Exist = os.path.exists('./'+save_results_folder)
    if not save_results_path_Exist:
        os.makedirs('./'+save_results_folder)
        print("The new directory: "+save_models_folder+ " for saving results is created!")
    if batchsize>label_batchsize:
        num_iter_in_one_epoch=max(int(trainset_only_labeled.__len__()/label_batchsize),int(trainset_only_unlabeled.__len__()/unlabel_batchsize))
    else:
        num_iter_in_one_epoch=int(trainset_only_labeled.__len__()/label_batchsize)
    if MANUAL_SET_num_iter_in_one_epoch>0:
        num_iter_in_one_epoch=MANUAL_SET_num_iter_in_one_epoch
    
    #balanced sampler is from https://github.com/khornlund/pytorch-balanced-sampler
    batch_sampler_labeled = SamplerFactory().get(
        class_idxs=trainset_only_labeled.class_idxs,
        batch_size=label_batchsize,
        n_batches=num_iter_in_one_epoch,#how many batches in one epoch
        alpha=1.0,#totally balanced
        kind='fixed'#fixed number of each label in one batch
        )
    if batchsize>label_batchsize:
        sampler_unlabeled = RandomSampler(trainset_only_unlabeled, replacement=True, num_samples=num_iter_in_one_epoch * unlabel_batchsize)
        batch_sampler_unlabeled = BatchSampler(sampler_unlabeled, unlabel_batchsize, drop_last=True)
        train_only_unlabeled_loader = DataLoader(trainset_only_unlabeled, batch_sampler=batch_sampler_unlabeled, num_workers=num_workers, pin_memory=True)
        
    train_only_labeled_loader = DataLoader(trainset_only_labeled, batch_sampler=batch_sampler_labeled, num_workers=num_workers, pin_memory=True)
    
    
    val_loader = DataLoader(valset, batch_size=batchsize, num_workers=num_workers, drop_last=False, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batchsize, num_workers=num_workers, drop_last=False, shuffle=False)

    my_CLASS_M_model = CLASS_M_Model(params_model)
    device = torch.device("cuda:0")
    my_CLASS_M_model.to(device)
    if ENABLE_EMA_MODEL_FOR_VAL_TEST:
        ema_my_CLASS_M_model = CLASS_M_Model(params_model)
        ema_my_CLASS_M_model.to(device)
        for ema_param in ema_my_CLASS_M_model.parameters():
            ema_param.detach_()


    #It enables benchmark mode in cudnn.
    #benchmark mode is good whenever your input sizes for your network do not vary. 
    #This way, cudnn will look for the optimal set of algorithms for that particular 
    # configuration (which takes some time). This usually leads to faster runtime.
    #But if your input sizes changes at each iteration, then cudnn will benchmark 
    # every time a new size appears, possibly leading to worse runtime performances.
    cudnn.benchmark = True

    if batchsize>label_batchsize:
        print('length of training set labeled, unlabeled, val set, test set')
    else:
        print('Fully supervised learning, length of training set labeled, val set, test set')
    print(trainset_only_labeled.__len__())
    if batchsize>label_batchsize:
        print(trainset_only_unlabeled.__len__())
    print(valset.__len__())
    print(testset.__len__())

    print('Train labeled each class sample number:')
    print(trainset_only_labeled.each_class_number)

    print('Validation each class sample number:')
    print(valset.each_class_number)

    print('Test each class sample number:')
    print(testset.each_class_number)
    get_best_model=0
    best_epoch_balanced_test_acc=0.0
    # train
    if program_mode !='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
        start_epoch=1
        best_acc=0.0
        best_epoch=1
        if program_mode=='resume_best_training':# 'normal_training', 'resume_best_training', 'resume_latest_training', 'only_test'
            #choose current_best for path2weights and current_result_path
            path2weights=save_models_folder+'/20231023_NorCanNecExp91_0_current_best.pt'
            my_CLASS_M_model.load_state_dict(torch.load(path2weights))
            if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                path2weights_ema=save_models_folder+'/20231023_NorCanNecExp91_0_current_best_ema.pt'
                ema_my_CLASS_M_model.load_state_dict(torch.load(path2weights_ema))
            current_result_path=save_results_folder+'/20231023_NorCanNecExp91_0_current_best_for_resuming.txt'
        elif program_mode=='resume_latest_training':# 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
            #choose current_latest for path2weights and current_result_path
            path2weights=save_models_folder+'/20231023_NorCanNecExp91_0_current_latest.pt'
            my_CLASS_M_model.load_state_dict(torch.load(path2weights))
            if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                path2weights_ema=save_models_folder+'/20231023_NorCanNecExp91_0_current_latest_ema.pt'
                ema_my_CLASS_M_model.load_state_dict(torch.load(path2weights_ema))
            current_result_path=save_results_folder+'/20231023_NorCanNecExp91_0_current_latest_for_resuming.txt'
        if program_mode=='resume_best_training' or program_mode=='resume_latest_training':
            current_result_file = open(current_result_path, 'r')
            current_epoch=current_result_file.readline()
            start_epoch=int(current_epoch)+1
            balanced_val_acc=float(current_result_file.readline())
            balanced_test_acc=float(current_result_file.readline())
            initial_lr=float(current_result_file.readline())
            best_epoch=int(current_result_file.readline())
            best_acc=float(current_result_file.readline())
            best_epoch_balanced_test_acc=float(current_result_file.readline())
            current_result_file.close()
        
        acc_list = []
        vote_pred = np.zeros(valset.__len__())
        vote_score = np.zeros(valset.__len__())

        if lr_strategy=='DecayingWithPatience':
            optimizer = optim.RMSprop(my_CLASS_M_model.parameters(), lr=initial_lr, weight_decay=1e-8, momentum=0.9)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=lr_decay_factor, patience=lr_patience)  # goal: minimize loss
            if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                ema_optimizer = WeightEMA(my_CLASS_M_model, ema_my_CLASS_M_model, alpha=ema_decay)
        else:
            #'WarmupCosine'
            optimizer = torch.optim.SGD(my_CLASS_M_model.parameters(), lr=fixed_initial_lr, weight_decay=5e-4,
                                momentum=0.9, nesterov=True)
            scheduler = WarmupCosineLrScheduler(
                optimizer, max_iter=restart_epoch_period*num_iter_in_one_epoch, warmup_iter=0
            )#for WarmupCosine, scheduler is updated in each iter.
            if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                ema_optimizer = WeightEMA(my_CLASS_M_model, ema_my_CLASS_M_model, alpha=ema_decay)
            if program_mode=='resume_best_training' or program_mode=='resume_latest_training':
                #change lr from fixed_initial_lr to initial_lr
                num_iters_to_resume=(start_epoch%restart_epoch_period)*num_iter_in_one_epoch
                for extra_iter in range(num_iters_to_resume):
                    scheduler.step()

        for epoch in range(start_epoch, total_epoch+1):
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))
            current_lr=optimizer.param_groups[0]['lr']
            if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                if batchsize>label_batchsize:
                    train(optimizer, epoch, num_iter_in_one_epoch, train_only_labeled_loader, train_only_unlabeled_loader, ema_optimizer)
                else:
                    train(optimizer, epoch, num_iter_in_one_epoch, train_only_labeled_loader, None, ema_optimizer)
            else:
                if batchsize>label_batchsize:
                    train(optimizer, epoch, num_iter_in_one_epoch, train_only_labeled_loader, train_only_unlabeled_loader)
                else:
                    train(optimizer, epoch, num_iter_in_one_epoch, train_only_labeled_loader)
            if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                #Avoid starting from random weights
                if epoch==start_epoch and program_mode=='normal_training':
                    ema_my_CLASS_M_model=copy.deepcopy(my_CLASS_M_model)
                    ema_optimizer = WeightEMA(my_CLASS_M_model, ema_my_CLASS_M_model, alpha=ema_decay)
            if lr_strategy=='WarmupCosine':
                if epoch%restart_epoch_period==0:
                    scheduler = WarmupCosineLrScheduler(
                        optimizer, max_iter=restart_epoch_period*num_iter_in_one_epoch, warmup_iter=0)

            if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                val_target_list, val_score_list, val_pred_list, raw_unbalanced_val_acc, val_loss = val(ema_my_CLASS_M_model)
            else:
                val_target_list, val_score_list, val_pred_list, raw_unbalanced_val_acc, val_loss = val(my_CLASS_M_model)
            if lr_strategy=='DecayingWithPatience':
                scheduler.step(val_loss)
            TP=[0]*NUM_CLASSES
            TN=[0]*NUM_CLASSES
            FN=[0]*NUM_CLASSES
            FP=[0]*NUM_CLASSES
            p=[0.0]*NUM_CLASSES
            r=[0.0]*NUM_CLASSES
            F1=[0.0]*NUM_CLASSES
            TPR=[0.0]*NUM_CLASSES
            TNR=[0.0]*NUM_CLASSES
            balanced_val_acc=0.0
            for class_index in range(NUM_CLASSES):
                TP[class_index]=((val_pred_list == class_index) & (val_target_list == class_index)).sum()
                TN[class_index] = ((val_pred_list != class_index) & (val_target_list != class_index)).sum()
                FN[class_index] = ((val_pred_list != class_index) & (val_target_list == class_index)).sum()
                FP[class_index] = ((val_pred_list == class_index) & (val_target_list != class_index)).sum()
                if TP[class_index]+FP[class_index]==0:
                    p[class_index]=1.0
                else:
                    p[class_index]=float(TP[class_index])/float(TP[class_index]+FP[class_index])
                if TP[class_index]+FN[class_index]==0:
                    r[class_index]=1.0
                else:
                    r[class_index]=float(TP[class_index]) / float(TP[class_index] + FN[class_index])
                if r[class_index]+p[class_index]==0.0:
                    F1[class_index]=0.0
                else:
                    F1[class_index] = 2 * r[class_index] * p[class_index] / (r[class_index] + p[class_index])
                TPR[class_index]=r[class_index]
                TNR[class_index]=float(TN[class_index])/float(TN[class_index]+FP[class_index])
                balanced_val_acc+=float(TP[class_index])/float(TP[class_index]+FN[class_index])#It is avg r
            balanced_val_acc/=NUM_CLASSES
            
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print('The epoch is {}, balanced val accuracy: {:.4f}, average loss(each sample): {}, previous best balanced val {} with balanced test{} at epoch{}\n'.format(epoch, balanced_val_acc, val_loss, best_acc, best_epoch_balanced_test_acc, best_epoch))
            f = open(f'{save_results_folder}/20231023_NorCanNecExp91_0_val01_{my_modelname}.txt', 'a+')
            f.write('Val: The epoch is {}, current_lr= {}, balanced accuracy: {:.4f}, average loss(each sample): {},TP= [{}], TN= [{}], FN= [{}], FP= [{}], precision= [{}], F1= [{}], TPR= [{}], TNR= [{}], Current Time ={}\n\n'.format(epoch, current_lr, balanced_val_acc,
                val_loss,", ".join(str(item) for item in TP),", ".join(str(item) for item in TN),", ".join(str(item) for item in FN),", ".join(str(item) for item in FP),
                ", ".join(str(item) for item in p),", ".join(str(item) for item in F1),", ".join(str(item) for item in TPR),", ".join(str(item) for item in TNR),current_time))
            f.close()

            if best_acc<balanced_val_acc:
                print('Currently saving models and results for ***current_best.pt, dont terminate program now!!!')
                best_acc=balanced_val_acc
                best_epoch=epoch
                my_best_model=copy.deepcopy(my_CLASS_M_model)
                get_best_model=1
                torch.save(my_CLASS_M_model.state_dict(), save_models_folder+"/20231023_NorCanNecExp91_0_current_best.pt")
                if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                    my_best_model_ema=copy.deepcopy(ema_my_CLASS_M_model)
                    torch.save(ema_my_CLASS_M_model.state_dict(), save_models_folder+"/20231023_NorCanNecExp91_0_current_best_ema.pt")
                if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                    test_target_list, test_score_list, test_pred_list, raw_unbalanced_test_acc, test_loss = test(ema_my_CLASS_M_model,epoch)
                else:
                    test_target_list, test_score_list, test_pred_list, raw_unbalanced_test_acc, test_loss = test(my_CLASS_M_model,epoch)
                
                TP=[0]*NUM_CLASSES
                TN=[0]*NUM_CLASSES
                FN=[0]*NUM_CLASSES
                FP=[0]*NUM_CLASSES
                p=[0.0]*NUM_CLASSES
                r=[0.0]*NUM_CLASSES
                F1=[0.0]*NUM_CLASSES
                TPR=[0.0]*NUM_CLASSES
                TNR=[0.0]*NUM_CLASSES
                balanced_test_acc=0.0
                for class_index in range(NUM_CLASSES):
                    TP[class_index]=((test_pred_list == class_index) & (test_target_list == class_index)).sum()
                    TN[class_index] = ((test_pred_list != class_index) & (test_target_list != class_index)).sum()
                    FN[class_index] = ((test_pred_list != class_index) & (test_target_list == class_index)).sum()
                    FP[class_index] = ((test_pred_list == class_index) & (test_target_list != class_index)).sum()
                    if TP[class_index]+FP[class_index]==0:
                        p[class_index]=1.0
                    else:
                        p[class_index]=float(TP[class_index])/float(TP[class_index]+FP[class_index])
                    if TP[class_index]+FN[class_index]==0:
                        r[class_index]=1.0
                    else:
                        r[class_index]=float(TP[class_index]) / float(TP[class_index] + FN[class_index])
                    if r[class_index]+p[class_index]==0.0:
                        F1[class_index]=0.0
                    else:
                        F1[class_index] = 2 * r[class_index] * p[class_index] / (r[class_index] + p[class_index])
                    TPR[class_index]=r[class_index]
                    TNR[class_index]=float(TN[class_index])/float(TN[class_index]+FP[class_index])
                    balanced_test_acc+=float(TP[class_index])/float(TP[class_index]+FN[class_index])#It is avg r
                balanced_test_acc/=NUM_CLASSES
                print('balanced test acc',balanced_test_acc)

                best_epoch_balanced_test_acc=balanced_test_acc
                current_result_path=save_results_folder+'/20231023_NorCanNecExp91_0_current_best_for_resuming.txt'
                current_result_file = open(current_result_path, 'w')#overwrite
                current_result_file.write(f'{epoch}\n')
                current_result_file.write(f'{balanced_val_acc}\n')
                current_result_file.write(f'{balanced_test_acc}\n')
                current_result_file.write(f'{current_lr}\n')
                current_result_file.write(f'{best_epoch}\n')
                current_result_file.write(f'{best_acc}\n')
                current_result_file.write(f'{best_epoch_balanced_test_acc}\n')
                current_result_file.close()

                now = datetime.now()
                current_time = now.strftime("%H:%M:%S")
                f = open(f'{save_results_folder}/20231023_NorCanNecExp91_0_test01_{my_modelname}.txt', 'a+')
                f.write('Test: The epoch is {}, balanced val accuracy: {:.4f}, balanced test accuracy: {:.4f}, average loss(each sample): {},TP= [{}], TN= [{}], FN= [{}], FP= [{}], precision= [{}], F1= [{}], TPR= [{}], TNR= [{}], Current Time ={}\n\n'.format(epoch, balanced_val_acc, balanced_test_acc,
                    test_loss,", ".join(str(item) for item in TP),", ".join(str(item) for item in TN),", ".join(str(item) for item in FN),", ".join(str(item) for item in FP),
                    ", ".join(str(item) for item in p),", ".join(str(item) for item in F1),", ".join(str(item) for item in TPR),", ".join(str(item) for item in TNR),current_time))
                f.close()
                print('Finished saving models and results for ***current_best.pt')
                
            if epoch%save_latest_epoch_frequency==0:
                print('Currently saving models and results for ***current_lstest.pt, dont terminate program now!!!')
                torch.save(my_CLASS_M_model.state_dict(), save_models_folder+"/20231023_NorCanNecExp91_0_current_latest.pt")
                if ENABLE_EMA_MODEL_FOR_VAL_TEST:
                    torch.save(ema_my_CLASS_M_model.state_dict(), save_models_folder+"/20231023_NorCanNecExp91_0_current_latest_ema.pt")
                current_result_path=save_results_folder+'/20231023_NorCanNecExp91_0_current_latest_for_resuming.txt'
                current_result_file = open(current_result_path, 'w')#overwrite
                current_result_file.write(f'{epoch}\n')
                current_result_file.write(f'{balanced_val_acc}\n')
                current_result_file.write(f'{balanced_test_acc}\n')#This test_acc may not be this epoch's test_acc
                current_result_file.write(f'{current_lr}\n')
                current_result_file.write(f'{best_epoch}\n')
                current_result_file.write(f'{best_acc}\n')
                current_result_file.write(f'{best_epoch_balanced_test_acc}\n')
                current_result_file.close()
                print('Finished saving models and results for ***current_latest.pt')

        f = open(f'{save_results_folder}/20231023_NorCanNecExp91_0_val01_{my_modelname}.txt', 'a+')
        f.write('best epoch: {} best balanced validation accuracy:{} balanced test acc at best val: {}'.format(best_epoch,best_acc,best_epoch_balanced_test_acc))
        f.close()
            
    if program_mode =='only_test': # 'normal_training', 'resume_best_training','resume_latest_training', 'only_test'
        path2test_weights=save_models_folder+"/20231023_NorCanNecExp91_0_current_best.pt"
        my_CLASS_M_model.load_state_dict(torch.load(path2test_weights))
        device = torch.device("cuda:0")
        my_CLASS_M_model.to(device) 
        if ENABLE_EMA_MODEL_FOR_VAL_TEST:
            path2test_weights_ema=save_models_folder+"/20231023_NorCanNecExp91_0_current_best_ema.pt"
            ema_my_CLASS_M_model.load_state_dict(torch.load(path2test_weights_ema))
            device = torch.device("cuda:0")
            ema_my_CLASS_M_model.to(device) 
        if ENABLE_EMA_MODEL_FOR_VAL_TEST:
            val_target_list, val_score_list, val_pred_list, raw_unbalanced_val_acc, val_loss = val(ema_my_CLASS_M_model)
        else:
            val_target_list, val_score_list, val_pred_list, raw_unbalanced_val_acc, val_loss = val(my_CLASS_M_model)
        TP=[0]*NUM_CLASSES
        TN=[0]*NUM_CLASSES
        FN=[0]*NUM_CLASSES
        FP=[0]*NUM_CLASSES
        p=[0.0]*NUM_CLASSES
        r=[0.0]*NUM_CLASSES
        F1=[0.0]*NUM_CLASSES
        TPR=[0.0]*NUM_CLASSES
        TNR=[0.0]*NUM_CLASSES
        balanced_val_acc=0.0
        for class_index in range(NUM_CLASSES):
            TP[class_index]=((val_pred_list == class_index) & (val_target_list == class_index)).sum()
            TN[class_index] = ((val_pred_list != class_index) & (val_target_list != class_index)).sum()
            FN[class_index] = ((val_pred_list != class_index) & (val_target_list == class_index)).sum()
            FP[class_index] = ((val_pred_list == class_index) & (val_target_list != class_index)).sum()
            if TP[class_index]+FP[class_index]==0:
                p[class_index]=1.0
            else:
                p[class_index]=float(TP[class_index])/float(TP[class_index]+FP[class_index])
            if TP[class_index]+FN[class_index]==0:
                r[class_index]=1.0
            else:
                r[class_index]=float(TP[class_index]) / float(TP[class_index] + FN[class_index])
            if r[class_index]+p[class_index]==0.0:
                F1[class_index]=0.0
            else:
                F1[class_index] = 2 * r[class_index] * p[class_index] / (r[class_index] + p[class_index])
            TPR[class_index]=r[class_index]
            TNR[class_index]=float(TN[class_index])/float(TN[class_index]+FP[class_index])
            balanced_val_acc+=float(TP[class_index])/float(TP[class_index]+FN[class_index])#It is avg r
        balanced_val_acc/=NUM_CLASSES
        print('On validation: ')
        print('TP=')
        print(TP)
        print('TN=')
        print(TN)
        print('FN=')
        print(FN)
        print('FP=')
        print(FP)
        print('precision')
        print(p)
        print('recall')
        print(r)
        print('F1')
        print(F1)
        print('balanced_val_acc')
        print(balanced_val_acc)
        print('\n')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f = open(f'{save_results_folder}/20231023_NorCanNecExp91_0_val01_load_model_{my_modelname}.txt', 'a+')
        f.write('Val: balanced accuracy: {:.4f}, average loss(each sample): {},TP= [{}], TN= [{}], FN= [{}], FP= [{}], p= [{}], r= [{}], F1= [{}], TPR= [{}], TNR= [{}], \
            Current Time ={}\n\n'.format(balanced_val_acc,
            val_loss,", ".join(str(item) for item in TP),", ".join(str(item) for item in TN),", ".join(str(item) for item in FN),", ".join(str(item) for item in FP),
            ", ".join(str(item) for item in p),", ".join(str(item) for item in r),", ".join(str(item) for item in F1),", ".join(str(item) for item in TPR),", ".join(str(item) for item in TNR),current_time))
        f.close()

        if ENABLE_EMA_MODEL_FOR_VAL_TEST:
            test_target_list, test_score_list, test_pred_list, raw_unbalanced_test_acc, test_loss = test(ema_my_CLASS_M_model,-1)
        else:
            test_target_list, test_score_list, test_pred_list, raw_unbalanced_test_acc, test_loss = test(my_CLASS_M_model,-1)

        TP=[0]*NUM_CLASSES
        TN=[0]*NUM_CLASSES
        FN=[0]*NUM_CLASSES
        FP=[0]*NUM_CLASSES
        p=[0.0]*NUM_CLASSES
        r=[0.0]*NUM_CLASSES
        F1=[0.0]*NUM_CLASSES
        TPR=[0.0]*NUM_CLASSES
        TNR=[0.0]*NUM_CLASSES
        balanced_test_acc=0.0
        for class_index in range(NUM_CLASSES):
            TP[class_index]=((test_pred_list == class_index) & (test_target_list == class_index)).sum()
            TN[class_index] = ((test_pred_list != class_index) & (test_target_list != class_index)).sum()
            FN[class_index] = ((test_pred_list != class_index) & (test_target_list == class_index)).sum()
            FP[class_index] = ((test_pred_list == class_index) & (test_target_list != class_index)).sum()
            if TP[class_index]+FP[class_index]==0:
                p[class_index]=1.0
            else:
                p[class_index]=float(TP[class_index])/float(TP[class_index]+FP[class_index])
            if TP[class_index]+FN[class_index]==0:
                r[class_index]=1.0
            else:
                r[class_index]=float(TP[class_index]) / float(TP[class_index] + FN[class_index])
            if r[class_index]+p[class_index]==0.0:
                F1[class_index]=0.0
            else:
                F1[class_index] = 2 * r[class_index] * p[class_index] / (r[class_index] + p[class_index])
            TPR[class_index]=r[class_index]
            TNR[class_index]=float(TN[class_index])/float(TN[class_index]+FP[class_index])
            balanced_test_acc+=float(TP[class_index])/float(TP[class_index]+FN[class_index])#It is avg r
        balanced_test_acc/=NUM_CLASSES
        print('On testing: ')
        print('TP=')
        print(TP)
        print('TN=')
        print(TN)
        print('FN=')
        print(FN)
        print('FP=')
        print(FP)
        print('precision')
        print(p)
        print('recall')
        print(r)
        print('F1')
        print(F1)
        print('balanced_test_acc')
        print(balanced_test_acc)
        print('\n')
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        f = open(f'{save_results_folder}/20231023_NorCanNecExp91_0_test01_load_model_{my_modelname}.txt', 'a+')
        f.write('Test: balanced accuracy: {:.4f}, average loss(each sample): {},TP= [{}], TN= [{}], FN= [{}], FP= [{}], precision=[{}], recall=[{}], F1= [{}], TPR= [{}], TNR= [{}], \
            Current Time ={}\n\n'.format(balanced_test_acc,
            test_loss,", ".join(str(item) for item in TP),", ".join(str(item) for item in TN),", ".join(str(item) for item in FN),", ".join(str(item) for item in FP),
            ", ".join(str(item) for item in p),", ".join(str(item) for item in r),", ".join(str(item) for item in F1),", ".join(str(item) for item in TPR),", ".join(str(item) for item in TNR),current_time))
        f.close()