
import torch
import torch.utils.data as data
from torchsummary import summary
from datasets import get_dataset, HyperX,sHyperX
from models import get_model, train,test
from utils import sample_gt,get_device,metrics,compute_imf_weights
import visdom
import argparse
import numpy as np
import cv2
from models_liu import train_liu



parser = argparse.ArgumentParser(description="Run deep learning experiments on"
                                             " various hyperspectral datasets")
parser.add_argument('--dataset', type=str, default=None, 
                    help="Dataset to use.")
parser.add_argument('--model', type=str, default=None,
                    help="Model to train. Available:\n"
                    "SVM (linear), "
                    "SVM_grid (grid search on linear, poly and RBF kernels), "
                    "baseline (fully connected NN), "
                    "hu (1D CNN), "
                    "hamida (3D CNN + 1D classifier), "
                    "lee (3D FCN), "
                    "chen (3D CNN), "
                    "li (3D CNN), "
                    "he (3D CNN), "
                    "luo (3D CNN), "
                    "sharma (2D CNN), "
                    "boulch (1D semi-supervised CNN), "
                    "liu (3D semi-supervised CNN), "
                    "mou (1D RNN)")
parser.add_argument('--folder', type=str, help="Folder where to store the "
                    "datasets (defaults to the current working directory).",
                    default="./Datasets/")
parser.add_argument('--cuda', type=int, default=0,
                    help="Specify CUDA device (defaults to -1, which learns on CPU)")
parser.add_argument('--runs', type=int, default=1, help="Number of runs (default: 1)")
parser.add_argument('--restore', type=str, default=None,
                    help="Weights to use for initialization, e.g. a checkpoint")

# Dataset options
group_dataset = parser.add_argument_group('Dataset')
group_dataset.add_argument('--training_sample', type=float, default=10,
                    help="Percentage of samples to use for training (default: 10%%)")
group_dataset.add_argument('--sampling_mode', type=str, default='random', help="Sampling mode"
                    " (random sampling or disjoint, default= random)"
                    )
group_dataset.add_argument('--train_set', type=str, default=None,
                    help="Path to the train ground truth (optional, this "
                    "supersedes the --sampling_mode option)")
group_dataset.add_argument('--test_set', type=str, default=None,
                    help="Path to the test set (optional, by default "
                    "the test_set is the entire ground truth minus the training)")
# Training options
group_train = parser.add_argument_group('Training')
group_train.add_argument('--epoch', type=int, help="Training epochs (optional, if"
                    " absent will be set by the model)")
group_train.add_argument('--patch_size', type=int,
                    help="Size of the spatial neighbourhood (optional, if "
                    "absent will be set by the model)")
group_train.add_argument('--lr', type=float,
                    help="Learning rate, set by the model if not specified.")
group_train.add_argument('--class_balancing', action='store_true',
                    help="Inverse median frequency class balancing (default = False)")
group_train.add_argument('--batch_size', type=int,
                    help="Batch size (optional, if absent will be set by the model")
group_train.add_argument('--test_stride', type=int, default=1,
                     help="Sliding window step stride during inference (default = 1)")
# Data augmentation parameters
group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true',
                    help="Random flips (if patch_size > 1)")
group_da.add_argument('--radiation_augmentation', action='store_true',
                    help="Random radiation noise (illumination)")
group_da.add_argument('--mixture_augmentation', action='store_true',
                    help="Random mixes between spectra")

parser.add_argument('--with_exploration', action='store_true',
                    help="See data exploration visualization")
parser.add_argument('--download', type=str, default=None, nargs='+',
                    
                    help="Download the specified datasets and quits.")



args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET =args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride











MODEL='hamida'
DATASET='train2'
vDATASET='test2'
MODE='disjoint'
ttest=True    
if ttest==False:
    SAMPLE_PERCENTAGE=0.2
else:
    SAMPLE_PERCENTAGE=1
# CHECKPOINT="checkpoints/hamida_et_al/name of the dataset used/True2020-07-02 02:12:54.440903_epoch18_0.72.pth"

viz = visdom.Visdom(env=DATASET + ' ' + MODEL)
hyperparams = vars(args)



img, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET,
                                                               FOLDER)

# top, bottom, left, right = [30]*4
# img=cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
# gt=cv2.copyMakeBorder(gt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])

# qmin=[]
# qmax=[]
img = img.astype('float32')

qmax=np.load("maxint16.npy")
qmin=np.load("minint16.npy")

for i in range(img.shape[-1]):
    # qmin.append(np.min(img[:,:,i]))
    # qmax.append(np.max(img[:,:,i]))
    img[:,:,i] = (img[:,:,i] - qmin[i]) /(qmax[i]  - qmin[i])
    



N_CLASSES=len(LABEL_VALUES)
# N_CLASSES = 24

 
N_BANDS=img.shape[-1]


hyperparams = vars(args)
hyperparams.update({'n_classes': N_CLASSES, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS, 'device': CUDA_DEVICE})
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)








if SAMPLE_PERCENTAGE!=1:
    test_gt, train_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=MODE)
    
    
    #######
    # test_gt, train_gt = sample_gt(test_gt, SAMPLE_PERCENTAGE, mode=MODE)
    # test_gt, train_gt = sample_gt(test_gt, SAMPLE_PERCENTAGE, mode=MODE)
    # test_gt, train_gt = sample_gt(test_gt, SAMPLE_PERCENTAGE, mode=MODE)
    ########
else:
    train_gt = test_gt= gt 
    
    #######



# weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
# hyperparams['weights'] = torch.from_numpy(weights).float()



##########



model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

train_dataset = HyperX(img, train_gt, **hyperparams)
train_loader = data.DataLoader(train_dataset,
                                       batch_size=hyperparams['batch_size'],
                                       pin_memory=hyperparams['device'],
                                       shuffle=True)


if ttest==True:
    del img
    
    vimg, vgt, vLABEL_VALUES, vIGNORED_LABELS, vRGB_BANDS, vpalette = get_dataset(vDATASET,
                                                                   FOLDER)
    # top, bottom, left, right = [30]*4
    # vimg=cv2.copyMakeBorder(vimg, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
    # vgt=cv2.copyMakeBorder(vgt, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
    
    
    
    # vqmin,vqmax=[],[]
    
    vimg = vimg.astype('float32')
    for i in range(vimg.shape[-1]):
        # vqmin.append(np.min(vimg[:,:,i]))
        # vqmax.append(np.max(vimg[:,:,i]))
        vimg[:,:,i] = (vimg[:,:,i] - qmin[i]) /(qmax[i]  - qmin[i])
    
    
    
    # vtrain_gt, vtest_gt = sample_gt(vgt, 0.1, mode='random')
    val_dataset = HyperX(vimg, vgt, **hyperparams)
    # del vimg
    val_loader = data.DataLoader(val_dataset,
                                         pin_memory=hyperparams['device'],
                                         batch_size=hyperparams['batch_size'],drop_last=True)
    del vimg
    
else:
    
    
    
    
  
    
    val_dataset = HyperX(img, test_gt, **hyperparams)
    val_loader = data.DataLoader(val_dataset,
                                         pin_memory=hyperparams['device'],
                                         batch_size=hyperparams['batch_size'],drop_last=True)
        
# del img        
print(np.count_nonzero(train_gt))
print(np.count_nonzero(test_gt))
print(np.count_nonzero(gt))
print(hyperparams)
print("Network :")





with torch.no_grad():
    for input, _ in train_loader:
        break
    summary(model.to(hyperparams['device']), input.size()[1:])
    
    
if CHECKPOINT is not None:
            
    model.load_state_dict(torch.load(CHECKPOINT))
    
if MODEL!="liu":
    try:
                  train(model, optimizer, loss, train_loader, hyperparams['epoch'],
                      scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                      supervision=hyperparams['supervision'], val_loader=val_loader,
                      display=viz,klepsia=klepsia)

    except KeyboardInterrupt:
                # Allow the user to stop the training
        pass
    
if MODEL=="liu":
    
    strain_dataset = sHyperX(img, train_gt, **hyperparams) 
        
                                      
    strain_loader = data.DataLoader(strain_dataset,
                                        batch_size=hyperparams['batch_size'],
                                        pin_memory=hyperparams['device'],
                                        shuffle=True,drop_last=True)                           
    try:
                  train_liu(model, optimizer, loss, train_loader, hyperparams['epoch'], val_loader,klepsia,strain_loader,
                      scheduler=hyperparams['scheduler'], device=hyperparams['device'],
                       
                      display=viz)

    except KeyboardInterrupt:
                # Allow the user to stop the training
        pass

# probabilities = test(model, vimg, hyperparams)
# prediction = np.argmax(probabilities, axis=-1)
# run_results = metrics(prediction, vgt, ignored_labels=[0], n_classes=N_CLASSES)
# cm=run_results['Confusion matrix']
# cmr = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
# cmp = cm.astype('float') / cm.sum(axis=0)[np.newaxis,:]
# np.savetxt(MODEL+'cmr.csv',cmr,delimiter=',')
# np.savetxt(MODEL+'cm.csv',cm,delimiter=',')
# np.savetxt(MODEL+'cmp.csv',cmp,delimiter=',')
# rep=run_results['report']
# np.savetxt(MODEL+'rep.csv',rep,delimiter=',')
# import cv2
# cv2.imwrite(MODEL+'pred.tif',prediction)
