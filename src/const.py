import time
import torch
import socket as _socket

_hostname = str(_socket.gethostname())

name = time.strftime('%m-%d %H:%M:%S', time.localtime())


USE_NET = 'VGG16'

TRAIN_DIR = 'runs/' + name
VAL_DIR = 'runs/' + name

FASHIONET_LOAD_VGG16_GLOBAL = False

DATASET_PROC_METHOD_TRAIN = 'RANDOM'
DATASET_PROC_METHOD_VAL = 'LARGESTCENTER'

# 0: no sigmoid 1: sigmoid
VGG16_ACT_FUNC_IN_POSE = 0

MODEL_NAME = 'vgg16.pkl'

if 'dlcs302-2' == _hostname:
    base_path = '/home/hzy/datasets/DeepFashion/Category and Attribute Prediction Benchmark/'
else:
    base_path = '/home/dl/datasets/DeepFashion/Category and Attribute Prediction Benchmark/'

NUM_EPOCH = 20
LEARNING_RATE = 0.0001
LEARNING_RATE_DECAY = 0.9
BATCH_SIZE = 16
VAL_BATCH_SIZE = 32

WEIGHT_ATTR_NEG = 0.1
WEIGHT_ATTR_POS = 1
WEIGHT_LANDMARK_VIS_NEG = 0.5
WEIGHT_LANDMARK_VIS_POS = 0.5


# LOSS WEIGHT
WEIGHT_LOSS_CATEGORY = 1
WEIGHT_LOSS_ATTR = 20
WEIGHT_LOSS_LM_POS = 100


# VAL
VAL_CATEGORY_TOP_N = (1, 3, 5)
VAL_ATTR_TOP_N = (3, 5)
VAL_LM_RELATIVE_DIS = 0.1

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

lm2name = ['L.Col', 'R.Col', 'L.Sle', 'R.Sle', 'L.Wai', 'R.Wai', 'L.Hem', 'R.Hem']
attrtype2name = {1: 'texture', 2: 'fabric', 3: 'shape', 4: 'part', 5: 'style'}

VAL_WHILE_TRAIN = True

USE_CSV = 'info.csv'

LM_TRAIN_USE = 'vis'
LM_EVAL_USE = 'vis'
