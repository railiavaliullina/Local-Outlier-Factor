import pathlib
from enums.dataset_enum import CurrentDataset
import numpy as np

ABSOLUTE_PROJECT_PATH = pathlib.Path(__file__).parent.absolute()
DATA_PATH = ABSOLUTE_PROJECT_PATH / 'data'

CUR_DATASET = CurrentDataset.cifar10.value

K_LIST = np.arange(1, 101) if CUR_DATASET == CurrentDataset.cifar10.value else np.arange(4000, 5803)

LOAD_SAVED_DISTS = True if CUR_DATASET == CurrentDataset.cifar10.value else False
CALCULATE_DISTS_IN_LOOP = True if CUR_DATASET == CurrentDataset.cifar10.value else False

USE_RESNET_AS_FEATURES_EXTRACTOR = False
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
RESNET_FEATURES_PICKLE_PATH = './data/saved_pickles/resnet_features.pickle'
CIFAR_DISTS_PICKLE_PATH = './data/saved_pickles/cifar_dists_resnet_features.pickle' \
    if LOAD_SAVED_DISTS and USE_RESNET_AS_FEATURES_EXTRACTOR \
    else 'data/saved_pickles/cifar_dists.pickle'

GET_DATASET_HALF = False

# табличка с результатами:
# https://docs.google.com/spreadsheets/d/1EOx6AbIy9VqNT96_V8UwkzATj7rVOlBCJu3NYdcerA0/edit?usp=sharing
