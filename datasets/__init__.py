import importlib
import inspect
import os
from argparse import Namespace

from datasets.utils.continual_dataset import ContinualDataset


def get_all_datasets():
    return [model.split('.')[0] for model in os.listdir('datasets')
            if not model.find('__') > -1 and 'py' in model]


NAMES = {}

for dataset in get_all_datasets():
    datas = importlib.import_module('datasets.' + dataset)
    dataset_classes_name = [x for x in datas.__dir__() if 'type' in str(type(getattr(datas, x))) and 'ContinualDataset' in str(inspect.getmro(getattr(datas, x))[1:])]
    for d in dataset_classes_name:
        c = getattr(datas, d)
        NAMES[c.NAME] = c
    
    gcl_dataset_classes_name = [x for x in datas.__dir__() if 'type' in str(type(getattr(datas, x))) and 'GCLDataset' in str(inspect.getmro(getattr(datas, x))[1:])]
    for d in gcl_dataset_classes_name:
        c = getattr(datas, d)
        NAMES[c.NAME] = c

def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES
    return NAMES[args.dataset](args)
