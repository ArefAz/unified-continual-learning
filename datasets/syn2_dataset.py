# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple
from argparse import Namespace
import os
import pickle

import torch
import torch.nn.functional as F
from backbones.mislnet import MISLNet
from torch.utils.data import DataLoader

from datasets.utils.continual_dataset import ContinualDataset
from utils.conf import base_path



class Syn2(ContinualDataset):
    NAME = 'syn-2'
    N_CLASSES_PER_TASK = 2
    INDIM = (3, 256, 256)
    N_TASKS = 20
    MAX_N_SAMPLES_PER_TASK = 1000
    
    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self.setup_loaders()

    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        current_train = self.train_loaders[self.i]
        current_test = self.test_loaders[self.i]

        next_train, next_test = None, None
        if self.i+1 < self.N_TASKS:
            next_train = self.train_loaders[self.i+1]
            next_test = self.test_loaders[self.i+1]
        
        return current_train, current_test, next_train, next_test

    def setup_loaders(self):
        self.test_loaders, self.train_loaders = [], []
        for pkl_file in [
                        "dataloaders/dn-real+dn-gan.pkl",
                        # self.args.new_pkl
                        "dataloaders/dn-real+dn-sd14.pkl",
                        "dataloaders/dn-real+dn-glide.pkl",
                        "dataloaders/dn-real+dn-mj.pkl",
                        "dataloaders/dn-real+dn-dallemini.pkl",
                        "dataloaders/dn-real+dn-tt.pkl",
                        "dataloaders/dn-real+dn-sd21.pkl",
                        "dataloaders/dn-real+dn-cips.pkl",
                        "dataloaders/dn-real+dn-biggan.pkl",
                        "dataloaders/dn-real+dn-vqdiff.pkl",
                        "dataloaders/dn-real+dn-diffgan.pkl",
                        "dataloaders/dn-real+dn-sg3.pkl",
                        "dataloaders/dn-real+dn-gansformer.pkl",
                        "dataloaders/dn-real+dn-dalle2.pkl",
                        "dataloaders/dn-real+dn-ld.pkl",
                        "dataloaders/dn-real+dn-eg3d.pkl",
                        "dataloaders/dn-real+dn-projgan.pkl",
                        "dataloaders/dn-real+dn-sd1.pkl",
                        "dataloaders/dn-real+dn-ddg.pkl",
                        "dataloaders/dn-real+dn-ddpm.pkl",
                    ]:
        
            pkl_file = os.path.join(base_path(), pkl_file)
            with open(pkl_file, 'rb') as f:
                dataloaders = pickle.load(f)

            train_loader = dataloaders[0]
            if self.args.validation:
                test_loader = dataloaders[1]
            else:
                test_loader = dataloaders[2]
            self.test_loaders.append(test_loader)
            self.train_loaders.append(train_loader)
            self.N_TASKS = len(self.train_loaders)

    @staticmethod
    def get_backbone():
        return MISLNet(num_classes=Syn2.N_CLASSES_PER_TASK)
    @staticmethod
    def get_transform():
        return None

    @staticmethod
    def get_normalization_transform():
        return None

    @staticmethod
    def get_denormalization_transform():
        return None

    @staticmethod
    def get_loss():
        return F.cross_entropy

    @staticmethod
    def get_epochs():
        return 1

    @staticmethod
    def get_scheduler(model, args):
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            model.opt,
            step_size=2,
            gamma=0.9,
        )
        return lr_scheduler

    @staticmethod
    def get_batch_size() -> int:
        return 64

    @staticmethod
    def get_minibatch_size() -> int:
        return Syn2.get_batch_size()
