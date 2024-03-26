import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, densenet121

from backbones.utils.continual_backbone import FwdContinualBackbone
from backbones.utils.modules import xavier

import ipdb

class DenseNet(FwdContinualBackbone):
    NAME = 'densenet'
    def __init__(self, indim, hiddim, outdim, args):
        super(DenseNet, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

        # dictionary for the output and the intermediate layers
        self.fwd_outputs = {}

        def get_feature(name):
            def hook(model, input, output):
                self.fwd_outputs[name] = output
            return hook

        self.net = densenet121(num_classes=outdim)

        # forward hook to store the intermediate results
        # self.net.features.register_forward_hook(get_feature('features'))
        # self.net.classifier.register_forward_hook(get_feature('logits'))

        lightning_checkpoint_path = '/media/nas2/Aref/share/continual_learning/models/densenet121/epoch=19-step=9380-v_loss=0.1256-v_acc=0.9575.ckpt'
        checkpoint = torch.load(lightning_checkpoint_path)
        model_state_dict = checkpoint['state_dict']
        model_state_dict = {key.replace('classifier.', '', 1): value for key, value in model_state_dict.items()}
        self.net.load_state_dict(model_state_dict)
        print('Densenet sucessfully loaded from', lightning_checkpoint_path)


    def forward(self, x, returnt='logits'):
        if returnt == 'features':
            self.net(x)
            return self.net.ff
        elif returnt == 'logits':
            return self.net(x)
        elif returnt == 'prob':
            return self.softmax(self.net(x))
        elif returnt == 'all':
            logits = self.net(x)
            probs = self.softmax(logits)
            return logits, probs, self.net.ff

    