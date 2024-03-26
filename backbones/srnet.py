import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from backbones.utils.continual_backbone import FwdContinualBackbone
from backbones.utils.modules import xavier
from .SRNet.model import Srnet

import torch
import torch.nn as nn

class SRNet2(nn.Module):
    def __init__(
            self,
            num_classes=0,
            patch_size=256,
            **args,
        ):
        super().__init__()
        self.srnet = Srnet()
        self.num_classes = num_classes
        if self.num_classes > 0:
            self.output = torch.nn.Linear(512, self.num_classes)
            
    def forward(self, x, returnt="logits"):
        features = self.srnet(x)
        logits = self.output(features)
        if returnt == "all":
            return logits, None, features
        elif returnt == "features":
            return features
        elif returnt == "logits":
            return logits
        elif returnt == "probs":
            return F.softmax(logits, dim=1)


class SRNet(FwdContinualBackbone):
    NAME = 'srnet'
    def __init__(self, indim, hiddim, outdim, args):
        
        super(SRNet, self).__init__()
        self.softmax = torch.nn.Softmax(dim=1)

        # dictionary for the output and the intermediate layers
        self.fwd_outputs = {}

        def get_feature(name):
            def hook(model, input, output):
                self.fwd_outputs[name] = output
            return hook

        self.net = SRNet2(num_classes=outdim)

        # forward hook to store the intermediate results
        # self.net.avgpool.register_forward_hook(get_feature('features'))
        # self.net.fc.register_forward_hook(get_feature('logits'))

        lightning_checkpoint_path = '/media/nas2/Aref/share/continual_learning/models/srnet/epoch=04-step=3845-v_loss=0.0104-v_acc=0.9966.ckpt'
        checkpoint = torch.load(lightning_checkpoint_path)
        model_state_dict = checkpoint['state_dict']
        model_state_dict = {key.replace('classifier.', '', 1): value for key, value in model_state_dict.items()}
        self.net.load_state_dict(model_state_dict)
        print('SRnet sucessfully loaded from', lightning_checkpoint_path)


    def forward(self, x, returnt='logits'):
        return self.net(x, returnt)

    