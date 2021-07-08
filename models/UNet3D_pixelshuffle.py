# -*- coding: utf-8 -*-
"""
# 3D Image Data Synthesis.
# Copyright (C) 2021 D. Eschweiler, M. Rethwisch, M. Jarchow, S. Koppers, J. Stegmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the Liceense at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Please refer to the documentation for more information about the software
# as well as for installation instructions.
#
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataloader.h5_dataloader import MeristemH5Dataset
from ThirdParty.radam import RAdam
from ThirdParty.layers import PixelShuffle3d



class UNet3D_pixelshuffle_module(nn.Module):
    """Implementation of the 3D U-Net architecture.
    """

    def __init__(self, patch_size, in_channels, out_channels, feat_channels=16, out_activation='sigmoid', norm_method='none', **kwargs):
        super(UNet3D_pixelshuffle_module, self).__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.out_activation = out_activation # relu | sigmoid | tanh | hardtanh | none
        self.norm_method = norm_method # instance | batch | none
        
        if self.norm_method == 'instance':
            self.norm = nn.InstanceNorm3d
        elif self.norm_method == 'batch':
            self.norm = nn.BatchNorm3d
        elif self.norm_method == 'none':
            self.norm = nn.Identity
        else:
            raise ValueError('Unknown normalization method "{0}". Choose from "instance|batch|none".'.format(self.norm_method))
        
        
        # Define layer instances       
        self.c1 = nn.Sequential(
            nn.Conv3d(in_channels, feat_channels//2, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels//2, feat_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d1 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )


        self.c2 = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels*2, kernel_size=3, padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d2 = nn.Sequential(
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=4, stride=2, padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.c3 = nn.Sequential(
            nn.Conv3d(feat_channels*2, feat_channels*2, kernel_size=3, padding=1),
            self.norm(feat_channels*2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*2, feat_channels*4, kernel_size=3, padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.d3 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=4, stride=2, padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )


        self.c4 = nn.Sequential(
            nn.Conv3d(feat_channels*4, feat_channels*4, kernel_size=3, padding=1),
            self.norm(feat_channels*4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*4, feat_channels*8, kernel_size=3, padding=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u1 = nn.Sequential(
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PixelShuffle3d(2),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c5 = nn.Sequential(
            nn.Conv3d(feat_channels*5, feat_channels*8, kernel_size=3, padding=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u2 = nn.Sequential(
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PixelShuffle3d(2),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c6 = nn.Sequential(
            nn.Conv3d(feat_channels*3, feat_channels*8, kernel_size=3, padding=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=3, padding=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.u3 = nn.Sequential(
            nn.Conv3d(feat_channels*8, feat_channels*8, kernel_size=1),
            self.norm(feat_channels*8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            PixelShuffle3d(2),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.c7 = nn.Sequential(
            nn.Conv3d(feat_channels*2, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        
        
        self.out = nn.Sequential(
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, feat_channels, kernel_size=3, padding=1),
            self.norm(feat_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv3d(feat_channels, out_channels, kernel_size=1)
            )
       
        if self.out_activation == 'relu':
            self.out_fcn = nn.ReLU()
        elif self.out_activation == 'sigmoid':
            self.out_fcn = nn.Sigmoid()
        elif self.out_activation == 'tanh':
            self.out_fcn = nn.Tanh()
        elif self.out_activation == 'hardtanh':
            self.out_fcn = nn.Hardtanh(0, 1)
        elif self.out_activation == 'none':
            self.out_fcn = None
        else:
            raise ValueError('Unknown output activation "{0}". Choose from "relu|sigmoid|tanh|hardtanh|none".'.format(self.out_activation))
        

    def forward(self, img):
        
        c1 = self.c1(img)
        d1 = self.d1(c1)
        
        c2 = self.c2(d1)
        d2 = self.d2(c2)
        
        c3 = self.c3(d2)
        d3 = self.d3(c3)
        
        c4 = self.c4(d3)
        
        u1 = self.u1(c4)
        c5 = self.c5(torch.cat((u1,c3),1))
        
        u2 = self.u2(c5)
        c6 = self.c6(torch.cat((u2,c2),1))
        
        u3 = self.u3(c6)
        c7 = self.c7(torch.cat((u3,c1),1))
        
        out = self.out(c7)
        if not self.out_fcn is None:
            out = self.out_fcn(out)
        
        return out
        
    
    
    
class UNet3D_pixelshuffle(pl.LightningModule):
    
    def __init__(self, hparams):
        super(UNet3D_pixelshuffle, self).__init__()
        
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.augmentation_dict = {}

        # networks
        self.network = UNet3D_pixelshuffle_module(patch_size=hparams.patch_size, in_channels=hparams.in_channels, out_channels=hparams.out_channels, feat_channels=hparams.feat_channels, out_activation=hparams.out_activation, norm_method=hparams.norm_method)
        
        # cache for generated images
        self.last_predictions = None
        self.last_imgs = None
        self.last_masks = None


    def forward(self, z):
        return self.network(z)
    
    
    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        
        # Load the state dict
        state_dict = torch.load(pretrained_file)['state_dict']
        
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)
            
        # Get parameter dict of current model
        param_dict = dict(self.network.named_parameters())
        
        layers = []
        for layer in param_dict:
            if strict and not 'network.'+layer in state_dict:
                if verbose:
                    print('Could not find weights for layer "{0}"'.format(layer))
                continue
            try:
                param_dict[layer].data.copy_(state_dict['network.'+layer].data)
                layers.append(layer)
            except (RuntimeError, KeyError) as e:
                print('Error at layer {0}:\n{1}'.format(layer, e))
        
        self.network.load_state_dict(param_dict)
        
        if verbose:
            print('Loaded weights for the following layers:\n{0}'.format(layers))
        
        
    def background_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)


    def boundary_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)
    
    
    def seed_loss(self, y_hat, y):
        loss = F.mse_loss(y_hat, y, reduction='none')
        weight = torch.clamp(y, min=0.1, max=1.0)
        loss = torch.mul(loss, weight)        
        loss = torch.sum(loss)/torch.sum(weight)
        return loss


    def training_step(self, batch, batch_idx):
        
        # Get image ans mask of current batch
        self.last_imgs, self.last_masks = batch['image'], batch['mask']
        
        # generate images
        self.predictions = self.forward(self.last_imgs)
                
        # get the losses
        loss_bg = self.background_loss(self.predictions[:,0,...], self.last_masks[:,0,...])
        loss_seed = self.seed_loss(self.predictions[:,1,...], self.last_masks[:,1,...])
        loss_boundary = self.boundary_loss(self.predictions[:,2,...], self.last_masks[:,2,...])
        
        loss = self.hparams.background_weight * loss_bg + \
               self.hparams.seed_weight * loss_seed + \
               self.hparams.boundary_weight * loss_boundary
        tqdm_dict = {'bg_loss': loss_bg, 'seed_loss':loss_seed, 'boundary_loss': loss_boundary, 'epoch': self.current_epoch}
        output = OrderedDict({
            'loss': loss,
            'progress_bar': tqdm_dict,
            'log': tqdm_dict
        })
        return output
        
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.forward(x)
        return {'test_loss': F.l1_loss(y_hat, y)} 

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        y_hat = self.forward(x)
        return {'val_loss': F.mse_loss(y_hat, y)} 

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt = RAdam(self.network.parameters(), lr=self.hparams.learning_rate)
        return [opt], []

    def train_dataloader(self):
         if self.hparams.train_list is None:
            return None
         else:
            dataset = MeristemH5Dataset(self.hparams.train_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict=self.augmentation_dict,\
                                        dist_handling='bool_inv', seed_handling='float', norm_method=self.hparams.data_norm)
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
    
    def test_dataloader(self):
        if self.hparams.test_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.test_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        dist_handling='bool_inv', seed_handling='float', norm_method=self.hparams.data_norm)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        if self.hparams.val_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.val_list, self.hparams.data_root, patch_size=self.hparams.patch_size,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        dist_handling='bool_inv', seed_handling='float', norm_method=self.hparams.data_norm)
            return DataLoader(dataset, batch_size=self.hparams.batch_size)


    def on_epoch_end(self):
        
        # log sampled images
        predictions = self.forward(self.last_imgs)
        prediction_grid = torchvision.utils.make_grid(predictions[:,:,int(self.hparams.patch_size[0]//2),:,:])
        self.logger.experiment.add_image('generated_images', prediction_grid, self.current_epoch)
        
        img_grid = torchvision.utils.make_grid(self.last_imgs[:,:,int(self.hparams.patch_size[0]//2),:,:])
        self.logger.experiment.add_image('raw_images', img_grid, self.current_epoch)
        
        mask_grid = torchvision.utils.make_grid(self.last_masks[:,:,int(self.hparams.patch_size[0]//2),:,:])
        self.logger.experiment.add_image('input_masks', mask_grid, self.current_epoch)
        
        
    def set_augmentations(self, augmentation_dict_file):
        self.augmentation_dict = json.load(open(augmentation_dict_file))
        
        
    @staticmethod
    def add_model_specific_args(parent_parser): 
        """
        Parameters you define here will be available to your model through self.hparams
        """
        parser = ArgumentParser(parents=[parent_parser])

        # network params
        parser.add_argument('--in_channels', default=2, type=int)
        parser.add_argument('--out_channels', default=1, type=int)
        parser.add_argument('--feat_channels', default=16, type=int)
        parser.add_argument('--patch_size', default=(64,128,128), type=int, nargs='+')
        parser.add_argument('--out_activation', default='sigmoid', type=str)
        parser.add_argument('--norm_method', default='instance', type=str)

        # data
        parser.add_argument('--data_norm', default='percentile', type=str)
        parser.add_argument('--data_root', default=r'data_root', type=str) 
        parser.add_argument('--train_list', default=r'datalist_train.csv', type=str)
        parser.add_argument('--test_list', default=r'datalist_test.csv', type=str)
        parser.add_argument('--val_list', default=r'datalist_val.csv', type=str)
        parser.add_argument('--image_groups', default=('data/image',), type=str, nargs='+')
        parser.add_argument('--mask_groups', default=('data/boundary', 'data/distance'), type=str, nargs='+')

        # training params
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)
        parser.add_argument('--background_weight', default=1, type=float)
        parser.add_argument('--seed_weight', default=100, type=float)
        parser.add_argument('--boundary_weight', default=1, type=float)
        
        return parser