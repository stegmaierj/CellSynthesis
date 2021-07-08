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

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
import numpy as np

from argparse import ArgumentParser, Namespace
from collections import OrderedDict
from torch.utils.data import DataLoader

from dataloader.h5_dataloader import MeristemH5Dataset
from dataloader.augmenter import intensity_augmenter_pytorch
from ThirdParty.radam import RAdam
from models.UNet3D_pixelshuffle import UNet3D_pixelshuffle_module
    
    

class Discriminator(nn.Module):
    
    def __init__(self, patch_size, in_channels, **kwargs):
        super(Discriminator, self).__init__()
        
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_size = tuple([int(p/2**4) for p in patch_size])

        # Define layer instances
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.leaky1 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.norm2 = nn.InstanceNorm3d(128)
        self.leaky2 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1)
        self.norm3 = nn.InstanceNorm3d(256)
        self.leaky3 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1)
        self.norm4 = nn.InstanceNorm3d(512)
        self.leaky4 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv5 = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1)
        self.norm5 = nn.InstanceNorm3d(512)
        self.leaky5 = nn.LeakyReLU(negative_slope=0.2)
        
        self.conv6 = nn.Conv3d(512, 1, kernel_size=3, stride=1, padding=1)
        self.sig6 = nn.Sigmoid()


    def forward(self, img):
        
        out = self.conv1(img)
        out = self.leaky1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.leaky2(out)
        
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.leaky3(out)
        
        out = self.conv4(out)
        out = self.norm4(out)
        out = self.leaky4(out)
        
        out = self.conv5(out)
        out = self.norm5(out)
        out = self.leaky5(out)
        
        out = self.conv6(out)
        out = self.sig6(out)

        return out




class GAN(pl.LightningModule):

    def __init__(self, hparams):
        super(GAN, self).__init__()
        
        if type(hparams) is dict:
            hparams = Namespace(**hparams)
        self.hparams = hparams
        self.augmentation_dict = {}
        self.augmenter = intensity_augmenter_pytorch(self.augmentation_dict)

        # networks
        self.generator = UNet3D_pixelshuffle_module(patch_size=self.hparams.patch_size, in_channels=self.hparams.in_channels, out_channels=self.hparams.out_channels,\
                                                    feat_channels=self.hparams.feat_channels, out_activation=self.hparams.out_activation, norm_method=hparams.norm)
        self.discriminator = Discriminator(patch_size=self.hparams.patch_size, in_channels=self.hparams.out_channels+1)

        # cache for generated images
        self.generated_imgs = None
        self.last_imgs = None
        
        self.ada_state = {'last_discriminator_outputs':[],
                              'ada_r':0,
                              'ada_prob':None}
        
        # Check if there already is an ada state dict
        if 'resume' in self.hparams:
            if os.path.isfile(os.path.join(self.hparams.output_path, 'ada_ckpt.json')) and self.hparams.resume:
                self.ada_state = json.load(open(os.path.join(self.hparams.output_path, 'ada_ckpt.json')))
            


    def forward(self, z):
        return self.generator(z)


    def load_pretrained(self, pretrained_file, strict=True, verbose=True):
        
        if isinstance(pretrained_file, (list,tuple)):
            pretrained_file = pretrained_file[0]
        
        # Load the state dict
        state_dict = torch.load(pretrained_file)['state_dict']
        
        # Make sure to have a weight dict
        if not isinstance(state_dict, dict):
            state_dict = dict(state_dict)
            
        # Get parameter dict of current model
        param_dict = dict(self.generator.named_parameters())
        
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
        
        self.generator.load_state_dict(param_dict)
        
        if verbose:
            print('Loaded weights for the following layers:\n{0}'.format(layers))
            

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def identity_loss(self, y_hat, y):
        return F.l1_loss(y_hat, y)

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        if self.current_epoch==0 and batch_idx==0 and optimizer_idx==0:
            print('Training started...')
        
        # Get image and mask of current batch
        imgs, masks = batch['image'], batch['mask']
        
        self.last_imgs = imgs.float()
        self.last_masks = masks.float()
        
        # Adaptive Discriminator Augmentation (ADA): 
        if self.hparams.ada_update_period>0:
            
            # Update states
            if self.current_epoch%self.hparams.ada_update_period==0 and optimizer_idx==0 and not self.current_epoch==0 and batch_idx==0:
                if len(self.ada_state['last_discriminator_outputs'])>0:
                    self.ada_state['ada_r'] = np.mean(self.ada_state['last_discriminator_outputs']).astype(np.float64) # (otherwise not serialisable)
                else:
                    self.ada_state['ada_r'] = self.hparams.ada_target
                if self.ada_state['ada_r'] > self.hparams.ada_target:
                    self.ada_state['ada_prob'] += self.hparams.ada_update
                elif self.ada_state['ada_r'] < self.hparams.ada_target:
                    self.ada_state['ada_prob'] -= self.hparams.ada_update
                self.ada_state['ada_prob'] = np.clip(self.ada_state['ada_prob'], 0, 1)
                self.set_augmentations()
                
                # Save current states
                with open(os.path.join(self.hparams.output_path, 'ada_ckpt.json'), 'w') as file_handle:
                    json.dump(self.ada_state, file_handle)
            
                # Reset discriminator outputs
                self.ada_state['last_discriminator_outputs'] = []

        # train generator
        if optimizer_idx == 0:
            
            # generate images
            self.generated_imgs = self.forward(self.last_masks)
            self.identity_imgs = self.forward(self.last_imgs)
            
            # identity loss (how well are sturctures preserved)
            identity_loss = self.identity_loss(self.identity_imgs, self.last_imgs[:,0:1,...])            

            # discriminator should recognize this as valid
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones((self.last_masks.shape[0],1)+self.discriminator.out_size, dtype=torch.float32)
            if self.on_gpu:
                valid = valid.cuda(self.last_imgs.device.index)
                
            # adversarial loss (how realistic does the image look like)
            self.generated_imgs = self.augmenter.apply(self.generated_imgs)
            adv_loss = self.adversarial_loss(self.discriminator(torch.cat((self.generated_imgs,self.last_masks[:,1:2,...]), 1)), valid)
            
            g_loss = (adv_loss + identity_loss) / 2
            tqdm_dict = {'adv_loss': adv_loss, 'identity_loss': identity_loss, 'epoch': self.current_epoch}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # generate images
            self.generated_imgs = self.forward(self.last_masks)
            self.generated_imgs = self.augmenter.apply(self.generated_imgs)

             # how well can it label as fake?
            fake = torch.zeros((self.last_masks.shape[0],1)+self.discriminator.out_size, dtype=torch.float32)
            if self.on_gpu:
                fake = fake.cuda(self.last_imgs.device.index)
                
            fake_loss = self.adversarial_loss(self.discriminator(torch.cat((self.generated_imgs,self.last_masks[:,1:2,...]), 1)), fake)

            # how well can it label as real?
            valid = torch.ones((self.last_masks.shape[0],1)+self.discriminator.out_size, dtype=torch.float32)
            if self.on_gpu:
                valid = valid.cuda(self.last_imgs.device.index)

            self.last_imgs[:,0,...] = self.augmenter.apply(self.last_imgs[:,0,...])
            real_loss = self.adversarial_loss(self.discriminator(self.last_imgs), valid)
           
            # Adaptive Discriminator Augmentation (ADA): append the current discriminator output
            if self.hparams.ada_update_period > 0:
                real_decision = self.discriminator(self.last_imgs)
                self.ada_state['last_discriminator_outputs'].append(float(torch.mean(real_decision).cpu().detach().numpy()))

            # discriminator loss is the average of these
            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'real_loss':real_loss, 'fake_loss':fake_loss, 'ada_r':self.ada_state['ada_r'], 'ada_prob':self.ada_state['ada_prob']}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
        
    def test_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        x_hat = self.forward(y)
        return {'test_loss': F.l1_loss(x_hat, x[:,0:1,...])} 

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        x_hat = self.forward(y)
        return {'val_loss': F.l1_loss(x_hat, x[:,0:1,...])} 

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        lr = self.hparams.learning_rate

        opt_g = RAdam(self.generator.parameters(), lr=lr)
        opt_d = RAdam(self.discriminator.parameters(), lr=lr)
        return [opt_g, opt_d], []

    def train_dataloader(self):
         if self.hparams.train_list is None:
            return None
         else:
            dataset = MeristemH5Dataset(self.hparams.train_list, self.hparams.data_root, patch_size=self.hparams.patch_size, norm_method=self.hparams.data_norm,\
                                        patches_from_fg=self.hparams.patches_from_fg, image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups,\
                                        augmentation_dict={}, dist_handling=self.hparams.dist_handling, dist_scaling=self.hparams.dist_scaling,\
                                        instance_handling=self.hparams.instance_handling, correspondence=False)
            print('Training on {0} images...'.format(dataset.__len__()))
            return DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)
    
    def test_dataloader(self):
        if self.hparams.test_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.test_list, self.hparams.data_root, patch_size=self.hparams.patch_size, norm_method=self.hparams.data_norm,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        dist_handling=self.hparams.dist_handling, dist_scaling=self.hparams.dist_scaling,\
                                        instance_handling=self.hparams.instance_handling, correspondence=False)
            print('Testing on {0} images...'.format(dataset.__len__()))
            return DataLoader(dataset, batch_size=self.hparams.batch_size)
    
    def val_dataloader(self):
        if self.hparams.val_list is None:
            return None
        else:
            dataset = MeristemH5Dataset(self.hparams.val_list, self.hparams.data_root, patch_size=self.hparams.patch_size, norm_method=self.hparams.data_norm,\
                                        image_groups=self.hparams.image_groups, mask_groups=self.hparams.mask_groups, augmentation_dict={},\
                                        dist_handling=self.hparams.dist_handling, dist_scaling=self.hparams.dist_scaling,\
                                        instance_handling=self.hparams.instance_handling, correspondence=False)
            print('Validating on {0} images...'.format(dataset.__len__()))
            return DataLoader(dataset, batch_size=self.hparams.batch_size)


    def on_epoch_end(self):
        
        # log sampled images
        generated_imgs = self.forward(self.last_masks)
        generated_grid = torchvision.utils.make_grid(generated_imgs[:,:,20,:,:])
        self.logger.experiment.add_image('generated_images', generated_grid, self.current_epoch)
        
        img_grid = torchvision.utils.make_grid(torch.cat((torch.abs(self.last_imgs[:,:,20,:,:]), torch.zeros((self.last_imgs.shape[0], 1)+self.last_imgs.shape[3:]).cuda(generated_imgs.device.index)), 1))
        self.logger.experiment.add_image('raw_images', img_grid, self.current_epoch)
        
        mask_grid = torchvision.utils.make_grid(torch.cat((torch.abs(self.last_masks[:,:,20,:,:]), torch.zeros((self.last_masks.shape[0], 1)+self.last_masks.shape[3:]).cuda(generated_imgs.device.index)), 1))
        self.logger.experiment.add_image('input_masks', mask_grid, self.current_epoch)
        
    
    def set_augmentations(self, augmentation_dict_file=None):
        if not augmentation_dict_file is None:
            self.augmentation_dict = json.load(open(augmentation_dict_file))
        self.augmenter = intensity_augmenter_pytorch(self.augmentation_dict)
        
        if not self.ada_state['ada_prob'] is None:
            self.augmenter.augmentation_dict['prob'] = self.ada_state['ada_prob']
        else:
            self.ada_state['ada_prob'] = self.augmenter.augmentation_dict['prob']
        
        
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
        parser.add_argument('--norm', default='instance', type=str)
        parser.add_argument('--out_activation', default='sigmoid', type=str)

        # ADA params
        parser.add_argument('--ada_update_period', default=1, type=int)
        parser.add_argument('--ada_update', default=0.05, type=float)
        parser.add_argument('--ada_target', default=0.6, type=float)

        # data
        parser.add_argument('--data_root', default='data_root', type=str) 
        parser.add_argument('--train_list', default='datalist_train.csv', type=str)
        parser.add_argument('--test_list', default='datalist_test.csv', type=str)
        parser.add_argument('--val_list', default='datalist_val.csv', type=str)
        parser.add_argument('--image_groups', default=('data/image', 'data/distance'), type=str, nargs='+')
        parser.add_argument('--mask_groups', default=('data/boundary', 'data/distance'), type=str, nargs='+')     
        parser.add_argument('--dist_handling', default='tanh', type=str)
        parser.add_argument('--dist_scaling', default=(100,100), type=float, nargs='+')
        parser.add_argument('--instance_handling', default='bool', type=str)
        parser.add_argument('--patches_from_fg', default=0.0, type=float)

        # training params (opt)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--learning_rate', default=0.001, type=float)

        return parser