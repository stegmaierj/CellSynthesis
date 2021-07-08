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
import pyshtools
import itertools
import numpy as np

from skimage import io, filters, morphology, measure
from scipy.stats import multivariate_normal
from scipy.ndimage import convolve, distance_transform_edt, gaussian_filter
from pyquaternion import Quaternion

from utils.utils import print_timestamp
from utils.harmonics import harmonics2sampling, sampling2instance
from utils.h5_converter import h5_writer



def generate_data(synthesizer, save_path, experiment_name='dummy_nuclei', num_imgs=50, img_shape=(140,140,1000), max_radius=40, min_radius=20, std_radius=10, psf=None,\
                  sh_order=20, num_cells=200, num_cells_std=50, circularity=5, smooth_std=0.5, noise_std=0.1, noise_mean=-0.1, position_std=3,\
                  cell_elongation=1.5, irregularity_extend=50, generate_images=False, theta_phi_sampling_file=r'utils/theta_phi_sampling_5000points_10000iter.npy'):
        
    # Set up the synthesizer
    synthesizer = synthesizer(img_shape=img_shape, max_radius=max_radius, min_radius=min_radius,\
                              smooth_std=smooth_std, noise_std=noise_std, noise_mean=noise_mean,\
                              sh_order=sh_order, circularity=circularity, num_cells=num_cells, psf=psf,\
                              position_std=position_std, theta_phi_sampling=theta_phi_sampling,\
                              cell_elongation=cell_elongation, irregularity_extend=irregularity_extend,
                              generate_images=generate_images)    
        
    # Set up the save directories
    if generate_images:
        os.makedirs(os.path.join(save_path, 'images'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'masks'), exist_ok=True)
    
    for num_data in range(num_imgs):
        
        current_radius = np.random.randint(min_radius, max_radius)
        synthesizer.max_radius = current_radius + std_radius
        synthesizer.min_radius = current_radius - std_radius
        
        cell_count = np.random.randint(num_cells-num_cells_std, num_cells+num_cells_std)
        synthesizer.num_cells = cell_count
        
        print_timestamp('_'*20)
        print_timestamp('Generating image {0}/{1} with {2} cells of size {3}-{4}', [num_data+1, num_imgs, cell_count, current_radius-std_radius, current_radius+std_radius])
        
        # Get the image and the corresponding mask
        processed_img, instance_mask = synthesizer.generate_data()
        
        ## Save the image
        for num_img,img in enumerate(processed_img):
          
            if not img is None:
                save_name_img = 'psf{0}_img_'.format(num_img)+experiment_name+'_{0}'.format(num_data)  
              
                # TIF
                io.imsave(os.path.join(save_path, 'images', save_name_img+'.tif'), 255*img.astype(np.uint8))
                
                # H5
                img = img.astype(np.float32)
                perc01, perc99 = np.percentile(img, [1,99])
                if not perc99-perc01 <= 0:
                    img -= perc01
                    img /= (perc99-perc01)
                else:
                    img /= img.max()
                img = np.clip(img, 0, 1)
                h5_writer([img], save_name_img+'.h5', group_root='data', group_names=['image'])
                
        ## Save the mask
        save_name_mask = 'mask_'+experiment_name+'_{0}'.format(num_data)
        
        # TIF
        io.imsave(os.path.join(save_path, 'masks', save_name_mask+'.tif'), instance_mask.astype(np.uint16))
    
        # H5
        h5_writer([instance_mask, synthesizer.dist_map], os.path.join(save_path, 'masks', save_name_mask+'.h5'), group_root='data', group_names=['nuclei', 'distance'])
        
        
        
        
def generate_data_from_masks(synthesizer_class, save_path, filelist, min_radius=8, max_radius=9, std_radius=1, psf=None,\
                             sh_order=20, circularity=5, smooth_std=0.5, noise_std=0.1, noise_mean=-0.1, position_std=3, bg_label=0,\
                             cell_elongation=1.5, irregularity_extend=50, generate_images=False, theta_phi_sampling_file=r'utils/theta_phi_sampling_5000points_10000iter.npy'):
        
    # Set up the synthesizer
    synthesizer = synthesizer_class(img_shape=(100,100,100), max_radius=max_radius, min_radius=min_radius,\
                                    smooth_std=smooth_std, noise_std=noise_std, noise_mean=noise_mean,\
                                    sh_order=sh_order, circularity=circularity, num_cells=0, psf=psf,\
                                    position_std=position_std, theta_phi_sampling_file=theta_phi_sampling_file,\
                                    cell_elongation=cell_elongation, irregularity_extend=irregularity_extend,
                                    generate_images=generate_images)    
        
    # Set up the save directories
    if generate_images:
        os.makedirs(os.path.join(save_path, 'images_h5'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'segmentation'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'segmentation_h5'), exist_ok=True)
    
    for num_file, file in enumerate(filelist):
        
        print_timestamp('_'*20)
        print_timestamp('Extracting statistics from image {0}/{1}', [num_file+1, len(filelist)])
        
        
        template = io.imread(file)
        synthesizer.img_shape = template.shape
        positions = []
        for props in measure.regionprops(template):
            positions.append([int(p) for p in props.centroid])
            
        synthesizer.num_cells = len(positions)
        
        current_radius = np.random.randint(min_radius, max_radius)
        synthesizer.max_radius = current_radius + std_radius
        synthesizer.min_radius = current_radius - std_radius
        
        print_timestamp('Generating image with {0} cells of size {1}-{2}', [len(positions), current_radius-std_radius, current_radius+std_radius])
        
        # Get the image and the corresponding mask
        processed_img, instance_mask = synthesizer.generate_data(foreground=template!=bg_label, positions=positions)
        
        ## Save the image
        for num_img,img in enumerate(processed_img):
          
            if not img is None:
                save_name_img = 'psf{0}_img_'.format(num_img)+os.path.split(file)[-1][:-4]
              
                # TIF
                io.imsave(os.path.join(save_path, 'images_h5', save_name_img+'.tif'), 255*img.astype(np.uint8))
                
                # H5
                img = img.astype(np.float32)
                perc01, perc99 = np.percentile(img, [1,99])
                if not perc99-perc01 <= 0:
                    img -= perc01
                    img /= (perc99-perc01)
                else:
                    img /= img.max()
                img = np.clip(img, 0, 1)
                h5_writer([img], save_name_img+'.h5', group_root='data', group_names=['image'])
                
        ## Save the mask
        save_name_mask = 'SimMask_'+os.path.split(file)[-1][:-4]
        
        # TIF
        io.imsave(os.path.join(save_path, 'segmentation', save_name_mask+'.tif'), instance_mask.astype(np.uint16))
    
        # H5
        h5_writer([instance_mask, synthesizer.dist_map], os.path.join(save_path, 'segmentation_h5', save_name_mask+'.h5'), group_root='data', group_names=['nuclei', 'distance'])
        
        


class SyntheticNuclei:
    
    def __init__(self, img_shape=(200,400,400), max_radius=50, min_radius=20, psf=None, sh_order=20, smooth_std=1,\
                 noise_std=0.1, noise_mean=0, num_cells=10, circularity=5, generate_images=False,\
                 theta_phi_sampling_file=r'utils/theta_phi_sampling_5000points_10000iter.npy', **kwargs):
        
        self.img_shape = img_shape
        self.max_radius = max_radius
        self.min_radius = min_radius
        self.sh_order = sh_order
        self.num_coefficients = (sh_order+1)**2
        self.smooth_std = smooth_std
        self.noise_std = noise_std
        self.noise_mean = noise_mean
        self.circularity = circularity
        self.num_cells = num_cells
        self.generate_images = generate_images
        self.theta_phi_sampling_file = theta_phi_sampling_file
        
        if not isinstance(psf, (tuple, list)):
            psf = [psf]
        
        self.psf = []
        for p in psf:
            if isinstance(p, str):
                if psf.endswith(('.tif', '.TIF', 'png')):
                    self.psf.append(io.imread(psf))
                elif psf.endswith(('.npz', '.npy')):
                    self.psf.append(np.load(p))
                else:
                    raise TypeError('Unknown PSF file format.')
            else:
                self.psf.append(p)
         
        self.fg_map = None
        self.instance_mask = None
        self.processed_img = [None]
        
        self._preparations()
        
        
    def _preparations(self):
        # Setting up the converter
        print_timestamp('Loading sampling angles...')
        self.theta_phi_sampling = np.load(self.theta_phi_sampling_file)
        print_timestamp('Setting up harmonic converter...')
        self.h2s = harmonics2sampling(self.sh_order, self.theta_phi_sampling)
        
        
    def generate_data(self, foreground=None, positions=None):
        
        if foreground is None:
            print_timestamp('Generating foreground...')
            self._generate_foreground()
        else:
            self.fg_map = foreground>0
            
        self._generate_distmap()
        
        if positions is None:
            print_timestamp('Determining cell positions...')
            self.positions = self._generate_positions()
        else:
            self.positions = positions
            
        print_timestamp('Starting cell generation...')
        self._generate_instances()
        
        if self.generate_images:
            print_timestamp('Starting synthesis process...')
            self._generate_image()  
            
        print_timestamp('Finished...')
        
        return self.processed_img, self.instance_mask
    
    
    def _generate_foreground(self):
        
        self.fg_map = np.zeros(self.img_shape, dtype=np.bool)
        
        
    def _generate_distmap(self):
        
        # generate foreground distance map
        fg_map = self.fg_map[::4,::4,::4]
        dist_map = distance_transform_edt(fg_map>=1)        
        dist_map = dist_map - distance_transform_edt(fg_map<1)
        dist_map = dist_map.astype(np.float32)
        
        # rescale to original size
        dist_map = np.repeat(dist_map, 4, axis=0)
        dist_map = np.repeat(dist_map, 4, axis=1)
        dist_map = np.repeat(dist_map, 4, axis=2)
        dim_missmatch = np.array(self.fg_map.shape)-np.array(dist_map.shape)
        if dim_missmatch[0]<0: dist_map = dist_map[:dim_missmatch[0],...]
        if dim_missmatch[1]<0: dist_map = dist_map[:,:dim_missmatch[1],:]
        if dim_missmatch[2]<0: dist_map = dist_map[...,:dim_missmatch[2]]
        dist_map = dist_map.astype(np.float32)
        
        self.dist_map = dist_map


    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations
        location_map = self.fg_map.copy()
        cell_size_est = (self.min_radius + self.max_radius) // 2
        slicing = tuple(map(slice, [cell_size_est,]*len(self.img_shape), [s-cell_size_est for s in self.img_shape]))
        location_map[slicing] = True
        
        for cell_count in range(self.num_cells):
            
            # Get random centroid
            location = np.array(np.nonzero(location_map))
            location = location[:,np.random.randint(0, location.shape[1])]
            positions[cell_count,:] = location
        
            # Exclude region of current cell from possible future locations
            slicing = tuple(map(slice, list(np.maximum(location-cell_size_est, 0)), list(location+cell_size_est)))
            location_map[slicing] = False      
                
        return positions
            
    
    def _generate_instances(self):
        
        assert self.circularity>=0, 'Circularity needs to be positive.'
                
        # Get the power per harmonic order
        power_per_order = np.arange(self.sh_order+1, dtype=np.float32)
        power_per_order[0] = np.inf
        power_per_order = power_per_order**-self.circularity
        
        coeff_list = np.zeros((len(self.positions), self.num_coefficients), dtype=np.float32)
        
        for cell_count in range(len(self.positions)):
                        
            # Get harmonic coefficients
            clm = pyshtools.SHCoeffs.from_random(power_per_order)
            coeffs = clm.coeffs
            coeffs[0,0,0] = 1
            
            # Get radius
            radius = np.random.randint(self.min_radius, self.max_radius)
            
            # Scale coefficients respectively
            coeffs *= radius
            coeffs = np.concatenate((np.fliplr(coeffs[0,...]), coeffs[1,...]), axis=1)
            coeffs = coeffs[np.nonzero(coeffs)]
            
            assert len(coeffs) == self.num_coefficients, 'Number of coefficients did not match the expected value.'
            
            coeff_list[cell_count,:] = coeffs
            
           
        # Reconstruct the sampling from the coefficients
        r_sampling = self.h2s.convert(coeff_list)
        
        # Reconstruct the intance mask
        instance_mask = sampling2instance(self.positions, r_sampling, self.theta_phi_sampling, self.img_shape, verbose=True)
        
        self.instance_mask = instance_mask
            
        
        
    def _generate_image(self):
        
        assert not self.instance_mask is None, 'There needs to be an instance mask.'
        
        # Generate image
        img_raw = np.zeros_like(self.instance_mask, dtype=np.float32)
        for label in np.unique(self.instance_mask):
            if label == 0: continue # exclude background
            img_raw[self.instance_mask == label] = np.random.uniform(0.5, 0.9)
            
        self.processed_img  = []
            
        for num_psf,psf in enumerate(self.psf):
            
            print_timestamp('Applying PSF {0}/{1}...', [num_psf+1, len(self.psf)])
        
            img = img_raw.copy()
            
            # Perform PSF smoothing
            if not psf is None:
                img = convolve(img, psf)
            
                # Add final additive noise
                noise = np.random.normal(self.noise_mean, self.noise_std, size=self.img_shape)
                img = img+noise
                img = img.clip(0, 1)
            
            # Final smoothing touch
            img = filters.gaussian(img, self.smooth_std)
            
            self.processed_img.append(img.astype(np.float32))
        
        
        
        
        
        
class SyntheticCElegansWorm(SyntheticNuclei):


    def __init__(self, img_shape=(140,140,1000), max_radius=20, min_radius=10, num_cells=400,\
                 psf=None, sh_order=20, smooth_std=0.5, noise_std=0.1, noise_mean=-0.1, circularity=5,\
                 theta_phi_sampling_file=r'utils/theta_phi_sampling_5000points_10000iter.npy', **kwargs):
        
        super().__init__(img_shape=img_shape, max_radius=max_radius, min_radius=min_radius, num_cells=num_cells,\
                         psf=psf, sh_order=sh_order, smooth_std=smooth_std, noise_mean=noise_mean,\
                         noise_std=noise_std, circularity=circularity, theta_phi_sampling_file=theta_phi_sampling_file)
    
        
    def _generate_foreground(self):
        
        # within ellipsoid equation: (x/a)^2 + (y/b)^2 + /z/c)^2 < 1
        a,b,c = [int(i*0.45) for i in self.img_shape]
        x,y,z = np.indices(self.img_shape)
        
        ellipsoid = ((x-self.img_shape[0]//2)/a)**2 + ((y-self.img_shape[1]//2)/b)**2 + ((z-self.img_shape[2]//2)/c)**2
        self.fg_map = ellipsoid<=1
        
                
    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations
        location_map = self.fg_map.copy()
        
        for cell_count in range(self.num_cells):
            
            print_timestamp('Placing cell {0}/{1}...', [cell_count+1, self.num_cells])
            
            # Get random centroid
            location = np.array(np.nonzero(location_map))
            if location.shape[1] == 0:
                print_timestamp('The maximum number of cells ({0}) was reached...', [cell_count+1])
                positions = positions[:cell_count-1,:]
                break
            location = location[:,np.random.randint(0, location.shape[1])]
            positions[cell_count,:] = location
            
            # Exclude region of current cell from possible future locations
            slicing = tuple(map(slice, list(np.maximum(location-self.min_radius, 0)), list(location+self.min_radius)))
            location_map[slicing] = False      
            
        return positions
        
    
    
    
    
class SyntheticTRIF(SyntheticNuclei):
    
    def __init__(self, img_shape=(900,1800,900), min_radius=13, max_radius=18, cell_elongation=2, num_cells=3500, psf=None,\
                 smooth_std=0.5, noise_std=0.1, noise_mean=-0.1, position_std=3, irregularity_extend=200, **kwargs):
        
        super().__init__(img_shape=img_shape, max_radius=max_radius, min_radius=min_radius, num_cells=num_cells,\
                         psf=psf, smooth_std=smooth_std, noise_mean=noise_mean,\
                         noise_std=noise_std)
        self.position_std = position_std
        self.cell_elongation = cell_elongation
        self.irregularity_extend = irregularity_extend
        
        
    def _preparations(self):
        pass
    
        
    def _generate_foreground(self):
        
        # determine ellipsoid parameters (adjusted to the image size)
        a,b,c = [int(i*0.4) for i in self.img_shape]
        x,y,z = np.indices(self.img_shape, dtype=np.float16)
        
        # distort the coordinates with random gaussian distributions to simulate random shape irregularities
        # coords = coords +/- extend * exp(-x_norm**2/sigma_x - y_norm**2/sigma_y**2 - z_norm**2/sigma_z**2)
        extend_x = (-1)**np.random.randint(0,2) * np.random.randint(self.irregularity_extend/2,np.maximum(self.irregularity_extend,1))
        extend_y = (-1)**np.random.randint(0,2) * np.random.randint(self.irregularity_extend/2,np.maximum(self.irregularity_extend,1))
        extend_z = (-1)**np.random.randint(0,2) * np.random.randint(self.irregularity_extend/2,np.maximum(self.irregularity_extend,1))
        
        distortion_x = np.exp(- np.divide(x-np.random.randint(0,2*a),np.random.randint(a/2,a),dtype=np.float16)**2 - np.divide(y-np.random.randint(0,2*b),np.random.randint(b/2,b),dtype=np.float16)**2 - np.divide(z-np.random.randint(0,2*c),np.random.randint(c/2,c),dtype=np.float16)**2, dtype=np.float16)
        distortion_y = np.exp(- np.divide(x-np.random.randint(0,2*a),np.random.randint(a/2,a),dtype=np.float16)**2 - np.divide(y-np.random.randint(0,2*b),np.random.randint(b/2,b),dtype=np.float16)**2 - np.divide(z-np.random.randint(0,2*c),np.random.randint(c/2,c),dtype=np.float16)**2, dtype=np.float16)
        distortion_z = np.exp(- np.divide(x-np.random.randint(0,2*a),np.random.randint(a/2,a),dtype=np.float16)**2 - np.divide(y-np.random.randint(0,2*b),np.random.randint(b/2,b),dtype=np.float16)**2 - np.divide(z-np.random.randint(0,2*c),np.random.randint(c/2,c),dtype=np.float16)**2, dtype=np.float16)
        
        x = x + extend_x * distortion_x
        y = y + extend_y * distortion_y
        z = z + extend_z * distortion_z
        
        # within ellipsoid equation: (x/a)^2 + (y/b)^2 + /z/c)^2 < 1
        ellipsoid = ((x-self.img_shape[0]//2)/a)**2 + ((y-self.img_shape[1]//2)/b)**2 + ((z-self.img_shape[2]//2)/c)**2
        self.fg_map = ellipsoid<=1
        
        self._generate_distmap()
        
        
        
    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations (outer ring)
        location_map = np.logical_xor(self.fg_map, morphology.binary_erosion(self.fg_map, selem=morphology.ball(self.position_std*2)))
        locations = np.array(np.nonzero(location_map))
        
        # Get cell parameters (*2 since we are looking for centroids)
        cell_shape = 2*np.array([self.max_radius, self.max_radius/self.cell_elongation, self.max_radius/self.cell_elongation])
        
        for cell_count in range(self.num_cells):
            
            print_timestamp('Placing cell {0}/{1}...', [cell_count+1, self.num_cells])
            
            # Get random centroid
            if locations.shape[1] == 0:
                print_timestamp('The maximum number of cells ({0}) was reached...', [cell_count+1])
                positions = positions[:cell_count-1,:]
                break
                        
            location = locations[:,np.random.randint(0, locations.shape[1])]
            positions[cell_count,:] = location
            
            # Exclude region of current cell from possible future locations
            distances = locations - location[:,np.newaxis]
            distances = distances / cell_shape[:,np.newaxis]
            distances = np.sum(distances**2, axis=0)
            locations = locations[:,distances>1]
                
        return positions
    
    
    
    def _generate_instances(self):
                
        # calculate the gradient direction at each position (used to orient each cell)
        grad_map_x, grad_map_y, grad_map_z = np.gradient(self.dist_map, 5)
        grad_map_x = gaussian_filter(grad_map_x, 5)
        grad_map_y = gaussian_filter(grad_map_y, 5)
        grad_map_z = gaussian_filter(grad_map_z, 5)
        
        # normalize the gradient vectors to unit length
        grad_norm = np.sqrt(grad_map_x**2 + grad_map_y**2 + grad_map_z**2)
        grad_map_x = grad_map_x/grad_norm
        grad_map_y = grad_map_y/grad_norm
        grad_map_z = grad_map_z/grad_norm        
        
        # create local coordinates
        cell_mask_shape = (self.max_radius*3,)*3
        coords_default = np.indices(cell_mask_shape)
        coords_default = np.reshape(coords_default, (3,-1))
        coords_default = np.subtract(coords_default, coords_default.max(axis=1, keepdims=True)//2)
        coords_default = coords_default.astype(np.float16)
        
        
        # place a cell at each position
        instance_mask = np.zeros(self.dist_map.shape, dtype=np.uint16)
        for num_cell, pos in enumerate(self.positions):
            
            print_timestamp('Generating cell {0}/{1}...', [num_cell+1, len(self.positions)])
            
            cell_size = np.random.randint(self.min_radius,self.max_radius)
            a,b,c = [cell_size,cell_size/self.cell_elongation,cell_size/self.cell_elongation]  
            coords = coords_default.copy()
            
            # rotation axis is perpendicular to gradient direction and the major axis of the cell
            grad_vec = [grad_map_x[tuple(pos)], grad_map_y[tuple(pos)], grad_map_z[tuple(pos)]]
            cell_vec = [0,]*3
            cell_vec[np.argmax([a,b,c])] = 1
            rot_axis = np.cross(grad_vec, cell_vec)
            axis_norm = np.sqrt(np.sum(rot_axis**2))
            
            if not axis_norm==0:
                
                # normalize the rotation axis
                rot_axis = rot_axis / axis_norm
            
                # calculate the angle from: a*b = ||a||*||b||*cos(angle)
                rot_angle = np.arccos(np.dot(grad_vec, cell_vec)/1)
            
                # rotate using the quaternion
                cell_quant = Quaternion(axis=rot_axis, angle=rot_angle)
                coords = np.matmul(cell_quant.rotation_matrix, coords)
            
            coords = coords.reshape((3,)+cell_mask_shape)
            x_new = coords[0,...]
            y_new = coords[1,...]
            z_new = coords[2,...]
                    
            ellipsoid = ((x_new/a)**2 + (y_new/b)**2 + (z_new/c)**2) <= 1
                        
            slice_start = [np.minimum(np.maximum(0,p-c//2),i-c) for p,c,i in zip(pos,cell_mask_shape,self.img_shape)]
            slice_end = [s+c for s,c in zip(slice_start,cell_mask_shape)]
            slicing = tuple(map(slice, slice_start, slice_end))
            instance_mask[slicing] = np.maximum(instance_mask[slicing], (num_cell+1)*ellipsoid.astype(np.uint16))
            
        self.instance_mask = instance_mask.astype(np.uint16)
        
        
        
   
    
class SyntheticDRO(SyntheticNuclei):
    
    def __init__(self, img_shape=(300,600,1200), min_radius=13, max_radius=18, cell_elongation=3, num_cells=1000, psf=None,\
                 smooth_std=0.5, noise_std=0.1, noise_mean=-0.1, position_std=3, irregularity_extend=200, **kwargs):
        
        super().__init__(img_shape=img_shape, max_radius=max_radius, min_radius=min_radius, num_cells=num_cells,\
                         psf=psf, smooth_std=smooth_std, noise_mean=noise_mean,\
                         noise_std=noise_std)
        self.position_std = position_std
        self.cell_elongation = cell_elongation
        self.irregularity_extend = irregularity_extend
        
        
    def _preparations(self):
        pass
    
        
    def _generate_foreground(self):
        
        # Determine positions
        coords = np.indices(self.img_shape, dtype=np.float16)
        coords[0,...] -= self.img_shape[0]//2
        coords[1,...] -= self.img_shape[1]//2
        coords[2,...] -= self.img_shape[2]//2
                
        # Rotate workspace around x- and y-axis between 0 and 10 degree
        coords = coords.reshape((3,-1))
        
        alpha_x = -np.radians(np.random.randint(5,10))
        alpha_y = -np.radians(np.random.randint(5,10))
        
        Rx = np.array([[1,0,0],[0,np.cos(alpha_x),-np.sin(alpha_x)],[0,np.sin(alpha_x),np.cos(alpha_x)]])
        Ry = np.array([[np.cos(alpha_y),0,np.sin(alpha_y)],[0,1,0],[-np.sin(alpha_y),0,np.cos(alpha_y)]])
        
        coords = np.matmul(Rx,coords)
        coords = np.matmul(Ry,coords)
        
        coords = coords.reshape((3,)+self.img_shape)
        
        # determine ellipsoid parameters (adjusted to the image size)
        a,b,c = [int(i*0.4) for i in self.img_shape]
        
        # distort the coordinates with large random gaussian distributions to simulate shape irregularities
        # coords = coords +/- extend * exp(-x_norm**2/sigma_x - y_norm**2/sigma_y**2 - z_norm**2/sigma_z**2)
        extend_x = (-1)**np.random.randint(0,2) * np.random.randint(self.irregularity_extend/2,np.maximum(self.irregularity_extend,1))
        extend_y = (-1)**np.random.randint(0,2) * np.random.randint(self.irregularity_extend/2,np.maximum(self.irregularity_extend,1))
        extend_z = (-1)**np.random.randint(0,2) * np.random.randint(self.irregularity_extend/2,np.maximum(self.irregularity_extend,1))
        
        distortion_x = np.exp(- np.divide(coords[0,...]-np.random.randint(0,2*a),np.random.randint(a/2,a),dtype=np.float16)**2\
                              - np.divide(coords[1,...]-np.random.randint(0,2*b),np.random.randint(b/2,b),dtype=np.float16)**2\
                              - np.divide(coords[2,...]-np.random.randint(0,2*c),np.random.randint(c/2,c),dtype=np.float16)**2, dtype=np.float16)
        distortion_y = np.exp(- np.divide(coords[0,...]-np.random.randint(0,2*a),np.random.randint(a/2,a),dtype=np.float16)**2\
                              - np.divide(coords[1,...]-np.random.randint(0,2*b),np.random.randint(b/2,b),dtype=np.float16)**2\
                              - np.divide(coords[2,...]-np.random.randint(0,2*c),np.random.randint(c/2,c),dtype=np.float16)**2, dtype=np.float16)
        distortion_z = np.exp(- np.divide(coords[0,...]-np.random.randint(0,2*a),np.random.randint(a/2,a),dtype=np.float16)**2\
                              - np.divide(coords[1,...]-np.random.randint(0,2*b),np.random.randint(b/2,b),dtype=np.float16)**2\
                              - np.divide(coords[2,...]-np.random.randint(0,2*c),np.random.randint(c/2,c),dtype=np.float16)**2, dtype=np.float16)
        
        coords[0,...] = coords[0,...] + extend_x * distortion_x
        coords[1,...] = coords[1,...] + extend_y * distortion_y
        coords[2,...] = coords[2,...] + extend_z * distortion_z
        
        
        # distort the coordinates with small gaussian distributions to simulate identations
        for i in range(np.random.randint(0,5)):
            extend_x = np.random.randint(a,a*2)
            extend_y = np.random.randint(b,b*2)
            extend_z = np.random.randint(c,c*2)
            
            distortion_x = np.exp(- np.divide(coords[0,...]-np.random.randint(a/2,a),np.random.randint(a/2,a),dtype=np.float16)**2\
                                  - np.divide(coords[1,...]-np.random.randint(b/2,b),np.random.randint(b/2,b),dtype=np.float16)**2\
                                  - np.divide(coords[2,...]-np.random.randint(c/2,c),np.random.randint(c/20,c/10),dtype=np.float16)**2, dtype=np.float16)
            distortion_y = np.exp(- np.divide(coords[0,...]-np.random.randint(a/2,a),np.random.randint(a/2,a),dtype=np.float16)**2\
                                  - np.divide(coords[1,...]-np.random.randint(b/2,b),np.random.randint(b/2,b),dtype=np.float16)**2\
                                  - np.divide(coords[2,...]-np.random.randint(c/2,c),np.random.randint(c/20,c/10),dtype=np.float16)**2, dtype=np.float16)
            distortion_z = np.exp(- np.divide(coords[0,...]-np.random.randint(a/2,a),np.random.randint(a/2,a),dtype=np.float16)**2\
                                  - np.divide(coords[1,...]-np.random.randint(b/2,b),np.random.randint(b/2,b),dtype=np.float16)**2\
                                  - np.divide(coords[2,...]-np.random.randint(c/2,c),np.random.randint(c/20,c/10),dtype=np.float16)**2, dtype=np.float16)
            
            coords[0,...] = coords[0,...] + np.sign(coords[0,...]) * extend_x * distortion_x
            coords[1,...] = coords[1,...] + np.sign(coords[1,...]) * extend_y * distortion_y
            coords[2,...] = coords[2,...] + np.sign(coords[2,...]) * extend_z * distortion_z
            
        
        # within ellipsoid equation: (x/a)^2 + (y/b)^2 + /z/c)^2 < 1
        ellipsoid = (coords[0,...]/a)**2 + (coords[1,...]/b)**2 + (coords[2,...]/c)**2
        self.fg_map = ellipsoid<=1
        
        self._generate_distmap()
        
        
        
    def _generate_positions(self):
        
        positions = np.zeros((self.num_cells, 3), dtype=np.uint16)
        
        # Get map of possible cell locations (outer ring)
        location_map = np.logical_xor(self.fg_map, morphology.binary_erosion(self.fg_map, selem=morphology.ball(self.position_std*2)))
        locations = np.array(np.nonzero(location_map))
        
        # Get cell parameters (*2 since we are looking for centroids)
        cell_shape = 2*np.array([self.max_radius, self.max_radius/self.cell_elongation, self.max_radius/self.cell_elongation])
        
        for cell_count in range(self.num_cells):
            
            print_timestamp('Placing cell {0}/{1}...', [cell_count+1, self.num_cells])
            
            # Get random centroid
            if locations.shape[1] == 0:
                print_timestamp('The maximum number of cells ({0}) was reached...', [cell_count+1])
                positions = positions[:cell_count-1,:]
                break
                        
            location = locations[:,np.random.randint(0, locations.shape[1])]
            positions[cell_count,:] = location
            
            # Exclude region of current cell from possible future locations
            distances = locations - location[:,np.newaxis]
            distances = distances / cell_shape[:,np.newaxis]
            distances = np.sum(distances**2, axis=0)
            locations = locations[:,distances>1]
                
        return positions
    
    
    
    def _generate_instances(self):
                
        # calculate the gradient direction at each position (used to orient each cell)
        grad_map_x, grad_map_y, grad_map_z = np.gradient(self.dist_map, 5)
        grad_map_x = gaussian_filter(grad_map_x, 5)
        grad_map_y = gaussian_filter(grad_map_y, 5)
        grad_map_z = gaussian_filter(grad_map_z, 5)
        
        # normalize the gradient vectors to unit length
        grad_norm = np.sqrt(grad_map_x**2 + grad_map_y**2 + grad_map_z**2)
        grad_map_x = grad_map_x/grad_norm
        grad_map_y = grad_map_y/grad_norm
        grad_map_z = grad_map_z/grad_norm        
        
        # create local coordinates
        cell_mask_shape = (self.max_radius*3,)*3
        coords_default = np.indices(cell_mask_shape)
        coords_default = np.reshape(coords_default, (3,-1))
        coords_default = np.subtract(coords_default, coords_default.max(axis=1, keepdims=True)//2)
        coords_default = coords_default.astype(np.float16)
        
        
        # place a cell at each position
        instance_mask = np.zeros(self.dist_map.shape, dtype=np.uint16)
        for num_cell, pos in enumerate(self.positions):
            
            print_timestamp('Generating cell {0}/{1}...', [num_cell+1, len(self.positions)])
            
            cell_size = np.random.randint(self.min_radius,self.max_radius)
            a,b,c = [cell_size,cell_size/self.cell_elongation,cell_size/self.cell_elongation]  
            coords = coords_default.copy()
            
            # rotation axis is perpendicular to gradient direction and the major axis of the cell
            grad_vec = [grad_map_x[tuple(pos)], grad_map_y[tuple(pos)], grad_map_z[tuple(pos)]]
            cell_vec = [0,]*3
            cell_vec[np.argmax([a,b,c])] = 1
            rot_axis = np.cross(grad_vec, cell_vec)
            axis_norm = np.sqrt(np.sum(rot_axis**2))
            
            if not axis_norm==0:
                
                # normalize the rotation axis
                rot_axis = rot_axis / axis_norm
            
                # calculate the angle from: a*b = ||a||*||b||*cos(angle)
                rot_angle = np.arccos(np.dot(grad_vec, cell_vec)/1)
            
                # rotate using the quaternion
                cell_quant = Quaternion(axis=rot_axis, angle=rot_angle)
                coords = np.matmul(cell_quant.rotation_matrix, coords)
            
            coords = coords.reshape((3,)+cell_mask_shape)
            x_new = coords[0,...]
            y_new = coords[1,...]
            z_new = coords[2,...]
                    
            ellipsoid = ((x_new/a)**2 + (y_new/b)**2 + (z_new/c)**2) <= 1
                        
            slice_start = [np.minimum(np.maximum(0,p-c//2),i-c) for p,c,i in zip(pos,cell_mask_shape,self.img_shape)]
            slice_end = [s+c for s,c in zip(slice_start,cell_mask_shape)]
            slicing = tuple(map(slice, slice_start, slice_end))
            instance_mask[slicing] = np.maximum(instance_mask[slicing], (num_cell+1)*ellipsoid.astype(np.uint16))
            
        self.instance_mask = instance_mask.astype(np.uint16)
            