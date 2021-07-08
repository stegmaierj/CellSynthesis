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
import numpy as np
import pandas as pd

from skimage import morphology, measure, io
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

from utils.h5_converter import h5_writer
from utils.utils import print_timestamp



def generate_data(syn_class, save_path='Segmentations_h5', experiment_name='synthetic_data', img_count=100, param_dict={}):

    synthesizer = syn_class(**param_dict)
    os.makedirs(save_path, exist_ok=True)
    
    for num_img in range(img_count):
        
        print_timestamp('_'*20)
        print_timestamp('Generating mask {0}/{1}', (num_img+1, img_count))
        
        # Generate a new mask
        synthesizer.generate_instances()
        
        # Get and save the instance, boundary, centroid and distance masks
        print_timestamp('Saving...')
        instance_mask = synthesizer.get_instance_mask().astype(np.uint16)
        instance_mask = np.transpose(instance_mask, [2,1,0])
        boundary_mask = synthesizer.get_boundary_mask().astype(np.uint8)
        boundary_mask = np.transpose(boundary_mask, [2,1,0])
        distance_mask = synthesizer.get_distance_mask().astype(np.float32)
        distance_mask = np.transpose(distance_mask, [2,1,0])
        centroid_mask = synthesizer.get_centroid_mask().astype(np.uint8)
        centroid_mask = np.transpose(centroid_mask, [2,1,0])
        
        save_name = os.path.join(save_path, experiment_name+'_'+str(num_img)+'.h5')        
        h5_writer([instance_mask, boundary_mask, distance_mask, centroid_mask], save_name, group_names=['instances', 'boundary', 'distance', 'seeds'])




### MERISTEM
#param_dict = {'gridsize':512,
#              'elongation':0.4,
#              'distance_weight':0.35,
#              'morph_radius':3,
#              'weights':None,
#              'cell_density':1/20,
#              'cell_density_decay':0.9,
#              'cell_position_smoothness':10,
#              'ring_density':1/20,
#              'ring_density_decay':0.9,
#              'angular_sampling_file':r'D:\LfB\pytorchRepo\utils\theta_phi_sampling_5000points_10000iter.npy',
#              'specimen_sampling_file':r'D:\LfB\pytorchRepo\utils\PNAS_sampling.csv'}



def analyze_cell_sizes(filelist, bg_label=0):
    
    area = []
    minor_axis = []
    major_axis = []
    img_shapes = []
    
    for num_file, file in enumerate(filelist):
        
        print_timestamp('Processing file {0}/{1}...', [num_file+1, len(filelist)])
        
        data = io.imread(file)
        
        img_shapes.append(np.array(data.shape))
        
        regions = measure.regionprops(data)
        
        for region in regions:
            
            if not region.label == bg_label:
                area.append(region.area)
                minor_axis.append(region.minor_axis_length)
                major_axis.append(region.major_axis_length)
            
    print_timestamp('Image shape: {0} +/- {1}', (np.mean(img_shapes, axis=0), np.std(img_shapes, axis=0)))
    print_timestamp('Area: {0:.2f}-{1:.2f}, {2:.2f} +/- {3:.2f}', (np.min(area), np.max(area), np.mean(area), np.std(area)))
    print_timestamp('Minor axis: {0:.2f}-{1:.2f}, {2:.2f} +/- {3:.2f}', (np.min(minor_axis), np.max(minor_axis), np.mean(minor_axis), np.std(minor_axis)))
    print_timestamp('Major axis: {0:.2f}-{1:.2f}, {2:.2f} +/- {3:.2f}', (np.min(major_axis), np.max(major_axis), np.mean(major_axis), np.std(major_axis)))
        
        
        
def analyze_organism_sizes(filelist, bg_label=0):
    
    area = []
    minor_axis = []
    major_axis = []
    img_shapes = []
    
    for num_file, file in enumerate(filelist):
        
        print_timestamp('Processing file {0}/{1}...', [num_file+1, len(filelist)])
        
        data = io.imread(file)
        
        img_shapes.append(np.array(data.shape))
        
        data = data>bg_label
        regions = measure.regionprops(data.astype(np.int))
        
        for region in regions:
            area.append(region.area)
            minor_axis.append(region.minor_axis_length)
            major_axis.append(region.major_axis_length)
            
    print_timestamp('Image shape: {0} +/- {1}', (np.mean(img_shapes, axis=0), np.std(img_shapes, axis=0)))
    print_timestamp('Area: {0:.2f}-{1:.2f}, {2:.2f} +/- {3:.2f}', (np.min(area), np.max(area), np.mean(area), np.std(area)))
    print_timestamp('Minor axis: {0:.2f}-{1:.2f}, {2:.2f} +/- {3:.2f}', (np.min(minor_axis), np.max(minor_axis), np.mean(minor_axis), np.std(minor_axis)))
    print_timestamp('Major axis: {0:.2f}-{1:.2f}, {2:.2f} +/- {3:.2f}', (np.min(major_axis), np.max(major_axis), np.mean(major_axis), np.std(major_axis)))
        
    
    
def agglomerative_clustering(x_samples, y_samples, z_samples, max_dist=10):
    
    samples = np.array([x_samples, y_samples, z_samples]).T
    
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=max_dist, linkage='complete').fit(samples)
    
    cluster_labels = clustering.labels_
    
    cluster_samples_x = []
    cluster_samples_y = []
    cluster_samples_z = []
    for label in np.unique(cluster_labels):
        
        cluster_samples_x.append(int(np.mean(x_samples[cluster_labels==label])))
        cluster_samples_y.append(int(np.mean(y_samples[cluster_labels==label])))
        cluster_samples_z.append(int(np.mean(z_samples[cluster_labels==label])))
        
    cluster_samples_x = np.array(cluster_samples_x)
    cluster_samples_y = np.array(cluster_samples_y)
    cluster_samples_z = np.array(cluster_samples_z)
        
    return cluster_samples_x, cluster_samples_y, cluster_samples_z




class SyntheticCellMembranes:
    """Parent class for generating synthetic cell membranes"""
    


    def __init__(self, gridsize=128, elongation=1, flatness=1, distance_weight=0.25, cell_density=1/20): 
        
        self.gridsize = gridsize
        self.elongation = elongation
        self.flatness = flatness
        self.distance_weight = distance_weight
        self.cell_density = cell_density
        
        self.instance_mask = None
        self.x_fg = []
        self.y_fg = []
        self.z_fg = []
        self.x_cell = []
        self.y_cell = []
        self.z_cell = []
       
        
        
    def _cart2sphere(self, x, y, z):
        
        n = x**2 + y**2 + z**2
        n = n.astype(np.float)
        n[n==0] += 1e-7 # prevent zero divisions
        r = np.sqrt(n)
        p = np.arctan2(y, x)
        t = np.arccos(z/np.sqrt(n))
                
        return r, t, p



    def _sphere2cart(self, r, t, p):

        x = np.sin(t) * np.cos(p) * r
        y = np.sin(t) * np.sin(p) * r
        z = np.cos(t) * r

        return x, y, z
    
    
    
    def generate_instances(self):
        
        print_timestamp('Generating foreground region...')
        self._generate_foreground()
        print_timestamp('Placing random centroids...')
        self._place_centroids()
        print_timestamp('Creating Voronoi tessellation...')
        self._voronoi_tessellation()
        print_timestamp('Morphological postprocessing...')
        self._post_processing()
        

    
    # sphere generation
    def _generate_foreground(self):
        
        # set the image size
        self.image_size = (int(self.gridsize*self.flatness), self.gridsize, int(self.gridsize*self.elongation))
        
        # Determine foreground region
        image_ind = np.indices(self.image_size, dtype=np.int)
        x_fg = image_ind[0].flatten()
        y_fg = image_ind[1].flatten()
        z_fg = image_ind[2].flatten()
            
        self.x_fg = x_fg
        self.y_fg = y_fg
        self.z_fg = z_fg

        
        
    def _place_centroids(self):
        
        # calculate the volume
        vol = len(self.x_fg)
        
        # estimate the number of cells
        cell_count = vol*(self.cell_density**3)
        cell_count = int(cell_count)
    
        # select random points within the foregound region
        rnd_idx = np.random.choice(np.arange(0,len(self.x_fg)), size=cell_count, replace=False)
        
        x_cell = self.x_fg[rnd_idx].copy()
        y_cell = self.y_fg[rnd_idx].copy()
        z_cell = self.z_fg[rnd_idx].copy()
            
        # perform clustering
        x_cell = np.array(x_cell)
        y_cell = np.array(y_cell)
        z_cell = np.array(z_cell)
        x_cell, y_cell, z_cell = agglomerative_clustering(x_cell, y_cell, z_samples=z_cell, max_dist=2*self.cell_density**-1)
        
        self.x_cell = x_cell
        self.y_cell = y_cell
        self.z_cell = z_cell
        
        
    
    def _voronoi_tessellation(self):
        
        # get each foreground sample to be tesselated
        samples = list(zip(self.x_fg, self.y_fg, self.z_fg))
        
        # get each cell centroid
        cells = list(zip(self.x_cell, self.y_cell, self.z_cell))
        cells_tree = cKDTree(cells)
        
        # determine weights for each cell (improved roundness)
        closest_cell_dist, _ = cells_tree.query(cells, k=2)
        weights = closest_cell_dist[:,1]/np.max(closest_cell_dist[:,1])*self.distance_weight + 1 
        
        # determine the closest cell candidates for each sampling point
        candidates_dist, candidates_idx = cells_tree.query(samples, k=5)
        candidates_dist = np.multiply(candidates_dist, weights[candidates_idx])
        candidates_closest = np.argmin(candidates_dist, axis=1)
        
        instance_mask = np.zeros(self.image_size, dtype=np.uint16)
        for sample, cand_idx, closest_idx in zip(samples, candidates_idx, candidates_closest):
            instance_mask[sample] = cand_idx[closest_idx]+1
            
        self.instance_mask = instance_mask
        
        
        
    def _post_processing(self):        
        pass
        
    
    
    def get_instance_mask(self):
        
        return self.instance_mask
        
    
    
    def get_centroid_mask(self, data=None):
        
        if data is None:            
            if self.instance_mask is None:
                return None
            else:
                data = self.instance_mask
        
        # create centroid mask
        centroid_mask = np.zeros(data.shape, dtype=np.bool)
        
        # find and place centroids
        regions = measure.regionprops(data)
        for props in regions:
            c = props.centroid
            centroid_mask[np.int(c[0]), np.int(c[1]), np.int(c[2])] = True
        
        return centroid_mask
    
    
    
    def get_boundary_mask(self, data=None):
        
        # get instance mask
        if data is None:            
            if self.instance_mask is None:
                return None
            else:
                data = self.instance_mask
            
        membrane_mask = morphology.dilation(data, selem=morphology.ball(3)) - data
        membrane_mask = membrane_mask != 0
        
        return membrane_mask
        
    
    def get_distance_mask(self, data=None):

        # get instance mask
        if data is None:            
            if self.instance_mask is None:
                return None
            else:
                data = self.instance_mask
            
        distance_encoding = np.zeros(data.shape, dtype=np.float32)
                        
        # get foreground distance
        distance_encoding = distance_transform_edt(data>0)
        
        # get background distance
        distance_encoding = distance_encoding - distance_transform_edt(data<=0)
                
        return distance_encoding

        


    

# Class for generation of synthetic meristem data
class SyntheticMeristem(SyntheticCellMembranes):
    """Child class for generating synthetic meristem membranes"""



    def __init__(self, gridsize=512, elongation=0.4, distance_weight=0.25, # general params 
                       morph_radius=3, weights=None, # foreground params
                       cell_density=1/23, cell_density_decay=0.9, cell_position_smoothness=10, # cell params
                       ring_density=1/23,  ring_density_decay=0.9, # ring params
                       angular_sampling_file=r'D:\LfB\pytorchRepo\utils\theta_phi_sampling_5000points_10000iter.npy',
                       specimen_sampling_file=r'D:\LfB\pytorchRepo\utils\PNAS_sampling.csv'):
        
        super().__init__(gridsize=gridsize, elongation=elongation, flatness=1, distance_weight=distance_weight, cell_density=cell_density)
        
        # foregound params
        self.morph_radius = morph_radius
        self.weights = weights
        
        # cell params
        self.cell_density_decay = cell_density_decay
        self.cell_position_smoothness = cell_position_smoothness
        
        # ring params
        self.ring_density = ring_density
        self.ring_density_decay = ring_density_decay
               
        self.angular_sampling_file = angular_sampling_file
        self.specimen_sampling_file = specimen_sampling_file
        
        # initialize the statistical shape model
        print_timestamp('Initializing the statistical shape model...')
        self.sampling_angles = np.load(angular_sampling_file)
        specimen_sampling = pd.read_csv(specimen_sampling_file, sep=';').to_numpy()
        self.specimen_mean = np.mean(specimen_sampling, axis=1)
        
        # calculate the PCA
        specimen_cov = np.cov(specimen_sampling)
        specimen_pca = PCA(n_components=3)
        specimen_pca.fit(specimen_cov)
        self.specimen_pca = specimen_pca
        
        if not self.weights is None:
            assert len(self.weights) == len(self.specimen_pca.singular_values_), 'Number of weights ({0}) does not match the number of eigenvectors {1}'.format(len(self.weights), len(self.specimen_pca.singular_values_))
    
    
    # sphere generation
    def _generate_foreground(self):
        
        # Construct the sampling tree
        sampling_tree = cKDTree(self.sampling_angles)

        if self.weights is None:
            # Generate random sampling
            weights = np.random.randn(len(self.specimen_pca.singular_values_))
        else:
            weights = np.array(self.weights)
        specimen_rnd = self.specimen_mean + np.matmul(self.specimen_pca.components_.T, np.sqrt(self.specimen_pca.singular_values_) * weights)
        
        # Normalize the radii
        specimen_rnd /= specimen_rnd.max()
        
        # Calculate the image size based on the shape model and the desired gridsize
        specimen_x, specimen_y, specimen_z = self._sphere2cart(specimen_rnd,self.sampling_angles[:,0],self.sampling_angles[:,1])
        specimen_dim_ratio = np.array([2*np.abs(specimen_x.max()), 2*np.abs(specimen_y.max()), np.abs(specimen_z.max())]) # *2 on x and y, since it's a hemisphere starting at the center
        specimen_dim_ratio /= specimen_dim_ratio.max()
        self.image_size = np.array(specimen_dim_ratio*self.gridsize, dtype=np.int)
                
        # Adjust to desired grid size       
        specimen_rnd *= self.gridsize/2*0.95  
        self.sampling_radii = specimen_rnd.copy()
        
        # Determine foreground region
        image_ind = np.indices(self.image_size, dtype=np.int)
        x = image_ind[0].flatten()-self.image_size[0]/2
        y = image_ind[1].flatten()-self.image_size[1]/2
        z = image_ind[2].flatten()
    
        r,t,p = self._cart2sphere(x,y,z)
    
        # Determine nearest sampling angles
        _, assignments = sampling_tree.query(np.array([t,p]).T, k=3)
        specimen_rnd[specimen_rnd==0] = np.nan
    
        # Determine foreground region
        foreground_ind = r <= np.nanmean(specimen_rnd[assignments], axis=1)
        x_fg = x[foreground_ind]+self.image_size[0]/2
        x_fg = x_fg.astype(np.int)
        y_fg = y[foreground_ind]+self.image_size[1]/2
        y_fg = y_fg.astype(np.int)
        z_fg = z[foreground_ind]
        z_fg = z_fg.astype(np.int)        
        
        self.x_fg = x_fg
        self.y_fg = y_fg
        self.z_fg = z_fg
       
                
        
    def _place_centroids(self):
        
        cell_density = self.cell_density
        ring_density = self.ring_density
        cluster_density = cell_density
             
        # initialize the first radius (small offset to place the first ring near the boundary)
        ring_radii = self.sampling_radii + (self.ring_density**-1)/2
        
        x_cell = []
        y_cell = []
        z_cell = []    
        ring_count = 0
        
        while ring_radii.max() - ring_density**-1 > 0:
            
            # set off the new ring
            ring_radii = np.maximum(0, ring_radii - ring_density**-1)
            
            # get the spherical coordinated of the new ring
            t = self.sampling_angles[ring_radii != 0, 0]
            p = self.sampling_angles[ring_radii != 0, 1]
            r = ring_radii[ring_radii != 0]
            
            # apply small offsets to the radii, depending on the depth of the current ring
            r = r + np.random.normal(loc=0, scale=ring_count/cell_density/self.cell_position_smoothness, size=len(r)) 
            r = np.maximum(0, r)
            
            # convert to cartesian coordinates
            x,y,z = self._sphere2cart(r,t,p)
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
                        
            # cluster cell centroid candidates to reduce computation afterwards
            if len(x) > 1:
                x, y, z = agglomerative_clustering(x, y, z_samples=z, max_dist=(cell_density**-1))
            
            # extend the centroid list
            x_cell.extend(x+self.image_size[0]//2)
            y_cell.extend(y+self.image_size[1]//2)
            z_cell.extend(z)
            
            cluster_density = np.max([cluster_density, cell_density, ring_density])
            cell_density = cell_density*self.cell_density_decay
            ring_density = ring_density*self.ring_density_decay
            ring_count += 1
        
        # perform clustering
        x_cell = np.array(x_cell)
        y_cell = np.array(y_cell)
        z_cell = np.array(z_cell)
        x_cell, y_cell, z_cell = agglomerative_clustering(x_cell, y_cell, z_samples=z_cell, max_dist=cluster_density**-1)
        
        self.x_cell = x_cell
        self.y_cell = y_cell
        self.z_cell = z_cell
    
           
        
    def _post_processing(self):
                    
        # get the memebrane mask
        membrane_mask = self.get_boundary_mask()
        
        # open the inner parts of the cells
        opened_fg = morphology.binary_opening(~membrane_mask, selem=morphology.ball(self.morph_radius))
        opened_fg[self.instance_mask==0] = False
        
        # erode the fg region 
        eroded_fg = morphology.binary_erosion(self.instance_mask>0, selem=morphology.ball(int(self.morph_radius*1.5)))
        
        # generate the enhanced foreground mask
        enhanced_fg = np.logical_or(opened_fg, eroded_fg)
        
        # generate the instance mask
        self.instance_mask[~enhanced_fg] = 0        
        self.x_fg, self.y_fg, self.z_fg = np.nonzero(self.instance_mask)
        
        






# Class for generation of synthetic meristem data
class SyntheticDrosophila(SyntheticCellMembranes):
    """Child class for generating synthetic drosophila membranes"""



    def __init__(self, gridsize=300, elongation=2, flatness=0.4, distance_weight=0.25, # general params 
                       morph_radius=1, # foreground params
                       cell_density=1/6, cell_density_decay=0.9, cell_position_smoothness=9, # cell params
                       ring_num=1, ring_shrink=0.3, # ring params
                       angular_sampling_file=r'D:\LfB\pytorchRepo\utils\theta_phi_sampling_5000points_10000iter.npy'):
        
        super().__init__(gridsize=gridsize, flatness=flatness, elongation=elongation, distance_weight=distance_weight, cell_density=cell_density)
        
        # foregound params
        self.morph_radius = morph_radius
        
        # cell params
        self.cell_density_decay = cell_density_decay
        self.cell_position_smoothness = cell_position_smoothness
        
        # ring params
        self.ring_num = ring_num
        self.ring_shrink = ring_shrink
               
        self.angular_sampling_file = angular_sampling_file
        
        # initialize the angular positions
        self.sampling_angles = np.load(angular_sampling_file)

    
    
    # sphere generation
    def _generate_foreground(self):

        # Generate sampling
        x_bound = self.flatness * np.sin(self.sampling_angles[:,0])*np.cos(self.sampling_angles[:,1])
        y_bound = 1 * np.sin(self.sampling_angles[:,0])*np.sin(self.sampling_angles[:,1])
        z_bound = self.elongation * np.cos(self.sampling_angles[:,0])
        
        r_outer,t_sampling,p_sampling = self._cart2sphere(x_bound, y_bound, z_bound)
        r_outer /= r_outer.max()
        r_outer *= 0.95*(np.max(self.image_size)/2)      
        self.r_outer = r_outer
        
        rnd_shrink = (1+np.random.choice(np.linspace(-0.2, 0.2, 50))) * self.ring_shrink
        self.r_inner = self.r_outer * (1 - rnd_shrink + rnd_shrink*np.abs(z_bound)/np.max(np.abs(z_bound))/self.elongation)
        
        # Construct the sampling tree
        self.sampling_angles_outer = np.array([t_sampling,p_sampling]).T
        sampling_tree = cKDTree(self.sampling_angles_outer)
            
        # Determine foreground region
        image_ind = np.indices(self.image_size, dtype=np.int)
        x = image_ind[0].flatten()-self.image_size[0]/2
        y = image_ind[1].flatten()-self.image_size[1]/2
        z = image_ind[2].flatten()-self.image_size[2]/2
    
        r,t,p = self._cart2sphere(x,y,z)
    
        # Determine nearest sampling angles
        num_neighbours = 9
        distances, assignments = sampling_tree.query(np.array([t,p]).T, k=num_neighbours)
        distances = distances / np.repeat(np.max(distances, axis=1, keepdims=True), num_neighbours, axis=1) 
    
        # Determine foreground region
        foreground_ind = np.logical_and(r >= np.average(self.r_inner[assignments], axis=1, weights=1-distances),\
                                        r <= np.average(self.r_outer[assignments], axis=1, weights=1-distances))
            
        x_fg = x[foreground_ind]+self.image_size[0]/2
        x_fg = x_fg.astype(np.int)
        y_fg = y[foreground_ind]+self.image_size[1]/2
        y_fg = y_fg.astype(np.int)
        z_fg = z[foreground_ind]+self.image_size[2]/2
        z_fg = z_fg.astype(np.int)
        
        self.x_fg = x_fg
        self.y_fg = y_fg
        self.z_fg = z_fg
        
        
        
    def _place_centroids(self):
        
        cell_density = self.cell_density
        cluster_density = cell_density
        ring_density = self.ring_num/(np.mean(self.r_outer)-np.mean(self.r_inner))
             
        # initialize the first radius (small offset to place the first ring near the boundary)
        ring_radii = self.r_outer + (ring_density**-1)/2
        
        x_cell = []
        y_cell = []
        z_cell = []    
        
        for ring_count in range(self.ring_num):
            
            # set off the new ring
            ring_radii = np.maximum(0, ring_radii - ring_density**-1)
            
            # get the spherical coordinated of the new ring
            t = self.sampling_angles_outer[ring_radii != 0, 0]
            p = self.sampling_angles_outer[ring_radii != 0, 1]
            r = ring_radii[ring_radii != 0]
            
            # apply small offsets to the radii, depending on the depth of the current ring
            r = r + np.random.normal(loc=0, scale=ring_count/cell_density/self.cell_position_smoothness, size=len(r)) 
            r = np.maximum(0, r)
            
            # convert to cartesian coordinates
            x,y,z = self._sphere2cart(r,t,p)
            x = np.array(x)
            y = np.array(y)
            z = np.array(z)
            
            # cluster cell centroid candidates to reduce computation afterwards
            x, y, z = agglomerative_clustering(x, y, z_samples=z, max_dist=(cell_density**-1))
            
            # extend the centroid list
            x_cell.extend(x+self.image_size[0]//2)
            y_cell.extend(y+self.image_size[1]//2)
            z_cell.extend(z+self.image_size[2]//2)
            
            cluster_density = np.max([cluster_density, cell_density])
            cell_density = cell_density*self.cell_density_decay
        
        # perform clustering
        x_cell = np.array(x_cell)
        y_cell = np.array(y_cell)
        z_cell = np.array(z_cell)
        x_cell, y_cell, z_cell = agglomerative_clustering(x_cell, y_cell, z_samples=z_cell, max_dist=cluster_density**-1)
        
        self.x_cell = x_cell
        self.y_cell = y_cell
        self.z_cell = z_cell
    
           
        
    def _post_processing(self):
                    
        # get the memebrane mask
        membrane_mask = self.get_boundary_mask()
        
        # open the inner parts of the cells
        opened_fg = morphology.binary_opening(~membrane_mask, selem=morphology.ball(self.morph_radius))
        opened_fg[self.instance_mask==0] = False
        
        # erode the fg region 
        eroded_fg = morphology.binary_erosion(self.instance_mask>0, selem=morphology.ball(int(self.morph_radius*1.5)))
        
        # generate the enhanced foreground mask
        enhanced_fg = np.logical_or(opened_fg, eroded_fg)
        
        # generate the instance mask
        self.instance_mask[~enhanced_fg] = 0        
        self.x_fg, self.y_fg, self.z_fg = np.nonzero(self.instance_mask)
        
        