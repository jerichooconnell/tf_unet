# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 28, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import random
import math
import pickle
import os

import matlab.engine
from scipy.io import loadmat

sigma = 10

plateau_min = -2
plateau_max = 2

r_min = 1
r_max = 200

def generate_voronoi_diagram(width, height, cnt=50):

    num_cells = cnt

    nbins = 4
    image = np.zeros((width, height, nbins))
    labels = np.zeros([width, height], dtype=np.bool)

    imgx, imgy = width, height
    nx = []  # hold the x value
    ny = []  # holds the y value
    nr = np.zeros([nbins, int(cnt / 2) - 20])  # low energy image
    nr2 = np.zeros([nbins, int(cnt / 2)])  # high energy image
    label = []  # labels whether or not the cell has cartilage

    target_SNR = 100
    # Finding the singnal in water that will be the basis for the noise
    signal = 3.1109 * 10**3
    # Finding the target noise
    sigma = signal / target_SNR
    # Finding the signal strength that would give the correct poisson noise
    correct_signal = (signal / sigma)**2.

    D = loadmat(
        "/home/jericho/Downloads/bin/pcd_planar_imaging/DES_radiography/DES_3.0/Y_ML_val.mat"
    )

    cal_values = D["Y_ML_val"][:, 1:]

    inds_0 = np.where(cal_values[0, :] == 0)[0]
    inds_1 = np.where(cal_values[0, :] != 0)[0]

    for i in range(int(num_cells / 2 - 20)):

        # choosing random place for the cell
        nx.append(random.randrange(imgx))
        ny.append(random.randrange(imgy))
        pp = random.randint(0, len(cal_values[cal_values == 0.]) - 1)

        # making a random value for the cartilage
        #     if i % 2 == 0:
        #         nr[:, i] = cal_values[1:5, rr]
        #         label.append(1)
        #     else:
        nr[:, i] = cal_values[1:5, inds_0[pp]]
        label.append(0)

    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx - 1, imgy - 1)
            j = -1
            for i in range(int(num_cells / 2 - 20)):
                d = math.hypot(nx[i] - x, ny[i] - y)
                if d < dmin:
                    dmin = d
                    j = i

            # Generating some poisson noise\
            #for kk in range(nbins):
            image[x, y, :] = np.nan_to_num(np.log(
                (np.exp(nr[:, j])) +
                np.random.normal(0, np.sqrt(np.exp(nr[:, j]) * 2500)) / 2500))

    labels[x, y] = label[j]

    r_min = 3
    r_max = 50
    border = 10
    sigma = 10

    #image = np.ones((nx, ny, 1))
    label = np.ones((width, height))
    mask = np.zeros([width, height], dtype=np.bool)
    for i in range(int(num_cells / 2)):
        a = np.random.randint(border, width - border)
        b = np.random.randint(border, height - border)
        r = np.random.randint(r_min, r_max)
        rr = random.randint(0, len(inds_1) - 1)
        nr2[:, i] = cal_values[1:5, inds_1[rr]]

        y, x = np.ogrid[-a:width - a, -b:height - b]
        m = x * x + y * y <= r * r
        mask = np.logical_or(mask, m)

        image[m] = np.nan_to_num(
            np.log((np.exp(nr2[:, i])) + np.random.normal(
                0, np.sqrt(np.exp(nr2[:, i]) * 2500), size=image.shape)[m] / 2500))

    labels[mask] = 1

    matlab_reshape = np.reshape(image,[image.size],order='F').copy()
    matlab_reshape = list(np.squeeze(matlab_reshape)).copy()
    matlab_reshape = matlab.double(matlab_reshape)

    eng = matlab.engine.start_matlab()
    ys = eng.net_val(matlab_reshape)
    
    image = np.zeros((width,height,1))
    image[:,:,0] = np.reshape(ys,[width,height],order='F').copy()
    
    image -= np.amin(image)
    image /= np.amax(image)
    
    return image, labels

def generate_voronoi_diagram_val(width, height, cnt=1, r = 2, t = 4, rr = 5):

    num_cells = cnt

    nbins = 4
    image = np.zeros((width, height, nbins))
    labels = np.zeros([width, height], dtype=np.bool)

    imgx, imgy = width, height
    nx = []  # hold the x value
    ny = []  # holds the y value
    nr = np.zeros([nbins, int(cnt / 2) - 20])  # low energy image
    nr2 = np.zeros([nbins, int(cnt / 2)])  # high energy image
    label = []  # labels whether or not the cell has cartilage

    target_SNR = 100
    # Finding the singnal in water that will be the basis for the noise
    signal = 3.1109 * 10**3
    # Finding the target noise
    sigma = signal / target_SNR
    # Finding the signal strength that would give the correct poisson noise
    correct_signal = (signal / sigma)**2.

    D = loadmat(
        "/home/jericho/Downloads/bin/pcd_planar_imaging/DES_radiography/DES_3.0/Y_ML_val.mat"
    )

    cal_values = D["Y_ML_val"][:, 1:]

    inds_0 = np.where(cal_values[0, :] == 0)[0]
    inds_1 = np.where(cal_values[0, :] != 0)[0]

    for i in range(5):

        # choosing random place for the cell
        nx.append(random.randrange(imgx))
        ny.append(random.randrange(imgy))
        pp = random.randint(0, len(cal_values[cal_values == 0.]) - 1)

        # making a random value for the cartilage
        #     if i % 2 == 0:
        #         nr[:, i] = cal_values[1:5, rr]
        #         label.append(1)
        #     else:
        nr[:, i] = cal_values[1:5, inds_0[pp]]
        label.append(0)

    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx - 1, imgy - 1)
            j = -1
            for i in range(int(num_cells / 2 - 20)):
                d = math.hypot(nx[i] - x, ny[i] - y)
                if d < dmin:
                    dmin = d
                    j = i

            # Generating some poisson noise\
            #for kk in range(nbins):
            image[x, y, :] = np.nan_to_num(np.log(
                (np.exp(nr[:, j])) +
                np.random.normal(0, np.sqrt(np.exp(nr[:, j]) * 2500)) / 2500))

    labels[x, y] = label[j]

    r_min = 3
    r_max = 50
    border = 10
    sigma = 10

    #image = np.ones((nx, ny, 1))
    label = np.ones((width, height))
    mask = np.zeros([width, height], dtype=np.bool)
    
    a = np.random.randint(border, width - border)
    b = np.random.randint(border, height - border)
    nr2 = cal_values[1:5, inds_1[rr]]

    y, x = np.ogrid[-a:width - a, -b:height - b]
    m = x * x + y * y <= r * r
    mask = np.logical_or(mask, m)

    image[m] = np.nan_to_num(
        np.log((np.exp(nr2[:, i])) + np.random.normal(
            0, np.sqrt(np.exp(nr2[:, i]) * 2500), size=image.shape)[m] / 2500))

    labels[mask] = 1

    matlab_reshape = np.reshape(image,[image.size],order='F').copy()
    matlab_reshape = list(np.squeeze(matlab_reshape)).copy()
    matlab_reshape = matlab.double(matlab_reshape)

    eng = matlab.engine.start_matlab()
    ys = eng.net_val(matlab_reshape)
    
    image = np.zeros((width,height,1))
    image[:,:,0] = np.reshape(ys,[width,height],order='F').copy()
    
    image -= np.amin(image)
    image /= np.amax(image)
    
    return image, labels

def create_image_and_label(nx,ny, cnt = 10):
    r_min = 5
    r_max = 50
    border = 92
    sigma = 20
    
    image = np.ones((nx, ny, 1))
    label = np.ones((nx, ny))
    mask = np.zeros((nx, ny), dtype=np.bool)
    for _ in range(cnt):
        a = np.random.randint(border, nx-border)
        b = np.random.randint(border, ny-border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1,255)

        y,x = np.ogrid[-a:nx-a, -b:ny-b]
        m = x*x + y*y <= r*r
        mask = np.logical_or(mask, m)

        image[m] = h
    label[mask] = 0
    
    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)
    
    return image, label



def get_image_gen(nx, ny, **kwargs):
    def create_batch(n_image):
        
        X = np.zeros((n_image,nx,ny,1))
        Y = np.zeros((n_image,nx,ny,2))
        
        for i in range(n_image):
            #X[i],Y[i,:,:,1] = create_image_and_label(nx,ny, **kwargs)
            X[i],Y[i,:,:,1] = generate_voronoi_diagram(nx,ny, **kwargs)
            Y[i,:,:,0] = 1-Y[i,:,:,1]
            
        return X,Y
    
    create_batch.channels = 1
    create_batch.n_class = 2
    return create_batch

def get_image_gen_val(nx, ny, **kwargs):
    def create_batch(n_image):
        
        X = np.zeros((n_image,nx,ny,1))
        Y = np.zeros((n_image,nx,ny,2))
        
        for i in range(n_image):
            #X[i],Y[i,:,:,1] = create_image_and_label(nx,ny, **kwargs)
            X[i],Y[i,:,:,1] = generate_voronoi_diagram_val(nx,ny, **kwargs)
            Y[i,:,:,0] = 1-Y[i,:,:,1]
            
        return X,Y
    
    create_batch.channels = 1
    create_batch.n_class = 2
    return create_batch

def get_image_gen_rgb(nx, ny, **kwargs):
    def create_batch(n_image):
            
            X = np.zeros((n_image, nx, ny, 3))
            Y = np.zeros((n_image, nx, ny,2))
            
            for i in range(n_image):
                x, Y[i,:,:,1] = create_image_and_label(nx,ny, **kwargs)
                X[i] = to_rgb(x)
                Y[i,:,:,0] = 1-Y[i,:,:,1]
                
            return X, Y
    create_batch.channels = 3
    create_batch.n_class = 2
    return create_batch

def to_rgb(img):
    img = img.reshape(img.shape[0], img.shape[1])
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    img /= np.amax(img)
    blue = np.clip(4*(0.75-img), 0, 1)
    red  = np.clip(4*(img-0.25), 0, 1)
    green= np.clip(44*np.fabs(img-0.5)-1., 0, 1)
    rgb = np.stack((red, green, blue), axis=2)
    return rgb

# def create_image_and_label(nx,ny):
#     x = np.floor(np.random.rand(1)[0]*nx).astype(np.int)
#     y = np.floor(np.random.rand(1)[0]*ny).astype(np.int)
# 
#     image = np.ones((nx,ny))
#     label = np.ones((nx,ny))
#     image[x,y] = 0
#     image_distance = ndimage.morphology.distance_transform_edt(image)
# 
#     r = np.random.rand(1)[0]*(r_max-r_min)+r_min
#     plateau = np.random.rand(1)[0]*(plateau_max-plateau_min)+plateau_min
# 
#     label[image_distance <= r] = 0 
#     label[image_distance > r] = 1
#     label = (1 - label)
#     
#     image_distance[image_distance <= r] = 0 
#     image_distance[image_distance > r] = 1
#     image_distance = (1 - image_distance)*plateau
# 
#     image = image_distance + np.random.randn(nx,ny)/sigma
#     
#     return image, label[92:nx-92,92:nx-92]
# 
# def get_image_gen(nx, ny):
#     def create_batch(n_image):
#         
#         X = np.zeros((n_image,nx,ny))
#         Y = np.zeros((n_image,nx-184,ny-184,2))
#         
#         for i in range(n_image):
#             X[i,:,:],Y[i,:,:,1] = create_image_and_label(nx,ny)
#             Y[i,:,:,0] = 1-Y[i,:,:,1]
#             
#         return X,Y
#     return create_batch