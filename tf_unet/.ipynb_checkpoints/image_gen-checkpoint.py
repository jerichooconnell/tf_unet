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

sigma = 10

plateau_min = -2
plateau_max = 2

r_min = 1
r_max = 200

def generate_voronoi_diagram(width, height, cnt=50):
    
    num_cells = cnt
    
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_path = 'Attenuation.pkl'
    abs_file_path = os.path.join(script_dir, rel_path)
    # This is the part that imports the spectra file
    with open(abs_file_path,'rb') as f:  # Python 3: open(..., 'rb')
        Attenuation, Attenuation2, Spec = pickle.load(f)
        
    Ev = Spec[:,0]
    Int = Spec[:,1]

    image = np.ones((width,height,1))
    image2 = np.zeros([width,height,1])
    labels = np.zeros([width,height],dtype=np.bool)
    imgx, imgy = width, height
    nx = [] # hold the x value
    ny = [] # holds the y value
    nr = [] # low energy image
    nr2 = [] # high energy image
    label = [] # labels whether or not the cell has cartilage

    total_intensity_low = sum(Int[30:48]) # the original intensities
    total_intensity_high = sum(Int[70:])

    for i in range(num_cells):

        # choosing random place for the cell
        nx.append(random.randrange(imgx))
        ny.append(random.randrange(imgy))
        rr = random.random()
        pp = random.random()

        # making a random value for the cartilage
        if i%4==0:
            t_cart = random.uniform(0.5,5)
            label.append(0)
        else:
            label.append(1)
            t_cart = 0.

        # Making a random value for the muscle
        t_muscle = 10 - t_cart

        # Attenuating
        pixel_value = 0.

        # finding the attenuation for the low image
        for ii,energy in enumerate(Ev[30:48]):

            mu = np.interp(energy/1000.,Attenuation[:,0],Attenuation[:,7])
            mu2 = np.interp(energy/1000.,Attenuation2[:,0],Attenuation2[:,7])

            pixel_value += Int[ii+30]*np.exp(-mu*t_muscle*1.05)*np.exp(-mu2*t_cart*1.1)

        nr.append(pixel_value)

        pixel_value = 0.

        # finding the attenuation for the high image
        for ii,energy in enumerate(Ev[70:]):

            mu = np.interp(energy/1000,Attenuation[:,0],Attenuation[:,7])
            mu2 = np.interp(energy/1000,Attenuation2[:,0],Attenuation2[:,7])

            pixel_value += Int[ii+70]*np.exp(-mu*t_muscle*1.05)*np.exp(-mu2*t_cart*1.1)

        nr2.append(pixel_value)


    for y in range(imgy):
        for x in range(imgx):
            dmin = math.hypot(imgx-1, imgy-1)
            j = -1
            for i in range(num_cells):
                d = math.hypot(nx[i]-x, ny[i]-y)
                if d < dmin:
                    dmin = d
                    j = i

            # Generating some poisson noise\
            
            image[x,y,0] = random.gauss(nr[j],(nr[j])**(0.5))#/(total_intensity_low)
            image2[x,y,0] = random.gauss(nr2[j],(nr2[j])**(0.5))#/(total_intensity_high)
            
            image[x,y,0] = image[x,y,0]/image2[x,y,0]
            labels[x,y] = label[j]
            
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