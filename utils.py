#
# MIT License
#
# Copyright (c) 2018 Matteo Poggi m.poggi@unibo.it
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from matplotlib import cm
from bilinear_sampler import *
import cv2

# Colormap wrapper
def applyColorMap(img, cmap):
  colormap = cm.get_cmap(cmap) 
  colored = colormap(img)
  return np.float32(cv2.cvtColor(np.uint8(colored*255),cv2.COLOR_RGBA2BGR))/255.

# 2D convolution wrapper
def count_text_lines(file_path):
  f = open(file_path, 'r')
  lines = f.readlines()
  f.close()
  return len(lines)    

#######################################
# Suite of utility functions, 
# credits mostly to Clement Godard
#######################################
  
def post_process_disparity(disp):
  _, h, w = disp.shape
  l_disp = disp[0,:,:]
  r_disp = np.fliplr(disp[1,:,:])
  m_disp = 0.5 * (l_disp + r_disp)
  l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
  l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
  r_mask = np.fliplr(l_mask)
  return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def reverse_post_process_disparity(disp):
  _, h, w = disp.shape
  l_disp = np.fliplr(disp[0,:,:])
  r_disp = disp[1,:,:]
  m_disp = 0.5 * (l_disp + r_disp)
  l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
  l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
  r_mask = np.fliplr(l_mask)
  return np.fliplr(r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp)  

def build_disparity(disp_cr, disp_cl):
  disparity = post_process_disparity(np.stack([disp_cr, np.fliplr(disp_cl)],0).squeeze())
  return disparity 

def build_disparity_pp(disp_cr, disp_cl): 
  disparity = post_process_disparity(np.stack([post_process_disparity(disp_cr.squeeze()),np.fliplr(reverse_post_process_disparity(disp_cr.squeeze()))],0).squeeze())
  return disparity
  
def generate_image_left(img, disp):
  return bilinear_sampler_1d_h(img, -disp)

def generate_image_right(img, disp):
  return bilinear_sampler_1d_h(img, disp)
    
