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
import tensorflow as tf
import tensorflow.contrib.slim as slim

def get_disp(x):
    disp = 0.3 * conv(x, 2, 3, 1, tf.nn.sigmoid)
    return disp

def conv(x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
  p = np.floor((kernel_size - 1) / 2).astype(np.int32)
  p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
  return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

def conv_block(x, num_out_layers, kernel_size):
  conv1 = conv(x, num_out_layers, kernel_size, 1)
  conv2 = conv(conv1, num_out_layers, kernel_size, 2)
  return conv2

def maxpool(x, kernel_size):
  p = np.floor((kernel_size - 1) / 2).astype(np.int32)
  p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
  return slim.max_pool2d(p_x, kernel_size)

def resconv(x, num_layers, stride):
  do_proj = tf.shape(x)[3] != num_layers or stride == 2
  shortcut = []
  conv1 = conv(x, num_layers, 1, 1)
  conv2 = conv(conv1, num_layers, 3, stride)
  conv3 = conv(conv2, 4 * num_layers, 1, 1, None)
  if do_proj:
    shortcut = conv(x, 4 * num_layers, 1, stride, None)
  else:
    shortcut = x
  return tf.nn.elu(conv3 + shortcut)

def resblock(x, num_layers, num_blocks):
  out = x
  for i in range(num_blocks - 1):
    out = resconv(out, num_layers, 1)
  out = resconv(out, num_layers, 2)
  return out

def upconv(x, num_out_layers, kernel_size, scale):
  upsample = upsample_nn(x, scale)
  convs = conv(upsample, num_out_layers, kernel_size, 1)
  return convs

def deconv(x, num_out_layers, kernel_size, scale):
  p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
  convs = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
  return convs[:,3:-1,3:-1,:]
  
def upsample_nn(x, ratio):
  s = tf.shape(x)
  h = s[1]
  w = s[2]
  return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])
