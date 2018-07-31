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


import tensorflow as tf
import sys
import os
import argparse
import time
import datetime
from utils import *
from trinet import *

# forces tensorflow to run on CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

parser = argparse.ArgumentParser(description='Argument parser')

""" Arguments related to network architecture"""
parser.add_argument('--width', dest='width', type=int, default=512, help='width of input images')
parser.add_argument('--height', dest='height', type=int, default=256, help='height of input images')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', type=str, default='checkpoint/3DV18/3net', help='checkpoint directory')
parser.add_argument('--mode', dest='mode', type=int, default=0, help='Select the demo mode [0: depth-from-mono, 1:view synthesis, 2:stereo]')

# Norm. factors for visualization 
DEPTH_FACTOR = 15
DISP_FACTOR = 6

args = parser.parse_args()

def main(_):

  with tf.Graph().as_default():
    height = args.height
    width = args.width
    placeholders = {'im0':tf.placeholder(tf.float32,[None, None, None, 3], name='im0')}
    model = trinet(placeholders,net='resnet50')

    init = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

    loader = tf.train.Saver()
    saver = tf.train.Saver()
    cam = cv2.VideoCapture(0)

    with tf.Session() as sess:
        sess.run(init)
        loader.restore(sess, args.checkpoint_dir)
        while True:
          for i in range(4):
            cam.grab()
          ret_val, img = cam.read() 

          # Prepare input to the network
          img = cv2.resize(img, (width, height)).astype(np.float32) / 255.
          img_batch = np.expand_dims(img, 0)

          # Run 3net!!!
          start = time.time()
          disp_cr, disp_cl, synt_left, synt_right = sess.run([model.disparity_cr, model.disparity_cl, model.warp_left, model.warp_right], feed_dict={placeholders['im0']: img_batch})
          disp = build_disparity(disp_cr, disp_cl)
          end = time.time()
          
          # Bring back images to uint8 and prepare visualization
          img = (img*255).astype(np.uint8)            
          synt_left = (synt_left*255).astype(np.uint8)
          synt_right = (synt_right*255).astype(np.uint8)          
          disp_color = (applyColorMap(disp*DEPTH_FACTOR, 'plasma')*255).astype(np.uint8)
          toShow_C = np.concatenate((img, disp_color), 1)

          narrow_disp = np.zeros_like(synt_left)
          wide_disp = np.zeros_like(synt_left)
          # If synthetic views are active
          if args.mode==0:
            synt_left = np.zeros_like(synt_left)
            synt_right = np.zeros_like(synt_left)
          else:
            # If SGM is active
            if args.mode>1:
              sgm = cv2.StereoSGBM(0,128,1)
              narrow_disp = sgm.compute(synt_left[:,:,0:1], img[:,:,0:1])
              wide_disp = sgm.compute(synt_left[:,:,0:1], synt_right[:,:,0:1])
              cv2.filterSpeckles(narrow_disp, 0, 4000, 16)
              cv2.filterSpeckles(wide_disp, 0, 4000, 16)
              mask = np.expand_dims(np.uint8(narrow_disp > 0),-1)            
              narrow_disp = cv2.applyColorMap(np.uint8(narrow_disp/DISP_FACTOR), 2)*mask       
              mask = np.expand_dims(np.uint8(wide_disp > 0),-1)            
              wide_disp = cv2.applyColorMap(np.uint8(wide_disp/DISP_FACTOR), 2)*mask

          # Build final output
          toShow_L = (np.concatenate((synt_left, narrow_disp), 1)).astype(np.uint8)
          toShow_R = (np.concatenate((synt_right, wide_disp), 1)).astype(np.uint8)
          toShow = np.concatenate((toShow_L, toShow_C, toShow_R), 0)
          toShow = cv2.resize(toShow, (width*2, height*2))

          # Show cool visualization
          cv2.imshow('3net', toShow)
                    
          k = cv2.waitKey(1)
          if k == 1048685: # 'm' to change mode (shifting 0->1->2->0)
            args.mode = (args.mode+1) % 3
          if k == 1048603 or k == 27: # esc to quit
            break
          if k == 1048688: # 'p' to pause
            cv2.waitKey(0)

          print("Time: " + str(end - start))
        cam.release()        

if __name__ == '__main__':
    tf.app.run()
