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

from layers import *
from utils import *

class trinet(object):

    def __init__(self, placeholders=None, net='vgg'):
        self.model_collection = ['3net']    
        self.placeholders = placeholders
        self.build_model(net)
        self.build_output()
    
    # Build model
    def build_model(self,net): 
        with tf.variable_scope('model') as scope:    
          with tf.variable_scope('shared-encoder') as encoder_scope:
            features_cr = self.build_encoder(self.placeholders['im0'],model_name=net)
            features_cl = features_cr
          with tf.variable_scope('encoder-C2R'):
            self.disp_cr = self.build_decoder(features_cr,model_name=net)
          with tf.variable_scope('encoder-C2L'):
            self.disp_cl = self.build_decoder(features_cl,model_name=net)
              
    def build_output(self):
        self.disparity_cr = self.disp_cr[0][0,:,:,0]
        self.disparity_cl = self.disp_cl[0][0,:,:,0]
        self.warp_left = generate_image_left(self.placeholders['im0'], self.disparity_cl)[0]
        self.warp_right = generate_image_right(self.placeholders['im0'], self.disparity_cr)[0]
      
      
    # Build shared encoder
    def build_encoder(self, model_input, model_name='vgg'):

        with tf.variable_scope('encoder'):
          if model_name == 'vgg':
            conv1 = conv_block(model_input,  32, 7) # H/2
            conv2 = conv_block(conv1,             64, 5) # H/4
            conv3 = conv_block(conv2,            128, 3) # H/8
            conv4 = conv_block(conv3,            256, 3) # H/16
            conv5 = conv_block(conv4,            512, 3) # H/32
            conv6 = conv_block(conv5,            512, 3) # H/64
            conv7 = conv_block(conv6,            512, 3) # H/128    
            return conv7, conv1, conv2, conv3, conv4, conv5, conv6

          elif model_name == 'resnet50':
            conv1 = conv(model_input, 64, 7, 2) # H/2  -   64D
            pool1 = maxpool(conv1,           3) # H/4  -   64D
            conv2 = resblock(pool1,      64, 3) # H/8  -  256D
            conv3 = resblock(conv2,     128, 4) # H/16 -  512D
            conv4 = resblock(conv3,     256, 6) # H/32 - 1024D
            conv5 = resblock(conv4,     512, 3) # H/64 - 2048D
            return conv5, conv1, pool1, conv2, conv3, conv4      

    def build_decoder(self, skip, model_name='vgg'):

        with tf.variable_scope('decoder'):
          if model_name == 'vgg':           
            upconv7 = upconv(skip[0],  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip[6]], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip[5]], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip[4]], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip[3]], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip[2], udisp4], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip[1], udisp3], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            disp1 = get_disp(iconv1)

          elif model_name == 'resnet50':            
            upconv6 = upconv(skip[0],   512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip[5]], 3)
            iconv6  = conv(concat6,   512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip[4]], 3)
            iconv5  = conv(concat5,   256, 3, 1)

            upconv4 = upconv(iconv5,  128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip[3]], 3)
            iconv4  = conv(concat4,   128, 3, 1)
            disp4 = get_disp(iconv4)
            udisp4  = upsample_nn(disp4, 2)

            upconv3 = upconv(iconv4,   64, 3, 2) #H/4
            concat3 = tf.concat([upconv3, skip[2], udisp4], 3)
            iconv3  = conv(concat3,    64, 3, 1)
            disp3 = get_disp(iconv3)
            udisp3  = upsample_nn(disp3, 2)

            upconv2 = upconv(iconv3,   32, 3, 2) #H/2
            concat2 = tf.concat([upconv2, skip[1], udisp3], 3)
            iconv2  = conv(concat2,    32, 3, 1)
            disp2 = get_disp(iconv2)
            udisp2  = upsample_nn(disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            concat1 = tf.concat([upconv1, udisp2], 3)
            iconv1  = conv(concat1,   16, 3, 1)
            disp1 = get_disp(iconv1)

          return disp1, disp2, disp3, disp4    
