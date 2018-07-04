#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-07-02
Modify Date: 2018-07-02
descirption: "some utils"
'''

import pickle

import numpy as np
from PCV.tools import imtools, pca
from PIL import Image
from pylab import *
from scipy import *
from scipy.cluster.vq import *

temp_image_path = "G:\python\pcv_data\data\selectedfontimages\\a_selected_thumbs"

imlist = imtools.get_imlist(temp_image_path)
print imlist[0]

# 获取图像列表和他们的尺寸
im = np.array(Image.open(imlist[0])) 
print type(im)
m, n = im.shape[:2]  
imnbr = len(imlist)  
print "The number of images is %d" % imnbr


immatrix = np.array([np.array(Image.open(imname)).flatten() for imname in imlist[:10]], 'f')
print immatrix.shape

V, S, immean = pca.pca(immatrix)
print V[0][:10]