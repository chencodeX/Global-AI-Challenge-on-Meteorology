#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-07-02
Modify Date: 2018-07-02
descirption: ""
'''
import os
import shutil
from config import *

test_path = os.path.join(DATA_PATH,'SRAD2018_Test_1')
P_path = os.path.join(DATA_PATH,'up_test')
samples = os.listdir(test_path)

for sample in samples:
    temp_data_path = os.path.join(test_path,sample)
    temp_last_image_path = os.path.join(temp_data_path,sample+'_030.png')
    if os.path.exists(temp_last_image_path):
        up_load_data_path = temp_data_path.replace('SRAD2018_Test_1','up_test')
        os.makedirs(up_load_data_path)
        for i in range(6):
            save_path = os.path.join(up_load_data_path,sample+'_f00%d.png'%(i+1))
            shutil.copy(temp_last_image_path,save_path)
