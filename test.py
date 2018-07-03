#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-07-03
Modify Date: 2018-07-03
descirption: ""
'''

import os

from utils.flow_extrapolate import Extrapolate
from config.config import *
work_list= os.listdir(PATH_TEST_FILE_PATH)
type_count = {0:0,1:0,2:0}

for work_name in work_list:
    work_path = os.path.join(PATH_TEST_FILE_PATH,work_name)
    work_file_path = os.path.join(work_path,work_name+'_030.png')
    print work_file_path
    ep = Extrapolate()
    ep.transform(work_file_path)
    print ep.data_type
    type_count[ep.data_type]+=1

print type_count