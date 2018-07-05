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
import multiprocessing
from utils.flow_extrapolate import Extrapolate
from config.config import *
work_list= os.listdir(PATH_TEST_FILE_PATH)
type_count = {0:0,1:0,2:0}

def extrapolate_single(file_path):
    ep = Extrapolate()
    ep.transform(file_path)
    print 'data_type: %d'%ep.data_type
    old_path = os.path.split(file_path)[0]
    new_path = old_path.replace('SRAD2018_Test_1','SRAD2018_Test_1_%d'%ep.data_type)
    os.makedirs(new_path)
    os.system('cp -r %s/* %s'%(old_path,new_path))
# all_param = []
#
#
# for work_name in work_list:
#     work_path = os.path.join(PATH_TEST_FILE_PATH,work_name)
#     work_file_path = os.path.join(work_path,work_name+'_030.png')
#     print work_file_path
#     all_param.append(work_file_path)
#
#     # type_count[ep.data_type]+=1
# pool = multiprocessing.Pool(14)
#
# pool.map(extrapolate_single, all_param)
# pool.close()
# pool.join()
# print type_count

extrapolate_single('/home/meteo/zihao.chen/data/IEEE_ICDM_2018/download/test_file_001/SRAD2018_Test_1/RAD_516482464219551/RAD_516482464219551_030.png')