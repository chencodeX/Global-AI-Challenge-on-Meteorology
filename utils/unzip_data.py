#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-07-01
Modify Date: 2018-07-01
descirption: ""
'''

import os


from config import DATA_PATH
os.chdir(DATA_PATH)
for i in range(2,11):
    os.system('mkdir train_file_%03d'%i)
    os.system('mv train_file_%03d.zip train_file_%03d' % (i,i))
    os.system('cd train_file_%03d'%i)
    os.system('unzip train_file_%03d.zip'%i)
    os.system('cd ..')