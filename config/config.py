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
RADAR_TIME_INTERVAL = 360 * 5
DATA_PATH ="/home/meteo/zihao.chen/data/IEEE_ICDM_2018/download/"
PATH_TEST_FILE_PATH = os.path.join(DATA_PATH,'test_file_001/SRAD2018_Test_1/')
PATH_PREV_FILE_PATH = os.path.join(DATA_PATH,'extrapolate_flow')

TIME_RADAR_EXTRAPOLATE = 6