#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-07-03
Modify Date: 2018-07-03
descirption: ""
'''

import numpy as np

def update_center(_points, _min_dist=0.5, _direc_scale_factor=0.1):
    """利用距离平方反比为权重，从均值点出发，找到全局的聚类中心"""
    _center = _points.mean(axis=0)
    for count in range(10):
        dist = np.sqrt((((_points - _center) * np.array([_direc_scale_factor, 1])) ** 2).sum(axis=2))
        dist[dist < _min_dist] = _min_dist
        weight = 1 / (dist ** 2)
        weight = weight / sum(weight)
        _center = np.ravel(np.dot(weight.transpose(), _points.transpose([1, 0, 2]))).reshape(1, -1)
    return _center



def calc_direc_amp(_optflow):
    _optflow = _optflow.transpose([0, 2, 1])
    # 计算光流场方向
    _flow_direc = (np.arctan(_optflow[:, 1]/(_optflow[:, 0]+10**-10))
                   + (_optflow[:, 0] < 0)*np.pi)/np.pi*180
    # 计算光流场振幅
    _flow_amp = np.sqrt(_optflow[:, 0]**2 + _optflow[:, 1]**2)
    _flow_direc_amp = np.hstack((_flow_direc, _flow_amp))
    _flow_direc_amp = _flow_direc_amp.reshape(_flow_direc_amp.shape + (1,))
    _flow_direc_amp = _flow_direc_amp.transpose([0, 2, 1])
    return _flow_direc_amp