#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017 - zihao.chen <zihao.chen@moji.com>
'''
Author: zihao.chen
Create Date: 2018-07-03
Modify Date: 2018-07-03
descirption: ""
'''

import datetime
from scipy.interpolate import griddata
import sys
import spopt
import os
import cv2
import numpy as np
import scipy as sy
import matplotlib.pyplot as plt
from config.config import *
import math

from math_utils import calc_direc_amp, update_center

Gray_Threshold = 0


class Extrapolate(object):
    """
    进行图像外推
    """

    def __init__(self):
        self._file_path = ''
        self._time_str = ''
        self._radar_code = ''
        self._time_intervals = []
        self._so = spopt.SparseOptiflow()
        self._radars = {}
        self._extra_num = TIME_RADAR_EXTRAPOLATE
        self.extra_radars = {}
        self.max_time_interval = 5
        self._max_iter = 1
        # self.flow_type = 'dense'
        self.flow_type = 'sparse'
        self.local_ratio = 0.5
        self.data_type = None
        self.error_data = None
        # self._extra_type = 'griddata'#interpolation
        # self._extra_type = 'interpolation'#interpolation

    def set_path(self, file_name_):
        # 设置图片文件路径
        self._file_path = file_name_

    def set_radar_info(self):
        self._time_str = self._file_path[-16:-4]
        self._radar_code = os.path.split(self._file_path)[0][-6:]
        self._sample_code = os.path.split(self._file_path)[0][-19:]


    def set_data_type(self,img_tmp):
        if self.data_type is None:
            black_block = img_tmp[476:]
            print black_block.sum()
            if black_block.sum() <=127755*2:
                self.data_type = 1
                self.error_data = img_tmp[470:]
                temp_data = img_tmp[:470]
                print temp_data.shape
                temp_data[temp_data==255]=0
                return  temp_data
            f_img_tmp = 255-img_tmp
            corp_data = f_img_tmp[21:-21]
            if np.abs(corp_data.sum()*1.0 - f_img_tmp.sum()) < (255. * 200):
                print 2
                self.data_type = 2
                self.error_data = img_tmp
                return img_tmp[22:-22,22:-22]
            self.data_type = 0
            img_tmp[img_tmp == 255] = 0
            return img_tmp
        else:
            if self.data_type == 1:
                temp_data = img_tmp[:470]
                temp_data[temp_data == 255] = 0
                return temp_data
            if self.data_type == 2:
                return img_tmp[22:-22, 22:-22]
            img_tmp[img_tmp == 255] = 0

            return img_tmp


    def _load_picture(self, time_interval=0):
        """根据图片路径加载图片"""
        _file_path = self._get_prev_file_path(time_interval)
        # print 'Loading picture:', _file_path
        if os.path.isfile(_file_path):
            # 0代表读取的是灰度图像
            img_tmp = cv2.imread(_file_path, 0)
            # img_tmp[img_tmp == 255] = 0
            self._radars[time_interval] = self.set_data_type(img_tmp)
            print type(self._radars[time_interval])
            print self._radars[time_interval].shape

            return True
        else:
            print 'Picture', _file_path, 'do not exits!'
            return False

    def _get_prev_file_path(self, _time_interval):
        """根据当前图片路径查找前_time_interval张图片的路径"""
        time_index = int(self._file_path[-7:-4])
        prev_time_index = time_index - _time_interval * 5
        SAMPLE_FORMAT_STR = self._sample_code + '/%s' % (self._sample_code)
        prev_file_path = PATH_TEST_FILE_PATH + SAMPLE_FORMAT_STR
        des_path_ex_ = prev_file_path + '_%03d.png'
        prev_file_path = des_path_ex_ % prev_time_index
        return prev_file_path

    def _extrapolate(self, global_flow_):
        if global_flow_ is not None:
            """利用全局光流场进行外推"""
            self.extra_radars[0] = self._radars[0]
            radar_length = self._radars[0].shape[0]
            grid_x, grid_y = np.meshgrid(range(radar_length), range(radar_length))
            # 为了不使用插值,进而加快时间
            grid_x_shift = {0: grid_x}
            grid_y_shift = {0: grid_y}
            # 为了在外推形变和取整误差之间找到平衡，采取了这种直接外推和递推外推结合的方式
            # 计算一个外推周期内的每步的位置偏移
            for i_iter in range(1, self._max_iter + 1):
                # 矩阵index的排序方式是y，x
                grid_x_shift[i_iter] = (grid_x_shift[i_iter - 1] +
                                        global_flow_[grid_y_shift[i_iter - 1], grid_x_shift[i_iter - 1], 0]).astype(
                    np.int)
                # y值是从上之下，与一般向量的画法不同,通过稠密光流法算出来的场向量与图片矩阵的索引方式相同
                grid_y_shift[i_iter] = (grid_y_shift[i_iter - 1] +
                                        global_flow_[grid_y_shift[i_iter - 1], grid_x_shift[i_iter - 1], 1]).astype(
                    np.int)
                grid_x_shift[i_iter][grid_x_shift[i_iter] < 0] = 0
                grid_x_shift[i_iter][grid_x_shift[i_iter] > radar_length - 1] = radar_length - 1
                grid_y_shift[i_iter][grid_y_shift[i_iter] < 0] = 0
                grid_y_shift[i_iter][grid_y_shift[i_iter] > radar_length - 1] = radar_length - 1
            # 用混合方式进行外推
            for i_cycle in range(1, int(math.ceil(float(self._extra_num) / self._max_iter) + 1)):
                if i_cycle == 1:
                    radar_base = self.extra_radars[0]
                else:
                    radar_base = self.extra_radars[self._max_iter * (i_cycle - 1)]
                for i_iter in range(1, min(self._max_iter, self._extra_num - (i_cycle - 1) * self._max_iter) + 1):
                    radar_tmp = -1 * np.ones_like(radar_base)
                    print radar_tmp.shape
                    print radar_base.shape
                    radar_tmp[grid_y_shift[i_iter], grid_x_shift[i_iter]] = radar_base
                    for width in 5 * np.ones(3, np.int):
                        kernel_ = np.ones((width, width))
                        # 对-1的值由不为-1的点的均值进行插值，为了进行卷积，进行先加1，最后再-1实现计算不为-1点的均值计算
                        non_minus_num = cv2.filter2D((radar_tmp >= 0).astype(np.float32), -1, kernel_)
                        radar_patch = cv2.filter2D(radar_tmp + 1, -1, kernel_) / \
                                      (non_minus_num + 10 ** -10) - 1
                        radar_patch[non_minus_num == 0] = -1
                        radar_tmp = (radar_tmp >= 0) * radar_tmp + (radar_tmp < 0) * radar_patch
                    radar_tmp[radar_tmp < 0] = 0
                    self.extra_radars[(i_cycle - 1) * self._max_iter + i_iter] = radar_tmp
        else:
            # 如果缺图导致无法计算光流，或者本时次就是空白图，则用不动法外推
            for i_extra in range(self._extra_num):
                self.extra_radars[i_extra] = self._radars[0]

    def transform(self, _src_path, time_interval=360):
        self.set_path(_src_path)
        self.set_radar_info()
        print self._load_picture(0)
        print self._radars.keys()
        # 保证图片像素点不会太少
        if self._radars[0].sum() > 100:
            # 尝试装载self.max_time_interval*6分钟内最近的一张图片
            for _time_interval in range(1, self.max_time_interval + 1):
                _picture_exist = self._load_picture(_time_interval)
            # 删去self.radars中的白图片
            for _time_interval in self._radars.keys():
                if self._radars[_time_interval].sum() <= 100:
                    del self._radars[_time_interval]
            self._time_intervals = self._radars.keys()
            print 'There is', len(self._radars), 'non-white pictures in self.radars！'
            if len(self._time_intervals) >= 2:
                global_flow = self._calc_global_flow()
            else:
                global_flow = None
        else:
            global_flow = None
        self._extrapolate(global_flow)
        des_path_ex = self.save_image()
        return des_path_ex

    def gray_filter(self, images, threshold):
        images[images < threshold] = 0
        # images[images == 0] = 255
        if self.data_type ==1:
            images[images == 0] = 255
            images = np.concatenate((images,self.error_data),axis=0)
        elif self.data_type == 2:
            self.error_data[22:-22, 22:-22] = images
            return self.error_data.copy()
        else:
            images[images == 0] = 255
            return images
        return images
    def save_image(self):
        # 分站点存储图片
        SAMPLE_FORMAT_STR = self._sample_code + '/%s' % (self._sample_code)
        prev_file_path = PATH_PREV_FILE_PATH + SAMPLE_FORMAT_STR
        des_path_ex_ = prev_file_path + '_f%03d.png'
        if not os.path.exists(os.path.split(des_path_ex_)[0]):
            os.makedirs(os.path.split(des_path_ex_)[0])
        extra_path = []
        for i_extra in self.extra_radars:
            des_path_ = des_path_ex_ % (i_extra)
            extra_path.append(des_path_)
            self.extra_radars[i_extra] = self.gray_filter(self.extra_radars[i_extra], Gray_Threshold)
            cv2.imwrite(des_path_, self.extra_radars[i_extra])
            print 'Write extrapolate picture:', des_path_
        return extra_path

    def _calc_global_flow(self):
        if self.flow_type == 'dense':
            self._so.set_pic(self._radars[self._time_intervals[0]], self._radars[self._time_intervals[1]])
            dense_flow = self._so.calc_dense_flow()
            scale = self._time_intervals[1] - self._time_intervals[0]
            dense_flow = -dense_flow / scale
        else:
            p0 = []
            p1 = []
            sparse_flow = []
            for i_t in range(len(self._time_intervals) - 1):
                self._so.set_pic(self._radars[self._time_intervals[i_t]], self._radars[self._time_intervals[i_t + 1]])
                p0_tmp, p1_tmp = self._so.calc_corr_points()
                # 防止p0_tmp为空
                if len(p0_tmp) > 0:
                    p0.append(p0_tmp)
                    p1.append(p1_tmp)
                    sparse_flow_tmp = -(p1_tmp - p0_tmp) / float(
                        self._time_intervals[i_t + 1] - self._time_intervals[i_t])
                    sparse_flow.append(sparse_flow_tmp)
            # 防止p0为空
            if len(p0) >= 1:
                p0_stack = np.concatenate(p0, axis=0).astype(np.int)
                sparse_flow_stack = np.concatenate(sparse_flow, axis=0)
                flow_direc_amp = calc_direc_amp(sparse_flow_stack)
                direc_amp_center = update_center(flow_direc_amp)
                # 过滤掉与主矢量差别太大的光流矢量
                index_ = self.flow_filter(direc_amp_center, flow_direc_amp)
                p0_stack = p0_stack[index_]
                sparse_flow_stack = sparse_flow_stack[index_]
                # 防止可用光流场的数目太少
                if len(p0_stack) < 5:
                    print ('Too few sparse optiflow')
                    dense_flow = None
                else:
                    global_flow_ave = sparse_flow_stack.mean(axis=0)
                    print global_flow_ave
                    dense_flow = np.zeros(self._radars[0].shape + (2,))
                    # 图像index是y在前
                    dense_flow[p0_stack[:, :, 1], p0_stack[:, :, 0]] = sparse_flow_stack
                    kernel_ = np.ones((51, 51))
                    for xy in range(2):
                        nonzero_num = cv2.filter2D((np.abs(dense_flow[:, :, xy]) > 0).astype(np.float32), -1, kernel_)
                        dense_flow[:, :, xy] = cv2.filter2D(dense_flow[:, :, xy], -1, kernel_) / \
                                               (nonzero_num + 10 ** -10)
                        dense_flow[nonzero_num < 10 ** -10, xy] = global_flow_ave[0, xy]
                        dense_flow[:, :, xy] = self.local_ratio * dense_flow[:, :, xy] + \
                                               (1 - self.local_ratio) * global_flow_ave[0, xy]
            else:
                dense_flow = None
        return dense_flow

    def flow_filter(self, center_, sparse_flow_, direc_threshold=60, amp_threshold=(0.5, 2)):
        # 角度差小于direc_threshold
        diff_direc = (sparse_flow_[:, :, 0] - center_[0, 0])
        diff_direc = diff_direc * (diff_direc < 180) + (360 - diff_direc) * (diff_direc >= 180)
        index1 = (abs(diff_direc) < direc_threshold).ravel()
        # 速度绝对值比值在amp_threshold之内
        ratio_amp = (sparse_flow_[:, :, 1] / center_[0, 1])
        index2 = ((ratio_amp > amp_threshold[0]) & (ratio_amp < amp_threshold[1])).ravel()
        index_ = index1 & index2
        return index_


def optiflow_histogram(_sparse_optflow):
    print _sparse_optflow.shape
    _flow_direc_amp = calc_direc_amp(_sparse_optflow)
    flow_hist, yedges, xedges = np.histogram2d(_flow_direc_amp[:, :, 0].reshape(-1),
                                               _flow_direc_amp[:, :, 1].reshape(-1),
                                               bins=(list(range(-90, 271, 10)), 40))
    print 'Direction std:', np.std(_flow_direc_amp[:, :, 0])
    print 'Amplitude std:', np.std(_flow_direc_amp[:, :, 1])





if __name__ == '__main__':
    ep = Extrapolate()
    ep.transform(
        '/home/meteo/zihao.chen/data/IEEE_ICDM_2018/download/test_file_001/SRAD2018_Test_1/RAD_296682434212531/RAD_296682434212531_030.png')
