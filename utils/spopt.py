#!/usr/bin/env python
# -*- coding:utf-8 -*-
import datetime
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

lk_params = dict(winSize=(35, 35),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=2000,
                      qualityLevel=0.01,
                      minDistance=2,
                      blockSize=11)

FLIP_CHECK = True


class SparseOptiflow(object):
    """计算稀疏光流场"""

    def __init__(self):
        self._p_prev = None
        self._p_next = None

    '''通过传入矩阵设定前后两张图片'''

    def set_pic(self, p_prev, p_next):
        self._p_prev = p_prev
        self._p_next = p_next

    '''计算前后两张图片的对应特征点'''

    def calc_corr_points(self, flip_check=FLIP_CHECK):
        """计算特征点"""
        p0 = cv2.goodFeaturesToTrack(self._p_prev, **feature_params)
        p1, trace_status1 = checked_trace(self._p_prev, self._p_next, p0)
        if flip_check:
            p0_reduced, p1_reduced = p0[trace_status1].copy(), p1[trace_status1].copy()
            # print "Throw points number:", len(trace_status1) - sum(trace_status1)
        else:
            p0_reduced, p1_reduced = p0, p1
        is_move = ((p0_reduced - p1_reduced) ** 2).sum(axis=2).sum(axis=1) > 0
        p0_reduced = p0_reduced[is_move]
        p1_reduced = p1_reduced[is_move]
        return p0_reduced, p1_reduced

    def calc_dense_flow(self):
        n_size = 35
        dense_flow_ = cv2.calcOpticalFlowFarneback(self._p_prev, self._p_next, 0.5, 3, n_size, 3, 5, 1.2, 0)
        # dense_flow_ = cv2.calcOpticalFlowFarneback(self._p_prev, self._p_next, None, 0.5, 3, n_size, 3, 5, 1.2, 0)
        return dense_flow_


def checked_trace(img0, img1, p0, back_threshold=2.0):
    p1, st, err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, st, err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0 - p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status



if __name__ == '__main__':
    p_prev, p_next = find_recent_pair(os.path.join('D:\pic_database_nowcast\denoised\AZ9010', '201606131048.png'))
    # p_prev, p_next = find_recent_pair(os.path.join('D:\pic_database_nowcast\denoised\AZ9357', '201606121554.png'))
    so_ = SparseOptiflow()
    so_.set_pic(p_prev, p_next)
    p0, p1 = so_.calc_corr_points()
    _sparse_optflow = p1 - p0
    dense_flow = so_.calc_dense_flow()
    np.set_printoptions(threshold='nan')
    print np.sqrt((dense_flow ** 2).sum(axis=2)).max(axis=0)
    print np.sqrt(_sparse_optflow ** 2).sum(axis=2)

    print (dense_flow != 0).sum()
    print p0.shape
    print dense_flow.shape
    p1 = plt.subplot(121)
    plt.imshow(p_prev)
    # dense_flow = np.fliplr(dense_flow)
    # dense_flow = np.flipud(dense_flow)
    x, y = np.meshgrid(range(0, 477, 16), range(0, 477, 16))
    plt.quiver(x, y, dense_flow[x, y, 0], dense_flow[x, y, 1], angles='xy', scale_units='xy', scale=0.25)
    p2 = plt.subplot(122)
    plt.imshow(p_prev)
    plt.quiver(p0[:, 0, 0].reshape((1, -1)), p0[:, 0, 1].reshape((1, -1)),
               _sparse_optflow[:, 0, 0].reshape((1, -1)), _sparse_optflow[:, 0, 1].reshape((1, -1)), angles='xy',
               scale_units='xy', scale=0.25)
    plt.show()
