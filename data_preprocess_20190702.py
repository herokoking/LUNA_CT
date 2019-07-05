#!/usr/bin/python3
# data preprocess
import numpy as np
import time
import glob
import SimpleITK as sitk
from skimage import morphology, measure, segmentation
import h5py
import pickle
import skimage
import matplotlib.pyplot as plt
import csv
import os
import skimage.io as io
import pandas as pd
import matplotlib.patches as patches


def isflip(self):
    """根据TransformMatrix来判断矩阵图像是否翻转"""
    with open(self.path) as f:
        contents = f.readlines()
    line = [k for k in contents if k.startswith('TransformMatrix')][0]
    transform = np.array(line.split(' = ')[1].split(' ')).astype('float')
    transform = np.round(transform)
    if np.any(transform != np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])):
        isflip = True
    else:
        isflip = False
    return isflip


def world_To_Voxel_Coord(worldCoord, origin, spacing):
    """世界坐标系转到图像中的坐标"""
    stretched_Voxel_Coord = np.absolute(worldCoord - origin)
    voxel_Coord = stretched_Voxel_Coord / spacing
    return voxel_Coord


def voxel_To_WorldCoord(voxelCoord, origin, spacing):
    """世界坐标系转到图像中的坐标系"""
    streched_Vocel_Coord = voxelCoord * spacing
    worldCoord = streched_Vocel_Coord + origin
    return worldCoord


def get_image_spacing_origin(mhd_file):
    itk_img = sitk.ReadImage(mhd_file)
    image = sitk.GetArrayFromImage(itk_img)
    spacing = np.array(itk_img.GetSpacing())
    origin = np.array(itk_img.GetOrigin())
    return itk_img, image, spacing, origin


def make_mask(c, origin, spacing, image):
    width = image.shape[2]
    height = image.shape[1]
    mask = np.zeros([height, width])
    center = np.array(c[1], c[2])
    v_diam_x = c[4] / 2 / spacing[0]
    v_diam_y = c[5] / 2 / spacing[1]
    v_node_x = int(abs(c[1] - origin[0]) / spacing[0])
    v_node_y = int(abs(c[2] - origin[1]) / spacing[1])
    v_xmin = np.max([0, int(v_node_x - v_diam_x) - 1])
    v_xmax = np.min([width - 1, int(v_node_x + v_diam_x) + 1])

    v_ymin = np.max([0, int(v_node_y - v_diam_y) - 1])
    v_ymax = np.min([height - 1, int(v_node_y + v_diam_y) + 1])

    v_xrange = range(v_xmin, v_xmax + 1)
    v_yrange = range(v_ymin, v_ymax + 1)

    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0] * v_x + origin[0]
            p_y = spacing[1] * v_y + origin[1]
            mask[int(abs(p_y - origin[1]) / spacing[1]),int(abs(p_x - origin[0]) / spacing[0])] = 1.0
    return mask


path = glob.glob("./train_part1/*.mhd")

id = []
annotations = np.array(pd.read_csv('./chestCT_round1_annotation.csv',))

i = 0
all_label_list = []
for mhd_file in path:
    # img = io.imread(p, plugin='simpleitk')
    id.append(os.path.basename(mhd_file).split(".")[0])
    itk_img, image, spacing, origin = get_image_spacing_origin(mhd_file)
    name = os.path.basename(mhd_file).split(".")[0]
    # 从全部annotation中获取到当前annotation的注释
    current_annotation = np.copy(annotations[annotations[:, 0] == int(name)])
    index_list = []
    label_list = []
    for c in current_annotation:
        # 坐标系转换
        pos = world_To_Voxel_Coord(c[1:4], origin=origin, spacing=spacing)
        # 这一步不是很确定
        diameterX = c[4] / spacing[0]
        diameterY = c[5] / spacing[1]
        # 获取切片对应的索引
        idx = int(np.absolute(c[3] - origin[-1]) / spacing[-1])
        index_list.append(idx)
        label_list.append(c[7])
        # 病灶中心位置
        node_x = int(abs(c[1] - origin[0]) / spacing[0])
        node_y = int(abs(c[2] - origin[1]) / spacing[1])
        node_z = int(abs(c[3] - origin[2]) / spacing[2])
        v_center = np.array([node_x, node_y, node_z])
        mask=make_mask(c, origin, spacing, image)
        '''
        # 显示每一个切片
        fig, ax = plt.subplots()
        ax.imshow(image[idx], cmap="gray")
        cir = patches.Ellipse(node_x, node_y), width=diameterX,height=diameterY, fill=None,color='red')
        ax.add_patch(cir)
        plt.show()
        plt.imshow(mask, cmap="gray")
        plt.show()
        '''
    b = []
    if len(index_list) == 0:
        continue
    for index in index_list:
        a = np.array(image[index])
        b.append(a)
    arr = np.array(b)
    if arr.shape[1] != 512:
        continue
    # print(arr.shape)
    all_label_list.append(label_list)
    if i == 0:
        all_arr = arr
    else:
        all_arr = np.concatenate((all_arr, arr), axis=0)
    i = i + 1

all_label_list = [i for k in all_label_list for i in k]

print(len(all_label_list), all_arr.shape)
