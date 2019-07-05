#!/usr/bin/python3
#data preprocess
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
	voxel_Coord = stretched_Voxel_Coord/spacing
	return voxel_Coord


def voxel_To_WorldCoord(voxelCoord, origin, spacing):
	"""世界坐标系转到图像中的坐标系"""
	streched_Vocel_Coord = voxelCoord*spacing
	worldCoord = streched_Vocel_Coord + origin
	return worldCoord


def get_image_spacing_origin(path):
	itk_img = sitk.ReadImage(path)
	image = sitk.GetArrayFromImage(itk_img)
	spacing = np.array(itk_img.GetSpacing())
	origin = np.array(itk_img.GetOrigin())
	return itk_img, image, spacing, origin

if __name__ == "__main__":
	path = glob.glob("F:/tianchi/train_part1/*.mhd")
	id = []
	annotations = np.array(pd.read_csv('F:/tianchi/chestCT_round1_annotation.csv',))
	for p in path:
		# img = io.imread(p, plugin='simpleitk')
		id.append(os.path.basename(p).split(".")[0])
		itk_img, image, spacing, origin = get_image_spacing_origin(p)
		name = os.path.basename(p).split(".")[0]

		### 从全部annotation中获取到当前annotation的注释
		current_annotation = np.copy(annotations[annotations[:, 0] == int(name)])
		for c in current_annotation:
			## 坐标系转换
			pos = world_To_Voxel_Coord(c[1:4], origin=origin, spacing=spacing)

			## 这一步不是很确定
			diameterX = c[4]/spacing[0]
			diameterY = c[5]/spacing[1]

			## 获取切片对应的索引
			idx = int(np.absolute(c[3] - origin[-1])/spacing[-1])

			## 显示每一个切片
			fig, ax = plt.subplots()
			ax.imshow(image[idx], cmap="gray")
			cir = patches.Ellipse(xy=(pos[0], pos[1]), width=diameterX,height=diameterY, fill=None)
			ax.add_patch(cir)
			plt.show()