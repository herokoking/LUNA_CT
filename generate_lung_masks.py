#!/usr/bin/python3
import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
import matplotlib.pyplot as plt


def extract_lung_mask(img):
    # normalization
    mean = np.mean(img)
    std = np.std(img)
    img = (img - mean) / std
    # 界限模糊
    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    #print(mean, min, max)
    img[img == max] = mean
    img[img == min] = mean
    # 聚类，抽取肺部区
    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    threshold_img = np.where(img < threshold, 1, 0)
    # 先腐蚀&后膨胀
    eroded = morphology.erosion(threshold_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))
    labels = measure.label(dilation)
    # 提取肺部mask
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)  # 获取连通区域
    good_labels = []
    for prop in regions:
        B = prop.bbox
        # print(B)
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    lung_mask = np.ndarray([512, 512], dtype=np.int8)
    lung_mask[:] = 0
    for N in good_labels:
        lung_mask = lung_mask + np.where(labels == N, 1, 0)  # 把属于肺部的连通区填1，其余填0
    lung_mask = morphology.dilation(lung_mask, np.ones([10, 10]))  # one last dilation
    # imgs_to_process[i] = mask
    return lung_mask

all_imgs = np.load("./all_imgs.npy")
all_node_masks = np.load("./all_node_masks.npy")

all_lung_masks = []
for count in range(all_imgs.shape[0]):
    img = all_imgs[count]
    lung_mask = extract_lung_mask(img)
    all_lung_masks.append(lung_mask)

all_lung_masks=np.array(all_lung_masks)
all_lung_masks=all_lung_masks.astype(np.int16)
np.save("all_lung_masks.npy",all_lung_masks)
