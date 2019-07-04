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
    kmeans = KMeans(n_clusters=2).fit(
        np.reshape(middle, [np.prod(middle.shape), 1]))
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
    lung_mask = morphology.dilation(
        lung_mask, np.ones([10, 10]))  # one last dilation
    # imgs_to_process[i] = mask
    return lung_mask


def resize_lung(img, lung_mask, node_mask):
    # we're scaling back up to the original size of the image
    new_size = [512, 512]
    img = lung_mask * img          # apply lung mask
    # 肺部区域均值和标准差
    new_mean = np.mean(img[lung_mask > 0])
    new_std = np.std(img[lung_mask > 0])
    # 重设背景颜色，这一步不明白为什么？
    old_min = np.min(img)       # background color
    img[img == old_min] = new_mean - 1.2 * \
        new_std   # resetting backgound color
    # normalization lung region
    img = img - new_mean
    img = img / new_std
    labels = measure.label(lung_mask)
    regions = measure.regionprops(labels)
    min_row = 512
    max_row = 0
    min_col = 512
    max_col = 0
    for prop in regions:
        B = prop.bbox
        if min_row > B[0]:
            min_row = B[0]
        if min_col > B[1]:
            min_col = B[1]
        if max_row < B[2]:
            max_row = B[2]
        if max_col < B[3]:
            max_col = B[3]
    width = max_col - min_col
    height = max_row - min_row
    # #print(width,height)
    if width > height:
        max_row = min_row + width
    else:
        max_col = min_col + height
    img = img[min_row:max_row, min_col:max_col]
    lung_mask = lung_mask[min_row:max_row, min_col:max_col]
    if max_row - min_row < 5 or max_col - min_col < 5:  # skipping all images with no god regions
        pass
    else:
        # moving range to -1 to 1 to accomodate the resize function
        mean = np.mean(img)
        img = img - mean
        min = np.min(img)
        max = np.max(img)
        img = img / (max - min)
        new_img = resize(img, [512, 512])
        new_lung_mask = resize(lung_mask, [512, 512])
        new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], [
                               512, 512])  # 肺结节mask也相应裁剪和resize
    return (new_img, new_lung_mask, new_node_mask)


all_imgs = np.load("./all_arr_imgs.npy")
all_node_masks = np.load("./all_arr_masks.npy")
all_new_imgs = []
all_new_lung_masks = []
all_new_node_masks = []
for count in range(all_imgs.shape[0]):
    img = all_imgs[count]
    node_mask = all_node_masks[count]
    lung_mask = extract_lung_mask(img)
    new_img, new_lung_mask, new_node_mask = resize_lung(
        img, lung_mask, node_mask)
    all_new_imgs.append(new_img)
    all_new_lung_masks.append(new_lung_mask)
    all_new_node_masks.append(new_node_mask)
    if count > 2:
        break
all_new_imgs = np.array(all_new_imgs)
all_new_lung_masks = np.array(all_new_lung_masks)
all_new_node_masks = np.array(new_node_mask)
