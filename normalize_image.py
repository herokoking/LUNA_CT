#!/usr/bin/python3
#this script is used to normalize image
import numpy as np
from skimage import morphology
from skimage import measure
from sklearn.cluster import KMeans
from skimage.transform import resize
from glob import glob
all_arr_imgs = np.load("./all_arr_imgs.npy")
all_arr_masks=np.load("./all_arr_masks.npy")