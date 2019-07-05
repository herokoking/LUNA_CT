#!/usr/bin/python
#this script is used to resize the img, lung_mask, node_mask

def resize_lung(img, lung_mask, node_mask):
    # we're scaling back up to the original size of the image
    new_size = [512, 512]
    img = lung_mask * img          # apply lung mask
    # 肺部区域均值和标准差
    new_mean = np.mean(img[lung_mask > 0])
    new_std = np.std(img[lung_mask > 0])
    # 重设背景颜色，这一步不明白为什么？
    old_min = np.min(img)       # background color
    img[img == old_min] = new_mean - 1.2 * new_std   # resetting backgound color
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
    mean = np.mean(img)
    img = img - mean
    min = np.min(img)
    max = np.max(img)
    img = img / (max - min)
    new_img = resize(img, [512, 512])
    new_lung_mask = resize(lung_mask, [512, 512])
    new_node_mask = resize(node_mask[min_row:max_row, min_col:max_col], [512, 512])  # 肺结节mask也相应裁剪和resize
    return (new_img, new_lung_mask, new_node_mask)

all_imgs = np.load("./all_arr_imgs.npy")
all_node_masks = np.load("./all_arr_masks.npy")
all_lung_masks=np.load("./all_lung_masks.npy")

all_new_imgs=[]
all_new_node_masks=[]
all_new_lung_masks=[]

for count in range(all_imgs.shape[0]):
    img=all_imgs[count]
    lung_mask=all_lung_masks[count]
    node_mask=all_node_masks[count]
    if len(np.unique(lung_mask))==2:
        new_img, new_lung_mask, new_node_mask=resize_lung(img,lung_mask,node_mask)
        all_new_imgs.append(new_img)
        all_lung_masks.append(new_lung_mask)
        all_new_node_masks.append(new_node_mask)
all_new_imgs=np.array(all_new_imgs)
all_new_lung_masks=np.array(all_new_lung_masks).astype(np.int8)
all_new_node_masks=np.array(all_new_node_masks).astype(np.int8)

np.save("all_new_imgs.npy",all_new_imgs)
np.save("all_new_lung_masks.npy",all_new_lung_masks)
np.save("all_new_node_masks.npy",all_new_node_masks)

