#!/usr/bin/python3
#data preprocess tianchi rank22

def normalize(image):
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def world_2_voxel(world_coordinates, origin, spacing):
    stretched_voxel_coordinates = np.absolute(world_coordinates - origin)
    voxel_coordinates = stretched_voxel_coordinates / spacing
    return voxel_coordinates

def voxel_2_world(voxel_coordinates, origin, spacing):
    stretched_voxel_coordinates = voxel_coordinates * spacing
    world_coordinates = stretched_voxel_coordinates + origin
    return world_coordinates


def make_mask(center,diam,z,width,height,spacing,origin): #只显示结节
    mask = np.zeros([height,width]) 
    diam = diam/2
    v_center = np.absolute((center-origin))/spacing
    v_diam = int(diam/spacing[0])
    v_xmin = np.max([0,int(v_center[0]-v_diam)-1])
    v_xmax = np.min([width-1,int(v_center[0]+v_diam)+1])
    v_ymin = np.max([0,int(v_center[1]-v_diam)-1]) 
    v_ymax = np.min([height-1,int(v_center[1]+v_diam)+1])

    v_xrange = range(v_xmin,v_xmax+1)
    v_yrange = range(v_ymin,v_ymax+1)

    x_data = [x*spacing[0]+origin[0] for x in range(width)]
    y_data = [x*spacing[1]+origin[1] for x in range(height)]

    for v_x in v_xrange:
        for v_y in v_yrange:
            p_x = spacing[0]*v_x + origin[0]
            p_y = spacing[1]*v_y + origin[1]
            if np.linalg.norm(center-np.array([p_x,p_y,z]))<=diam:
                mask[np.absolute(int((p_y-origin[1]))/spacing[1]),int((p_x-origin[0])/spacing[0])] = 1.0
    return(mask)

def create_samples(df_node,img_file,pic_path):
    mini_df = df_node[df_node["file"]==img_file] #get all nodules associate with file
    if mini_df.shape[0]>0: # some files may not have a nodule--skipping those 
        # load the data once
        patient_id = img_file.split('/')[-1][:-4]
        itk_img = SimpleITK.ReadImage(data_path + img_file) 
        img_array = SimpleITK.GetArrayFromImage(itk_img) # indexes are z,y,x (notice the ordering)
        num_z, height, width = img_array.shape        #heightXwidth constitute the transverse plane
        origin = np.array(itk_img.GetOrigin())      # x,y,z  Origin in world coordinates (mm)
        spacing = np.array(itk_img.GetSpacing())    # spacing of voxels in world coor. (mm)
        # go through all nodes (why just the biggest?)
        for node_idx, cur_row in mini_df.iterrows():       
            node_x = cur_row["coordX"]
            node_y = cur_row["coordY"]
            node_z = cur_row["coordZ"]
            diam = cur_row["diameter_mm"]
            center = np.array([node_x, node_y, node_z])   # nodule center
            v_center = np.rint(np.absolute(center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
            for i, i_z in enumerate(np.arange(int(v_center[2])-1,
                             int(v_center[2])+2).clip(0, num_z-1)): # clip prevents going out of bounds in Z
                img = img_array[i_z]
                seg_img, overlap = helpers.get_segmented_lungs(img.copy())
                img = normalize(img)
                mask = make_mask(center, diam, i_z*spacing[2]+origin[2],
                                 width, height, spacing, origin)
                if img.shape[0] > 512: 
                    print patient_id
                else:                    
                    cv2.imwrite(pic_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_i.png',img*255)
                    cv2.imwrite(pic_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_m.png',mask*255)
                    cv2.imwrite(pic_path + str(patient_id)+'_'+str(node_idx)+'_'+str(i)+ '_o.png',overlap*255)
    return