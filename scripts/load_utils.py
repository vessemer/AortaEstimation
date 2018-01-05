from glob import glob 
import os
import tqdm as tqdm 

import numpy as np
import scipy.ndimage
import stl


def npz_load(path):
    npz_data = np.load(path)
    data = {key: value for (key, value) in npz_data.iteritems()}
    npz_data.close()
    return data


def load_mesh(path, meta):
    """
    Created on Mon Dec  4 14:04:02 2017
    @author: srivak1
    """
    
    #import STL file
    mesh = stl.mesh.Mesh.from_file(path)
    #pull data points associated with mesh
    seg = mesh.points
    #Mesh is defined by a number of triangles. Each triangle has three points defined by 3 coordinates.
    #We will average the coordinates of each triangle to form an outer boundary for the mask
    #I've found that the X and Y coordinates for the STL may be flipped relative to the pixel array in dicoms.
    #be sure that they line up appropriately. This is why first coordinate is made up of 1, 4 and 7 instead of 0, 3 and 6
    
    seg = np.array([[(temp[2] + temp[5] + temp[8]) / 3,
                     (temp[1] + temp[4] + temp[7]) / 3,
                     (temp[0] + temp[3] + temp[6]) / 3] 
                    for temp in seg])
    #stl coordinates are the same as actual space on the CT. I am only adjusting to make them relative to the first
    #slice. You will likely have a different way of doing this. Don't worry about first_vx,first_vy, and first_vz being
    #undefined here
    
    seg = np.abs(seg - meta['ImagePosPat']) 

    #initialize mask the same size as patient data (undefined here)
    mask = np.zeros(np.ceil(meta['Shape'] * meta['PixSpac']).astype(np.uint), dtype=np.bool)
    #extract only unique coordinates from segmentation, as there are a lot of repeats from small triangles
    sub_seg = np.unique(np.round(seg), axis=0).astype(int)
    #for each point in the segmentation, set the mask to 1. You may find a better way to do this
    sub_seg[:, 0] = np.clip(sub_seg[:, 0], 0, mask.shape[0] - 1)
    sub_seg[:, 1] = np.clip(sub_seg[:, 1], 0, mask.shape[1] - 1)
    sub_seg[:, 2] = np.clip(sub_seg[:, 2], 0, mask.shape[2] - 1)
    for g in sub_seg:
        mask[g[0], g[1], g[2]] = True
    #segmentation is only the border. To fill in segmentation, use binary_fill_holes    
    mask = scipy.ndimage.binary_fill_holes(mask)
    mask = scipy.ndimage.zoom(mask, 1 / meta['PixSpac'], order=0)
    return mask


# Load the scans in given folder path
def load_scan(root_dir, patient_id):
    paths = glob(os.path.join(root_dir, patient_id, '*.npz'))
    slices = [npz_load(path) for path in paths]
    slices = sorted(slices, key=lambda x: x['SliceLoc'])
    return slices


def get_pixels_hu(slices):
    image = np.stack([s['dat'] for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image <= -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number]['ResInt']
        slope = slices[slice_number]['ResSlo']
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def load_patient(root_dir, patient_id, metadata=False):
    slices = load_scan(root_dir, patient_id)
    out = get_pixels_hu(slices)
    if metadata:
        meta = dict(slices[0])
        meta['SliceThickness'] = (slices[-1]['SliceLoc'] - slices[0]['SliceLoc']) / (len(slices) - 1)
        meta['PixSpac'] = np.array([meta['SliceThickness']] + meta['PixSpac'][::-1].tolist())
        meta['ImagePosPat'] = meta['ImagePosPat'][::-1]
        meta['Shape'] = np.array(out.shape)
        out = [out, meta]
    return out
