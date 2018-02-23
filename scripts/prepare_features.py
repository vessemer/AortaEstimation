import argparse
from glob import glob 
import os
from tqdm import tqdm as tqdm 
from itertools import product

from sklearn.mixture import GaussianMixture
import numpy as np
import scipy.ndimage

import sys
sys.path.append('../scripts')
sys.path.append('../nets')
import load_utils
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


from skimage import measure
from skimage.morphology import convex_hull_image

import pickle
from skimage.measure import regionprops


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare features for valve size classification.')
    parser.add_argument('patdir', type=str, help='input directory should contains patients\' CT scans')
    parser.add_argument('mdir', type=str, help='masks input directory')
    parser.add_argument('ndir', type=str, help='masks input directory')
    parser.add_argument('vdir', type=str, help='masks input directory')
    parser.add_argument('opath', type=str, help='path to output csv file')

    parser.add_argument('--exclude_paths', metavar='EP', type=str, help='path to the pickled version of excluded paths')
    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of samples to be processed')
    parser.add_argument('--s', metavar='S', type=int, 
                        help='Skip first S samples')

    args = parser.parse_args()

    # create a directory if there isn't any
    try:
        os.mkdir(os.path.dirname(args.opath))
    except:
        pass

    features_names = [
        'longs_0', 'shorts_0', 'ratio_0',
        'longs_1', 'shorts_1', 'ratio_1',
        'large_max', 'tangs_max', 'areas_max',
        'areas_min', 'gm_min', 'gm_max', 'class'
    ]

    SIDE = 224

    try:
        exclude_paths = pickle.load(open(args.exclude_paths, 'rb'))
    except:
        pass

    try:
        labels_df = pd.read_excel(args.labels_path)
    except:
        pass

    # extracted patients ids inside os a given `idir`
    # leave only those which have extrated normals in `ndir`
    # and predicted valve in `vdir`
    paths = glob(os.path.join(args.ndir, 'prods', '*'))
    ids = [os.path.basename(path) for path in paths]
    ids = [pid for pid in ids if os.path.isfile(os.path.join(args.ndir, 'slices', pid))]
    ids = [pid for pid in ids if os.path.isfile(os.path.join(args.ndir, 'masks', pid))]
    ids = [pid for pid in ids if os.path.isfile(os.path.join(args.vdir, pid))]
    ids = [pid for pid in ids if os.path.isfile(os.path.join(args.patdir, pid))]

    if args.s:
        paths = paths[args.s:]
    if args.n:
        paths = paths[:args.n]

    # create features_df to store features
    seriesuid = pd.Series([pid.split('.npy')[0] for pid in ids])
    features_df = pd.DataFrame()
    features_df['seriesuid'] = seriesuid

    # extract features for each patient id
    for pid in tqdm(ids):
        # load patient along with predicted aorta's mask
        patient = np.load(os.path.join(args.patdir, pid))
        mask = np.load(os.path.join(args.mdir, pid)) > .5
        
        # prods is coordinates of format: [N, 3, length * length], where N is amount of cropped slices
        prods = np.load(os.path.join(args.ndir, 'prods', pid))
        # stacked masks extracted along the planes normal to curve
        mis = np.load(os.path.join(args.ndir, 'masks', pid))
        planes = np.load(os.path.join(args.ndir, 'planes', pid))
        
        # load valve mask
        vmask = np.load(os.path.join(args.vdir, pid))

        # get lower and upper bounds for valve
        idxs = np.where(vmask)[0]
        # idx1 should depict an idx of annulus plane
        idx0, idx1 = idxs.min(), idxs.max()
        
        # create a directory if there isn't any
        try:
            os.mkdir(os.path.join(args.ndir, "annulus"))
        except:
            pass

        # annulu plane are in format: [origin_x, origin_y, origin_z, vector_x, vector_y, vector_z]
        # plane is consisted of origin-{x, y, z} point in 3D and vector-{x, y, z} (plane's normal vector) 
        np.save(os.path.join(args.ndir, "annulus", pid), planes[idx1])

        length = int(prods[0].shape[-1] ** .5)

        tangs = list()
        large = list()
        small = list()
        areas = list()

        # extracts features for each cropped plane
        for mi in mis:
            # label to leave the biggest connected component (in case of noise)
            mask_, _ = scipy.ndimage.label(mi > .5)
            try:
                idx = np.argmax(np.bincount(mask_.flatten())[1:]) + 1
                mask_ = mask_ == idx
                # extract major_axis_length, minor_axis_length, area of connected component
                rgroup = regionprops(mask_.astype(np.int))[0]
                large.append(rgroup.major_axis_length * length / SIDE)
                small.append(rgroup.minor_axis_length * length / SIDE)
                areas.append(rgroup.area * ((length / SIDE) ** 2))
            except:
                tangs.append(0)
                continue

        # Cut off zeros (outside of an aorta's mask)
        tangs = np.array([0 if np.isnan(l) or np.isnan(s) else s for l, s in zip(large, small)])
        large = np.array([0 if np.isnan(l) or np.isnan(s) else l for l, s in zip(large, small)])
        small = tangs.copy()

        # compute gaussian features 
        gm = GaussianMixture(n_components=2)
        gm.fit(patient[
            :mask.shape[0], 
            :mask.shape[1], 
            :mask.shape[2]
        ][mask[:patient.shape[0], :patient.shape[1], :patient.shape[2]]].reshape(-1, 1))

        # gather collected fetures and add them to `features_df`
        for feature, el in zip(features_names, 
                               [large[idx0], small[idx0], areas[idx0], 
                                large[idx1], small[idx1], areas[idx1], 
                                large[idx0: idx1].max(), 
                                tangs[idx0: idx1].max(), 
                                max(areas[idx0: idx1]), 
                                min(areas[idx0: idx1]),
                                gm.means_.min(),
                                gm.means_.max()]):
            features_df.loc[features_df['seriesuid'] == pid.split('.npy')[0], feature] = el
            
        features_df = features_df.dropna()
        features_df['ratio_idx0'] = features_df.longs_0 / features_df.shorts_0 
        features_df['ratio_idx1'] = features_df.longs_1 / features_df.shorts_1 
        features_df['gm_diff'] = features_df.gm_min - features_df.gm_max

        # save `features_df` on each iteration
        features_df.to_csv(args.opath, index=False)
