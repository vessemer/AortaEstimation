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
    parser.add_argument('idir', type=str, help='input directory')
    parser.add_argument('zpdir', type=str, help='directory should contains zis.npy and prods.npy')
    parser.add_argument('valve_path', type=str, help='path to the valve.csv')
    parser.add_argument('opath', type=str, help='output directory')
    
    parser.add_argument('--labels_path', metavar='LP', type=str, help='path to the REPRISE III Sizes.xlsx')
    parser.add_argument('--exclude_paths', metavar='EP', type=str, help='path to the pickled version of excluded paths')
    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of samples to be processed')
    parser.add_argument('--s', metavar='S', type=int, 
                        help='Skip first S samples')

    args = parser.parse_args()

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

    valve_df = pd.read_csv(args.valve_path)
    exclude_paths = pickle.load(open(args.exclude_paths, 'rb'))
    labels_df = pd.read_excel(args.labels_path)

    ids = valve_df.seriesuid.isin(labels_df['Subject name or identifier'])

    paths = valve_df.seriesuid.apply(lambda x: os.path.join(args.zpdir, x)).values
    paths = paths[ids]
    paths = [el for i, el in enumerate(paths) if not os.path.basename(el) in exclude_paths]

    zis_paths = {os.path.basename(path): os.path.join(path, 'zis.npy') for path in paths}
    prods_paths = {os.path.basename(path): os.path.join(path, 'prods.npy') for path in paths}


    if ars.s:
        paths = paths[args.s:]
    if args.n:
        paths = paths[:args.n]

    for path in tqdm(paths):
        try:
            mask = np.load(os.path.join(args.idir, os.path.basename(path), 'mask.npy'))
            patient = np.load(os.path.join(args.idir, os.path.basename(path), 'patient.npy'))

            prods = np.load(prods_paths[os.path.basename(path)])
            length = int(prods[0].shape[-1] ** .5)

            mis = list()
            for prod in tqdm(prods):
                mi = scipy.ndimage.map_coordinates(mask.astype(np.float), prod)
                mis.append(mi.reshape((length, length)))

            mis = np.stack(mis, 0)

            tangs = list()
            large = list()
            small = list()
            areas = list()

            row = valve_df[valve_df['seriesuid'] == os.path.basename(path)]

            for mi in mis:
                mask_, _ = scipy.ndimage.label(mi > .5)
                try:
                    idx = np.argmax(np.bincount(mask_.flatten())[1:]) + 1
                    mask_ = mask_ == idx
                    rgroup = regionprops(mask_.astype(np.int))[0]
                    large.append(rgroup.major_axis_length * length / SIDE)
                    small.append(rgroup.minor_axis_length * length / SIDE)
                    areas.append(rgroup.area * ((length / SIDE) ** 2))
                except:
                    tangs.append(0)
                    continue

            tangs = np.array([0 if np.isnan(l) or np.isnan(s) else s for l, s in zip(large, small)])
            large = np.array([0 if np.isnan(l) or np.isnan(s) else l for l, s in zip(large, small)])
            small = tangs.copy()

            idx0 = row['idx0'].values[0]
            idx1 = row['idx1'].values[0]

            gm = GaussianMixture(n_components=2)
            gm.fit(patient[
                :mask.shape[0], 
                :mask.shape[1], 
                :mask.shape[2]
            ][mask[:patient.shape[0], :patient.shape[1], :patient.shape[2]]].reshape(-1, 1))

            if labels_df[labels_df['Subject name or identifier'] == os.path.basename(path)].shape[0] == 1:
                label = labels_df[labels_df['Subject name or identifier'] == os.path.basename(path)]['Valve Size (Model number)']
                valve_df.loc[valve_df['seriesuid'] == os.path.basename(path), 'class'] = label.values[0]

            for feature, el in zip(features_names, 
                                   [large[idx0], small[idx0], areas[idx0], 
                                    large[idx1], small[idx1], areas[idx1], 
                                    large[idx0: idx1].max(), 
                                    tangs[idx0: idx1].max(), 
                                    max(areas[idx0: idx1]), 
                                    min(areas[idx0: idx1]),
                                    gm.means_.min(),
                                    gm.means_.max()]):
                valve_df.loc[valve_df['seriesuid'] == os.path.basename(path), feature] = el

            features = pd.merge(
                labels_df[['Subject name or identifier', 'Valve Size (Model number)']], 
                valve_df, 
                left_on=['Subject name or identifier'], 
                right_on=['seriesuid'],
            )

            features['class'] = features['Valve Size (Model number)']
            features = features.drop(['Valve Size (Model number)', 'Subject name or identifier', 'idx0', 'idx1'], axis=1)
            features = features.dropna()
            features['ratio_idx0'] = features.longs_0 / features.shorts_0 
            features['ratio_idx1'] = features.longs_1 / features.shorts_1 
            features['gm_diff'] = features.gm_min - features.gm_max
            features.to_csv(args.opath, index=False)
        except:
            print(path)
    print(features.head())