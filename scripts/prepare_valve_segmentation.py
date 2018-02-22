import argparse

import numpy as np
from tqdm import tqdm

import pandas as pd
import scipy.ndimage
from glob import glob
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for valve segmentation.')

    parser.add_argument('ndir', type=str, help='directory should contains prods & slices directories (output from extract_normals)')
    parser.add_argument('mdir', type=str, help='directory with valve masks')
    parser.add_argument('patdir', type=str, help='directory with CTs, processed w.r.t. valve masks')
    parser.add_argument('odir', type=str, help='output directory')
        
    parser.add_argument('--valvecsv', type=str, help='csv file: [seriesuid, idx0, idx1] w.r.t. processed CTs')
    parser.add_argument('--test', metavar='T', type=bool, 
                        help='Process in test mode')
    parser.add_argument('--s', metavar='S', type=int, 
                        help='Skip first S samples')
    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of samples to be processed')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.odir))
    except:
        pass

    paths = glob(os.path.join(args.ndir, 'prods', '*'))
    ids = [os.path.basename(path) for path in paths]
    ids = [pid for pid in ids if os.path.isfile(os.path.join(args.ndir, 'slices', pid))]
    if not args.test:
        valve_df = pd.read_csv(args.valvecsv)
        ids = [pid for pid in ids if pid.split('.npy')[0] in valve_df.seriesuid.values]

    if args.s:
        ids = ids[args.s:]
    if args.n:
        ids = ids[:args.n]

    for idx, pid in tqdm(enumerate(ids), total=len(ids)):
        patient = np.load(os.path.join(args.patdir, pid))
        mask = np.load(os.path.join(args.mdir, pid)) > .5
        prods = np.load(os.path.join(args.ndir, 'prods', pid))
        
        if not args.test:
            imask = np.zeros_like(mask)
            values = scipy.ndimage.map_coordinates(mask, np.rollaxis(prods, 1, 0).reshape(3, -1))
            values = values.reshape((prods.shape[0], prods.shape[-1]))
            idx0, idx1 = valve_df[valve_df.seriesuid == pid.split('.npy')[0]][['idx0', 'idx1']].values[0]

            for i in range(idx0, idx1):
                coords = np.array([prods[i][0], prods[i][1], prods[i][2]])
                coords = np.clip(coords.T, 0, np.array(mask.shape) - 1)
                coords = np.round(coords[values[i] > .5]).astype(np.int).T
                coords = tuple(c for c in coords)

                imask[coords] = True

            imask = mask & scipy.ndimage.binary_closing(imask, iterations=8)

            if not imask.sum():
                print('Empty mask: ', pid)
                continue

        prods_ = np.rollaxis(prods, 1, 0).reshape(3, -1)
        mapped_mask = scipy.ndimage.map_coordinates(mask.astype(np.float), prods_, order=0) > .5
        if not args.test:
            imapped_mask = scipy.ndimage.map_coordinates(imask.astype(np.float), prods_, order=0) > .5
        mapped_patient = scipy.ndimage.map_coordinates(patient.astype(np.float), prods_)

        mapped_mask = mapped_mask.reshape((prods.shape[0], int(prods.shape[-1] ** .5), int(prods.shape[-1] ** .5)))
        if not args.test:
            imapped_mask = imapped_mask.reshape((prods.shape[0], int(prods.shape[-1] ** .5), int(prods.shape[-1] ** .5)))
        mapped_patient = mapped_patient.reshape((prods.shape[0], int(prods.shape[-1] ** .5), int(prods.shape[-1] ** .5)))
        mapped_patient = ((mapped_patient + 199.) / 461.) * mapped_mask
        if not args.test:
            mapped_mask = imapped_mask

        if not mapped_mask.sum():
            print(idx, pid, ' empty')
            continue

        try:
            os.mkdir(os.path.join(args.odir, pid.split('.npy')[0]))
        except:
            pass

        z, x, y = np.where(mapped_mask)
        x_mean, y_mean = (x.mean(), y.mean())
        x_std, y_std = (x.std(), y.std())

        for i in range(10):
            try:
                x_point = int(x_mean + np.random.uniform(-x_std, x_std))
                y_point = int(y_mean + np.random.uniform(-y_std, y_std))
            except:
                print(idx, pid)
                for i in range(10):
                    try:
                        os.remove(os.path.join(args.odir, pid.split('.npy')[0], 'patch_' + str(i)))
                    except:
                        pass

            if np.random.randint(2):
                patch = mapped_patient[:, x_point]
                mask = mapped_mask[:, x_point]
            else:
                patch = mapped_patient[:, :, y_point]
                mask = mapped_mask[:, :, y_point]

            np.save(os.path.join(args.odir, pid.split('.npy')[0], 'patch_' + str(i)), patch)
            if not args.test:
                np.save(os.path.join(args.odir, pid.split('.npy')[0], 'mask_' + str(i)), mask)
