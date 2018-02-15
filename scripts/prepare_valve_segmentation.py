import argparse

import numpy as np
from tqdm import tqdm

import scipy.ndimage
from glob import glob
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for valve segmentation.')
    parser.add_argument('idir', type=str, help='input directory (should contains zis.npy and prods.npy)')
    parser.add_argument('mdir', type=str, help='directory with valve masks')
    parser.add_argument('odir', type=str, help='output directory')
    
    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of samples to be processed')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.odir))
    except:
        pass

    paths = glob(os.path.join(args.idir, '*', 'zis.npy'))
    ids = [os.path.basename(os.path.dirname(path)) for path in paths]
    ids = [pid for pid in ids if os.path.isfile(os.path.join(args.idir, pid, 'prods.npy'))]
    ids = [pid for pid in ids if os.path.isfile(os.path.join(args.mdir, pid, 'valve_mask.npy'))]
    if args.n:
        ids = ids[:args.n]

    for idx, pid in tqdm(enumerate(ids), total=len(ids)):
        patient = np.load(os.path.join(args.mdir, pid, 'patient.npy'))
        mask = np.load(os.path.join(args.mdir, pid, 'mask.npy')) > .5
        imask = np.load(os.path.join(args.mdir, pid, 'valve_mask.npy')) > .5
        prods = np.load(os.path.join(args.idir, pid, 'prods.npy'))

        prods_ = np.rollaxis(prods, 1, 0).reshape(3, -1)
        mapped_mask = scipy.ndimage.map_coordinates(mask.astype(np.float), prods_, order=0) > .5
        imapped_mask = scipy.ndimage.map_coordinates(imask.astype(np.float), prods_, order=0) > .5
        mapped_patient = scipy.ndimage.map_coordinates(patient.astype(np.float), prods_)

        mapped_mask = mapped_mask.reshape((prods.shape[0], int(prods.shape[-1] ** .5), int(prods.shape[-1] ** .5)))
        imapped_mask = imapped_mask.reshape((prods.shape[0], int(prods.shape[-1] ** .5), int(prods.shape[-1] ** .5)))
        mapped_patient = mapped_patient.reshape((prods.shape[0], int(prods.shape[-1] ** .5), int(prods.shape[-1] ** .5)))
        mapped_patient = ((mapped_patient + 199.) / 461.) * mapped_mask
        mapped_mask = imapped_mask

        if not mapped_mask.sum():
            print(idx, pid, ' empty')
            continue

        try:
            os.mkdir(os.path.join(args.odir, 'patches', pid))
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
                        os.remove(os.path.join(mdir, 'patches', pid, 'pathc_' + str(i)))
                    except:
                        pass

            if np.random.randint(2):
                patch = mapped_patient[:, x_point]
                mask = mapped_mask[:, x_point]
            else:
                patch = mapped_patient[:, :, y_point]
                mask = mapped_mask[:, :, y_point]

            np.save(os.path.join(args.odir, 'patches', pid, 'pathc_' + str(i)), patch)
            np.save(os.path.join(args.odir, 'patches', pid, 'mask_' + str(i)), mask)
