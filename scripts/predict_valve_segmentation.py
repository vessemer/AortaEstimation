import argparse

import numpy as np
from tqdm import tqdm

import scipy.ndimage
from glob import glob
import os

import sys
sys.path.append('nets')
import elu_unet


SIDE = 128


def preprocess_test(patch):
    return  np.expand_dims(scipy.ndimage.zoom(patch,  (SIDE / patch.shape[0], SIDE / patch.shape[1])), -1)


def load_test(rootdir, pid):
    batch = list()
    for i in range(10):
        patch = np.load(os.path.join(rootdir, pid, 'patch_' + str(i) + '.npy'))
        shape = patch.shape
        batch.append(preprocess_test(patch))
    return np.stack(batch), shape


def generator_test(rootdir, ids):
    while True:
        for pid in ids:
            yield load_test(rootdir, pid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Valve segmentation model inference over prepared dataset')
    parser.add_argument('idir', type=str, help='directory with prepared data (should contains mask_*.npy)')
    parser.add_argument('odir', type=str, help='output directory')
    parser.add_argument('mpath', type=str, help='path to the model')
    
    parser.add_argument('--valvedir', metavar='VSCV', type=str, 
                        help='Directory to store predicted valve mapped on 3D')
    parser.add_argument('--ndir', type=str, help='directory should contains prods & slices directories (output from extract_normals)')
    parser.add_argument('--mdir', type=str, help='directory with valve masks')

    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of epochs to be trained')
    parser.add_argument('--s', metavar='S', type=int, 
                    help='Skip first S samples')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.odir))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.valvedir))
    except:
        pass

    paths = glob(os.path.join(args.idir, '*', 'patch_0.npy'))
    ids = [os.path.basename(os.path.dirname(path)) for path in paths]
    
    if args.s:
        ids = ids[args.s:]
    if args.n:
        ids = ids[:args.n]

    model = elu_unet.get_unet()
    model.load_weights(args.mpath)
    
    test_gen = generator_test(args.idir, ids)
    
    for pid in tqdm(ids):
        t, shape = next(test_gen)
        pred = model.predict(t).mean(0)
        
        # scale back
        pred = scipy.ndimage.zoom(pred[..., 0],  (shape[0] / SIDE, shape[1] / SIDE))
        
        # maskes post-processing by selecting the largest connected component ..
        lpred, _ = scipy.ndimage.label(pred > .5)
        idx = np.argmax(np.bincount(lpred.flatten())[1:]) + 1
        lpred[lpred != idx] = 0
        
        # .. and binary closing
        lpred = scipy.ndimage.binary_closing(lpred, iterations=5)
        np.save(os.path.join(args.odir, pid), lpred)
        
        if args.valvedir:
            idxs = np.where(lpred)[0]
            idx0, idx1 = idxs.min(), idxs.max()

            mask = np.load(os.path.join(args.mdir, pid + '.npy')) > .5
            prods = np.load(os.path.join(args.ndir, 'prods', pid + '.npy'))
            
            imask = np.zeros_like(mask)
            values = scipy.ndimage.map_coordinates(mask, np.rollaxis(prods, 1, 0).reshape(3, -1))
            values = values.reshape((prods.shape[0], prods.shape[-1]))

            for i in range(idx0, idx1):
                coords = np.array([prods[i][0], prods[i][1], prods[i][2]])
                coords = np.clip(coords.T, 0, np.array(mask.shape) - 1)
                coords = np.round(coords[values[i] > .5]).astype(np.int).T
                coords = tuple(c for c in coords)

                imask[coords] = True

            imask = mask & scipy.ndimage.binary_closing(imask, iterations=8)
            np.save(os.path.join(args.valvedir, pid), imask)
