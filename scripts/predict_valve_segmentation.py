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
        patch = np.load(os.path.join(rootdir, pid, 'pathc_' + str(i) + '.npy'))
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
    
    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of epochs to be trained')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.odir))
    except:
        pass

    paths = glob(os.path.join(args.idir, '*', 'mask_0.npy'))
    ids = [os.path.basename(os.path.dirname(path)) for path in paths]

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
