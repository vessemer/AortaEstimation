import argparse
from functools import partial

from glob import glob 
import os
from tqdm import tqdm 
from itertools import product

import numpy as np
import scipy.ndimage
import cv2
from multiprocessing import Pool

import load_utils

import sys
sys.path.append('nets')
import unet

import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def preprocess_test(patch):
    """
    Preprocess patch for the network input (in test mode)
    patch: ndarray
    """
    # crop a central window in 2D
    window = min(patch.shape[1], int(1.7 * SIDE))
    point = np.array(patch.shape) // 2 - window // 2
    point = np.clip(point, 0, np.array(patch.shape) - window)
    patch = patch[
        point[0]: point[0] + window, 
        point[1]: point[1] + window
    ]

    # stack and zoom back cropped windows to the desired receptive fields
    clip = cv2.resize(patch, dsize=(SIDE, SIDE))
    return clip
      

def test_generator(patient, batch_size=32, train_mode=False):
    """
    Generator function for Keras (in eval mode)
    patient: ndarray
    batch_size: imperically chosen parameter
    """
    for i in range(len(patient) // batch_size + 1):
        batch = patient[i * batch_size: (i + 1) * batch_size]
        processed = list(map(preprocess_test, batch))
        processed = np.array(processed)
        yield (np.expand_dims(processed, -1) + 199.) / 461.
        

def postprocess_test(pred, patient):
    """
    Postprocess predicted output (inserts and resizes back a cropped prediction)
    """
    window = min(patient.shape[1], int(1.7 * SIDE))
    
    pred = scipy.ndimage.zoom(pred > .5, [1, window / SIDE, window / SIDE], order=0)
    
    point = np.array(patient.shape[1:3]) // 2 - window // 2
    point = np.clip(point, 0, np.array(patient.shape[1:3]) - window)
    return np.pad(pred, [[0, 0], point[[0, 0]], point[[1, 1]]], mode='constant')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='UNet inference over CT scans, ROI segmentation.')
    parser.add_argument('idir', type=str, help='input directory')
    parser.add_argument('odir', type=str, help='output directory')
    parser.add_argument('mpath', type=str, help='path to the model')
    
    parser.add_argument('--spacing', metavar='S', type=float,
                        help='if included isotropic spacing of CT will be forced, '\
                        + 'otherwise original spacing will be preserved')
    parser.add_argument('--batch-size', metavar='N', type=int,
                        help='batch size to load in RAM')
    parser.add_argument('--TTA', metavar='T', type=int,
                        help='whether to have test time augmentations, T in {0, 1, 2, 3}')
    parser.add_argument('--j', metavar='J', type=int, 
                        help='number of process to run simultaneously')
    parser.add_argument('--pdir', type=str, help='output directory for CT postprocessed data')
    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of samples to be processed')
    parser.add_argument('--s', metavar='S', type=int, 
                        help='Skip first S samples')

    args = parser.parse_args()

    # creates a directories if there isn't any
    try:
        os.mkdir(os.path.join(args.odir))
        os.mkdir(os.path.join(args.pdir))
    except:
        pass
    
    SIDE = 224

    # creates model and read weights from --mpath 
    model = unet.get_unet(SIDE, SIDE)
    model.load_weights(args.mpath)

    # reads available paths from idir
    paths = glob(os.path.join(args.idir, '*'))

    if args.s:
        paths = paths[args.s:]
    if args.n:
        paths = paths[:args.n]

    BATCH_SIZE = 32
    if args.batch_size:
        BATCH_SIZE = args.batch_size

    j = 1
    if args.j:
        j = args.j

    TTA = 0
    if args.TTA:
        TTA = args.TTA
        assert TTA in [0, 1, 2, 3], "Test time augmentation should lie in {0, 1, 2, 3}"

    for path in tqdm(paths):
        # loads patient CT scan along with its meta data
        patient, meta = load_utils.load_patient(os.path.dirname(path), os.path.basename(path), metadata=True)
        fact = meta['PixSpac']

        # prepares slices of CT scan with `test_generator`
        length = int(np.ceil(len(patient) / BATCH_SIZE))
        test_gen = test_generator(patient)

        # makes prediction
        pred = [model.predict_generator(test_gen, length)]

        # applys requested amount of test time augmentations
        if TTA > 0:
            test_gen = test_generator(np.flip(patient, 1))
            pred.append(model.predict_generator(test_gen, length))
        if TTA > 1:
            test_gen = test_generator(np.flip(patient, 2))
            pred.append(model.predict_generator(test_gen, length))
        if TTA > 2:
            test_gen = test_generator(np.flip(np.flip(patient, 1), 2))
            pred.append(model.predict_generator(test_gen, length))

        # restores augmentations if any
        try:
            pred[0] = pred[0] 
            pred[1] = np.flip(pred[1], 1) 
            pred[2] = np.flip(pred[2], 2) 
            pred[3] = np.flip(np.flip(pred[3], 1), 2)
        except:
            pass
        
        # averages the results
        pred = np.mean(np.stack([pred], -1), -1)
        lpred = postprocess_test(np.squeeze(pred) > .5, patient)

        # maskes post-processing by selecting the largest connected component ..
        lpred, _ = scipy.ndimage.label(lpred)
        idx = np.argmax(np.bincount(lpred.flatten())[1:]) + 1
        lpred[lpred != idx] = 0
        
        # .. and binary closing
        lpred = scipy.ndimage.binary_closing(lpred, iterations=5)
        
        # final stage of postprocessing is zoom to the desired scale provided with --spacing
        if args.spacing:
            lpred = scipy.ndimage.zoom(lpred, fact / args.spacing, order=0)
            if args.pdir:
                patient = scipy.ndimage.zoom(patient, fact / args.spacing, order=3)
        if args.pdir:
            np.save(os.path.join(args.pdir, os.path.basename(path)), patient)
        np.save(os.path.join(args.odir, os.path.basename(path)), lpred)
