import argparse
from multiprocessing import Pool
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
import watereshed
import generator_3D

import sys
sys.path.append('nets')
import unet

from keras.callbacks import ModelCheckpoint
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from skimage import measure
from skimage.morphology import convex_hull_image

import pickle
from skimage.measure import regionprops


def load(path, standardize=False):
    patches = glob(os.path.join(path, 'patch*.npy'))
    patch = np.load(patches[np.random.randint(len(patches))])
    if standardize:
        return np.dstack([
            scipy.ndimage.zoom(patch[..., 0], .7),
            scipy.ndimage.zoom(patch[..., -1], .7, order=0)
        ])

    return patch


def preprocess_test(patch):
    window = min(patch.shape[1], int(1.7 * SIDE))
    point = np.array(patch.shape) // 2 - window // 2
    point = np.clip(point, 0, np.array(patch.shape) - window)
    patch = patch[
        point[0]: point[0] + window, 
        point[1]: point[1] + window
    ]

    clip = cv2.resize(patch, dsize=(SIDE, SIDE))
    return clip


def preprocess_val(patch):
    window = min(patch.shape[1], int(1.7 * SIDE))
    point = np.array(patch.shape[:-1]) // 2 - window // 2
    point = np.clip(point, 0, np.array(patch.shape[:-1]) - window)
    
    patch = patch[
        point[0]: point[0] + window, 
        point[1]: point[1] + window
    ]

    return np.dstack([
        scipy.ndimage.zoom(patch[..., 0], SIDE / patch.shape[0]),
        scipy.ndimage.zoom(patch[..., -1], SIDE / patch.shape[0], order=0)
    ])


def preprocess_train(patch):
    if np.random.randint(2):
        patch = np.flip(patch, 0)
    
    if np.random.randint(2):
        patch = np.flip(patch, 1)
    
    if np.random.randint(3):
        shift = np.random.uniform(-.2, .2, size=2)
        shift *= np.array(patch.shape[:2])
        patch = np.dstack([
            scipy.ndimage.shift(patch[..., 0], shift),
            scipy.ndimage.shift(patch[..., -1], shift, order=0)
        ])
        
    if np.random.randint(3):
        rotate = np.random.uniform(-40, 40)
        patch = np.dstack([
            scipy.ndimage.rotate(patch[..., :-1], rotate),
            scipy.ndimage.rotate(patch[..., -1:], rotate, order=0)
        ])
    
    scale = np.random.uniform(.5, 1.5)
    window = min(min(patch.shape[:-1]), int(SIDE * scale))
    if np.count_nonzero(patch[..., 1]):
        coords = np.array(np.where(patch[..., 1]))
        cmin, cmax = coords.min(1) - window, coords.max(1)
        point = np.array([
            np.random.randint(cmin[0], cmax[0]),
            np.random.randint(cmin[1], cmax[1])
        ])
    else:
        point = np.random.randint(0, min(patch.shape[:-1]) - window + 1)
    point = np.clip(point, 0, np.array(patch.shape[:-1]) - window)
    
    patch = patch[
        point[0]: point[0] + window, 
        point[1]: point[1] + window
    ]

    return np.dstack([
        scipy.ndimage.zoom(patch[..., 0], SIDE / patch.shape[0]),
        scipy.ndimage.zoom(patch[..., -1], SIDE / patch.shape[0], order=0)
    ])


def eval_generator(patient, batch_size=32):
    for i in range(len(patient) // batch_size + 1):
        batch = patient[i * batch_size: (i + 1) * batch_size]
        processed = list(map(preprocess_val, batch))
        processed = np.array(processed)
        yield (np.expand_dims(processed[..., 0], -1) + 199.) / 461., np.expand_dims(processed[..., 1], -1) > 0
        

def test_generator(patient, batch_size=32, train_mode=False):
    for i in range(len(patient) // batch_size + 1):
        batch = patient[i * batch_size: (i + 1) * batch_size]
        processed = list(map(preprocess_test, batch))
        processed = np.array(processed)
        yield (np.expand_dims(processed, -1) + 199.) / 461.
        

def generator(patients_ids, idir, batch_size=32, train_mode=False, shuffle_coef=.7, j=1):
    while True:
        if train_mode:
            np.random.shuffle(patients_ids)

        paths = [
            os.path.join(idir, patient_id) 
            for i, patient_id in enumerate(patients_ids)
        ]

        for i in range(len(paths) // batch_size + 1):
            batch = paths[i * batch_size: (i + 1) * batch_size]
            with Pool(j) as pool:
                processed = pool.map(load, batch)

            if train_mode:
                with Pool(j) as pool:
                    processed = pool.map(preprocess_train, processed)
            else:
                with Pool(j) as pool:
                    processed = pool.map(preprocess_val, processed)
            processed = np.array(processed)
            yield (np.expand_dims(processed[..., 0], -1) + 199.) / 461., np.expand_dims(processed[..., 1], -1) > 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train UNet over CT scans for ROI segmentation.')
    parser.add_argument('idir', type=str, help='input directory')
    parser.add_argument('mdir', type=str, help='output model directory')
    parser.add_argument('--batch-size', metavar='N', type=int,
                        help='batch size to load in RAM')
    parser.add_argument('--epochs', metavar='E', type=int,
                        help='number of epochs')
    parser.add_argument('--split', metavar='S', type=float,
                        help='train / test split: train = patient_ids[int(SPLIT * len(patient_ids)):]')
    parser.add_argument('--j', metavar='J', type=int, 
                        help='number of process to run simultaneously')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.model_dir))
    except:
        pass


    patient_ids = glob(os.path.join(args.idir, '*', 'patch_0.npy'))
    patient_ids = [os.path.basename(os.path.dirname(pid)) for pid in patient_ids]

    np.random.RandomState(seed=42)
    np.random.shuffle(patient_ids)

    SIDE = 224
    BATCH_SIZE = 32
    if args.batch_size:
        BATCH_SIZE = args.batch_size

    SPLIT = .7
    if args.split:
        SPLIT = args.split

    train = patient_ids[int(SPLIT * len(patient_ids)):]
    test = patient_ids[:int(SPLIT * len(patient_ids))]

    model = unet.get_unet(224, 224)

    j = 1
    if args.j:
        j = args.j

    cval = 0.0
    NUM = 300
    if args.epochs:
        NUM = args.epochs

    for i in range(NUM):
        train_gen = generator(train, args.idir, train_mode=True, j=j)

        model.fit_generator(
            train_gen,
            steps_per_epoch= 10 * len(train) // BATCH_SIZE + 1, 
            verbose=1, 
        )

        test_gen = generator(test, args.idir, train_mode=False, j=j)
        valeval = model.evaluate_generator(test_gen, 3)

        print('Iteration %s/%s \n val dice: %s'%(i, NUM, valeval[0]))
        if valeval[0] < cval:
            cval = valeval[0]
            model.save(os.path.join(args.mdir, 'unet_model'))
            print('-' * 50)
