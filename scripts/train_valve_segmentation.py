import argparse

import numpy as np
from tqdm import tqdm

import scipy.ndimage
from glob import glob
import os

import sys
sys.path.append('nets')
import elu_unet


def preprocess_val(patch):
    return np.dstack(
        [scipy.ndimage.zoom(patch[..., 0],  (SIDE / patch.shape[0], SIDE / patch.shape[1])),
         scipy.ndimage.zoom(patch[..., -1],  (SIDE / patch.shape[0], SIDE / patch.shape[1]), order=0)]
    )


def preprocess(patch):
    scale = np.random.uniform(.8, 1.1, size=2)
    if np.random.randint(2):
        patch = np.flip(patch, 0)
    if np.random.randint(2):
        patch = np.flip(patch, 1)
    if np.random.randint(2):
        patch = np.rot90(patch, k=np.random.randint(3) + 1)
    
    if not np.random.randint(3):
        patch = np.concatenate([np.flip(patch, 0), patch, np.flip(patch, 0)])
    if not np.random.randint(3):
        patch = np.concatenate([np.flip(patch, 1), patch, np.flip(patch, 1)], 1)
    
    window = np.clip(SIDE * scale, 0, min(patch.shape[:-1]))
    if np.count_nonzero(patch[..., -1]):
        coords = np.array(np.where(patch[..., -1]))
        cmin, cmax = coords.min(1) - window, coords.max(1)
        point = np.array([
            np.random.randint(cmin[0], cmax[0]),
            np.random.randint(cmin[1], cmax[1])
        ])
    else:
        point = np.array([
            np.random.randint(0, patch.shape[0] - window[0] + 1),
            np.random.randint(0, patch.shape[1] - window[1] + 1)
        ]) 
    point = np.clip(point, 0, np.array(patch.shape[:-1]) - window).astype(np.int)

    patch = patch[
        point[0]: point[0] + int(window[0]), 
        point[1]: point[1] + int(window[1])
    ]
    
    return np.dstack(
        [scipy.ndimage.zoom(patch[..., 0], (SIDE / patch.shape[0], SIDE / patch.shape[1])), 
         scipy.ndimage.zoom(patch[..., -1],  (SIDE / patch.shape[0], SIDE / patch.shape[1]), order=0)]
    )


def load(rootdir, pid, test_mode=False, val_mode=False):
    i = 0
    if (not test_mode) and (not val_mode):
        i = np.random.randint(10)

    patch = np.load(os.path.join(rootdir, pid, 'pathc_' + str(i) + '.npy'))
    if test_mode:
        return patch

    mask = np.load(os.path.join(rootdir, pid, 'mask_' + str(i) + '.npy'))
    return np.dstack([
        patch,
        mask,
    ])


def load_test(rootdir, pid):
    batch = list()
    for i in range(10):
        patch = np.load(os.path.join(rootdir, pid, 'pathc_' + str(i) + '.npy'))
        batch.append(preprocess_test(patch))
    return np.stack(batch)


def generator_test(rootdir, ids):
    while True:
        for pid in ids:
            yield load_test(rootdir, pid)


def generator(rootdir, ids, test_mode, val_mode, preprocess, batch_size=8):
    while True:
        if (not test_mode) and (not val_mode):
            np.random.shuffle(ids)
        iterations = int(np.ceil(len(ids) / batch_size))
        for i in range(iterations):
            if test_mode:
                processed = np.zeros((batch_size, SIDE, SIDE, 1))
            else:
                processed = np.zeros((batch_size, SIDE, SIDE, 2))
            batch = ids[i * batch_size: (i + 1) * batch_size]
            
            for j, pid in enumerate(batch):
                patch = load(rootdir, pid, test_mode, val_mode)
                processed[j] = preprocess(patch)

            yield processed[..., :-1], np.expand_dims(processed[..., -1], -1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train valve segmentation model.')
    parser.add_argument('idir', type=str, help='input directory (should contains folders with zis.npy and prods.npy)')
    parser.add_argument('mdir', type=str, help='directory with prepared data (should contains mask_*.npy)')
    parser.add_argument('mpath', type=str, help='path to the model')
    
    parser.add_argument('--epochs', metavar='E', type=int, 
                        help='maximum number of epochs to be trained')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.odir))
    except:
        pass

    paths = glob(os.path.join(args.idir, '*', 'zis.npy'))
    ids = [os.path.basename(os.path.dirname(path)) for path in paths]
    ids = [pid for pid in ids if os.path.isfile(os.path.join(args.mdir, pid, 'mask_0.npy'))]

    EPOCHS = 100
    if args.epochs:
        EPOCHS = args.epochs

    SPLIT = .8
    model = elu_unet.get_unet()

    train_ids = ids[:int(SPLIT * len(ids))]
    test_ids = ids[int(SPLIT * len(ids)):]

    batch_size = 16
    train_gen = generator(args.mdir, train_ids, test_mode=False, val_mode=False, preprocess=preprocess, batch_size=batch_size)
    test_gen = generator(args.mdir, test_ids, test_mode=False, val_mode=True, preprocess=preprocess_val, batch_size=batch_size)

    from keras.callbacks import ModelCheckpoint

    model.fit_generator(
        train_gen, 
        steps_per_epoch=int(np.ceil(len(train_ids) / batch_size)), 
        epochs=EPOCHS, 
        validation_data=test_gen,
        validation_steps=int(np.ceil(len(test_ids) / batch_size)),
        callbacks=[ModelCheckpoint(args.mpath, verbose=True, save_best_only=True)]
    )