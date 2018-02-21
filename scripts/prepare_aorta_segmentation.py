import os
from glob import glob 

import numpy as np
import scipy.ndimage

import argparse
from multiprocessing import Pool
from functools import partial

import sys
import load_utils
import watereshed
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')


def rot(image, xy, angle, reshape=True):
    im_rot = scipy.ndimage.interpolation.rotate(image, angle, reshape=reshape) 
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy - org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a)])
    return im_rot, new + rot_center


def slice_along(arr, ids, axis):
    arr = np.swapaxes(arr, axis, 0)
    arr = arr[slice(*ids)]
    return np.swapaxes(arr, axis, 0)


def process(patient_id, idir, odir):
    patient, meta = load_utils.load_patient(idir, patient_id, metadata=True)

    meshs = glob(os.path.join(idir, patient_id, '*_A.stl'))
    meshs = load_utils.load_mesh(meshs[0], meta) 

    bbox = np.array(np.where(meshs))
    bbox = np.array([bbox.min(1), bbox.max(1)])
    bbox_shape = np.diff(bbox, axis=0).astype(np.int).flatten()
    bbox_centroid = bbox.mean(0).astype(np.int)

    bbox_axes = [0] 
    for bb_axis in bbox_axes:
        bb_ids = (
            bbox_centroid[bb_axis] - bbox_shape[bb_axis] // 2, 
            bbox_centroid[bb_axis] + bbox_shape[bb_axis] // 2
        )
        bb_ids = np.clip(bb_ids, 0, patient.shape[bb_axis])
        meshs = slice_along(meshs, bb_ids, bb_axis)
        patient = slice_along(patient, bb_ids, bb_axis)

    try:
        os.mkdir(os.path.join(odir, patient_id))
    except:
        pass

    for i, (mes, pat) in enumerate(zip(meshs, patient)):
        patch = np.dstack([pat, mes[:pat.shape[0], :pat.shape[1]]])
        np.save(os.path.join(odir, patient_id, 'patch_' + str(i)), patch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess CT scans for ROI segmentation.')
    parser.add_argument('idir', type=str, help='input directory')
    parser.add_argument('odir', type=str, help='output directory')
    parser.add_argument('--batch-size', metavar='N', type=int,
                        help='batch size to load in RAM')
    parser.add_argument('--j', metavar='J', type=int, 
                        help='number of process to run simultaneously')
    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of samples to be processed')
    parser.add_argument('--s', metavar='S', type=int, 
                        help='Skip first S samples')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.odir))
    except:
        pass

    BATCH_SIZE = 1
    if args.batch_size:
        BATCH_SIZE = args.batch_size

    patient_ids = glob(os.path.join(args.idir, '*'))
    patient_ids = [os.path.basename(path) for path in patient_ids]
    

    if ars.s:
        patient_ids = patient_ids[args.s:]
    if args.n:
        patient_ids = patient_ids[:args.n]


    if args.j:
        process = partial(process, idir=args.idir, odir=args.odir)

    processeds = list()
    for i in tqdm(range(len(patient_ids) // BATCH_SIZE + 1)):
        batch = patient_ids[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
        if args.j:

            with Pool(args.j) as pool:
                processed = pool.map(process, batch)
        else:
            for patch in batch:
                processeds.append(process(patch, idir=args.idir, odir=args.odir))
