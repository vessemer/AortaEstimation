import argparse

from glob import glob 
import os
from tqdm import tqdm 
from itertools import product

import numpy as np
import scipy.ndimage
import cv2
import load_utils
import pandas as pd

import warnings
warnings.filterwarnings('ignore')


def multi_linespace(start, stop, length=50):
    """
    Generalize np.linespace function for a multi dimensional case.
    start : scalar
        The starting value of the sequence.
    stop : scalar
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    length : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    """
    shape = start.shape
    start = start.flatten()
    stop = stop.flatten()
    lspace = np.array([np.linspace(v, e, length) for v, e in np.stack([start, stop]).T])
    return lspace.reshape(shape + (-1, ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract normal planes of CT scans and predicted masks.')
    parser.add_argument('maskdir', type=str, help='masks input directory')
    parser.add_argument('patdir', type=str, help='input directory should contains processed patients\' CT scans')
    parser.add_argument('odir', type=str, help='output directory')
    parser.add_argument('--side', type=str, help='output directory')
    
    parser.add_argument('--j', metavar='J', type=int, 
                        help='number of process to run simultaneously')
    parser.add_argument('--s', metavar='S', type=int, 
                        help='Skip first S samples')
    parser.add_argument('--n', metavar='N', type=int, 
                        help='maximum number of samples to be processed')


    args = parser.parse_args()

    # creates a directories if there isn't any
    try:
        os.mkdir(args.odir)
        os.mkdir(os.path.join(args.odir, 'prods'))
        os.mkdir(os.path.join(args.odir, 'slices'))
        os.mkdir(os.path.join(args.odir, 'planes'))
        os.mkdir(os.path.join(args.odir, 'masks'))
    except:
        pass
    
    
    SIDE = 224
    if args.side:
        SIDE = args.side

    # extracted patients ids inside os a given `idir`
    # liave only those which have predicted masks in `mdir`
    mask_paths = glob(os.path.join(args.maskdir, '*.npy'))
    mask_paths = [os.path.basename(path) for path in mask_paths]
    paths = glob(os.path.join(args.patdir, '*.npy'))
    paths = [path for path in paths if os.path.basename(path) in mask_paths]
    
    j = 1
    if args.j:
        j = args.j

    if args.s:
        paths = paths[args.s:]
    if args.n:
        paths = paths[:args.n]

    for i, path in enumerate(paths):
        print("Iteration %d/%d, patient id: %s" % (i + 1, len(paths), os.path.basename(path)))
        lpred = np.load(os.path.join(args.maskdir, os.path.basename(path)))
        patient = np.load(os.path.join(args.patdir, os.path.basename(path)))
        
        x, y = np.asarray(np.where(lpred))[[0, 2]]
        z = np.polyfit(x, y, 2)
        p = np.poly1d(z)

        # To check whether CT is flipped, inflection point is derived 
        init_d = p.deriv()
        opt = p(init_d.roots)
        if np.abs(p(x.min()) - opt) > np.abs(p(x.max()) - opt):
            patient = np.flip(patient, 0)
            lpred = np.flip(lpred, 0)
            x, y = np.asarray(np.where(lpred))[[0, 2]]
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)

        # extract mask's points to approximate its centroid with a curve
        cx, cy = np.asarray(np.where(lpred))[[0, 1]]
        cz = np.polyfit(cx, cy, 1)
        cp = np.poly1d(cz)

        init_p = np.poly1d(p.coefficients)
        length = min(min(patient.shape[1:]), SIDE)
        # size with which to iterate along a curve (in mm)
        step_size = .5

        deriv_p = init_p.deriv()
        point = np.array([x.min() + 20, init_p(x.min() + 20)])
        points = [point]
        
        # collect points along a curve w.r.t. `step_size`
        while points[-1][0] <= x.max():
            v = np.array([1, deriv_p(point[0] + 1)])
            v = step_size * v / np.linalg.norm(v, ord=2)
            point = points[-1] + v
            point = np.array([point[0], p(point[0])])
            points.append(point)

        points = np.array(points)

        zis = list()
        mis = list()
        prods = list()
        planes = list()

        for point in tqdm(points[:, 0]):
            # in order to find normal vector curve's tangent line has been employed
            v = np.array([1, cp(point + 1) - cp(point), init_p(point + 1) - init_p(point)])
            # cross product with any planar vector will result in a new vector which is orthonormal to the origin one
            v = np.stack([np.cross(v, np.array([0, 1, 0])), v])
            v = np.stack([np.cross(v[0], v[1]), v[0]])
            # after plane's basis has been found, scale basis vectors
            v = length * v / np.expand_dims(np.linalg.norm(v, ord=2, axis=1), -1)

            origin = np.array([point, cp(point), init_p(point)])
            planes.append(np.concatenate([origin, np.cross(v[0], v[1])]))

            # to find slice's coordinates, univariate grid was generated from basis vectors
            lz = multi_linespace((-v[0] / 2 + origin), (v[0] / 2 + origin), length)
            lz = multi_linespace((lz.T - v[1] / 2), (lz.T + v[1] / 2), length)
            lz = np.rollaxis(lz, 1, 0).reshape(3, -1)

            # crop out slices from a CT image
            zi = scipy.ndimage.map_coordinates(patient, lz)
            zi = zi.reshape((length, length))
            zis.append(zi)

            # crop out slices from mask of a CT image 
            mi = scipy.ndimage.map_coordinates(lpred.astype(np.float), lz)
            mi = mi.reshape((length, length))
            mis.append(np.expand_dims(mi, -1))

            prods.append(lz)

        # stack resulted planes and coordinates into np.ndarray
        zis = np.array(zis)
        mis = np.array(mis)
        planes = np.array(planes)
        prods = np.array(prods)

        # save resulted arrays into sub directories of --odir
        # prods is coordinates of format: [N, 3, length * length], where N is amount of cropped slices
        np.save(os.path.join(args.odir, 'prods', os.path.basename(path)), prods)
        # zis is stacked slices of patient
        np.save(os.path.join(args.odir, 'slices', os.path.basename(path)), zis)
        # mis is stacked masks
        np.save(os.path.join(args.odir, 'masks', os.path.basename(path)), mis)
        # planes are in format: [origin_x, origin_y, origin_z, vector_x, vector_y, vector_z]
        # planes elements are consisted of origin-{x, y, z} (point on a curve) and vector-{x, y, z} (plane's normal vector) 
        np.save(os.path.join(args.odir, 'planes', os.path.basename(path)), planes)
