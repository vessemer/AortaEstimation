import argparse

from tqdm import tqdm
from glob import glob
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prediction pipeline')
    parser.add_argument('idir', type=str, help='directory with CT scans in .npz format')
    parser.add_argument('tmpdir', type=str, help='intermediate output directory (aorta & valve segmentation will be placed there)')
    parser.add_argument('mvpath', type=str, help='path to the valve model')
    parser.add_argument('mapath', type=str, help='path to the aorta model')

    args = parser.parse_args()

    try:
        os.mkdir(os.path.join(args.odir))
    except:
        pass
    try:
        os.mkdir(os.path.join(args.tmpdir))
    except:
        pass

    params = "--batch-size=32 --spacing=1.0"
    os.system("python scripts/predict_aorta_segmentation.py %s %s/predicted_aorta %s --pdir=%s/processed_patient %s" 
              % (args.idir, args.tmpdir, args.mapath, args.tmpdir, params))
    
    os.system("python scripts/extract_normals.py %s %s %s" 
              % (os.path.join(args.tmpdir, 'predicted_aorta'), 
                 os.path.join(args.tmpdir, 'processed_patient'), 
                 os.path.join(args.tmpdir, 'normals_extracted')))

    params = "--test=True"
    os.system("python scripts/prepare_valve_segmentation.py %s %s %s %s %s" 
              % (os.path.join(args.tmpdir, 'normals_extracted'), 
                 os.path.join(args.tmpdir, 'predicted_aorta'),
                 os.path.join(args.tmpdir, 'processed_patient'),
                 os.path.join(args.tmpdir, 'prepared_valve'),
                 params))

    os.system("python scripts/predict_valve_segmentation.py %s %s %s --valvedir=%s --ndir=%s --mdir=%s" 
              % (os.path.join(args.tmpdir, 'prepared_valve'),
                 os.path.join(args.tmpdir, 'predicted_valve'),
                 args.mvpath,
                 os.path.join(args.tmpdir, 'prepared_valve_3d'),
                 os.path.join(args.tmpdir, 'normals_extracted'),
                 os.path.join(args.tmpdir, 'predicted_aorta')))

    os.system("python scripts/prepare_features.py %s %s %s %s %s" 
              % (os.path.join(args.tmpdir, 'processed_patient'),
                 os.path.join(args.tmpdir, 'predicted_aorta'),
                 os.path.join(args.tmpdir, 'normals_extracted'),
                 os.path.join(args.tmpdir, 'predicted_valve'),
                 os.path.join(args.tmpdir, 'features.csv')))

    
    
    
    
    
    
    