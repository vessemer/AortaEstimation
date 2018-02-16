# bsc-project


### Preprocess CT scans for ROI segmentation

```bash
$ python scripts/prepare_aorta_segmentation.py -h
usage: prepare_2D.py [-h] [--batch-size N] [--j J] idir odir

Preprocess CT scans for ROI segmentation.

positional arguments:
  idir            input directory
  odir            output directory

optional arguments:
  -h, --help      show this help message and exit
  --batch-size N  batch size to load in RAM
  --j J           number of process to run simultaneously

```

```bash
$ python scripts/prepare_aorta_segmentation.py ../DM_Data/RIII/ ../DM_Data/RIII_test/
  100%|██████████████████████████████████████████████████████████████████| 768/768 [2:16:30<00:00, 10.76s/it]
```


### Train UNet over CT scans for ROI segmentation

```bash
$ python scripts/train_aorta_segmentation.py -h
usage: train_2D.py [-h] [--batch-size N] [--epochs E] [--split S] [--j J]
                   idir mdir

Train UNet over CT scans for ROI segmentation.

positional arguments:
  idir            input directory
  mdir            output model directory

optional arguments:
  -h, --help      show this help message and exit
  --batch-size N  batch size to load in RAM
  --epochs E      number of epochs
  --split S       train / test split: train = patient_ids[int(SPLIT *
                  len(patient_ids)):]
  --j J           number of process to run simultaneously

```

```bash
$ python scripts/train_aorta_segmentation.py ../DM_Data/RIII_test/ ../DM_Data/RIII_test_model/
Epoch 1/1
1/1 [==============================] - 4s 4s/step - loss: -0.1278 - dice_coef: 0.1278
Iteration 0/300 
 val dice: -0.0497503019869
--------------------------------------------------
Epoch 1/1
1/1 [==============================] - 1s 686ms/step - loss: -0.1426 - dice_coef: 0.1426
Iteration 1/300 
 val dice: -0.0524366907775
--------------------------------------------------
```


### UNet inference over CT scans, ROI segmentation

```bash
$ python scripts/predict_aorta_segmentation.py -h
usage: segment_2D.py [-h] [--spacing S] [--batch-size N] [--TTA T] [--j J]
                     idir odir mpath

UNet inference over CT scans for ROI segmentation.

positional arguments:
  idir            input directory
  odir            output directory
  mpath           path to the model

optional arguments:
  -h, --help      show this help message and exit
  --spacing S     if included isotropic spacing of CT will be forced,
                  otherwise original spacing will be preserved
  --batch-size N  batch size to load in RAM
  --TTA T         whether to have test time augmentations, T in {0, 1, 2, 3}
  --j J           number of process to run simultaneously

```

```bash
$ python scripts/predict_aorta_segmentation.py ../DM_Data/RIII ../DM_Data/RIII_test ../DM_Data/RIII_models/unet_model --batch-size=32 --spacing=1.0
100%|████████████████████████████████████████████████████████████████████| 12/12 [03:05<00:00, 15.45s/it]
```

![](https://image.ibb.co/hVuJi7/part.png) ![](https://image.ibb.co/dbvPO7/mask.png) 


### Extract normal planes of CT scans and predicted masks

```bash
$ python scripts/extract_normals.py -h
usage: extract_normals.py [-h] [--side SIDE] [--j J] maskdir patdir odir

Extract normal planes of CT scans and predicted masks.

positional arguments:
  maskdir      masks input directory
  patdir       input directory should contains patients' CT scans
  odir         output directory

optional arguments:
  -h, --help   show this help message and exit
  --side SIDE  output directory
  --j J        number of process to run simultaneously

```

```bash
$ python scripts/extract_normals.py ../DM_Data/RIII_test ../DM_Data/RIII_test_patients ../DM_Data/RIII_test_planes
Iteration 1/5, patient id: 0679R3035.npy
100%|██████████████████████████████████████████████████████████████████| 130/130 [02:01<00:00,  1.07it/s]
Iteration 2/5, patient id: 0304R3039.npy
100%|██████████████████████████████████████████████████████████████████| 211/211 [02:08<00:00,  1.64it/s]
Iteration 3/5, patient id: 0303R3023.npy
 10%|██████████████████████████████████████████████████████████████████| 149/149 [01:06<00:00,  2.19it/s]
```
![](https://preview.ibb.co/cueDJ7/113.png)
![](https://preview.ibb.co/jph0y7/112.png)


### Prepare dataset for valve segmentation (require valve annotated data)

```bash
$ python scripts/prepare_valve_segmentation.py  -h
usage: prepare_valve_segmentation.py [-h] [--n N] idir mdir odir

Prepare dataset for valve segmentation.

positional arguments:
  idir        input directory (should contains zis.npy and prods.npy)
  mdir        directory with valve masks
  odir        output directory

optional arguments:
  -h, --help  show this help message and exit
  --n N       maximum number of samples to be processed

```

```bash
$ python scripts/prepare_valve_segmentation.py  ../DM_Data/RIII /home/ubuntu/edata/ ../DM_Data/RIII_test --n=4
100%|██████████████████████████████████████████████████████████████████████| 4/4 [00:44<00:00, 11.03s/it]
```  

![](https://preview.ibb.co/n0xsfn/prepared.png)  


### Train valve segmentation model

```bash
$ python scripts/train_valve_segmentation.py -h
usage: train_valve_segmentation.py [-h] [--epochs E] idir mdir mpath

Train valve segmentation model.

positional arguments:
  idir        input directory (should contains folders with zis.npy and
              prods.npy)
  mdir        directory with prepared data (should contains folders with
              mask_*.npy)
  mpath       path to the model

optional arguments:
  -h, --help  show this help message and exit
  --epochs E  maximum number of epochs to be trained

```

```bash
$ python scripts/train_valve_segmentation.py ../DM_Data/RIII ~/edata/patchs/ ../DM_Data/RIII_test_model/valve --epochs=2
Epoch 1/2
14/14 [==============================] - 6s 401ms/step - loss: 1.0654 - binary_crossentropy: 0.3117 - dice_coef: 0.2463 - val_loss: 0.7435 - val_binary_crossentropy: 0.0818 - val_dice_coef: 0.3383
Epoch 2/2
14/14 [==============================] - 3s 206ms/step - loss: 0.7789 - binary_crossentropy: 0.1465 - dice_coef: 0.3677 - val_loss: 0.7318 - val_binary_crossentropy: 0.0872 - val_dice_coef: 0.3554
```


### Predict valve segmentation

```bash
$ python scripts/predict_valve_segmentation.py -h
usage: predict_valve_segmentation.py [-h] [--n N] idir odir mpath

Valve segmentation model inference over prepared dataset.

positional arguments:
  idir        directory with prepared data (should contains mask_*.npy)
  odir        output directory
  mpath       path to the model

optional arguments:
  -h, --help  show this help message and exit
  --n N       maximum number of epochs to be trained
  
```  

```bash
$ python scripts/predict_valve_segmentation.py ~/edata/patches ~/edata/valve_output ~/edata/xception_valve --n=4
100%|██████████████████████████████████████████████████████████████████████| 4/4 [00:00<00:00,  4.16it/s]
```  

![](https://preview.ibb.co/dkApRS/valve_predicted.png)


### Prepare features  

```bash
$ python scripts/prepare_features.py -h
usage: prepare_features.py [-h] [--labels_path LABELS_PATH]
                           [--exclude_paths EXCLUDE_PATHS] [--n N]
                           idir zpdir valve_path odir

Prepare dataset for valve segmentation.

positional arguments:
  idir                  input directory
  zpdir                 directory should contains zis.npy and prods.npy
  valve_path            path to the valve.csv
  odir                  output directory

optional arguments:
  -h, --help            show this help message and exit
  --labels_path LABELS_PATH
                        path to the REPRISE III Sizes.xlsx
  --exclude_paths EXCLUDE_PATHS
                        path to the pickled version of excluded paths
  --n N                 maximum number of samples to be processed

```

```bash
$ python scripts/prepare_features.py ~/edata/ ~/cdata/DM_Data/RIII valve.csv ~/edata/features.csv --labels_path=../DM_Data/REPRISE\ III\ Sizes.xlsx --exclude_paths=exclude_paths --n=3
100%|██████████████████████████████████████████████████████████████████████| 3/3 [01:03<00:00, 21.28s/it]
```


### Make a decision

```python
from sklearn.model_selection import LeaveOneOut
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
import pandas as pd
from tqdm import tqdm
import pickle


features = pd.read_csv('features')
features = features.drop(['Unnamed: 0'], axis=1)

labels = features[['class']]
for label in labels['class'].unique():
    labels[label] = features['class'] == label
labels = labels.drop(['class'], axis=1).values
features = features.drop(['class', 'seriesuid'], axis=1).values

loo = LeaveOneOut()
predicted = dict()
gt = list()
clfs = [
        GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=10)
    ]
for clf in clfs:
    predicted[str(clf.__class__)] = list()
    for split, lo in tqdm(loo.split(features)):
        clf.fit(features[split], np.argmax(labels[split], axis=1))

        predicted[str(clf.__class__)].append(clf.predict_proba(features[lo]))
        
for i, clf in enumerate(clfs):
    pred = np.array(predicted[str(clf.__class__)])
    pred = np.argmax(pred, axis=-1)
    gt = np.argmax(labels, axis=-1)
    print((len(labels) - (np.abs(np.squeeze(pred) - gt) > 0).sum()) / len(labels))


pickle.dump(clfs, open('../clfs', 'wb'))
```
