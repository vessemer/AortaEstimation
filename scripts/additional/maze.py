import numpy as np


def maze(patient, mask, frequency=3, n=7, cell_size=100):
    # one out of `frequency` will be positive
    zss = np.array(np.where(mask)).T
    cell = np.zeros(shape=(2, cell_size, cell_size))
    maze = np.zeros(shape=(2, cell_size * n, cell_size * n))
    slices = np.random.choice(zss[:, 0], size=n ** 2)
    zxy = np.zeros(shape=(3,), dtype=np.int16)

    for i, zslice in enumerate(slices):
        if not np.random.randint(frequency):
            zxy = zss[zss[:, 0] == zslice]
            zxy = zxy[np.random.choice(len(zxy))]
        else:
            zxy[0] = zslice
            zxy[1:] = np.random.choice(patient.shape[-1], size=2)

        zxy[1:] = np.clip(zxy[1:] - cell_size // 2, 0, patient.shape[-1] - cell_size)
        cell[0] = patient[
            zxy[0], 
            zxy[1]: zxy[1] + cell_size, 
            zxy[2]: zxy[2] + cell_size
        ]
        cell[1] = mask[
            zxy[0], 
            zxy[1]: zxy[1] + cell_size, 
            zxy[2]: zxy[2] + cell_size
        ]
        if np.random.randint(2):
            np.flip(cell, axis=1)
        if np.random.randint(2):
            np.flip(cell, axis=2)
        maze[
            :,
            (i // n) * cell_size: (i // n + 1) * cell_size, 
            (i % n) * cell_size: (i % n + 1) * cell_size
        ] = cell.copy()
    
    return maze
