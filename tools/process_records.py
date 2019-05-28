import os
import argparse

import numpy as np
import pandas as pd
import mxnet as mx

from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MXNet records')
    parser.add_argument('--folder', help='full path for folder with records')
    args = parser.parse_args()

    folder = args.folder
    
    image_folder = os.path.join(folder, 'train')
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
        
    path_imgidx = os.path.join(folder, 'train.idx')
    path_imgrec = os.path.join(folder, 'train.rec')
    imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')

    init, _ = mx.recordio.unpack(imgrec.read_idx(0))
    train = []
    for i in tqdm(range(1, int(init.label[0]))):
        header, blob = mx.recordio.unpack(imgrec.read_idx(i))
        train.append({'label': int(header.label[0] if isinstance(header.label, np.ndarray) else header.label),
                      'image': '{:07d}.jpg'.format(header.id)})

        with open(os.path.join(image_folder, 
                               '{:07d}.jpg'.format(header.id)), 'wb') as f:
            f.write(blob)

    train_df = pd.DataFrame(train)
    train_df.to_csv(os.path.join(folder, 'train.csv'), index=None)
