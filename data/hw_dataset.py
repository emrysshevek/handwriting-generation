import os

import torch
from torch.utils import data
from torch.nn.utils.rnn import pad_sequence

import matplotlib.pyplot as plt
import numpy as np
import cv2


start_token = '<start>'
end_token = '<end>'
unknown_token = '<unk>'

train_path = 'data/training_lines.json'
PADDING_CONSTANT = 0
scale_factor = 8

def collate(batch):
    batch = [b for b in batch if b is not None]
    # These all should be the same size or error
    assert len(set([b.shape[0] for b in batch])) == 1
    assert len(set([b.shape[2] for b in batch])) == 1

    dim0 = batch[0].shape[0]
    dim1 = max([b.shape[1] for b in batch])
    dim2 = batch[0].shape[2]

    # make the width a multiple of scale_factor (8) so it isn't truncated during encoding/decoding
    dim1 += scale_factor - (dim1 % scale_factor)

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]
        input_batch[i, :, :b_img.shape[1], :] = b_img

    line_imgs = input_batch.transpose([0, 3, 1, 2])
    line_imgs = torch.from_numpy(line_imgs)

    return line_imgs


class HWDataset(data.Dataset):

    def __init__(self, dir='./hw_images', img_height=60):
        self.img_height = img_height
        self.data = []
        for root, dirs, files in os.walk(dir):
            self.data.extend([os.path.join(root, file) for file in files])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = self.data[index]

        img = cv2.imread(img_path, 0)

        if img is None:
            print("Warning: image is None:", img_path)
            return None

        percent = float(self.img_height) / img.shape[0]
        img = cv2.resize(img, (0, 0), fx=percent, fy=percent, interpolation=cv2.INTER_CUBIC)

        img = img.astype(np.float32)
        # img = img / 128.0 - 1
        min_val = np.min(img)
        max_val = np.max(img)
        img = (img - min_val) / (max_val - min_val)

        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)

        return img

