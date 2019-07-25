import time
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.hw_dataset import HWDataset, collate
from models.cnn import CNN
from models.upsample import Upsampler


def run_epoch(model, loader, loss_fct, optimizer):
    losses = []
    min_loss = np.inf
    best_img = None
    for i, x in enumerate(loader):
        # f, axarr = plt.subplots(len(x), 1)
        # for j, x in enumerate(x):
        #     img = x.squeeze().detach().numpy()
        #     print(img)
        #     axarr[j].imshow(img, cmap='gray')
        # plt.show()
        # break

        print(f'batch {i}: ', end='', flush=True)
        encoding = model[0](x)
        output = model[1](encoding)

        shape_diff = output.shape[-1] - x.shape[-1]
        left_diff = int(np.ceil(np.abs(shape_diff)/2))
        right_diff = int(np.floor(np.abs(shape_diff)/2))
        if shape_diff > 0:
            output = output[..., left_diff:output.shape[-1]-right_diff]
        elif shape_diff < 0:
            output = nn.functional.pad(output, (left_diff, right_diff))

        loss = loss_fct(output, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        print(loss)
        losses.append(loss)
        if loss < min_loss:
            min_loss = loss
            best_img = output[0].squeeze().detach().numpy()
            gt_img = x[0].squeeze().detach().numpy()
            f, axarr = plt.subplots(2, 1)
            axarr[0].imshow(best_img, cmap='gray')
            axarr[1].imshow(gt_img, cmap='gray')
            plt.show()

    return np.mean(losses), best_img


def train(model, loader, loss_fct, optimizer):
    model.train()
    for i in range(10):
        print(f'Epoch {i}:', end='', flush=True)

        loss, img = run_epoch(model, loader, loss_fct, optimizer)
        print(f'Epoch {i} loss: {loss}')

        plt.imshow(img, cmap='gray')


def main():
    dataset = HWDataset(dir='data/hw_images')
    loader = DataLoader(dataset, batch_size=8, collate_fn=collate, shuffle=True)

    encoder = CNN()
    decoder = Upsampler()
    model = nn.ModuleList([encoder, decoder])
    loss_fct = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    train(model, loader, loss_fct, optimizer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()



