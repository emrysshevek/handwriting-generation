import datetime
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

if torch.cuda.is_available():
    print('Using GPU\n')
    device = torch.device('cuda')
else:
    print('GPU unavailable\n')
    device = torch.device('cpu')


def run_epoch(model, loader, loss_fct, optimizer, epoch, name):
    losses = []
    min_loss = np.inf
    best_img = None
    for i, gt in enumerate(loader):
        x = gt.to(device)

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

        loss = loss.detach().cpu().item()
        print(loss)
        losses.append(loss)
        if loss < min_loss:
            min_loss = loss
            best_img = output[0].squeeze().detach().cpu().numpy()
            gt_img = gt[0].squeeze().detach().numpy()
            f, axarr = plt.subplots(2, 1)
            axarr[0].imshow(best_img, cmap='gray')
            axarr[1].imshow(gt_img, cmap='gray')
            f.suptitle(f'Batch {i}')

            if not os.path.exists(f'{name}/epoch_{epoch}'):
                os.mkdir(f'{name}/epoch_{epoch}')
            plt.savefig(f'{name}/epoch_{epoch}/batch_{i}.png')
            plt.close()

    return np.mean(losses), best_img


def train(model, loader, loss_fct, optimizer, name):
    if not os.path.exists(f'{name}/best'):
        os.mkdir(f'{name}/best')

    losses = []
    model.train()
    for i in range(1000):
        print('------------------------------------------')
        print(f'EPOCH {i}', flush=True)

        loss, img = run_epoch(model, loader, loss_fct, optimizer, i, name)
        losses.append(loss)
        print(f'Epoch {i} loss: {loss}')
        print('------------------------------------------')

        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.imsave(f'{name}/best/epoch_{i}.png', img, cmap='gray')

        plt.figure()
        plt.plot(losses)
        plt.title('Loss')
        plt.savefig(f'{name}/loss.png')
        plt.show()


def main():
    experiment = 'sigmoid_activation'
    dataset = HWDataset(dir='data/hw_images')
    loader = DataLoader(dataset, batch_size=16, collate_fn=collate, shuffle=True)

    encoder = CNN()
    decoder = Upsampler()
    model = nn.ModuleList([encoder, decoder]).to(device)
    loss_fct = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.002)

    name = 'results/encoding'
    if not os.path.exists(name):
        os.mkdir(name)
    name = f'{name}/{experiment}_{datetime.datetime.now()}'
    if not os.path.exists(name):
        os.mkdir(name)

    train(model, loader, loss_fct, optimizer, name)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main()



