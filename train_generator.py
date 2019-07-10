import time
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from data.dataset import WordsDataset, LetterDataset, collate
from models.rnn import Encoder, Decoder
from models.upsample import Upsampler
from models.hwr import create_model
from models.hw_generator import HWGenerator
from utils.string_utils import label2str, naive_decode
from utils.error_rates import cer


def train(n_epochs, model, optimizer, criterion, data_generator, dataset, online, result_dir):
    overall_best = np.inf
    for epoch in range(n_epochs):
        print("---------------------------------------")
        print("EPOCH", epoch)
        print("---------------------------------------")
        sum_loss = 0.0
        steps = 0.0

        model.train()
        epoch_best = np.inf
        for labels, label_lengths in data_generator:
            preds, images = model(labels, online)
            preds = preds.cpu()
            images = images.detach().numpy()

            preds_size = torch.IntTensor([preds.size(0)] * preds.size(1))

            output_batch = preds.permute(1, 0, 2)
            out = output_batch.data.cpu().numpy()

            loss = criterion(preds, labels, preds_size, label_lengths)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print("BATCH LOSS:", loss)
            print()

            for j in range(out.shape[0]):
                logits = out[j, ...]
                pred, raw_pred = naive_decode(logits)
                pred_str = label2str(pred, dataset.idx_to_char, False)

                gt_str = label2str(labels[j].cpu().numpy(), dataset.idx_to_char, False)
                print("GROUND TRUTH:", gt_str)
                print("PREDICTION:", pred_str)
                cer_loss = cer(gt_str, pred_str)
                print("CER:", cer_loss)
                sum_loss += cer_loss
                steps += 1

                if cer_loss < epoch_best:
                    epoch_best = cer_loss
                    print('saving epoch best')
                    plt.imshow(images[j])
                    plt.title(gt_str)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    plt.xlabel(pred_str)
                    plt.savefig(os.path.join(result_dir, f'epoch_{epoch}.png'))

                if cer_loss < overall_best:
                    overall_best = cer_loss
                    print('saving new overall best!!!!')
                    plt.savefig(os.path.join(result_dir, 'best.png'))

                print()

        training_cer = sum_loss / steps
        print("CER:", training_cer)

def main(n_epochs, results_dir, dataset, name):

    result_dir = os.path.join('results', name)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)

    img_height = 8  # with upsampling this becomes 60
    start_token = torch.zeros(img_height)
    end_token = torch.zeros(img_height)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    online = False

    if dataset == 'words':
        dataset = WordsDataset('data/raw_text_10000.txt')
    elif dataset == 'letters':
        dataset = LetterDataset(size=10)
    else:
        raise ValueError(f'dataset must be either "words" or "letters", not {dataset}')

    data_generator = DataLoader(dataset, batch_size=10, shuffle=True, collate_fn=collate)

    encoder = Encoder(num_embeddings=dataset.n_chars + 1, embed_size=12, output_size=img_height)
    decoder = Decoder(input_size=img_height, output_size=img_height, start_token=start_token, end_token=end_token)
    upsampler = Upsampler()
    hwr = create_model({
        'cnn_out_size': 1024,
        'num_of_channels': 3,
        'num_of_outputs': 80
    })
    hwr.load_state_dict(torch.load('models/hwr.pt', map_location=device))
    hw_generator = HWGenerator(encoder, decoder, upsampler, hwr)

    criterion = torch.nn.CTCLoss()
    optimizer = torch.optim.Adam(params=torch.nn.ModuleList([encoder, decoder, upsampler]).parameters(), lr=0.001)

    train(n_epochs, hw_generator, optimizer, criterion, data_generator, dataset, online, result_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', metavar='E', default=1000)
    parser.add_argument('--results_dir', metavar='R', default='./results')
    parser.add_argument('--dataset', metavar='D', default='letters')
    parser.add_argument('--name', metavar='N', default=str(time.time()))
    args = parser.parse_args()

    main(args.n_epochs, args.results_dir, args.dataset, args.name)



