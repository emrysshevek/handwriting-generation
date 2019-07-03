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

img_height = 8
start_token = torch.zeros(img_height)
end_token = torch.zeros(img_height)

device = torch.device('cpu')
online = False
n_epochs = 100000

# dataset = WordsDataset('data/raw_text_10000.txt')
dataset = LetterDataset(size=10)

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
overall_best = np.inf

for epoch in range(n_epochs):
    print("---------------------------------------")
    print("EPOCH", epoch)
    print("---------------------------------------")
    sum_loss = 0.0
    steps = 0.0
    hw_generator.train()
    epoch_best = np.inf
    for labels, label_lengths in data_generator:
        preds, images = hw_generator(labels, online)
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
                plt.xlabel(pred_str)
                plt.savefig(f'results/epoch_{epoch}.png')

            if cer_loss < overall_best:
                overall_best = cer_loss
                print('saving new overall best!!!!')
                plt.savefig(f'results/best.png')

            print()

    training_cer = sum_loss / steps
    print("CER:", training_cer)



