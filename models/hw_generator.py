import torch
from torch import nn


class HWGenerator(nn.Module):

    def __init__(self, encoder, decoder, upsampler, hwr):
        super(HWGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.upsampler = upsampler
        self.hwr = hwr
        self.activation = nn.functional.log_softmax

    def forward(self, labels, online):
        embedding_vectors = self.encoder(labels)
        feature_vectors = self.decoder(embedding_vectors)
        images = self.upsampler(feature_vectors)
        preds = self.hwr(images, online)
        return self.activation(preds, dim=1), images.permute(0, 2, 3, 1)
