from torch.utils.data.dataset import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
import pysam
import torch

class Mouse(Dataset):

    split_list = {
        'train': '/srv/scratch/soumyak/inputs/mouse_liver_adult.train.bed',
        'validate': '/srv/scratch/soumyak/inputs/mouse_liver_adult.validate.bed',
        'test': '/srv/scratch/soumyak/inputs/mouse_liver_adult.test.bed'
    }

    def __init__(self, split='train', transform=None, upsample=0, epoch_size=100000):
        self.split = split
        self.transform = transform
        self.upsample = upsample
        self.epoch_size = epoch_size

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="train_small or split="test" or split="test_small"')

        self.filename = self.split_list[self.split]
        self.data = pd.read_csv(self.filename,header=0,sep='\t',index_col=[0,1,2])
        self.ones = self.data.loc[(self.data > 0).any(axis=1)]
        self.zeros = self.data.loc[(self.data < 1).all(axis=1)]
        self.ref = pysam.FastaFile('/srv/scratch/soumyak/metadata/mm10.genome.fa')
        self.num_gen = 0
        self.total = self.data.shape[0]
        self.pos_num_gen = 0
        self.pos_total = self.ones.shape[0]
        self.neg_num_gen = 0
        self.neg_total = self.zeros.shape[0]
        self.ltrdict = {'a':[1,0,0,0],
               'c':[0,1,0,0],
               'g':[0,0,1,0],
               't':[0,0,0,1],
               'n':[0,0,0,0],
               'A':[1,0,0,0],
               'C':[0,1,0,0],
               'G':[0,0,1,0],
               'T':[0,0,0,1],
               'N':[0,0,0,0]}

    def __getitem__(self, index):

        if self.upsample > 0:

            if index % self.upsample == 0:
                entry = self.ones.index[self.pos_num_gen]
                seq = self.ref.fetch(entry[0], entry[1], entry[2])
                onehot = np.array([self.ltrdict.get(x,[0,0,0,0]) for x in seq])
                label = self.ones[self.pos_num_gen:(self.pos_num_gen + 1)]
                label = np.asarray(label)
                label = label[0]
                self.pos_num_gen += 1
                if self.pos_num_gen == self.pos_total:
                    self.pos_num_gen = 0
                onehot = np.transpose(onehot, (1, 0))
                onehot = np.expand_dims(onehot, 1)
                onehot = torch.tensor(onehot).type(torch.FloatTensor)
                label = torch.tensor(label).type(torch.FloatTensor)

            else:
                entry = self.zeros.index[self.neg_num_gen]
                seq = self.ref.fetch(entry[0], entry[1], entry[2])
                onehot = np.array([self.ltrdict.get(x,[0,0,0,0]) for x in seq])
                label = self.zeros[self.neg_num_gen:(self.neg_num_gen + 1)]
                label = np.asarray(label)
                label = label[0]
                self.neg_num_gen += 1
                if self.neg_num_gen == self.neg_total:
                    self.neg_num_gen = 0
                onehot = np.transpose(onehot, (1, 0))
                onehot = np.expand_dims(onehot, 1)
                onehot = torch.tensor(onehot).type(torch.FloatTensor)
                label = torch.tensor(label).type(torch.FloatTensor)

        else:

            entry = self.data.index[self.num_gen]
            seq = self.ref.fetch(entry[0], entry[1], entry[2])
            onehot = np.array([self.ltrdict.get(x,[0,0,0,0]) for x in seq])
            label = self.data[self.num_gen:(self.num_gen + 1)]
            label = np.asarray(label)
            label = label[0]
            self.num_gen += 1
            if self.num_gen == self.total:
                self.num_gen = 0
            onehot = np.transpose(onehot, (1, 0))
            onehot = np.expand_dims(onehot, 1)
            onehot = torch.tensor(onehot).type(torch.FloatTensor)
            label = torch.tensor(label).type(torch.FloatTensor)

        return onehot, label

    def __len__(self):
        return self.epoch_size
