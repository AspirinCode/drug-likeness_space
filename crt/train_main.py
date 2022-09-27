# This code was adapted from LigGPT https://github.com/devalab/molgpt
# with modifications.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import pandas as pd
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.cuda.amp import GradScaler

from model_reg import GPT, GPTConfig
from training import Trainer, TrainerConfig
from dataset import SmileDataset
from utils import SmilesEnumerator

import math
import re
import random
import time

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug')

    parser.add_argument('--scaffold', action='store_true',
            default=False,
            help='condition on scaffold')

    parser.add_argument('--lstm', action='store_true',
            default=False,
            help='use lstm for transforming scaffold')

    parser.add_argument('--data_name', type=str,
        default = '../data/train_data.csv',
        help="name of the dataset to train on", required=False)

    parser.add_argument('--val_data_name', type=str,
        default = '../data/val_data.csv',
        help="name of the validation dataset to train on", required=False)

    #MACCS fingerprint property files broken into 4 files because csv
    #is limited to 1m entries.  csv loaded faster than np.txt
    #experimented with txt file format, which had slower load time
    parser.add_argument('--prop1', type=str,
        default = '../data/maccs_train1.csv',
        help="name of MACCS fingerprint dataset to train on", required=False)

    parser.add_argument('--prop2', type=str,
        default = '../data/maccs_train2.csv',
        help="name of MACCS fingerprint dataset to train on", required=False)

    parser.add_argument('--prop3', type=str,
        default = '../data/maccs_train3.csv',
        help="name of MACCS fingerprint dataset to train on", required=False)

    parser.add_argument('--prop4', type=str,
        default = '../data/maccs_train4.csv',
        help="name of MACCS fingerprint dataset to train on", required=False)

    parser.add_argument('--val_prop', type=str,
        default = '../data/val_167maccs.csv',
        help="name of MACCS fingerprint dataset to train on", required=False)

    parser.add_argument('--num_props', type=int,
            default = 1,
            help="number of properties to use for condition", required=False)

    parser.add_argument('--prop1_unique', type=int, default = 0,
            help="unique values in that property", required=False)

    parser.add_argument('--n_layer', type=int, default = 8,
            help="number of layers", required=False)

    parser.add_argument('--n_head', type=int, default = 8,
            help="number of heads", required=False)

    parser.add_argument('--n_embd', type=int, default = 256,
            help="embedding dimension", required=False)

    parser.add_argument('--fingerprint', type=int, default = 167,
            help="fingerprint dimension", required=False)

    parser.add_argument('--max_epochs', type=int, default = 20,
            help="total epochs", required=False)

    parser.add_argument('--batch_size', type=int,
            default = 512,
            help="batch size", required=False)
            #batch_size = 256

    parser.add_argument('--learning_rate', type=int, default = 12e-4,
            help="learning rate", required=False)

    parser.add_argument('--lstm_layers', type=int, default = 2,
            help="number of layers in lstm", required=False)

    parser.add_argument('--char_save', type=str,
        help="path where save characters file",
        default = '../data/prop167_chars.csv')

    parser.add_argument('--ck_path', type=str,
        help="path where save model",
        default = '../data/maccsprop.pt')


    ts = time.time()
    print()
    print('Arguments list:')
    args = parser.parse_args()

    for k, v in vars(args).items():
        print(k, ' ',v)
    print()
    set_seed(42)

    data = pd.read_csv(args.data_name)

    train_data = data.dropna(axis=0).reset_index(drop=True)

    print('Sample training data: ',train_data[:5])
    print()
    vdata = pd.read_csv(args.val_data_name)
    val_data = vdata.reset_index(drop=True)
    val_data.columns = val_data.columns.str.lower()
    print('Sample validation data: ',val_data[:5])
    print()
    print('Number of training examples: ',len(train_data))
    print('Number of validation examples: ',len(val_data))
    print()
    smiles = train_data["0"]

    vsmiles = val_data['smiles']

    print('Sample training set smiles: ',smiles[:5])
    print()

    print('Loading MACCS fingerprint files...')

    d1 = pd.read_csv(args.prop1)
    d1_data = d1.reset_index(drop=True)
    d1_data.columns = d1_data.columns.str.lower()
    #print(d1_data[:5])

    d2 = pd.read_csv(args.prop2)
    d2_data = d2.reset_index(drop=True)
    d2_data.columns = d2_data.columns.str.lower()
    #print(d2_data[:5])

    d3 = pd.read_csv(args.prop3)
    d3_data = d3.reset_index(drop=True)
    d3_data.columns = d3_data.columns.str.lower()
    #print(d3_data[:5])

    d4 = pd.read_csv(args.prop4)
    d4_data = d4.reset_index(drop=True)
    d4_data.columns = d4_data.columns.str.lower()

    d = [d1,d2,d3,d4]

    prop = pd.concat(d, axis=0)

    vprop = pd.read_csv(args.val_prop)
    vprop = vprop.reset_index(drop=True)
    vprop.columns = vprop.columns.str.lower()
    #print('vprop ',vprop.shape,vprop.head())
    print()
    tel = time.time()
    print('Time to load: %.2f min' %((tel-ts)/60))
    print('Finished loading data...')
    print()
    print('Pre-Processing data...')
    print()
    scaffold = smiles
    vscaffold = vsmiles

    pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)

    lens = [len(regex.findall(i.strip())) for i in (list(scaffold.values) + list(vscaffold.values))]
    scaffold_max_len = max(lens)

    smiles = [ i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [ i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in vsmiles]

    scaffold = [ i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [ i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold]

    whole_string = ' '.join(smiles + vsmiles + scaffold + vscaffold)
    whole_string = sorted(list(set(regex.findall(whole_string))))

    df = pd.DataFrame(whole_string)
    df.to_csv(args.char_save, index=False)

    scaffold = smiles
    vscaffold = vsmiles

    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop = prop,
        aug_prob = 0, scaffold = scaffold, scaffold_maxlen = scaffold_max_len)

    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop = vprop,
        aug_prob = 0, scaffold = vscaffold, scaffold_maxlen = scaffold_max_len)

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.max_len,
        num_props = args.num_props,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,fingerprint=args.fingerprint,
                   scaffold = args.scaffold, scaffold_maxlen = scaffold_max_len,
	               lstm = args.lstm, lstm_layers = args.lstm_layers)

    model = GPT(mconf)

    print()
    print('Starting training...')
    print()

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size,
                          learning_rate=args.learning_rate,
	                      lr_decay=True, warmup_tokens=0.1*len(train_data)*max_len,
                          final_tokens=args.max_epochs*len(train_data)*max_len,
                          num_workers=0,
                          ckpt_path = args.ck_path)

    trainer = Trainer(model, train_dataset, valid_dataset, tconf)

    tt = time.time()
    print('Time to start training: %.2f min' %((tt-ts)/60))
    print()
    trainer.train()
