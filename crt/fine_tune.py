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
        #active analogs
        default='../data/finetune_data.csv',
        help="name of the dataset to train on")

    parser.add_argument('--property_data', type=str,
        #Morgan fingerprints of active analogs
        default  = '../data/finetune_data_167maccs.csv',
        help="name of the property dataset with MACCS fingerprints")#, required=False)

    parser.add_argument('--block_size', type=int,
                    default = 100,
                    help="number of layers", required=False)
                    #default = 100

    parser.add_argument('--vocab_size', type=int,
                    default = 59, #for ChEMBL
                    #default = 81, #for ChEMBL
                    #default = 79, #for  ChEMBL
                    help="number of layers", required=False)

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

    parser.add_argument('--max_epochs', type=int, default = 55,
            help="total epochs", required=False)

    #the batch size is affected by the size of the underlying dataset
    parser.add_argument('--batch_size', type=int,
            default = 35, #analogs; larger dataset，default = 130
            help="batch size", required=False)

    parser.add_argument('--learning_rate', type=int,
            default = 1e-4,
            help="learning rate", required=False)

    parser.add_argument('--lstm_layers', type=int, default = 2,
            help="number of layers in lstm", required=False)

    #changed NA to 2 vs models folder; and 1 to 2
    parser.add_argument('--char_save', type=str,
        help="path where to save characters file",
        #for character size 81 - Analogs
        default = '../models/prop167_chars.csv')

    parser.add_argument('--mpath', type=str, help="path to load trained model",
        # pre-trained model
        default = "../models/maccsprop.pt")

    parser.add_argument('--ck_path', type=str,
        help="path where save model",
        #active analogs
        default="../models/maccsprop_ft.pt")

    args = parser.parse_args()

    set_seed(42)

    data = pd.read_csv(args.data_name)
    data = data.dropna(axis=0).reset_index(drop=True)
    data.columns = data.columns.str.lower()

    trn = int(len(data)*.8)
    trn
    val = len(data) - trn

    train_data = data[:trn]
    val_data = data[trn:]

    print('trn ',train_data.head())
    print('val ',val_data.head())

    smiles = train_data['smiles']
    smiles[:5]
    vsmiles = val_data['smiles']
    print('smiles ',smiles.shape,smiles[:5])
    print('vsmiles ',vsmiles.shape)

    allprop = pd.read_csv(args.property_data)
    allprop = allprop.reset_index(drop=True)

    print('allprop ',allprop.shape)

    prop = allprop[:trn]
    vprop = allprop[trn:]
    print('len prop ',len(prop))
    print()
    print('len vprop ',len(vprop))
    print()

    print('finished loading data files')
    print()

    scaffold = smiles
    vscaffold = vsmiles

    pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)

    lens = [len(regex.findall(i.strip())) for i in (list(smiles.values) + list(vsmiles.values))]
    max_len = max(lens)
    print('max len ',max_len)
    lens = [len(regex.findall(i.strip())) for i in (list(scaffold.values) + list(vscaffold.values))]
    scaffold_max_len = max(lens)

    smiles = [ i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in smiles]
    vsmiles = [ i + str('<')*(max_len - len(regex.findall(i.strip()))) for i in vsmiles]
    print('smiles ',len(smiles),smiles[:5])
    print('vsmiles ',len(vsmiles))

    print(len(smiles[1]),smiles[1])
    scaffold = [ i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in scaffold]
    vscaffold = [ i + str('<')*(scaffold_max_len - len(regex.findall(i.strip()))) for i in vscaffold]
                     
    if args.vocab_size == 59:
        whole_string = ['#', '(', ')', '-', '1', '2', '3','4', '5', '6', '7', '8', '9', '<', 
    '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH2-]', '[BH3-]', '[C-]', 
    '[C+]', '[I+]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[O+]', '[O-]', '[OH+]', '[O]',
    '[P+]', '[PH]', '[S+]', '[S-]', '[SH]', '[SH2]', '[c+]', '[c-]', '[cH-]', '[n+]', '[n-]', '[nH+]',
    '[nH]', '[o+]', '[s+]', 'c', 'n', 'o', 's']

    if args.vocab_size == 81:

        whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3',
                      '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I',
                      'N', 'O', 'P', 'S', '[11CH2]', '[11CH3]', '[11C]', '[123I]', '[125I]',
                      '[13CH]', '[13C]', '[17F]', '[18F]', '[19F]', '[2H]', '[35S]', '[3H]',
                      '[76Br]', '[As]', '[B-]', '[BH3-]', '[C-]', '[C@@H]', '[C@H]', '[CH+]',
                      '[F+]', '[IH2]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]',
                      '[O+]', '[O-]', '[O]', '[P+]', '[PH]', '[S+]', '[S-]', '[SH]', '[Se+]',
                      '[SeH]', '[Se]', '[Si]', '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]',
                      '[se]', '[te]', 'c', 'n', 'o', 's']

    if args.vocab_size == 79:

        whole_string = ['#', '%10', '%11', '%12', '(', ')', '-', '1', '2', '3',
                      '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C',
                      'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[11CH2]', '[11CH3]',
                      '[11C]', '[123I]', '[125I]', '[13CH]', '[13C]', '[17F]',
                      '[18F]', '[19F]', '[2H]', '[35S]', '[3H]', '[76Br]',
                      '[As]', '[B-]', '[BH3-]', '[C-]', '[CH+]', '[F+]',
                      '[IH2]', '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]',
                      '[NH3+]', '[O+]', '[O-]', '[O]', '[P+]', '[PH]', '[S+]',
                      '[S-]', '[SH]', '[Se+]', '[SeH]', '[Se]', '[Si]', '[n+]',
                      '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '[se]', '[te]',
                      'c', 'n', 'o', 's']

    if args.vocab_size == 86:

        whole_string = ['#', '(', ')', '-', '1', '2', '3', '4', '5', '6', '7', 
                      '<', '=', 'B', 'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S',
                      '[B-]', '[BH-]', '[BH2-]', '[BH3-]', '[Br+2]', '[C+]', '[C-]',
                      '[CH+]', '[CH-]', '[CH2+]', '[CH2-]', '[Cl+2]', '[Cl+3]',
                      '[Cl+]', '[ClH+]', '[I+]', '[I-]', '[IH+]', '[N+]', '[N-]', '[NH+]',
                      '[NH-]', '[NH2+]', '[NH3+]', '[O+]', '[O-]', '[OH+]', '[OH2+]',
                      '[P+]', '[P-]', '[PH+]', '[PH-]', '[PH2+]', '[PH3+]', '[PH]',
                      '[S+]', '[S-]', '[SH+]', '[SH2+]', '[SH2]', '[SH4]', '[SH]',
                      '[b-]', '[c+]', '[c-]', '[cH-]', '[cH+]', '[n+]', '[n-]', '[nH+]',
                      '[nH]', '[o+]', '[oH+]', '[p+]', '[pH]', '[s+]', '[sH+]', 'b', 'p',
                      'c', 'n', 'o', 's']

    if args.vocab_size == 76:

        whole_string = ['#', '%10', '%11', '%12', '%13', '%14', '%15', '%16', 
                      '%17', '%18', '(', ')', '-', '1', '2', '3',
                      '4', '5', '6', '7', '8', '9', '<', '=', 'B', 'Br', 'C',
                      'Cl', 'F', 'I', 'N', 'O', 'P', 'S', '[B-]', '[BH2-]',
                      '[BH3-]', '[C+]', '[C-]', '[CH-]', '[CH2]', '[CH]',
                      '[I+]', '[IH]', '[N+]', '[N-]', '[NH+]', '[NH-]',
                      '[NH2+]', '[O+]', '[O-]', '[OH+]', '[O]', '[P+]',
                      '[PH]', '[S+]', '[S-]', '[SH-]', '[SH2]', '[SH]',
                      '[b-]', '[c+]', '[c-]', '[cH-]', '[n+]', '[n-]', '[nH+]',
                      '[nH]', '[o+]', '[s+]', 'b', 'p', 'c', 'n', 'o', 's']
                      
    print('whole string ',len(whole_string),whole_string)
    print()

    scaffold = smiles
    vscaffold = vsmiles

    train_dataset = SmileDataset(args, smiles, whole_string, max_len, prop = prop,
        aug_prob = 0, scaffold = scaffold, scaffold_maxlen = scaffold_max_len)

    valid_dataset = SmileDataset(args, vsmiles, whole_string, max_len, prop = vprop,
        aug_prob = 0, scaffold = vscaffold, scaffold_maxlen = scaffold_max_len)

    mconf = GPTConfig(args.vocab_size, args.block_size, num_props = args.num_props,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,fingerprint=args.fingerprint,
                   scaffold = args.scaffold,
                   #scaffold_maxlen = scaffold_max_len,
                   scaffold_maxlen = args.block_size,
	               lstm = args.lstm, lstm_layers = args.lstm_layers)
    
    print(mconf.fingerprint,args.fingerprint)

    model = GPT(mconf)

    model.load_state_dict(torch.load(args.mpath), False)# + args.model_weight))

    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    decay=True, warmup_tokens=0.1*len(train_data)*max_len,
                    final_tokens=args.max_epochs*len(train_data)*max_len,
                    num_workers=0,
                    ckpt_path = args.ck_path)
    trainer = Trainer(model, train_dataset, valid_dataset, tconf)

    trainer.train()