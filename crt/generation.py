# This code was adapted from LigGPT https://github.com/devalab/molgpt
# with modifications.

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from utils import check_novelty, sample, canonic_smiles, get_mol
from dataset import SmileDataset

#model version with diversity; used for molecule generation
from model_div import GPT, GPTConfig 

from rdkit.Chem import QED
from rdkit.Chem import Crippen
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit import Chem
from rdkit import RDConfig
from rdkit.Chem import FragmentCatalog

import os
import math
from tqdm import tqdm
import argparse
import pandas as pd
import torch
import numpy as np
import re

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--temp', type=float,
        help="temperature to vay generation",
        default =0.9)
            
    parser.add_argument('--mpath', type=str, help="path to load trained model",
        #pre-trained model
        default = "../models/maccsprop_ft.pt")
    
    parser.add_argument('--scaffold', action='store_true', default=False,
                     help='condition on scaffold')
    
    parser.add_argument('--lstm', action='store_true', default=False,
                     help='use lstm for transforming scaffold')
    
    parser.add_argument('--cpath', type=str,
        help="name to save the generated mols in csv format",
        default = '../gen/gen_5kTNFa.csv')
        
    parser.add_argument('--property_data', type=str,      
        #Morgan fingerprints of seed active analogs
        default  = '../data/seeds_167maccs.csv',
        help="name of the property dataset with MACCS fingerprints", required=False)
    
    parser.add_argument('--molecule_data', type=str,         
        #seed active analogs
        default  =  '../data/seeds.csv',
        help="name of the property dataset with MACCS fingerprints", required=False)
    
    parser.add_argument('--gen_size', type=int, 
                    default = 1000, 
                    help="number of times to generate from a batch",
                    required=False)
    
    parser.add_argument('--vocab_size', type=int, 
                        #default = 79, #for  ChEMBL
                        #default = 81, #for ChEMBL
                        default = 59, #for ChEMBL
                     help="number of layers", required=False)  
    
    parser.add_argument('--block_size', type=int, 
                    default = 100, 
                    help="block size", required=False)   
    
    parser.add_argument('--num_props', type=int,
                        default = 1,
                     help="number of properties to use for condition",
                     required=False)
    
    parser.add_argument('--n_layer', type=int, default = 8,
                     help="number of layers", required=False)
    
    parser.add_argument('--n_head', type=int, default = 8,
                     help="number of heads", required=False)
    
    parser.add_argument('--n_embd', type=int, default = 256,
                     help="embedding dimension", required=False)

    parser.add_argument('--fingerprint', type=int, default = 167,
            help="fingerprint dimension", required=False)
    
    parser.add_argument('--lstm_layers', type=int, default = 2,
                     help="number of layers in lstm", required=False)
    
    parser.add_argument('--char_save', type=str, 
        help="path where to save characters file",
        #for character size 81 - Analogs
        default = '../models/prop167_chars.csv')    

    args = parser.parse_args()
    
    
    pattern =  "(\[[^\]]+]|<|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    
    chars =  pd.read_csv(args.char_save,
                #usecols=['SMILES'],
                squeeze=True).astype(str).tolist()
    
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    
    mconf = GPTConfig(args.vocab_size, args.block_size, num_props = args.num_props,
	               n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,fingerprint=args.fingerprint,
                   scaffold = args.scaffold, 
                   scaffold_maxlen = args.block_size,
	               lstm = args.lstm, lstm_layers = args.lstm_layers)
    
    model = GPT(mconf)
    
    model.load_state_dict(torch.load(args.mpath))
    
    model.to('cuda')
    
    gen_iter = math.ceil(args.gen_size / 512)
    
    all_dfs = []
    
    prop =  pd.read_csv(args.property_data)
    
    #file later used to compare uniqueness of generated molecules
    gen =  pd.read_csv(args.molecule_data)
    
    glen = len(prop)
    
    mult = int(args.gen_size / (glen*4)) + 1
    
    prop = prop.values
   
    prop = np.vstack([prop]*mult)
    
    prop_smiles = np.vstack([gen]*mult*4)
    prop_smiles=pd.DataFrame(data=prop_smiles,columns=['input_smiles'])
    
    batch_size = len(prop)
    print('Batch size: ',batch_size)
    context = "C"
    
    molecules = []
        
    all_comp = []
    all_mol = []   
        
    count = 0
    for c in range(4):
        print('generating molecules...')
        
        x = torch.tensor([stoi[s] for s in regex.findall(context)],
            dtype=torch.long)[None,...].repeat(int(batch_size), 1).to('cuda')
        
        p = torch.tensor([prop],dtype=torch.float).to('cuda') 
        p = p.permute(1,0,2)
                
        sca = None
        
        y = sample(model, x, args.block_size, temperature=args.temp, sample=True, 
                top_k=None, prop = p, scaffold = sca)
        
        for gen_mol in y:
            
            completion = ''.join([itos[int(i)] for i in gen_mol])
            
            completion = completion.replace('<', '')
            
            all_comp.append(completion)
            mol = get_mol(completion)
            all_mol.append(mol)
            
            if mol:
                molecules.append(mol)
                
        count+=batch_size
       
    
    print('Number of valid molecules generated: ',len(molecules)) 
    
    mol_dict_all = []
    mol_dict = []
   
    for i in all_mol:
        if i ==None:
            
            mol_dict.append({'molecule' : i, 'gen_smiles': None})
        else:
            mol_dict.append({'molecule' : i, 'gen_smiles': Chem.MolToSmiles(i)})
    
    r = pd.DataFrame(mol_dict)
    
    results = pd.concat([prop_smiles, r],axis=1)
    
    canon_smiles = [canonic_smiles(s) for s in results['gen_smiles']]
    
    unique_smiles = list(set(canon_smiles))
    
    novel_ratio = check_novelty(unique_smiles, set(gen))

    print('Valid ratio: ', np.round(len(molecules)/count, 3))
    
    ins = results['input_smiles'].to_list()
    gs = results['gen_smiles'].to_list()
    
    unis = []
    ungs = []
    
    for i in range(len(gs)):
        if gs[i] not in ungs and gs[i]!=None:
            ungs.append(gs[i])
            unis.append(ins[i])
            
    pdunis = pd.DataFrame(data=unis,columns=['input_smiles'])
    pdungs = pd.DataFrame(data=ungs,columns=['gen_smiles'])
    
    results = pd.concat([pdunis, pdungs],axis=1)
    
    results.to_csv(args.cpath, index = False, mode='a')