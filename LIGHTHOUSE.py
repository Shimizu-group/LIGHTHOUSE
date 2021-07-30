# -*- coding: utf-8 -*-
"""
This is a script to setup LIGHTHOUSE.
"""

import os
import re
import math
import pickle
import codecs
import copy
import sys
from time import time


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import rdkit
import rdkit.Chem as Chem


from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate


torch.manual_seed(2)
np.random.seed(3)


from subword_nmt.apply_bpe import BPE

# AACs
AALetter = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "E",
    "Q",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]


def Getkmers():
    kmers = list()
    for i in AALetter:
        for j in AALetter:
            for k in AALetter:
                kmers.append(i + j + k)
    return kmers


def GetSpectrumDict(proteinsequence):
    result = {}
    kmers = Getkmers()
    for i in kmers:
        result[i] = len(re.findall(i, proteinsequence))
    return result

def CalculateAAComposition(ProteinSequence):
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        Result[i] = round(float(ProteinSequence.count(i)) / LengthSequence * 100, 3)
    return Result

def CalculateDipeptideComposition(ProteinSequence):
    LengthSequence = len(ProteinSequence)
    Result = {}
    for i in AALetter:
        for j in AALetter:
            Dipeptide = i + j
            Result[Dipeptide] = round(
                float(ProteinSequence.count(Dipeptide)) / (LengthSequence - 1) * 100, 2
            )
    return Result


def CalculateAADipeptideComposition(ProteinSequence):
    result = {}
    result.update(CalculateAAComposition(ProteinSequence))
    result.update(CalculateDipeptideComposition(ProteinSequence))
    result.update(GetSpectrumDict(ProteinSequence))

    return np.array(list(result.values()))

#MPNN
def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.Kekulize(mol)
    return mol

ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown']
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6

def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def atom_features(atom):
    return torch.Tensor(onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5])
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + onek_encoding_unk(int(atom.GetChiralTag()), [0,1,2,3])
            + [atom.GetIsAromatic()])


def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

vocab_path = './LIGHTHOUSE_data/LIGHTHOUSE_protein_codes.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv('./LIGHTHOUSE_data/LIGHTHOUSE_subword_units.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))


def create_var(tensor, requires_grad=None):
    if requires_grad is None:
        return Variable(tensor)
    else:
        return Variable(tensor, requires_grad=requires_grad)


def length_func(list_or_tensor):
	if type(list_or_tensor)==list:
		return len(list_or_tensor)
	return list_or_tensor.shape[0]

def index_select_ND(source, dim, index):
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)


def smiles2mpnnfeature(smiles):

	try:
		padding = torch.zeros(ATOM_FDIM + BOND_FDIM)
		fatoms, fbonds = [], [padding]
		in_bonds,all_bonds = [], [(-1,-1)]
		mol = get_mol(smiles)
		n_atoms = mol.GetNumAtoms()
		for atom in mol.GetAtoms():
			fatoms.append( atom_features(atom))
			in_bonds.append([])

		for bond in mol.GetBonds():
			a1 = bond.GetBeginAtom()
			a2 = bond.GetEndAtom()
			x = a1.GetIdx()
			y = a2.GetIdx()

			b = len(all_bonds)
			all_bonds.append((x,y))
			fbonds.append( torch.cat([fatoms[x], bond_features(bond)], 0) )
			in_bonds[y].append(b)

			b = len(all_bonds)
			all_bonds.append((y,x))
			fbonds.append( torch.cat([fatoms[y], bond_features(bond)], 0) )
			in_bonds[x].append(b)

		total_bonds = len(all_bonds)
		fatoms = torch.stack(fatoms, 0)
		fbonds = torch.stack(fbonds, 0)
		agraph = torch.zeros(n_atoms,MAX_NB).long()
		bgraph = torch.zeros(total_bonds,MAX_NB).long()
		for a in range(n_atoms):
			for i,b in enumerate(in_bonds[a]):
				agraph[a,i] = b

		for b1 in range(1, total_bonds):
			x,y = all_bonds[b1]
			for i,b2 in enumerate(in_bonds[x]):
				if all_bonds[b2][0] != y:
					bgraph[b1,i] = b2
	except:
		fatoms = torch.zeros(0,39)
		fbonds = torch.zeros(0,50)
		agraph = torch.zeros(0,6)
		bgraph = torch.zeros(0,6)
	Natom, Nbond = fatoms.shape[0], fbonds.shape[0]
	shape_tensor = torch.Tensor([Natom, Nbond]).view(1,-1)
	return [fatoms.float(), fbonds.float(), agraph.float(), bgraph.float(), shape_tensor.float()]


# Data process
def data_process(X_drug, X_target = None, y=None, drug_encoding='MPNN', target_encoding=None, random_seed = 1, sample_frac = 1):

	y = [-1]*len(X_drug)

	if X_target is not None:
		if isinstance(X_target, str):
			X_target = [X_target]
		if len(X_target) == 1:
			X_target = np.tile(X_target, (length_func(X_drug), ))

	if X_target is not None:
		df_data = pd.DataFrame(zip(X_drug, X_target, y))
		df_data.rename(columns={0:'SMILES',
								1: 'Target Sequence',
								2: 'Label'},
								inplace=True)
	unique = pd.Series(df_data['SMILES'].unique()).apply(smiles2mpnnfeature)
	unique_dict = dict(zip(df_data['SMILES'].unique(), unique))
	df_data['drug_encoding'] = [unique_dict[i] for i in df_data['SMILES']]


	if target_encoding == 'AAC':

		AA = pd.Series(df_data['Target Sequence'].unique()).apply(CalculateAADipeptideComposition)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]

	elif target_encoding == 'CNN':
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(trans_protein)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]
		# the embedding is large and not scalable but quick, so we move to encode in dataloader batch.

	elif target_encoding == 'Transformer':
		AA = pd.Series(df_data['Target Sequence'].unique()).apply(protein2emb_encoder)
		AA_dict = dict(zip(df_data['Target Sequence'].unique(), AA))
		df_data['target_encoding'] = [AA_dict[i] for i in df_data['Target Sequence']]

	return df_data.reset_index(drop=True)

def LIGHTHOUSE_preprocess(candidate_compound, target, drug_encoding, target_encoding):


	target = target
	df = data_process(candidate_compound, target, drug_encoding = drug_encoding,
								target_encoding = target_encoding)

	return df

class data_process_loader(data.Dataset):

	def __init__(self, list_IDs, labels, df, **config):
		self.labels = labels
		self.list_IDs = list_IDs
		self.df = df
		self.config = config

	def __len__(self):
		return len(self.list_IDs)

	def __getitem__(self, index):
		index = self.list_IDs[index]
		v_d = self.df.iloc[index]['drug_encoding']
		if self.config['drug_encoding'] == 'CNN' or self.config['drug_encoding'] == 'CNN_RNN':
			v_d = drug_2_embed(v_d)
		v_p = self.df.iloc[index]['target_encoding']
		if self.config['target_encoding'] == 'CNN' or self.config['target_encoding'] == 'CNN_RNN':
			v_p = protein_2_embed(v_p)
		y = self.labels[index]
		return v_d, v_p, y

# Embeddings
def protein2emb_encoder(x):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i, np.asarray(input_mask)

def drug2emb_encoder(x):

    max_d = 50
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


amino_char = ['?', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I', 'H', 'K', 'M', 'L', 'O',
       'N', 'Q', 'P', 'S', 'R', 'U', 'T', 'W', 'V', 'Y', 'X', 'Z']

smiles_char = ['?', '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
       '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
       'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
       'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
       'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y']


enc_protein = OneHotEncoder().fit(np.array(amino_char).reshape(-1, 1))
enc_drug = OneHotEncoder().fit(np.array(smiles_char).reshape(-1, 1))

MAX_SEQ_PROTEIN = 1000
MAX_SEQ_DRUG = 100

def trans_protein(x):
	temp = list(x.upper())
	temp = [i if i in amino_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_PROTEIN:
		temp = temp + ['?'] * (MAX_SEQ_PROTEIN-len(temp))
	else:
		temp = temp [:MAX_SEQ_PROTEIN]
	return temp

def protein_2_embed(x):
	return enc_protein.transform(np.array(x).reshape(-1,1)).toarray().T

def trans_drug(x):
	temp = list(x)
	temp = [i if i in smiles_char else '?' for i in temp]
	if len(temp) < MAX_SEQ_DRUG:
		temp = temp + ['?'] * (MAX_SEQ_DRUG-len(temp))
	else:
		temp = temp [:MAX_SEQ_DRUG]
	return temp

def drug_2_embed(x):
	return enc_drug.transform(np.array(x).reshape(-1,1)).toarray().T

# Utils
def load_dict(path):
	with open(path, 'rb') as f:
		return pickle.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Layers
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, variance_epsilon=1e-12):

        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_size, dropout_rate):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_size, hidden_size)

        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, hidden_size, hidden_dropout_prob):
        super(SelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Attention, self).__init__()
        self.self = SelfAttention(hidden_size, num_attention_heads, attention_probs_dropout_prob)
        self.output = SelfOutput(hidden_size, hidden_dropout_prob)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = F.relu(hidden_states)
        return hidden_states

class Output(nn.Module):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class Encoder(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder, self).__init__()
        self.attention = Attention(hidden_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.intermediate = Intermediate(hidden_size, intermediate_size)
        self.output = Output(intermediate_size, hidden_size, hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, n_layer, hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder(hidden_size, intermediate_size, num_attention_heads, attention_probs_dropout_prob, hidden_dropout_prob)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layer)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            #if output_all_encoded_layers:
            #    all_encoder_layers.append(hidden_states)
        #if not output_all_encoded_layers:
        #    all_encoder_layers.append(hidden_states)
        return hidden_states


class transformer(nn.Sequential):
	def __init__(self, encoding, **config):
		super(transformer, self).__init__()

		self.emb = Embeddings(config['input_dim_protein'], config['transformer_emb_size_target'], 545, config['transformer_dropout_rate'])
		self.encoder = Encoder_MultipleLayers(config['transformer_n_layer_target'],
													config['transformer_emb_size_target'],
													config['transformer_intermediate_size_target'],
													config['transformer_num_attention_heads_target'],
													config['transformer_attention_probs_dropout'],
													config['transformer_hidden_dropout_rate'])

	### parameter v (tuple of length 2) is from utils.drug2emb_encoder
	def forward(self, v):
		e = v[0].long().to(device)
		e_mask = v[1].long().to(device)
		ex_e_mask = e_mask.unsqueeze(1).unsqueeze(2)
		ex_e_mask = (1.0 - ex_e_mask) * -10000.0

		emb = self.emb(e)
		encoded_layers = self.encoder(emb.float(), ex_e_mask.float())
		return encoded_layers[:,0]


class CNN(nn.Sequential):
	def __init__(self, encoding, **config):
		super(CNN, self).__init__()

		in_ch = [26] + config['cnn_target_filters']
		kernels = config['cnn_target_kernels']
		layer_size = len(config['cnn_target_filters'])
		self.conv = nn.ModuleList([nn.Conv1d(in_channels = in_ch[i],
												out_channels = in_ch[i+1],
												kernel_size = kernels[i]) for i in range(layer_size)])
		self.conv = self.conv.double()
		n_size_p = self._get_conv_output((26, 1000))

		self.fc1 = nn.Linear(n_size_p, config['hidden_dim_protein'])

	def _get_conv_output(self, shape):
		bs = 1
		input = Variable(torch.rand(bs, *shape))
		output_feat = self._forward_features(input.double())
		n_size = output_feat.data.view(bs, -1).size(1)
		return n_size

	def _forward_features(self, x):
		for l in self.conv:
			x = F.relu(l(x))
		x = F.adaptive_max_pool1d(x, output_size=1)
		return x

	def forward(self, v):
		v = self._forward_features(v.double())
		v = v.view(v.size(0), -1)
		v = self.fc1(v.float())
		return v


class MLP(nn.Sequential):
	def __init__(self, input_dim, output_dim, hidden_dims_lst):
		super(MLP, self).__init__()
		layer_size = len(hidden_dims_lst) + 1
		dims = [input_dim] + hidden_dims_lst + [output_dim]

		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v):
		# predict
		v = v.float().to(device)
		for i, l in enumerate(self.predictor):
			v = F.relu(l(v))
		return v

class MPNN(nn.Sequential):

	def __init__(self, mpnn_hidden_size, mpnn_depth):
		super(MPNN, self).__init__()
		self.mpnn_hidden_size = mpnn_hidden_size
		self.mpnn_depth = mpnn_depth


		self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, self.mpnn_hidden_size, bias=False)
		self.W_h = nn.Linear(self.mpnn_hidden_size, self.mpnn_hidden_size, bias=False)
		self.W_o = nn.Linear(ATOM_FDIM + self.mpnn_hidden_size, self.mpnn_hidden_size)

	def forward(self, feature):
		fatoms, fbonds, agraph, bgraph, atoms_bonds = feature
		agraph = agraph.long()
		bgraph = bgraph.long()
		atoms_bonds = atoms_bonds.long()
		batch_size = atoms_bonds.shape[0]
		N_atoms, N_bonds = 0, 0
		embeddings = []
		for i in range(batch_size):
			n_a = atoms_bonds[i,0].item()
			n_b = atoms_bonds[i,1].item()
			if (n_a == 0):
				embed = create_var(torch.zeros(1, self.mpnn_hidden_size))
				embeddings.append(embed.to(device))
				continue
			sub_fatoms = fatoms[N_atoms:N_atoms+n_a,:].to(device)
			sub_fbonds = fbonds[N_bonds:N_bonds+n_b,:].to(device)
			sub_agraph = agraph[N_atoms:N_atoms+n_a,:].to(device)
			sub_bgraph = bgraph[N_bonds:N_bonds+n_b,:].to(device)
			embed = self.single_molecule_forward(sub_fatoms, sub_fbonds, sub_agraph, sub_bgraph)
			embed = embed.to(device)
			embeddings.append(embed)
			N_atoms += n_a
			N_bonds += n_b
		try:
			embeddings = torch.cat(embeddings, 0)
		except:
			print(embeddings)
		return embeddings


	def single_molecule_forward(self, fatoms, fbonds, agraph, bgraph):
		fatoms = create_var(fatoms)
		fbonds = create_var(fbonds)
		agraph = create_var(agraph)
		bgraph = create_var(bgraph)
		binput = self.W_i(fbonds)
		message = F.relu(binput)
		for i in range(self.mpnn_depth - 1):
			nei_message = index_select_ND(message, 0, bgraph)
			nei_message = nei_message.sum(dim=1)
			nei_message = self.W_h(nei_message)
			message = F.relu(binput + nei_message)

		nei_message = index_select_ND(message, 0, agraph)
		nei_message = nei_message.sum(dim=1)
		ainput = torch.cat([fatoms, nei_message], dim=1)
		atom_hiddens = F.relu(self.W_o(ainput))
		return torch.mean(atom_hiddens, 0).view(1,-1).to(device)

class Classifier(nn.Sequential):
	def __init__(self, model_drug, model_protein, **config):
		super(Classifier, self).__init__()
		self.input_dim_drug = config['hidden_dim_drug']
		self.input_dim_protein = config['hidden_dim_protein']
		self.model_drug = model_drug
		self.model_protein = model_protein
		self.dropout = nn.Dropout(0.1)
		self.hidden_dims = config['cls_hidden_dims']
		layer_size = len(self.hidden_dims) + 1
		dims = [self.input_dim_drug + self.input_dim_protein] + self.hidden_dims + [1]

		self.predictor = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(layer_size)])

	def forward(self, v_D, v_P):
		# each encoding
		v_D = self.model_drug(v_D)
		v_P = self.model_protein(v_P)
		# concatenate and classify
		v_f = torch.cat((v_D, v_P), 1)
		for i, l in enumerate(self.predictor):
			if i==(len(self.predictor)-1):
				v_f = l(v_f)
			else:
				v_f = F.relu(self.dropout(l(v_f)))
		return v_f

def model_initialize(**config):
	model = DBTA(**config)
	return model

def LIGHTHOUSE_load_pretrained(path = None):
	config = load_dict(path + '_settings.pkl')
	model = DBTA(**config)
	model.load_pretrained(path + '_parameters.pt')
	return model

def LIGHTHOUSE_compute (candidate_compound, target_sequence, model):

	data = LIGHTHOUSE_preprocess(candidate_compound, target_sequence, model.drug_encoding, model.target_encoding)
	y_pred = model.predict(data)
	return y_pred


def mpnn_feature_collate_func(x):
	return [torch.cat([x[j][i] for j in range(len(x))], 0) for i in range(len(x[0]))]

def mpnn_collate_func(x):
	mpnn_feature = [i[0] for i in x]
	mpnn_feature = mpnn_feature_collate_func(mpnn_feature)
	x_remain = [[i[1], i[2]] for i in x]
	x_remain_collated = default_collate(x_remain)
	return [mpnn_feature] + x_remain_collated


class DBTA:

	def __init__(self, **config):
		target_encoding = config['target_encoding']
		drug_encoding = config['drug_encoding']

		self.model_drug = MPNN(config['hidden_dim_drug'], config['mpnn_depth'])

		if target_encoding == 'AAC':
			self.model_protein = MLP(config['input_dim_protein'], config['hidden_dim_protein'], config['mlp_hidden_dims_target'])
		elif target_encoding == 'CNN':
			self.model_protein = CNN('protein', **config)
		elif target_encoding == 'Transformer':
			self.model_protein = transformer('protein', **config)

		self.model = Classifier(self.model_drug, self.model_protein, **config)
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		self.drug_encoding = drug_encoding
		self.target_encoding = target_encoding
		if 'num_workers' not in self.config.keys():
			self.config['num_workers'] = 0
		if 'decay' not in self.config.keys():
			self.config['decay'] = 0

	def test_(self, data_generator, model, test = False):
		y_pred = []
		y_label = []
		model.eval()
		for i, (v_d, v_p, label) in enumerate(data_generator):
			if self.drug_encoding == "MPNN" or self.drug_encoding == 'Transformer':
				v_d = v_d
			else:
				v_d = v_d.float().to(self.device)
			if self.target_encoding == 'Transformer':
				v_p = v_p
			else:
				v_p = v_p.float().to(self.device)
			score = self.model(v_d, v_p)
			logits = torch.squeeze(score).detach().cpu().numpy()
			label_ids = label.to('cpu').numpy()
			y_label = y_label + label_ids.flatten().tolist()
			y_pred = y_pred + logits.flatten().tolist()
			outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
		return y_pred

	def predict(self, df_data):
		info = data_process_loader(df_data.index.values, df_data.Label.values, df_data, **self.config)
		self.model.to(device)
		params = {'batch_size': self.config['batch_size'],
				'shuffle': False,
				'num_workers': self.config['num_workers'],
				'drop_last': False,
				'sampler':SequentialSampler(info)}

		if (self.drug_encoding == "MPNN"):
			params['collate_fn'] = mpnn_collate_func

		generator = data.DataLoader(info, **params)
		score = self.test_(generator, self.model)
		return score


	def load_pretrained(self, path):
		if not os.path.exists(path):
			os.makedirs(path)

		if self.device == 'cuda':
			state_dict = torch.load(path)
		else:
			state_dict = torch.load(path, map_location = torch.device('cpu'))

		self.model.load_state_dict(state_dict)

# LIGHTHOUSE
def setups ():
    global module1_CNN, module1_AAC, module1_Transformer, module2_CNN, module2_AAC, module2_Transformer
    print ("LIGHTHOUSE is now loading pre-trained parameters.")
    module1_CNN = LIGHTHOUSE_load_pretrained(path = './LIGHTHOUSE_data/MODULE1_MPNN-CNN')
    module1_AAC = LIGHTHOUSE_load_pretrained(path = './LIGHTHOUSE_data/MODULE1_MPNN-AAC')
    module1_Transformer = LIGHTHOUSE_load_pretrained(path = './LIGHTHOUSE_data/MODULE1_MPNN-Transformer')
    module2_CNN = LIGHTHOUSE_load_pretrained(path = './LIGHTHOUSE_data/MODULE2_MPNN-CNN')
    module2_AAC = LIGHTHOUSE_load_pretrained(path = './LIGHTHOUSE_data/MODULE2_MPNN-AAC')
    module2_Transformer = LIGHTHOUSE_load_pretrained(path = './LIGHTHOUSE_data/MODULE2_MPNN-Transformer')
    print ("Loading finished! Now you can use LIGHTHOUSE.")

def predict_confidence (chemical, protein):
    print ("LIGHTHOUSE is now calculating confidence score between")
    print ("chemical: {}".format (chemical))
    print ("and")
    print ("protein: {}".format (protein))
    print ("Please wait for seconds...")
    print ("---------------------------")
    CNN_score = LIGHTHOUSE_compute (candidate_compound = [chemical], target_sequence = [protein] , model = module1_CNN)
    Transformer_score = LIGHTHOUSE_compute (candidate_compound = [chemical], target_sequence = [protein] , model = module1_Transformer)
    AAC_score = LIGHTHOUSE_compute (candidate_compound = [chemical], target_sequence = [protein] , model = module1_AAC)
    confidence_score = 3/ (1/CNN_score[0] + 1/Transformer_score[0] + 1/AAC_score[0])
    print ("Finished. The confidence score is {:.3f}".format (confidence_score))


def predict_interaction (chemical, protein):
    scaleIC50 = {4: "100 uM", 5: "10 uM", 6: "1 uM", 7: "100 nM", 8: "10 nM", 9: "1 nM", 10: "pM scale"}
    print ("LIGHTHOUSE is now calculating interaction score between")
    print ("chemical: {}".format (chemical))
    print ("and")
    print ("protein: {}".format (protein))
    print ("Please wait for seconds...")
    print ("---------------------------")
    CNN_score =LIGHTHOUSE_compute (candidate_compound = [chemical], target_sequence = [protein] , model = module2_CNN)
    Transformer_score = LIGHTHOUSE_compute (candidate_compound = [chemical], target_sequence = [protein] , model = module2_Transformer)
    AAC_score = LIGHTHOUSE_compute (candidate_compound = [chemical], target_sequence = [protein] , model = module2_AAC)
    IC50_score = 3/ (1/CNN_score[0] + 1/Transformer_score[0] + 1/AAC_score[0])
    print ("Finished. The interaction score is {:.3f}".format (IC50_score))
    try:
        print ("Estimated IC50 is between {} and {}.".format (scaleIC50[math.ceil (IC50_score)], scaleIC50[math.floor (IC50_score)]))
    except:
        print ("The chemical would NOT inhibit the protein.")

def main ():
    print ("Done")
    sys.path.append(os.getcwd())
    
if __name__ == "__main__":
    main()