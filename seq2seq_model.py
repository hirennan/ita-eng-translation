import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import spacy
import random
import pandas as pd
import unicodedata
import re
import random
from utils import get_batch, translate_sentence
import sys
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

if torch.backends.mps.is_available():
    device = torch.device("mps")

spacy_ita = spacy.load('it_core_news_sm')

spacy_eng = spacy.load('en_core_web_sm')

def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

def tokenizer_ita(text):
    return [tok.text for tok in spacy_ita.tokenizer(text)]

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s) \
            if unicodedata.category(c) != 'Mn'
        )

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"[^a-zA-Z!?']+", r" ", s)
    return s.strip()        

#a module that returns datasets for data analysis

def get_data():
    
    source_text = []
    eng_pair = []
    ita_pair = []

    #read file

    with open('ita-eng/ita.txt', 'r') as file:
        source_text = file.read().splitlines()

    #split eng, ita pairs

    for idx, _ in enumerate(source_text):
        eng_pair.append(normalize_string(_.split('\t')[0]))
        ita_pair.append(normalize_string(_.split('\t')[1]))
        
    #tokenize our sentences & remove duplicates

    inp_lang = []
    out_lang = []

    for _eng in eng_pair:
        inp_lang.append(' '.join(tokenizer_eng(_eng)))

    for _ita in ita_pair:
        out_lang.append(' '.join(tokenizer_ita(_ita)))

    pairs = list(zip(inp_lang, out_lang))
    unique_pairs = list(set(pairs))

    inp_lang, out_lang = zip(*unique_pairs)

    inp_lang = list(inp_lang)
    out_lang = list(out_lang)

    #filter out sentences for faster training and reducing dataset size

    filtered_inp_lang = []
    filtered_out_lang = []

    MAX_LENGTH = 10

    eng_prefixes = (
        "i am ", "i 'm ",
        "he is", "he 's ",
        "she is", "she 's ",
        "you are", "you 're ",
        "we are", "we 're ",
        "they are", "they 're "
    )

    def filter_pair(p):
        return len(p[0].split(' ')) < MAX_LENGTH and \
            len(p[1].split(' ')) < MAX_LENGTH and \
                p[0].startswith(eng_prefixes)

    for _idx, _p in enumerate(zip(inp_lang, out_lang)):
        if filter_pair(_p) == True:
            filtered_inp_lang.append(_p[0])
            filtered_out_lang.append(_p[1])

    #build our vocab

    inp_vocab = set()
    out_vocab = set()

    def build_vocab(filtered_inp_lang=filtered_inp_lang, \
                    filtered_out_lang=filtered_out_lang):
        for _eng in filtered_inp_lang:
            inp_vocab.update(_eng.split(' '))
        for _ita in filtered_out_lang:
            out_vocab.update(_ita.split(' '))
        inp_vocab.add('_pad')
        out_vocab.update(['_pad', '_start', '_end'])

    build_vocab()
    
    return filtered_inp_lang, filtered_out_lang, inp_vocab, out_vocab

"""
use above data preprocessing functions in other scripts for data analysis etc.,

use the below script for training, validating & testing
"""

if __name__ == "__main__":
    #get our data
    
    filtered_inp_lang, filtered_out_lang, inp_vocab, out_vocab = \
        get_data()
    # references = [_s.split() for _s in filtered_out_lang]
    '''create word to index and vice versa, add _start, _end tokens &
    pad sentences for batching'''
    
    eng_to_index = {_w: _idx for _idx, _w in \
                    enumerate(inp_vocab)}
    index_to_eng = {_idx: _w for _idx, _w in \
                    enumerate(inp_vocab)}
        
    ita_to_index = {_w: _idx for _idx, _w in \
                    enumerate(out_vocab)}
    index_to_ita = {_idx: _w for _idx, _w in \
                    enumerate(out_vocab)}
    references = filtered_out_lang
    filtered_inp_lang = [_s + ((9-len(_s.split())) * ' _pad') \
                         for _s in filtered_inp_lang      
        ]

    filtered_out_lang = ['_start ' + _s + ' _end' for _s \
                         in filtered_out_lang]

    """
    to verify max length after adding _start, _end tokens
    
    max_len_ita = max([len(_s.split()) for _s in filtered_out_lang])
    """
    
    filtered_out_lang = [_s + ((11-len(_s.split())) * ' _pad') \
                         for _s in filtered_out_lang      
        ]

    #create & randomly split datasets
    
    size_of_dataset = len(filtered_inp_lang)
    random.seed(42)
    temp = list(zip(filtered_inp_lang, filtered_out_lang))
    random.shuffle(temp)
    filtered_inp_lang, filtered_out_lang = zip(*temp)
    filtered_inp_lang, filtered_out_lang = list(filtered_inp_lang), \
        list(filtered_out_lang)
    references = references[int(size_of_dataset*0.8):]

    
    train_x, train_y, val_x, val_y, test_x, test_y = \
                filtered_inp_lang[:int(size_of_dataset*0.7)], \
                filtered_out_lang[:int(size_of_dataset*0.7)], \
                filtered_inp_lang[int(size_of_dataset*0.7):int(size_of_dataset*0.8)], \
                filtered_out_lang[int(size_of_dataset*0.7):int(size_of_dataset*0.8)], \
                filtered_inp_lang[int(size_of_dataset*0.8):], \
                filtered_out_lang[int(size_of_dataset*0.8):]
    train_x, train_y, val_x, val_y, test_x, test_y = \
                torch.tensor([[eng_to_index[_w] for _w in _s.split()] \
                            for _s in train_x]).to(device), \
                torch.tensor([[ita_to_index[_w] for _w in _s.split()] \
                            for _s in train_y]).to(device), \
                torch.tensor([[eng_to_index[_w] for _w in _s.split()] \
                                    for _s in val_x]).to(device), \
                torch.tensor([[ita_to_index[_w] for _w in _s.split()] \
                                    for _s in val_y]).to(device), \
                torch.tensor([[eng_to_index[_w] for _w in _s.split()] \
                                    for _s in test_x]).to(device), \
                torch.tensor([[ita_to_index[_w] for _w in _s.split()] \
                                    for _s in test_y]).to(device)
                
    
    a, b = get_batch(train_x, train_y, 32, len(train_x))
        
    #numericalize & tensorise seq or data handler class
    
    def tensorise():
        pass
    
    def numericalize(s):
        return torch.tensor([])
    
    #build our model
     
    class Encoder(nn.Module):
        def __init__(self, input_size, embedding_size, hidden_size):
            super(Encoder, self).__init__()
            self.hidden_size = hidden_size
            self.emb = nn.Embedding(input_size, embedding_size)
            self.rnn = nn.LSTM(embedding_size, hidden_size)
            
        def forward(self, x):
            emb = self.emb(x)
            outputs, (hn, cn) = self.rnn(emb)
            return hn, cn
    
    class BahdanauAttn(nn.Module):
        def __init__(self):
            super(BahdanauAttn, self).__init__()
            pass

    class Decoder(nn.Module):
        def __init__(self, input_size, emb_size, hidden_size,\
                      output_size):
            super(Decoder, self).__init__()
            self.hidden_size = hidden_size
            self.emb = nn.Embedding(input_size, 128)
            self.rnn = nn.LSTM(emb_size, hidden_size)
            self.fc = nn.Linear(hidden_size, output_size)
        def forward(self, x, hidden, cell_state):
            x = x.unsqueeze(0)
            emb = self.emb(x)
            output, (hn, cn) = self.rnn(emb,\
                                        (hidden, cell_state))
            predictions = self.fc(output)
            predictions = predictions.squeeze(0)
            # print(f'predictions sahpe: {predictions.shape}')
            return predictions, hn, cn

    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder):
            super(Seq2Seq, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            
        def forward(self, input, target, teacher_forcing_prob=0.5):
            batch_size = input.shape[1]
            MAX_LENGTH = target.shape[0]
            hidden, cell = self.encoder(input)
            # print(f'encoder hidden shape: {hidden.shape}')
            # print(f'encoder cell shape: {cell.shape}')
            outputs = torch.zeros(MAX_LENGTH, batch_size, len(out_vocab)).to(device)
            
            x = target[0]
            
            # print(f'x shape: {x.shape}')
            for i in range(1, MAX_LENGTH):
                output, hidden, cell = self.decoder(x, hidden, cell)
                outputs[i] = output
                x = target[i] if random.random() < teacher_forcing_prob and \
                    mode == "train" else output.argmax(1)
                    
            return outputs

    #training hyperparameters
    
    EPOCHS = 40
    batch_size = 32
    NUM_OF_ITER = int(train_x.shape[0] / batch_size)
    loss = 0.0
    learning_rate = 0.001
    running_loss = []
    total_loss_per_epoch = 0.0
    criterion = nn.CrossEntropyLoss()
    
    #define our model  
    encoder = Encoder(len(inp_vocab), 128, 128).to(device)
    decoder = Decoder(len(out_vocab), 128, 128, len(out_vocab)).to(device)
    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    #save/load checkpoints
    
    #save checkpoints
    load_checkpoint = False
    
    if load_checkpoint:
        checkpoint = torch.load("checkpoint.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    mode = 'test'
    
    #training loop
    if mode == "train":
        model.train()
        for epoch in range(EPOCHS):
            total_loss = 0.0
            for iter in range(NUM_OF_ITER):
                input, target = get_batch(train_x, train_y, 32, len(train_x))
                # print(f'input shape: {input.shape}')
                outputs = model(input, target)
                # print(f'outputs shape: {outputs.shape}')
                # print(f'target shape: {target.shape}')
                optimizer.zero_grad()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                loss = criterion(outputs[1:].reshape(-1, outputs.shape[2]), \
                                           target[1:].reshape(-1))
                loss.backward()
                optimizer.step()
                
                running_loss.append(loss.item())
                total_loss += loss.item()
            total_loss /= NUM_OF_ITER
            
            print(f'Total loss: {total_loss}')
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
            }
        torch.save(checkpoint, 'checkpoint.pth.tar')
    elif mode == "eval":
        def model_eval(_idx):
            model.eval()
            checkpoint = torch.load("checkpoint.pth.tar")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            hypothesis = []
            input, _ = get_batch(val_x, val_y, 1, len(val_x))
            output_start = torch.full((11, 1), ita_to_index['_start']).to(device)
            outputs = model(input, output_start)
            input = input.squeeze(1)
            outputs = torch.argmax(outputs.view(-1, outputs.shape[2]), 1)
            for _i in input:
                print(index_to_eng[_i.item()], end=' ') \
                    if index_to_eng[_i.item()] != '_pad' else print('', end='')
            print('\n')
            for _o in outputs[1:]:
                print(index_to_ita[_o.item()], end=' ') if \
                    index_to_ita[_o.item()] != '_pad' and \
                    index_to_ita[_o.item()] != '_end' \
                        else print('', end='')
            print('\n')   
        
        for _idx in range(10):
            model_eval(_idx)

    elif mode == "test":
        running_bleu_score = []
        def model_test(_idx):
            model.eval()
            checkpoint = torch.load("checkpoint.pth.tar")
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            hypothesis = []
            input, _ = test_x[_idx].unsqueeze(1), test_y[_idx].unsqueeze(1)
            references = [[index_to_ita[value.item()] for value in _ \
                        if index_to_ita[value.item()] not in \
                            ("_start", "_pad", "_end")]]
            output_start = torch.full((11, 1), ita_to_index['_start']).to(device)
            outputs = model(input, output_start)
            outputs = torch.argmax(outputs.view(-1, outputs.shape[2]), 1)
            for _o in outputs[1:]:
                if index_to_ita[_o.item()] != '_pad' and \
                    index_to_ita[_o.item()] != '_end':
                        hypothesis.append(index_to_ita[_o.item()])
            smooth = SmoothingFunction().method1
            bleu_score = sentence_bleu(references, hypothesis, \
                                       smoothing_function=smooth)
            running_bleu_score.append(bleu_score)
        for _idx in range(len(test_x)):
            model_test(_idx)
            
            
        plt.figure(figsize=(20, 12))
        plt.hist(running_bleu_score, bins=20)
        
        less_than_03 = sum(1 for value in running_bleu_score if value <= 0.3)
        less_than_06 = sum(1 for value in running_bleu_score if value > 0.3 \
                            and value <= 0.6)
        close_to_1 = sum(1 for value in running_bleu_score if value > 0.6)
        
        labels = ["less_than_03", "less_than_06", "close_to_1"]
        sizes = [less_than_03, less_than_06, close_to_1]
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.legend(title = "Blue score of test set:")

