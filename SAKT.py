# -*- coding: utf-8 -*-
"""
Lastmodefied on Fri Apr 23 17:20:38 2021

@author: Fusion
"""
import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

def _getmask(window_size:int):
    return torch.from_numpy(np.triu(np.ones((window_size, window_size)), k=1).astype('bool')).to(device);

class SAKT(nn.Module):
    def __init__(self, device, num_skills, window_size, dim=64, heads=8, dropout=0.2):
        super(SAKT, self).__init__()
        self.window_size = window_size;
        self.loss_function = nn.BCELoss();
        self.activation =[nn.ReLU(), nn.Sigmoid()];


        self.Interation_embedding = nn.Embedding(num_embeddings=2*num_skills+1, embedding_dim=dim, padding_idx=0);
        self.Question_embedding = nn.Embedding(num_embeddings=num_skills+1, embedding_dim=dim, padding_idx=0);
        self.Position_embedding = nn.Embedding(num_embeddings=window_size+1, embedding_dim=dim, padding_idx=0);
        self.Projection = nn.ModuleList([nn.Linear(in_features=dim, out_features=dim, bias=False) for x in range(3)]);
        self.Attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout);
        self.Feed_forward = nn.Sequential(nn.Linear(in_features=dim, out_features=dim, bias=True),
                                          nn.ReLU(),
                                          nn.Linear(in_features=dim, out_features=dim, bias=True),
                                          nn.Dropout(dropout));
        self.Layer_norm = nn.LayerNorm(normalized_shape=dim);
        self.Dropout = nn.Dropout(dropout);
        self.Prediction = nn.Linear(in_features=dim, out_features=1, bias=True);

    def forward(self, input_in, input_ex):
        position = self.Position_embedding( torch.arange(self.window_size).unsqueeze(0).to(device) + 1 );
        interaction = self.Interation_embedding(input_in);
        key_value = interaction + (interaction != 0) * position;
        question = self.Question_embedding(input_ex);
        
        value = self.Layer_norm(self.Projection[0](key_value)).permute(1,0,2);
        key = self.Layer_norm(self.Projection[1](key_value)).permute(1,0,2);
        query = self.Layer_norm(self.Projection[2](question)).permute(1,0,2);
        
        atn, _ = self.Attention(query, key, value, attn_mask=_getmask(self.window_size));
        res = (self.Layer_norm(atn)+query).permute(1,0,2);
        ffn = self.Feed_forward(res);
        
        return self.activation[1](self.Prediction(ffn+res));

import math
import torch
import argparse
import torch.optim as optim
from sklearn import metrics
from utils import getdata, dataloader
from tqdm import tqdm

def train_sakt(window_size:int, dim:int, heads:int, dropout:float, lr:float, train_path:str, valid_path:str, save_path:str):
    
    print("using {}".format(device));
    
    batch_size = 128;
    epochs = 100;

    train_data,N_train,E_train,unit_list_train = getdata(window_size=window_size, path=train_path, model_type='sakt')
    valid_data,N_val,E_test,unit_list_val = getdata(window_size=window_size, path=valid_path, model_type='sakt');        
    train_loader = dataloader(train_data, batch_size=batch_size, shuffle=True);
    train_steps=len(train_loader);
    E = max(E_train, E_test);

    model = SAKT(device = device, num_skills=E, window_size=window_size, dim=dim, heads=heads, dropout=dropout);
    model.to(device);

    optimizer = optim.Adam(model.parameters(), lr=1e-3);
    best_auc = 0.0;
    
    for epoch in range(epochs):    
        model.train();
        running_loss = 0.0;
        train_bar = tqdm(train_loader);
        for data in train_bar:
            logits = model(data[0].to(device), data[1].to(device));
            correct = data[2].float().unsqueeze(-1).to(device);
            
            loss = model.loss_function(logits, correct);
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();
            running_loss += loss.item();
            
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss);
        print('[epoch %d] train_loss: %.3f' %(epoch + 1, running_loss / train_steps));       
            
        model.eval();
        with torch.no_grad():
            predict = model(valid_data[0].to(device), valid_data[1].to(device)).squeeze(-1).to("cpu");
            correct = valid_data[2];
            pred = [];
            cort = [];
            for i in range(N_val):
                pred.extend(predict[i][0:unit_list_val[i]].cpu().numpy().tolist());
                cort.extend(correct[i][0:unit_list_val[i]].numpy().tolist());
                
            rmse = math.sqrt(metrics.mean_squared_error(cort, pred));
            fpr, tpr, thresholds = metrics.roc_curve(cort, pred, pos_label=1);
            pred = torch.Tensor(pred) > 0.5;
            cort = torch.Tensor(cort) == 1;
            acc = torch.eq(pred, cort).sum();
            auc = metrics.auc(fpr, tpr);
            if auc > best_auc:
                best_auc = auc;
                torch.save(model, save_path); 
            print('val_auc: %.3f mse: %.3f acc: %.3f' %(auc, rmse, acc / len(pred)));
            
    print('best: %.3f' %(best_auc));
 
if __name__ =="__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("-ws", "--window_size", type=int);
    parser.add_argument("-d", "--dim", type=int);
    parser.add_argument("--heads", type=int);
    parser.add_argument("-drp", "--dropout", type=float);
    parser.add_argument("-lr", "--learn_rate", type=float);
    parser.add_argument("-t", "--train_data", required=True);
    parser.add_argument("-v", "--val_data", required=True);
    parser.add_argument("-s", "--save_path", required=True);
    args = parser.parse_args();

    lr = 0.001;
    window_size = 50;
    dim = 64;
    dropout = 0.2;
    heads = 8;

    if args.window_size:
        window_size = args.window_size;
    if args.dim:
        dim = args.dim;
    if args.heads:
        heads = args.heads;
    if args.dropout:
        dropout = args.dropout;
    if args.learn_rate:
        lr = args.learn_rate;
    
    train_sakt(window_size=window_size, 
               dim=dim, 
               heads=heads, 
               dropout=dropout, 
               lr=lr, 
               train_path=args.train_data, 
               valid_path=args.val_data, 
               save_path=args.save_path);
