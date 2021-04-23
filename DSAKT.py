import torch
import torch.nn as nn
import numpy as np
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

def _getmask(window_size:int):
    return torch.from_numpy(np.triu(np.ones((window_size, window_size)), k=1).astype('bool')).to(device);

class Encoder(nn.Module):
    def __init__(self, dim:int, heads:int, dropout:float, window_size:int):
        self.window_size = window_size;
        super(Encoder, self).__init__();
        self.MHA = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout);
        self.FFN = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.Dropout(dropout));
        self.LN = nn.ModuleList([nn.LayerNorm(normalized_shape=dim) for x in range (2)]);
        
    def forward(self, data_in):
        data_per = data_in.permute(1, 0, 2);
        data_out, _ = self.MHA(data_per, data_per, data_per, attn_mask=_getmask(self.window_size));
        data_out = self.LN[0](data_out + data_per).permute(1, 0, 2);
        temp = data_out;
        data_out = self.FFN(data_out);
        data_out = self.LN[1](data_out + temp);
        return data_out;

class Decoder(nn.Module):
    def __init__(self, dim:int, heads:int, dropout:float, window_size:int):
        self.window_size = window_size;
        super(Decoder, self).__init__();
        self.MHA = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout);
        self.FFN = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.Dropout(dropout));
        self.LN = nn.ModuleList([nn.LayerNorm(normalized_shape=dim) for x in range (2)]);
        
    def forward(self, data_in, encoded_data):
        data_per = data_in.permute(1, 0, 2);
        encoded_per = encoded_data.permute(1, 0, 2);
        data_out, _ = self.MHA(data_per, encoded_per, encoded_per, attn_mask=_getmask(self.window_size));
        data_out = self.LN[0](data_out + data_per).permute(1, 0, 2);
        temp = data_out;
        data_out = self.FFN(data_out);
        data_out = self.LN[1](data_out + temp);
        return data_out;

class DSAKT(nn.Module):
    def __init__(self, device, num_skills:int, num_layers:list, window_size:int, dim:int, heads:int, dropout:float):
        super(DSAKT, self).__init__();
        self.device = device;
        self.num_layers = num_layers;
        self.window_size = window_size;
        self.dim = dim;
        self.loss_function = nn.BCELoss();
        
        self.Exerc_embedding = nn.Embedding(num_embeddings=2*num_skills+1, embedding_dim=dim, padding_idx=0);
        #self.Score_embedding = nn.Embedding(num_embeddings=3, embedding_dim=dim, padding_idx=0);
        self.Posit_embedding = nn.Embedding(num_embeddings=window_size+1, embedding_dim=dim, padding_idx=0);
        self.Query_embedding = nn.Embedding(num_embeddings=num_skills+1, embedding_dim=dim, padding_idx=0);
        self.Prediction = nn.Sequential(nn.Linear(in_features=dim, out_features=1, bias=True),
                                        nn.Sigmoid());
        
        self.Encoders = nn.ModuleList([copy.deepcopy(Encoder(dim=dim, heads=heads, dropout=dropout, window_size=window_size)) for x in range(num_layers[0])]);
        self.Decoders = nn.ModuleList([copy.deepcopy(Decoder(dim=dim, heads=heads, dropout=dropout, window_size=window_size)) for x in range(num_layers[1])]);
     
    def forward(self, ex_in, ex_qu):
        with torch.no_grad():
            posi = torch.arange(self.window_size).unsqueeze(0).to(self.device) + 1;
            posi = posi * (ex_in != 0);
        
        interation = self.Exerc_embedding(ex_in) + self.Posit_embedding(posi);
        question = self.Query_embedding(ex_qu) + self.Posit_embedding(posi);
        
        for x in range(self.num_layers[0]):
            interation = self.Encoders[x](interation);
        for x in range(self.num_layers[1]):
            question = self.Decoders[x](question, interation);
        
        return self.Prediction(question);
    
import math
import torch
import argparse
import torch.optim as optim
from sklearn import metrics
from utils import getdata, dataloader, NoamOpt
from tqdm import tqdm

def train_dsakt(num_layers:list, window_size:int, dim:int, heads:int, dropout:float, lr:float, train_path:str, valid_path:str, save_path:str):
    
    print("using {}".format(device));
    
    batch_size = 128;
    epochs = 300;
    
    train_data,N_train,E,unit_list_train = getdata(window_size=window_size, path=train_path, model_type='sakt')
    valid_data,N_val,E,unit_list_val = getdata(window_size=window_size, path=valid_path, model_type='sakt');
    train_loader = dataloader(train_data, batch_size=batch_size, shuffle=True);
    
    model = DSAKT(device=device, num_skills=E, num_layers=num_layers, window_size=window_size, dim=dim, heads=heads, dropout=dropout);
    model.to(device);
    
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8);
    scheduler = NoamOpt(optimizer, warmup=120, dimension=dim, factor=lr);
    best_auc = 0.0;
    train_steps=len(train_loader);
    
    for epoch in range(epochs):
        model.train();
        running_loss = 0.0;
        scheduler.step();
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
    
        if (epoch + 1) % 5 == 0:
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
 
                
if __name__ =="__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("-l", "--layers", type=int, nargs=2);
    parser.add_argument("-ws", "--window_size", type=int);
    parser.add_argument("-d", "--dim", type=int);
    parser.add_argument("--heads", type=int);
    parser.add_argument("-drp", "--dropout", type=float);
    parser.add_argument("-lr", "--learn_rate", type=float);
    parser.add_argument("-t", "--train_data", required=True);
    parser.add_argument("-v", "--val_data", required=True);
    parser.add_argument("-s", "--save_path", required=True);
    args = parser.parse_args();

    lr = 0.9;
    num_layers = [2,2];
    window_size = 40;
    dim = 16;
    dropout = 0.2;
    heads = 8;

    if args.layers:
        num_layers = args.layers;
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
    
    train_dsakt(num_layers=num_layers,
                window_size=window_size, 
                dim=dim, 
                heads=heads, 
                dropout=dropout, 
                lr=lr, 
                train_path=args.train_data, 
                valid_path=args.val_data, 
                save_path=args.save_path);