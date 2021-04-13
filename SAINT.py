import torch
import torch.nn as nn
import numpy as np
import copy

device = torch.device("cuda");
    
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
        data_in = data_in.permute(1, 0, 2);
        data_out, temp = self.MHA(data_in, data_in, data_in, attn_mask=_getmask(self.window_size));
        data_out = self.LN[0](data_in + data_out).permute(1, 0, 2);
        temp = data_out;
        data_out = self.FFN(data_out);
        data_out = self.LN[1](data_out + temp);
        return data_out;

class Decoder(nn.Module):
    def __init__(self, dim:int, heads:int, dropout:float, window_size:int):
        super(Decoder, self).__init__();
        self.window_size = window_size;
        self.MHA = nn.ModuleList([nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout) for x in range(2)]);
        self.FFN = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=dim, out_features=dim, bias=True),
            nn.Dropout(dropout));
        self.LN = nn.ModuleList([nn.LayerNorm(normalized_shape=dim) for x in range (3)]);
        
    def forward(self, data_in, encoded_data):
        data_in = data_in.permute(1, 0, 2);
        data_out, _ = self.MHA[0](data_in, data_in, data_in, attn_mask=_getmask(self.window_size));
        data_out = self.LN[0](data_in + data_out);
        temp = data_out;
        encoded_data = encoded_data.permute(1, 0, 2);
        data_out, _ = self.MHA[1](data_out, encoded_data, encoded_data);
        data_out = self.LN[1](data_out + temp).permute(1, 0, 2);
        temp = data_out;
        data_out = self.FFN(data_out);
        data_out = self.LN[2](data_out + temp);
        return data_out;

class SAINT(nn.Module):
    def __init__(self, device, num_layers:tuple, num_skills:int, window_size:int, dim:int, heads:int, dropout:float):
        super(SAINT, self).__init__();
        self.device = device;
        self.num_layers = num_layers;
        self.window_size = window_size;
        self.loss_function = nn.BCELoss();
        
        self.Exerc_embedding = nn.Embedding(num_embeddings=num_skills+1, embedding_dim=dim, padding_idx=0);
        self.Score_embedding = nn.Embedding(num_embeddings=2, embedding_dim=dim);
        self.Posit_embedding = nn.Embedding(num_embeddings=window_size+1, embedding_dim=dim, padding_idx=0);
        self.Query_embedding = nn.Embedding(num_embeddings=num_skills+1, embedding_dim=dim, padding_idx=0);
        self.Prediction = nn.Sequential(nn.Linear(in_features=dim, out_features=1, bias=True),
                                        nn.Sigmoid());
        
        self.Encoders = nn.ModuleList([copy.deepcopy(Encoder(dim=dim, heads=heads, dropout=dropout, window_size=window_size)) for x in range(num_layers[0])]);
        self.Decoders = nn.ModuleList([copy.deepcopy(Decoder(dim=dim, heads=heads, dropout=dropout, window_size=window_size)) for x in range(num_layers[1])]);
        
    def forward(self, ex_in, sc_in, po_in, qu_in):
        interation = self.Exerc_embedding(ex_in) + self.Score_embedding(sc_in) + self.Posit_embedding(po_in);
        question = self.Query_embedding(qu_in);
        
        for x in range(self.num_layers[0]):
            interation = self.Encoders[x](interation);
        
        for x in range(self.num_layers[1]):
            question = self.Decoders[x](question, interation);
            
        return self.Prediction(question);

'''    
def randomdata(E, n, N):
    input_ex = torch.randint(1, E+1, (N, n+1));
    output_q = torch.randint(0 , 2 , (N, n+1));
    return torch.stack((input_ex[:, :-1], output_q[:, :-1], input_ex[:, 1:]),0);

model = SAINT(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), 
              num_layers=(2,2), 
              num_skills=50, 
              window_size=10,
              dim=8,
              heads=4,
              dropout=0.1);

model.to(device);

test = randomdata(50, 10, 2).to(device);
test = model(test[0], test[1], torch.arange(10).unsqueeze(0).to(device) + 1, test[2]);
print(model);
print(test);
'''