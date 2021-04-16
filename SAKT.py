import torch
import torch.nn as nn
import numpy as np

class SAKT(nn.Module):
    def __init__(self, device, num_skills, window_size, dim=64, heads=8, dropout=0.2):
        super(SAKT, self).__init__()
        self.window_size = window_size;
        self.loss_function = nn.BCELoss();
        self.activation =[nn.ReLU(), nn.Sigmoid()];
        self.device = device;

        self.Interation_embedding = nn.Embedding(num_embeddings=2*num_skills+1, embedding_dim=dim, padding_idx=0);
        self.Question_embedding = nn.Embedding(num_embeddings=num_skills+1, embedding_dim=dim, padding_idx=0);
        self.Position_embedding = nn.Embedding(num_embeddings=window_size+1, embedding_dim=dim, padding_idx=0);
        self.Projection = nn.ModuleList([nn.Linear(in_features=dim, out_features=dim, bias=False) for x in range(3)]);
        self.Attention = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout);
        self.Feed_forward = nn.ModuleList([nn.Linear(in_features=dim, out_features=dim, bias=True) for x in range(2)]);
        self.Layer_norm = nn.ModuleList([nn.LayerNorm(normalized_shape=dim) for x in range(2)]);
        self.Dropout = nn.Dropout(dropout);
        self.Prediction = nn.Linear(in_features=dim, out_features=1, bias=True);
        
        self._weight_init();

    def forward(self, input_in, input_ex):
        position = self.Position_embedding( torch.arange(self.window_size).unsqueeze(0).to(self.device) + 1 );
        interaction = self.Interation_embedding(input_in);
        interaction = interaction + (interaction != 0) * position;
        question = self.Question_embedding(input_ex);
        
        value = self.Projection[0](interaction).permute(1,0,2);
        key = self.Projection[1](interaction).permute(1,0,2);
        query = self.Projection[2](question).permute(1,0,2);
        
        atn, _ = self.Attention(query, key, value, attn_mask=torch.from_numpy(np.triu(np.ones((self.window_size, self.window_size)), k=1).astype('bool')).to(self.device));
        
        atn = self.Layer_norm[0](atn + query).permute(1,0,2);
        
        ffn = self.Dropout(self.Feed_forward[1]( self.activation[0](self.Feed_forward[0](atn)) ));
        ffn = self.Layer_norm[1](ffn + atn);
        
        return self.activation[1](self.Prediction(ffn));
    
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight);