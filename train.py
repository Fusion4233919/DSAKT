import math
import torch
import argparse
import torch.optim as optim
from sklearn import metrics
import readdata

from SAKT import SAKT, NoamOpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
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
window_size = 40;
dim = 16;
dropout = 0.2;
heads = 8;
epochs = 1000;

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
    
print("using {}".format(device));

def train_model(train_path, test_path):
    
    train_data,N_train,E,unit_list_train = readdata.getdata(window_size=window_size,path=train_path)
    pre_data,N_val,E,unit_list_val = readdata.getdata(window_size=window_size,path=test_path)

    model = SAKT(device = device, num_skills=E, window_size=window_size, dim=dim, heads=heads, dropout=dropout);
    model.to(device);

    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8);
    scheduler = NoamOpt(optimizer, warmup=400, dimension=dim);
    best_auc = 0.0;
    
    for epoch in range(epochs):
        model.train();
        logits = model(train_data[0].to(device), train_data[1].to(device)).squeeze(-1);
        correctness = train_data[2].float().to(device);
        
        loss = model.loss_function(logits, correctness);
        optimizer.zero_grad()
        loss.backward();
        scheduler.step();
        optimizer.step();
        
        model.eval();
        with torch.no_grad():
            pred = [];
            cort = [];
            predict = model(train_data[0].to(device), train_data[1].to(device)).squeeze(-1);
            correctness = train_data[2];
            for i in range(N_train):
                pred.extend(predict[i][0:unit_list_train[i]].cpu().numpy().tolist());
                cort.extend(correctness[i][0:unit_list_train[i]].numpy().tolist());
                
            rmse = math.sqrt(metrics.mean_squared_error(cort, pred));
            fpr, tpr, thresholds = metrics.roc_curve(cort, pred, pos_label=1);
            train_auc = metrics.auc(fpr, tpr);
            print('[epoch %d] train_loss: %.3f  train_auc: %.3f mse: %.3f' %(epoch + 1, loss.item(), train_auc, rmse)); 
            
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                predict = model(pre_data[0].to(device), pre_data[1].to(device)).squeeze(-1).to("cpu");
                correctness = pre_data[2];
                pred = [];
                cort = [];
                for i in range(N_val):
                    pred.extend(predict[i][0:unit_list_val[i]].cpu().numpy().tolist());
                    cort.extend(correctness[i][0:unit_list_val[i]].numpy().tolist());
                
                rmse = math.sqrt(metrics.mean_squared_error(cort, pred));
                fpr, tpr, thresholds = metrics.roc_curve(cort, pred, pos_label=1);
                pred = torch.Tensor(pred) > 0.5;
                cort = torch.Tensor(cort) == 1;
                acc = torch.eq(pred, cort).sum();
                auc = metrics.auc(fpr, tpr);
                if auc > best_auc:
                    best_auc = auc;
                    torch.save(model, args.save_path); 
                print('val_auc: %.3f mse: %.3f acc: %.3f' %(auc, rmse, acc / len(pred)));
    
if __name__ =="__main__":
    train_model(train_path=args.train_data,test_path=args.val_data);
