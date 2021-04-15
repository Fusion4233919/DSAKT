import math
import torch
import argparse
import torch.optim as optim
from sklearn import metrics
from utils import getdata, dataloader, NoamOpt
from tqdm import tqdm

from SAKT import SAKT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu");

def train_sakt(window_size:int, dim:int, heads:int, dropout:float, lr:float, train_path:str, valid_path:str, save_path:str):
    
    print("using {}".format(device));
    
    batch_size = 128;
    epochs = 300;
    
    train_data,N_train,E,unit_list_train = getdata(window_size=window_size, path=train_path, model_type='sakt')
    valid_data,N_val,E,unit_list_val = getdata(window_size=window_size, path=valid_path, model_type='sakt');
    train_loader = dataloader(train_data, batch_size=batch_size, shuffle=True);

    model = SAKT(device = device, num_skills=E, window_size=window_size, dim=dim, heads=heads, dropout=dropout);
    model.to(device);

    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), eps=1e-8);
    scheduler = NoamOpt(optimizer, warmup=400, dimension=dim, factor=lr);
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
            
        if (epoch + 1) % 50 == 0:
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
    window_size = 40;
    dim = 16;
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