import os
import math
import torch
import argparse
from sklearn import metrics
from utils import getdata, dataloader
from DKT import DKT

def predict(window_size:int, model_path:str, data_path:str):
    
    batch_size = 128;
    dim = 200;
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    pre_data,N_val,E,unit_list_val = getdata(window_size=window_size,path=data_path,model_type='sakt')
    pre_loader = dataloader(pre_data, batch_size=batch_size, shuffle=False);
    
    model = torch.load(model_path);
    model.to(device);
    model.eval();

    with torch.no_grad():
        
        h = torch.zeros((1, batch_size, dim)).to(device);
        c = torch.zeros((1, batch_size, dim)).to(device);
        
        pred = [];
        cort = [];  

        for batch in range(len(pre_loader)):
            data = pre_loader[batch];
            predict, (h,c) = model(data[0].to(device), (h,c));
            predict = torch.gather(predict.to('cpu'), 2, data[1].unsqueeze(2)).squeeze(-1);
            correct = data[2];

            for i in range(batch_size):
                pos = i + batch_size * batch;
                pred.extend(predict[i][0:unit_list_val[pos]].numpy().tolist());
                cort.extend(correct[i][0:unit_list_val[pos]].numpy().tolist());

        rmse = math.sqrt(metrics.mean_squared_error(cort, pred));
        fpr, tpr, thresholds = metrics.roc_curve(cort, pred, pos_label=1);
        pred = torch.Tensor(pred) > 0.5;
        cort = torch.Tensor(cort) == 1;
        acc = torch.eq(pred, cort).sum();
        auc = metrics.auc(fpr, tpr);
        print('val_auc: %.3f mse: %.3f acc: %.3f' %(auc, rmse, acc / len(pred)))
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("-ws", "--window_size", required=True);
    parser.add_argument("-d", "--data_path", required=True);
    parser.add_argument("-m", "--model_path", required=True);
    args = parser.parse_args();
    
    assert os.path.exists(args.data_path);
    assert os.path.exists(args.model_path);
    
    predict(int(args.window_size), args.model_path, args.data_path);