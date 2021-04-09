import os
import math
#import wandb
import torch
import argparse
from sklearn import metrics
from readdata import getdata

def predict(window_size:int, model_path:str, data_path:str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    pre_data,N_val,E,unit_list_val = getdata(window_size=window_size,path=data_path)

    model = torch.load(model_path);
    assert model.window_size == window_size;
    model.to(device);
    model.eval();

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
        print(len(pred), len(cort),'val_auc: %.3f mse: %.3f acc: %.3f' %(auc, rmse, acc / len(pred)));
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser();
    parser.add_argument("-ws", "--window_size", required=True);
    parser.add_argument("-d", "--data_path", required=True);
    parser.add_argument("-m", "--model_path", required=True);
    args = parser.parse_args();
    
    assert os.path.exists(args.data_path);
    assert os.path.exists(args.model_path);
    
    predict(int(args.window_size), args.model_path, args.data_path);