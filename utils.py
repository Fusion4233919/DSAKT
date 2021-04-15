import torch
import random

def getdata(window_size,path,drop=False):
    N=0
    count=0
    E=-1
    units=[]
    input_1=[]
    input_2=[]
    input_3=[]
    input_4=[]
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        if count%3==0:
            pass
        elif count%3==1:
            tlst=line.split('\n')[0].split(',')
            if drop:
                Elst=tlst[0:window_size+1]
            else:
                Elst=tlst
            for item in Elst:
                if int(item) >E:
                    E=int(item)

            tlst_1=tlst[0:len(tlst)-1]
            tlst_2=tlst[1:len(tlst)]

            if drop:
                if len(tlst_1)>window_size:
                    tlst_1=tlst_1[0:window_size]
                    tlst_2=tlst_2[0:window_size]


            while len(tlst_1)>window_size:
                input_1.append([int(i)+1 for i in tlst_1[0:window_size]])
                N+=1
                tlst_1= tlst_1[window_size:len(tlst_1)]
                units.append(window_size)
            tlst_1=[int(i)+1 for i in tlst_1]+[0]*(window_size - len(tlst_1))
            units.append(len(tlst_1))
            N+=1
            input_1.append(tlst_1)

            while len(tlst_2)>window_size:
                input_3.append([int(i)+1 for i in tlst_2[0:window_size]])
                tlst_2= tlst_2[window_size:len(tlst_2)]
            tlst_2=[int(i)+1 for i in tlst_2]+[0]*(window_size - len(tlst_2))
            input_3.append(tlst_2)
        else:   #1:False 2:True
            tlst=line.split('\n')[0].split(',')

            tlst_1=tlst[0:len(tlst)-1]
            tlst_2=tlst[1:len(tlst)]

            if drop:
                if len(tlst_1)>window_size:
                    tlst_1=tlst_1[0:window_size]
                    tlst_2=tlst_2[0:window_size]

            while len(tlst_1)>window_size:
                input_2.append([int(i)+1 for i in tlst_1[0:window_size]])
                tlst_1= tlst_1[window_size:len(tlst_1)]
            tlst_1=[int(i)+1 for i in tlst_1]+[0]*(window_size - len(tlst_1))
            input_2.append(tlst_1)

            while len(tlst_2)>window_size:
                input_4.append([int(i)+1 for i in tlst_2[0:window_size]])
                tlst_2= tlst_2[window_size:len(tlst_2)]
            tlst_2=[int(i)+1 for i in tlst_2]+[0]*(window_size - len(tlst_2))
            input_4.append(tlst_2)
        count+=1;
    file.close()
    E+=1

    input_1=torch.tensor(input_1)
    input_2=torch.tensor(input_2)
    input_3=torch.tensor(input_3)
    input_4=torch.tensor(input_4)

    return torch.stack((input_1,input_2,input_3,input_4),0),N,E,units

def dataloader(data, batch_size, shuffle:bool):
    data = data.permute(1,0,2);
    lis = [x for x in range(len(data))];
    if shuffle:
        random.shuffle(lis);
    lis = torch.Tensor(lis).long();
    ret = [];
    for i in range(int(len(data)/batch_size)):
        temp = torch.index_select(data, 0, lis[i*batch_size : (i+1)*batch_size]);
        ret.append(temp);
    return torch.stack(ret, 0).permute(0,2,1,3);

class NoamOpt:
    def __init__(self, optimizer:torch.optim.Optimizer, warmup:int, dimension:int, factor=0.1):
        self.optimizer = optimizer;
        self._steps = 0;
        self._warmup = warmup;
        self._factor = factor;
        self._dimension = dimension;
        
    def step(self):
        self._steps += 1;
        rate = self._factor * (self._dimension**(-0.5) * min(self._steps**(-0.5), self._steps * self._warmup**(-1.5)));
        for x in self.optimizer.param_groups:
            x['lr'] = rate;