import torch

def getdata(window_size,path):
    N=0
    count=0
    E=-1
    units=[]
    input_ex=[]
    output_q=[]
    file = open(path)
    while 1:
        line = file.readline()
        if not line:
            break
        if count%3==0:
            pass
        elif count%3==1:
            tlst=line.split('\n')[0].split(',')
            for item in tlst:
                if int(item) >E:
                    E=int(item)
            while len(tlst)>window_size+1:
                input_ex.append([int(i)+1 for i in tlst[0:window_size+1]])
                N+=1
                tlst= tlst[window_size:len(tlst)]
                units.append(window_size)
            units.append(len(tlst)-1)
            tlst=[int(i)+1 for i in tlst]+[0]*(window_size + 1 - len(tlst))
            N+=1
            input_ex.append(tlst)
        else:
            tlst=line.split('\n')[0].split(',')
            while len(tlst)>window_size+1:
                output_q.append([int(i) for i in tlst[0:window_size+1]])
                tlst= tlst[window_size:len(tlst)]
            tlst=[int(i) for i in tlst]+[0]*(window_size + 1 - len(tlst))
            output_q.append(tlst)
            pass
        count+=1;
    file.close()

    E+=1

    input_ex=torch.tensor(input_ex)
    output_q=torch.tensor(output_q)
    input_in = output_q[:,:int(-1)]* E + input_ex[:,:int(-1)];
    for _ in range(N):
        if len(input_in[_]) > units[_]:
            input_in[_][units[_]] = 0;
    return torch.stack((input_in, input_ex[:, 1:], output_q[:, 1:]),0),N,E,units


if __name__ == '__main__':
  datas,N,E=getdata(50,'./dataset/0910_a_test.csv')
  print(datas)
