import os
import os.path
import sys

folder='RESULT'
fname=''
data=''
measure=''
test_type=None
n_iter = 100
n_init = 5
n=4096
d=8
k=-1
num=-1
o="out"
verbose=0
n_group=2
init_type ='best'
def InitParameters(args):
    global n_iter,n_init,num,n,d,k,test_type,data,measure,folder,verbose,n_group,init_type,fname
    index =1
    while index < len(args):
        if args[index]== '-n_init': n_init = int(args[index+1]); index+=2; continue;
        if args[index]== '-n_iter': n_iter = int(args[index+1]); index+=2; continue;
        if args[index]== '-n': n = int(args[index+1]); index+=2; continue;
        if args[index]== '-d': d = int(args[index+1]); index+=2; continue;
        if args[index]== '-k': k = int(args[index+1]); index+=2; continue;
        if args[index]== '-num': num = int(args[index+1]); index+=2; continue;
        if args[index]== '-o': o = args[index+1]; index+=2; continue;
        if args[index]== '-test_type': test_type = args[index+1]; index+=2; continue;
        if args[index]== '-data': data = args[index+1]; index+=2; continue;
        if args[index]== '-measure': measure = args[index+1]; index+=2; continue;
        if args[index]== '-folder': folder = args[index+1]; index+=2; continue;
        if args[index]== '-fname': fname = args[index+1]; index+=2; continue;
        if args[index]== '-verbose': verbose = int(args[index+1]); index+=2; continue;
        if args[index]== '-n_group': n_group = int(args[index+1]); index+=2; continue;
        if args[index]== '-init_type': init_type = args[index+1]; index+=2; continue;
        print('ERROR: Cannot understand ',args[index] ) 
        index+= 1
if __name__ == "__main__":
    print(sys.argv)
    InitParameters(sys.argv)
