import argparse
import random
from random import randrange

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('--n',type=int, required=True)
    parser.add_argument('--f',type=int, required=True)
    parser.add_argument('--d',type=int, required=True)

    args = parser.parse_args()

    n = args.n # number of workers
    f = args.f # number of faulty workers
    d = args.d # number of datapoints

    M = [[] for each in range(n)]
    
    for datapoint in range(d):

        for fault in range(f+1):
            
            dest = randrange(0,n)
            while datapoint in M[dest]: 
                dest = randrange(0,n)

            M[dest].append(datapoint)

    print("M <-- datapoints for each worker :")
    print(M)
