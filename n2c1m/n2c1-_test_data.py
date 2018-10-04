import numpy as np
from numpy.random import RandomState
import matplotlib
matplotlib.use('Agg')
from matplotlib import cm
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras import initializers
import subprocess
import matplotlib.pyplot as plt
import re

#plt.style.use('ggplot')
RNG = RandomState()

#Helper function to split Data into training and test sample

def split(X, test_size):
   
    #This just splits data to training and testing parts
    
    ntrn = round(X.shape[0] * (1 - test_size))

    X_train= X[0:ntrn]
    X_test= X[ntrn:]

    return X_train, X_test

with open("K27.txt") as f2:
    LO=[x2.strip('\n') for x2 in f2.readlines()]

N=0
size=676163
for j in range(0,size):
    if abs(float(LO[28*j+12])) > 50 and float(LO[28*j+13]) > 100 and float(LO[28*j+14]) > 500 and float(LO[28*j+15]) > 500 and float(LO[28*j+16]) > 500 and float(LO[28*j+22]) > 200 and float(LO[28*j+24]) > 1/(1000*3000): 
        N=N+1

X=np.empty([N,21])
y=np.ones(N)
print(N)
k=0
for j in range(0, size):
	if abs(float(LO[28*j+12])) > 50 and float(LO[28*j+13]) > 100 and float(LO[28*j+14]) > 500 and float(LO[28*j+15]) > 500 and float(LO[28*j+16]) > 500 and float(LO[28*j+22]) > 200 and float(LO[28*j+24]) > 1/(1000*3000):
		X[k,0]=LO[28*j] #mix1
		X[k,1]=LO[28*j+1] #mix2
		X[k,2]=LO[28*j+4] #mix3
		X[k,3]=LO[28*j+5] #mix4
		X[k,4]=LO[28*j+8] #mix5
		X[k,5]=LO[28*j+9] #mix6
		X[k,6]=LO[28*j+10] #mix7
		X[k,7]=LO[28*j+11] #mix8
		X[k,8]=LO[28*j+12] #mneut
		X[k,9]=LO[28*j+13] #mchar
		X[k,10]=LO[28*j+14] #sq1
		X[k,11]=LO[28*j+15] #sq2
		X[k,12]=LO[28*j+16] #sq3
		X[k,13]=LO[28*j+17] #sq4
		X[k,14]=LO[28*j+18] #sq5
		X[k,15]=LO[28*j+19] #sq6
		X[k,16]=LO[28*j+20] #sq7
		X[k,17]=LO[28*j+21] #sq8
		X[k,18]=LO[28*j+22] #gluino
		X[k,19]=LO[28*j+25]
		X[k,20]=LO[28*j+24] #LO
		y[k]=LO[28*j+23] #k
		k=k+1
X3=X[:,:19]
NLO=X[:,19]

X1=np.empty([N,12])
for i in range(0, N):
        X1[i,0]=X[i,0]
        X1[i,1]=X[i,1]
        X1[i,2]=X[i,2]
        X1[i,3]=X[i,3]
        X1[i,4]=X[i,4]
        X1[i,5]=X[i,5]
        X1[i,6]=X[i,6]
        X1[i,7]=X[i,7]
        X1[i,8]=X[i,8]
        X1[i,9]=X[i,9]
        X1[i,10]=(X[i,11]+X[i,12]+X[i,13]+X[i,14]+X[i,15]+X[i,16]+X[i,17]+X[i,10])/8
        X1[i,11]=X[i,18]
print(X1[:10])
mu_x=np.mean(X1, axis=0)
print(mu_x[:10])
sigma_x=np.sqrt(np.var(X1, axis=0))
print(sigma_x[:10])
X1=(X1-mu_x)/sigma_x
mu_y=np.mean(y)
sigma_y=np.sqrt(np.var(y))
print(mu_y)
print(sigma_y)
c=mu_y/sigma_y
print(c)
print(len(y))

N=len(y)
y=y/4
train_size = N-10000
X3_test = X3[train_size:N]
NLO_test = NLO[train_size:N]
X_test=X1[train_size:N]
y_test=y[train_size:N]
X1=X1[:train_size]
y=y[:train_size]
X=X[train_size:N]

np.savetxt('n2c1-_NLO_X_test_data.txt', X3_test)
np.savetxt('n2c1-_NLO_y_test_data.txt', NLO_test)