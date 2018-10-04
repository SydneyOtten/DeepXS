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

with open("K25.txt") as f2:
    LO=[x2.strip('\n') for x2 in f2.readlines()]

N=0
size=182568
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
        X[k,20]=LO[28*j+24] #LO
        y[k]=LO[28*j+23] #k
        k=k+1
		
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
# X=X[y>1.6]
# X1=X1[y>1.6]
# y=y[y>1.6]
# print(len(y))
N=len(y)
y=y/4
train_size = N-10000
X_test=X1[train_size:N]
y_test=y[train_size:N]
X1=X1[:train_size]
y=y[:train_size]
X=X[:train_size]

#Split the data into training and test sample
X1, Xt=split(X1,0.10)
y, yt=split(y,0.10)
X, Xt_2 = split(X, 0.2)

#Now we start building the Net.
model=Sequential()
for i in range(0,8):
	model.add(Dense(32, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.176177), input_dim=12))
	model.add(Activation('selu'))
model.add(Dense(1, init='uniform',activation='linear'))

history = History()
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

tlog=open("KNet25_1.txt", "w")
learnrate=0.001
pat=50
iterations = 7
epochs = 250
batch_size = 120
c_lr = 2
lr_limit=learnrate/(c_lr**iterations)
k=0
tlog.write("Performing a regression on the K-Factor with randomly sampled inputs within certain parameter intervals.\n")
tlog.write("The total number of used samples is: " + str(N) + " No. of test samples: " + str(len(y_test)) + " No. of validation samples: " + str(len(yt)) + "\n")
tlog.write("The initial learning rate is: " + str(learnrate) + " LR-factor: " + str(c_lr) + " The number of iterations is: " + str(iterations) + " Epochs: " + str(epochs) + " Batch size: " + str(batch_size) + " patience: " + str(pat) + "\n")
tlog.write("Model summary: \n") 
tlog.write(str(model.summary()))
while learnrate > lr_limit:
	early_stopping = EarlyStopping(monitor='val_loss', patience=pat)
	opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	checkpointer = ModelCheckpoint(filepath='weightsK25_1_' + str(k) + '.hdf5', verbose=1, save_best_only=True)
	model.compile(loss="mape", optimizer=opt)
	model.fit(X1, y, validation_data=(Xt,yt), nb_epoch=epochs, batch_size=batch_size, verbose=2, callbacks=[history,checkpointer,early_stopping])
	model.load_weights('weightsK25_1_' + str(k) + '.hdf5')
	learnrate /= c_lr
	k=k+1
	#This is how you use your net to predict stuff. Here I predict the test points.
	y_pred=model.predict(Xt)
	y_pred=y_pred*4
	y_true=yt*4
	y_pred=y_pred.reshape(y_true.shape)

	#Find the biggest deviation with backtransformation
	percdev=np.zeros(len(y_true))
	percdevsort=np.zeros(len(y_true))
	for i in range(len(y_true)):
	  percdev[i]=(y_true[i]-y_pred[i])/y_true[i]
	percdev.reshape(y_true.shape)
	percdevsort=np.sort(percdev)
	print(percdevsort)
	tlog.write(str(percdevsort) + "\n")
	five_pc=0
	ten_pc=0
	for i in range(0,len(percdev)):
		if abs(percdev[i]) > 0.05:
			five_pc = five_pc + 1
		if abs(percdev[i]) > 0.1:
			ten_pc = ten_pc + 1
	print(five_pc)
	print(ten_pc)
	test_mape = np.mean(np.abs(percdev))
	print(test_mape)
	ideal = np.arange(np.min(y_true), np.max(y_true), 0.005)
	fig3=plt.figure(figsize=(10, 10))
	plt.plot(y_true, y_pred, 'ro', ideal, ideal, 'r--')
	plt.title('True vs. predicted cross-section')
	plt.xlabel('true value')
	plt.ylabel('predicted value')
	plt.tight_layout()
	plt.savefig('25_K_general_1_' + str(k) + '.png')
	plt.close()
	tlog.write("iteration: " + str(k) + " mape: " + str(test_mape) + " over 5 percent: " + str(five_pc) + " over 10 percent: " + str(ten_pc) + "\n")
	indices5=[]
	indices10=[]
	for i in range(0,len(percdev)):
		if abs(percdev[i]) > 0.05:
			indices5.append(i)
		if abs(percdev[i]) > 0.1:
			indices10.append(i)
	print(y_true[indices5])
	print(y_true[indices10])
	X_true = Xt*sigma_x+mu_x
	print(X_true[indices5])
	print(X_true[indices10])
	tlog.write("\n" + "target-values for samples with errors above 5 percent:\n")
	tlog.write(str(y_true[indices5]))
	tlog.write("\n" + "target-values for samples with errors above 10 percent:\n")
	tlog.write(str(y_true[indices10]))
	tlog.write("\n" + "representation inputs for samples with errors above 5 percent:\n")
	tlog.write(str(X_true[indices5]))
	tlog.write("\n" + "representation inputs for samples with errors above 10 percent:\n")
	tlog.write(str(X_true[indices10]))
	tlog.write("\n" + "The LO cross-sections:\n")
	tlog.write(str(Xt_2[indices5,20]))
	
tlog.close()