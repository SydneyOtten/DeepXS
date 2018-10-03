import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam, Adadelta, SGD, RMSprop, Adamax, Nadam
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
import keras.backend as K
from keras import initializers

#Helper function to split Data into training and test sample

def split(X, test_size):
   
    #This just splits data to training and testing parts
    
    ntrn = round(X.shape[0] * (1 - test_size))

    X_train= X[0:ntrn]
    X_test= X[ntrn:]

    return X_train, X_test

with open("22.txt") as f:
    LO=[x1.strip('\n') for x1 in f.readlines()]

with open("22_2.txt") as f:
    LO2=[x2.strip('\n') for x2 in f.readlines()]
Ntrain_b=int(len(LO2)/9)
Ntrain_a=int(9900000)
N=Ntrain_a+Ntrain_b

#initializing input arrays
X=np.empty([N,8])
y=np.zeros(N)

for k in range(0, 9900000):    
	X[k,0]=LO[14*k]
	X[k,1]=LO[14*k+1]
	X[k,2]=LO[14*k+2]
	X[k,3]=LO[14*k+3]
	X[k,4]=abs(float(LO[14*k+4]))
	X[k,5]=LO[14*k+5]
	X[k,6]=LO[14*k+6]
	X[k,7]=LO[14*k+7]
	y[k]=LO[14*k+13]
for k in range(0, Ntrain_b):    
	X[k+Ntrain_a,0]=LO2[9*k]
	X[k+Ntrain_a,1]=LO2[9*k+1]
	X[k+Ntrain_a,2]=LO2[9*k+2]
	X[k+Ntrain_a,3]=LO2[9*k+3]
	X[k+Ntrain_a,4]=abs(float(LO2[9*k+4]))
	X[k+Ntrain_a,5]=LO2[9*k+5]
	X[k+Ntrain_a,6]=LO2[9*k+6]
	X[k+Ntrain_a,7]=LO2[9*k+7]
	y[k+Ntrain_a]=LO2[9*k+8]
X=X[y > 1/3000000]
y=y[y > 1/3000000]
a=len(y)
y=14.92+np.log(y)

X[:,2] = X[:,2]**2-X[:,3]**2
X=np.delete(X, 3, 1)
mean = np.mean(X, axis = 0)
mean = np.reshape(mean, (-1,))
std = np.sqrt(np.var(X, axis = 0))
std = np.reshape(std, (-1,))
X=(X-mean)/std
np.savetxt('22_LO_X.txt', X)
np.savetxt('22_LO_y.txt', y)
z=0
Xt2=np.empty([10000,8])
yt2=np.zeros(10000)
for k in range(9900000, 9910000):    
	Xt2[z,0]=LO[14*k]
	Xt2[z,1]=LO[14*k+1]
	Xt2[z,2]=LO[14*k+2]
	Xt2[z,3]=LO[14*k+3]
	Xt2[z,4]=abs(float(LO[14*k+4]))
	Xt2[z,5]=LO[14*k+5]
	Xt2[z,6]=LO[14*k+6]
	Xt2[z,7]=LO[14*k+7]
	yt2[z]=LO[14*k+13]
	z=z+1
Xt2[:,2] = Xt2[:,2]**2-Xt2[:,3]**2
Xt2=np.delete(Xt2, 3, 1)
yt2=14.92+np.log(yt2) 
Xt2=(Xt2-mean)/std

#Split the data into training and test sample
X, Xt=split(X,0.1)
y, yt=split(y,0.1)

#Now we start building the Net.
model=Sequential()
for i in range(0,8):
	model.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.1), input_dim=7))
	model.add(Activation('selu'))
model.add(Dense(1, init='uniform',activation='linear'))

learnrate=0.0004
epochs=300
z=0
pat=50
bat=1536
c_lr = 2
iterations=6
lr_limit = learnrate/c_lr**iterations
model.load_weights('n2n2_LO_0.hdf5')
k=1
tlog=open("n2n2_LO.txt", "w")
while learnrate > lr_limit:
	early_stopping = EarlyStopping(monitor='val_loss', patience=pat)
	checkpointer = ModelCheckpoint(filepath="n2n2_LO_" + str(k) + ".hdf5", verbose=1, save_best_only=True)
	opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss="exp_xsec", optimizer=opt)
	model.fit(X, y, validation_data=(Xt,yt), nb_epoch=epochs, batch_size=bat, verbose=2, callbacks=[checkpointer,early_stopping])
	model.load_weights('n2n2_LO_' + str(k) + '.hdf5')
	learnrate /= c_lr
	y_pred = model.predict(Xt2)
	y_pred = np.exp(y_pred-14.92)
	y_true = np.exp(yt2-14.92)
	y_true = np.reshape(y_true, (-1,))
	y_pred = np.reshape(y_pred, (-1,))
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
	plt.savefig('n2n2_LO_general_' + str(k) + '.png')
	plt.close()
	tlog.write(" sigma > 1/(3*10^(-6)) " + "iteration: " + str(k) + " mape: " + str(test_mape) + " over 5 percent: " + str(five_pc) + " over 10 percent: " + str(ten_pc) + "\n")
	c=6.6*10**(-5)
	y_pred = y_pred[y_true > c]
	y_true = y_true[y_true > c]
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
	tlog.write(" sigma > 6.6*10^(-5) " +  "iteration: " + str(k) + " mape: " + str(test_mape) + " over 5 percent: " + str(five_pc) + " over 10 percent: " + str(ten_pc) + "\n")
	k=k+1
tlog.close()	