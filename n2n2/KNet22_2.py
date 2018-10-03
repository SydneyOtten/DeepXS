import numpy as np
from numpy.random import RandomState
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, Adadelta, SGD, RMSprop
from keras.callbacks import History, ModelCheckpoint, EarlyStopping
from keras import initializers

#plt.style.use('ggplot')
RNG = RandomState()

#Helper function to split Data into training and test sample

def split(X, test_size):
   
    #This just splits data to training and testing parts
    
    ntrn = round(X.shape[0] * (1 - test_size))

    X_train= X[0:ntrn]
    X_test= X[ntrn:]

    return X_train, X_test

with open("K22.txt") as f2:
    LO=[x2.strip('\n') for x2 in f2.readlines()]

X2=np.empty([100000,15])
y1=np.ones(100000)

for j in range(0, 100000):
    X2[j,0]=LO[19*j] #mix1
    X2[j,1]=LO[19*j+1] #mix2
    X2[j,2]=LO[19*j+2] #mix3
    X2[j,3]=LO[19*j+3] #mix4
    X2[j,4]=LO[19*j+4] #mneut
    X2[j,5]=LO[19*j+5] #sq1
    X2[j,6]=LO[19*j+6] #sq2
    X2[j,7]=LO[19*j+7] #sq3
    X2[j,8]=LO[19*j+8] #sq4
    X2[j,9]=LO[19*j+9] #sq5
    X2[j,10]=LO[19*j+10] #sq6
    X2[j,11]=LO[19*j+11] #sq7
    X2[j,12]=LO[19*j+12] #sq8
    X2[j,13]=LO[19*j+13] #gluino
    X2[j,14]=LO[19*j+15] #predLO
    y1[j]=LO[19*j+14] #k

cnt=100000
print(cnt)
X1=np.empty([cnt,6])
X=np.empty([cnt,6])
y=np.ones(cnt)
LOs = X2[:,14]
for i in range(0, cnt):
        X1[i,0]=X2[i,0]
        X1[i,1]=X2[i,1]
        X1[i,2]=X2[i,2]**2-X2[i,3]**2
        X1[i,3]=np.abs(X2[i,4])
        X1[i,4]=(X2[i,5]+X2[i,6]+X2[i,7]+X2[i,8]+X2[i,9]+X2[i,10]+X2[i,11]+X2[i,12])/8
        X1[i,5]=X2[i,13]

mean=np.empty(6)
std=np.empty(6)

for i in range(0,6):
    mean[i] = np.mean(X1[:,i])
    std[i] = np.std(X1[:,i])

for i in range(0, cnt):
    X[i,0]=(X1[i,0]-mean[0])/std[0]
    X[i,1]=(X1[i,1]-mean[1])/std[1]
    X[i,2]=(X1[i,2]-mean[2])/std[2]
    X[i,3]=(X1[i,3]-mean[3])/std[3]
    X[i,4]=(X1[i,4]-mean[4])/std[4]
    X[i,5]=(X1[i,5]-mean[5])/std[5]
    y[i]=y1[i]/2

#Split the data into training and test sample
X, Xt=split(X,0.2)
y, yt=split(y,0.2)
X_test = Xt[:10000] 
y_test = yt[:10000]
Xt = Xt[10000:]
yt = yt[10000:]
LOs1, LOs2 = split(LOs, 0.2)
LO_K = LOs2[:10000]

model=Sequential()
for i in range(0,8):
	model.add(Dense(32, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.176177), input_dim=6))
	model.add(Activation('selu'))
model.add(Dense(1, init='uniform',activation='linear'))


history = History()
checkpointer = ModelCheckpoint(filepath="n2n2_K.hdf5", verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=50)

tlog=open("n2n2_K.txt", "w")
learnrate=0.001
pat=50
iterations = 7
epochs = 250
batch_size = 120
c_lr = 2
lr_limit=learnrate/(c_lr**iterations)
k=0
tlog.write("Performing a regression on the K-Factor with randomly sampled inputs within certain parameter intervals.\n")
tlog.write("The initial learning rate is: " + str(learnrate) + " LR-factor: " + str(c_lr) + " The number of iterations is: " + str(iterations) + " Epochs: " + str(epochs) + " Batch size: " + str(batch_size) + " patience: " + str(pat) + "\n")
tlog.write("Model summary: \n") 
tlog.write(str(model.summary()))
while learnrate > lr_limit:
	early_stopping = EarlyStopping(monitor='val_loss', patience=pat)
	opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	checkpointer = ModelCheckpoint(filepath='n2n2_K_' + str(k) + '.hdf5', verbose=1, save_best_only=True)
	model.compile(loss="mape", optimizer=opt)
	model.fit(X, y, validation_data=(Xt,yt), nb_epoch=epochs, batch_size=batch_size, verbose=2, callbacks=[history,checkpointer,early_stopping])
	model.load_weights('n2n2_K_' + str(k) + '.hdf5')
	learnrate /= c_lr
	k=k+1
	#This is how you use your net to predict stuff. Here I predict the test points.
	y_pred=model.predict(X_test)
	y_pred=y_pred*2
	y_true=y_test*2
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
	plt.savefig('n2n2_K_' + str(k) + '.png')
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
	X_true = X_test*std+mean
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
	tlog.write("\n" + "The LO cross-sections corresponding to the samples with an error above 5 percent:\n")
	tlog.write(str(LO_K[indices5]))
tlog.close()