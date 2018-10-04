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

def split(X, test_size):
    
    ntrn = round(X.shape[0] * (1 - test_size))

    X_train= X[0:ntrn]
    X_test= X[ntrn:]

    return X_train, X_test

X=[]
with open("25_9p9m.txt", "r", 16777216) as f:
	for line in f:
		X.append(line.strip('\n'))
X=np.reshape(X, (9900000,12))
X=X.astype(np.float)
y=X[:,11]
X=X[:,:11]
X=X[y > 1/3000000]
y=y[y > 1/3000000]
y=14.92+np.log(y)
print(np.shape(y))
mu_x=np.mean(X, axis=0)
mu_y=np.mean(y)
sigma_x=np.sqrt(np.var(X, axis=0))
sigma_y=np.sqrt(np.var(y))
X=(X-mu_x)/sigma_x
X=X[:1000000]
y=y[:1000000]

X_0, X_train = split(X, 0.1)
y_0, y_train = split(y, 0.1)
X_train, X_test = split(X_train, 0.1)
y_train, y_test = split(y_train, 0.1)
y_0 = np.reshape(y_0, (-1,1))
#load the actively sampled data
AL=[]
X_AL=[]
y_AL=[]
with open('25_PBAL_X_13.txt', 'r', 16777216) as f:
	for line in f:
		AL.append(line.strip('\n'))
for item in AL:
	for word in re.split(r'\s+', item):
		if len(word) > 1:
			X_AL.append(word)
X_AL = np.reshape(X_AL, (2300000, 11))
X_AL = X_AL.astype(np.float)
with open('25_PBAL_y_13.txt', 'r', 16777216) as f:
	for line in f:
		y_AL.append(line.strip('\n'))
y_AL = np.reshape(y_AL, (2300000, 1))
y_AL = y_AL.astype(np.float)

### first model: validation data during training is random data

model=Sequential()
for i in range(0,8):
	model.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.125), input_dim=11))
	model.add(Activation('selu'))
model.add(Dense(1, init='uniform',activation='linear'))
j=0
history = History()

learnrate=0.0008
epochs=50
pat=75
bat=120
iterations=6
lim=learnrate/(2**iterations)

mapes = open('25_AL_mapes_120bs.txt', 'w')
while learnrate > lim:
	checkpointer = ModelCheckpoint(filepath="25_ALdata_120bs_" + str(j) + ".hdf5", verbose=1, save_best_only=True)
	early_stopping = EarlyStopping(monitor='val_loss', patience=pat)
	opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss="exp_xsec", optimizer=opt)
	model.fit(X_AL, y_AL, validation_data=(X_train,y_train), nb_epoch=epochs, batch_size=bat, verbose=2, callbacks=[history,checkpointer,early_stopping])
	model.load_weights("25_ALdata_120bs_" + str(j) + ".hdf5")
	learnrate /= 2
	j = j + 1
	#evaluation
	y_pred=model.predict(X_test)
	print(np.shape(y_pred))
	y_pred=np.exp(y_pred-14.92)
	print(np.shape(y_pred))
	y_true=np.exp(y_test-14.92)
	print(np.shape(y_true))
	y_true=np.reshape(y_true, (-1,1))
	y_true=np.array(y_true)
	y_pred=np.array(y_pred)
	#true vs. prediction figure for the 10k sample test set
	ideal = np.arange(np.min(y_true), np.max(y_true), 0.005)
	fig3=plt.figure(figsize=(10, 10))
	plt.plot(y_true, y_pred, 'ro', ideal, ideal, 'r--')
	plt.title('True vs. predicted cross-section')
	plt.xlabel('true value')
	plt.ylabel('predicted value')
	plt.tight_layout()
	plt.savefig('25_prediction_120bs_' + str(j) + '.png')
	plt.close()
	#sorted percentage deviations
	y_diff = y_true - y_pred
	print(np.shape(y_diff))
	percdev = np.true_divide(y_diff, y_true)
	print(np.shape(percdev))
	percdev = np.abs(percdev)
	print(np.shape(percdev))
	percdev = np.sort(-percdev, axis = 0)
	percdev = -percdev
	print(np.shape(percdev))
	print(percdev)
	five_pc=0
	ten_pc=0
	for i in range(0,len(percdev)):
		if abs(percdev[i]) > 0.05:
			five_pc = five_pc + 1
		if abs(percdev[i]) > 0.1:
			ten_pc = ten_pc + 1
	print(five_pc)
	print(ten_pc)
	test_mape = np.mean(percdev, axis = 0)
	print(test_mape)
	mapes.write("iteration: " + str(j) + " mape: " + str(test_mape) + " over 5 percent: " + str(five_pc) + " over 10 percent: " + str(ten_pc) + "\n")
mapes.close()

### second model: validation data during training is acitvely sampled data
# indices = np.random.randint(1, 2299999, size = 299999)
# X_AL_val = X_AL[indices]
# y_AL_val = y_AL[indices]
# X_AL = np.delete(X_AL, indices, axis = 0)
# y_AL = np.delete(y_AL, indices)

# model=Sequential()
# for i in range(0,8):
	# model.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.125), input_dim=11))
	# model.add(Activation('selu'))
# model.add(Dense(1, init='uniform',activation='linear'))
# j=0
# history = History()

# learnrate=0.0008
# epochs=250
# pat=75
# bat=1000
# iterations=10
# lim=learnrate/(2**iterations)

# mapes = open('25_AL_mapes_1.txt', 'w')
# while learnrate > lim:
	# checkpointer = ModelCheckpoint(filepath="25_ALdata_2_" + str(j) + ".hdf5", verbose=1, save_best_only=True)
	# early_stopping = EarlyStopping(monitor='val_loss', patience=pat)
	# opt = Adam(lr=learnrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	# model.compile(loss="exp_xsec", optimizer=opt)
	# model.fit(X_AL, y_AL, validation_data=(X_AL_val,y_AL_val), nb_epoch=epochs, batch_size=bat, verbose=2, callbacks=[history,checkpointer,early_stopping])
	# model.load_weights("25_ALdata_2_" + str(j) + ".hdf5")
	# learnrate /= 2
	# j = j + 1
	#evaluation
	# y_pred=model.predict(X_test)
	# print(np.shape(y_pred))
	# y_pred=np.exp(y_pred-14.92)
	# print(np.shape(y_pred))
	# y_true=np.exp(y_test-14.92)
	# print(np.shape(y_true))
	# y_true=np.reshape(y_true, (-1,1))
	# y_true=np.array(y_true)
	# y_pred=np.array(y_pred)
	#true vs. prediction figure for the 10k sample test set
	# ideal = np.arange(np.min(y_true), np.max(y_true), 0.005)
	# fig3=plt.figure(figsize=(10, 10))
	# plt.plot(y_true, y_pred, 'ro', ideal, ideal, 'r--')
	# plt.title('True vs. predicted cross-section')
	# plt.xlabel('true value')
	# plt.ylabel('predicted value')
	# plt.tight_layout()
	# plt.savefig('25_prediction_' + str(j) + '.png')
	# plt.close()
	#sorted percentage deviations
	# y_diff = y_true - y_pred
	# print(np.shape(y_diff))
	# percdev = np.true_divide(y_diff, y_true)
	# print(np.shape(percdev))
	# percdev = np.abs(percdev)
	# print(np.shape(percdev))
	# percdev = np.sort(-percdev, axis = 0)
	# percdev = -percdev
	# print(np.shape(percdev))
	# print(percdev)
	# five_pc=0
	# ten_pc=0
	# for i in range(0,len(percdev)):
		# if abs(percdev[i]) > 0.05:
			# five_pc = five_pc + 1
		# if abs(percdev[i]) > 0.1:
			# ten_pc = ten_pc + 1
	# print(five_pc)
	# print(ten_pc)
	# test_mape = np.mean(percdev, axis = 0)
	# print(test_mape)
	# mapes.write("iteration: " + str(j) + " mape: " + str(test_mape) + " over 5 percent: " + str(five_pc) + " over 10 percent: " + str(ten_pc) + "\n")
# mapes.close()


