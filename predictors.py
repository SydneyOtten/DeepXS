from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K
import numpy as np
#predict and post-process
#c1c1
def predict_c1c1(LO_data,NLO_data,LO_model,NLO_model,LO=0,NLO=1):
	LO_pred=LO_model.predict(LO_data)
	LO_pred=np.exp(3.568339158293295288*LO_pred-4.958795503662296156)
	LO_pred=LO_pred.astype(np.float)
	print("\n The Chargino1/Chargino1 pair production cross-sections are: \n")
	print("At leading order: " + str(LO_pred) + " pb\n")
	if LO == 1 and NLO == 0:
		print(LO_pred)
		return LO_pred
	if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
		K_pred=NLO_model.predict(NLO_data)
		K_pred=K_pred*2
		K_pred=K_pred.astype(np.float)
		print("With a K-factor of: " + str(K_pred) + " \n")
		NLO_pred=LO_pred*K_pred
		NLO_pred = NLO_pred.astype(np.float)
		print("Therefore, at next-to-leading order: " + str(NLO_pred) + " pb\n")
		np.savetxt('c1c1_LO_pred.txt', LO_pred)
		np.savetxt('c1c1_K_pred.txt', K_pred)
		np.savetxt('c1c1_NLO_pred.txt', NLO_pred)
		return LO_pred, K_pred, NLO_pred
#n2n2
def predict_n2n2(LO_data,NLO_data,LO_model,NLO_model,LO=0,NLO=1):
	LO_pred=LO_model.predict(LO_data)
	LO_pred=np.exp(LO_pred-14.92)
	LO_pred=LO_pred.astype(np.float)
	print("\n The Neutralino2/Neutralino2 pair production cross-sections are: \n")
	print("At leading order: " + str(LO_pred) + " pb\n")
	if LO == 1 and NLO == 0:
		return LO_pred
	if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
		K_pred=NLO_model.predict(NLO_data)
		K_pred = K_pred*2
		K_pred = K_pred.astype(np.float)
		print("With a K-factor of: " + str(K_pred) + " \n")
		NLO_pred=LO_pred*K_pred
		NLO_pred = NLO_pred.astype(np.float)
		print("Therefore, at next-to-leading order: " + str(NLO_pred) + " pb\n")
		np.savetxt('n2n2_LO_pred.txt', LO_pred)
		np.savetxt('n2n2_K_pred.txt', K_pred)
		np.savetxt('n2n2_NLO_pred.txt', NLO_pred)
		return LO_pred, K_pred, NLO_pred
#n2c1+
def predict_n2c1p(LO_data,NLO_data,LO_model_gen,LO_model_spec,NLO_model_gen,NLO_model_spec,LO,NLO):
	LO_pred=LO_model_gen.predict(LO_data)
	LO_pred = np.exp(LO_pred-14.92)
	LO_pred = LO_pred.astype(np.float)
	print("\n The Neutralino2/Chargino1+ pair production cross-sections are: \n")
	if len(LO_pred) == 1:
		LO_pred = np.float(LO_pred)
		if LO_pred > 0.0001:
			if LO_pred < 0.4:
				LO_pred = LO_model_spec.predict(LO_data)
				LO_pred = np.exp(LO_pred-9.22)
				LO_pred = LO_pred.astype(np.float)
		print("At leading order: " + str(LO_pred) + " pb\n")
		if LO == 1 and NLO == 0:
			return LO_pred
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			l = 6.6*10**(-5)
			if LO_pred > l:
				K_pred = NLO_model_spec.predict(NLO_data)
				K_pred = K_pred*4
				K_pred = K_pred.astype(np.float)
				NLO_pred=LO_pred*K_pred
			else:
				K_pred=NLO_model_gen.predict(NLO_data)
				K_pred = K_pred*4
				K_pred = K_pred.astype(np.float)
				NLO_pred=LO_pred*K_pred
			print("With a K-factor of: " + str(K_pred) + " \n")
			print("Therefore, at next-to-leading order: " + str(NLO_pred) + " pb\n")
			np.savetxt('n2c1+_LO_pred.txt', LO_pred)
			np.savetxt('n2c1+_K_pred.txt', K_pred)
			np.savetxt('n2c1+_NLO_pred.txt', NLO_pred)
			return LO_pred, K_pred, NLO_pred
	else:
		ind = []
		for i in range(0,LO_pred.shape[0]):
			if LO_pred[i] > 0.0001:
				if LO_pred[i] < 0.4:
					ind.append(i)
		LO_data_2 = LO_data[ind]
		LO_pred_2 = LO_model_spec.predict(LO_data_2)
		LO_pred_2 = np.exp(LO_pred_2-9.22)
		LO_pred_2 = LO_pred_2.astype(np.float)
		LO_pred[ind] = LO_pred_2
		if LO == 1 and NLO == 0:
			return LO_pred
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			l = 6.6*10**(-5)
			K_ind = []
			for i in range(0, LO_pred.shape[0]):
				if LO_pred[i] > l:
					K_ind.append(i)
			K_pred=NLO_model_gen.predict(NLO_data)
			K_pred=K_pred*4
			K_pred = K_pred.astype(np.float)
			NLO_pred=LO_pred*K_pred
			NLO_data_2 = NLO_data[K_ind]
			K_pred_2 = NLO_model_spec.predict(NLO_data_2)
			K_pred_2 = K_pred_2*4
			K_pred_2 = K_pred_2.astype(np.float)
			K_pred[K_ind] = K_pred_2
			np.savetxt('n2c1+_LO_pred.txt', LO_pred)
			np.savetxt('n2c1+_K_pred.txt', K_pred)
			np.savetxt('n2c1+_NLO_pred.txt', NLO_pred)
			return LO_pred, K_pred, NLO_pred
		
#n2c1-
def predict_n2c1m(LO_data,NLO_data,LO_model_gen,LO_model_spec,NLO_model,LO,NLO):
	LO_pred=LO_model_gen.predict(LO_data)
	LO_pred = np.exp(LO_pred-14.92)
	y_hybrid_mean = 3.103633007237718497e-02
	y_hybrid_std = 4.452607505787019998e-02
	print("\n The Neutralino2/Chargino1- pair production cross-sections are: \n")
	if len(LO_pred) == 1:
		LO_pred = np.float(LO_pred)
		if LO_pred > 0.001:
			if LO_pred < 0.2:
				LO_pred = LO_model_spec.predict(LO_data)
				LO_pred = np.float(LO_pred*y_hybrid_std+y_hybrid_mean)
		print("At leading order: " + str(LO_pred) + " pb\n")
		if LO == 1 and NLO == 0:
			return LO_pred
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			K_pred=NLO_model.predict(NLO_data)
			K_pred=np.float(K_pred*4)
			NLO_pred=LO_pred*K_pred
			print("With a K-factor of: " + str(K_pred) + " \n")
			print("Therefore, at next-to-leading order: " + str(NLO_pred) + " pb\n")
			np.savetxt('n2c1-_LO_pred.txt', LO_pred)
			np.savetxt('n2c1-_K_pred.txt', K_pred)
			np.savetxt('n2c1-_NLO_pred.txt', NLO_pred)
			return LO_pred, K_pred, NLO_pred
	else:
		LO_pred = LO_pred.astype(np.float)
		ind = []
		for i in range(0,LO_pred.shape[0]):
			if LO_pred[i] > 0.001:
				if LO_pred[i] < 0.2:
					ind.append(i)
		LO_data_2 = LO_data[ind]
		LO_pred_2 = LO_model_spec.predict(LO_data_2)
		LO_pred_2 = LO_pred_2*y_hybrid_std+y_hybrid_mean
		LO_pred_2 = LO_pred_2.astype(np.float)
		LO_pred[ind] = LO_pred_2
		if LO == 1 and NLO == 0:
			return LO_pred
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			K_pred=NLO_model.predict(NLO_data)
			K_pred=K_pred*4
			K_pred = K_pred.astype(np.float)
			NLO_pred=LO_pred*K_pred
			np.savetxt('n2c1-_LO_pred.txt', LO_pred)
			np.savetxt('n2c1-_K_pred.txt', K_pred)
			np.savetxt('n2c1-_NLO_pred.txt', NLO_pred)
			return LO_pred, K_pred, NLO_pred