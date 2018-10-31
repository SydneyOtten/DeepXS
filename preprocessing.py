import numpy as np

#c1c1
def preprocess_c1c1(input, LO=0, NLO=1, array = 0):
	mean_LO = np.empty(6)
	mean_LO[0] = -5.102739078010142260e-01
	mean_LO[1] = 2.583048919518980170e-03
	mean_LO[2] = -5.519746080358358675e-01
	mean_LO[3] = 5.539420961848278147e-01
	mean_LO[4] = 5.973604356655324636e+02
	mean_LO[5] = 2.831598598306486565e+03
	std_LO = np.empty(6)
	std_LO[0] = 4.629770922954990486e-01
	std_LO[1] = 7.247524259259684465e-01
	std_LO[2] = 4.392149272409718308e-01
	std_LO[3] = 4.422244157060932768e-01
	std_LO[4] = 4.091132673021468236e+02
	std_LO[5] = 1.269605008557467954e+03
	K_mean = np.empty(7)
	K_std = np.empty(7)
	K_mean[0] = -5.088403499832118149e-01
	K_mean[1] = 6.159000596876967534e-03
	K_mean[2] = -5.505627206159491305e-01
	K_mean[3] = 5.551189231571198590e-01
	K_mean[4] = 5.963042241893368782e+02
	K_mean[5] = 2.829201442946224688e+03
	K_mean[6] = 2.102373181263182687e+03
	K_std[0] = 4.631882721324815932e-01
	K_std[1] = 7.256033279320006635e-01
	K_std[2] = 4.394594816465790532e-01
	K_std[3] = 4.422657976934696311e-01
	K_std[4] = 4.100095581086242760e+02
	K_std[5] = 7.838001655706688098e+02
	K_std[6] = 9.362587992830435724e+02
	if array == 0:
		X = np.empty([1,6])
		X[0,0] = input[0,0]
		X[0,1] = input[0,1]
		X[0,2] = input[0,2]
		X[0,3] = input[0,3]
		X[0,4] = input[0,9]
		X[0,5] = input[0,10]
		c1c1_LO = (X-mean_LO)/std_LO
		if LO == 1 and NLO == 0:
			return c1c1_LO
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			X_NLO = np.empty([1,7])
			avg_sq = (input[0,10] + input[0,11] + input[0,12] + input[0,13] + input[0,14] + input[0,15] + input[0,16] + input[0,17])/8
			X_NLO[0,0] = X[0,0]
			X_NLO[0,1] = X[0,1]
			X_NLO[0,2] = X[0,2]
			X_NLO[0,3] = X[0,3]
			X_NLO[0,4] = X[0,4]
			X_NLO[0,5] = avg_sq
			X_NLO[0,6] = input[0,18]
			c1c1_NLO = (X_NLO-K_mean)/K_std
			return c1c1_LO, c1c1_NLO
	elif array == 1:
		X = np.empty([input.shape[0], 6])
		X[:,0] = input[:,0]
		X[:,1] = input[:,1]
		X[:,2] = input[:,2]
		X[:,3] = input[:,3]
		X[:,4] = input[:,4]
		X[:,5] = input[:,5]
		c1c1_LO = (X-mean_LO)/std_LO
		if LO == 1 and NLO == 0:
			return c1c1_LO
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			X_NLO = np.empty([input.shape[0],7])
			avg_sq = (input[:,5] + input[:,6] + input[:,7] + input[:,8] + input[:,9] + input[:,10] + input[:,11] + input[:,12])/8
			X_NLO[:,0] = X[:,0]
			X_NLO[:,1] = X[:,1]
			X_NLO[:,2] = X[:,2]
			X_NLO[:,3] = X[:,3]
			X_NLO[:,4] = X[:,4]
			X_NLO[:,5] = avg_sq
			X_NLO[:,6] = input[:,13]
			c1c1_NLO = (X_NLO-K_mean)/K_std
			return c1c1_LO, c1c1_NLO
#n2n2
def preprocess_n2n2(input, LO=0, NLO=1, array=0):
	mean_LO = np.empty(7)
	mean_LO[0] = 2.808734133596148611e-02
	mean_LO[1] = -1.311933578625973200e-02
	mean_LO[2] = -3.641857329865700915e-04
	mean_LO[3] = 6.132428251880508014e+02
	mean_LO[4] = 2.745410157255745617e+03
	mean_LO[5] = 2.800873082515320675e+03
	mean_LO[6] = 2.733281149497417573e+03
	std_LO = np.empty(7)
	std_LO[0] = 5.060287376220525823e-01
	std_LO[1] = 5.965248685939138484e-01
	std_LO[2] = 1.787471153814866598e-02
	std_LO[3] = 4.049841443600027446e+02
	std_LO[4] = 1.284149362675619386e+03
	std_LO[5] = 1.284389590288986483e+03
	std_LO[6] = 1.275291753946552262e+03
	K_mean = np.empty(6)
	K_std = np.empty(6)
	K_mean[0] = 3.818234395841604845e-02
	K_mean[1] = -2.289492807797044102e-02
	K_mean[2] = -8.735390882528014421e-04
	K_mean[3] = 6.124078503198890076e+02
	K_mean[4] = 2.759063492316725842e+03
	K_mean[5] = 2.072033230780390113e+03
	K_std[0] = 5.220634712147383949e-01
	K_std[1] = 6.445110254751014178e-01
	K_std[2] = 1.846261325975284695e-02
	K_std[3] = 3.870430305177503101e+02
	K_std[4] = 7.814676667444624627e+02
	K_std[5] = 9.495355646774295337e+02
	if array == 0:
		X = np.empty([1,7])
		X[0,0] = input[0,4]
		X[0,1] = input[0,5]
		X[0,2] = input[0,6]**2-input[0,7]**2
		X[0,3] = np.abs(input[0,8])
		X[0,4] = input[0,10]
		X[0,5] = input[0,11]
		X[0,6] = input[0,12]
		n2n2_LO = (X-mean_LO)/std_LO
		if LO == 1 and NLO == 0:
			return n2n2_LO
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			X_NLO = np.empty([1,6])
			avg_sq = (input[0,10] + input[0,11] + input[0,12] + input[0,13] + input[0,14] + input[0,15] + input[0,16] + input[0,17])/8
			X_NLO[0,0] = X[0,0]
			X_NLO[0,1] = X[0,1]
			X_NLO[0,2] = X[0,2]
			X_NLO[0,3] = X[0,3]
			X_NLO[0,4] = avg_sq
			X_NLO[0,5] = input[0,18]
			n2n2_NLO = (X_NLO-K_mean)/K_std
			return n2n2_LO, n2n2_NLO
	elif array == 1:
		X = np.empty([input.shape[0], 7])
		X[:,0] = input[:,0]
		X[:,1] = input[:,1]
		X[:,2] = input[:,2]**2-input[:,3]**2
		X[:,3] = np.abs(input[:,4])
		X[:,4] = input[:,5]
		X[:,5] = input[:,6]
		X[:,6] = input[:,7]
		n2n2_LO = (X-mean_LO)/std_LO
		if LO == 1 and NLO == 0:
			return n2n2_LO
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			X_NLO = np.empty([input.shape[0],6])
			avg_sq = (input[:,5] + input[:,6] + input[:,7] + input[:,8] + input[:,9] + input[:,10] + input[:,11] + input[:,12])/8
			X_NLO[:,0] = X[:,0]
			X_NLO[:,1] = X[:,1]
			X_NLO[:,2] = X[:,2]
			X_NLO[:,3] = X[:,3]
			X_NLO[:,4] = avg_sq
			X_NLO[:,5] = input[:,13]
			n2n2_NLO = (X_NLO-K_mean)/K_std
			return n2n2_LO, n2n2_NLO
#n2c1+
def preprocess_n2c1p(input, LO=0, NLO=1, array=0):
	mean_LO = np.empty(11)
	mean_LO[0] = -4.537002888949617785e-01
	mean_LO[1] = 4.597430389806048123e-03
	mean_LO[2] = -5.002146448363983255e-01
	mean_LO[3] = 6.081531493106234754e-01
	mean_LO[4] = 2.176255481718071885e-02
	mean_LO[5] = -2.348948826829861097e-03
	mean_LO[6] = 2.967871627075927173e-03
	mean_LO[7] = 9.697539393311536970e-02
	mean_LO[8] = 1.004278354827091420e+02
	mean_LO[9] = 5.688475700591559416e+02
	mean_LO[10] = 2.850835013058896038e+03
	std_LO = np.empty(11)
	std_LO[0] = 4.570470965193643709e-01
	std_LO[1] = 7.650116751094742096e-01
	std_LO[2] = 4.340662148358088168e-01
	std_LO[3] = 4.376317826813462264e-01
	std_LO[4] = 3.638953943364385268e-01
	std_LO[5] = 4.165887397718208152e-01
	std_LO[6] = 5.885984932336143505e-01
	std_LO[7] = 5.811141161562578539e-01
	std_LO[8] = 7.793744063258113783e+02
	std_LO[9] = 3.874963921375447740e+02
	std_LO[10] = 1.273623621766418410e+03
	K_mean = np.empty(12)
	K_std = np.empty(12)
	K_mean[0] = -4.955848563684884245e-01
	K_mean[1] = 2.146729600782480620e-03
	K_mean[2] = -5.379458903095909461e-01
	K_mean[3] = 5.706769058124540051e-01
	K_mean[4] = 1.517059504712743936e-02
	K_mean[5] = 2.162442702591358887e-03
	K_mean[6] = 5.752247356418870375e-03
	K_mean[7] = 9.502773894843294378e-02
	K_mean[8] = 1.366698952505785769e+02
	K_mean[9] = 5.986018752343833285e+02
	K_mean[10] = 2.818999073715641771e+03
	K_mean[11] = 2.105691372604444950e+03
	K_std[0] = 4.611028291501528265e-01
	K_std[1] = 7.360538177606378296e-01
	K_std[2] = 4.375610078697738659e-01
	K_std[3] = 4.398664031993423928e-01
	K_std[4] = 3.853855775783795456e-01
	K_std[5] = 4.309885665985029868e-01
	K_std[6] = 5.770732720647054892e-01
	K_std[7] = 5.686959364128325589e-01
	K_std[8] = 8.402302685138949983e+02
	K_std[9] = 4.067668872853457742e+02
	K_std[10] = 7.805322832904921597e+02
	K_std[11] = 9.336197652346548921e+02
	if array == 0:
		X = np.empty([1,11])
		X[0,0] = input[0,0]
		X[0,1] = input[0,1]
		X[0,2] = input[0,2]
		X[0,3] = input[0,3]
		X[0,4] = input[0,4]
		X[0,5] = input[0,5]
		X[0,6] = input[0,6]
		X[0,7] = input[0,7]
		X[0,8] = input[0,8]
		X[0,9] = input[0,9]
		X[0,10] = input[0,10]
		n2c1p_LO = (X-mean_LO)/std_LO
		if LO == 1 and NLO == 0:
			return n2c1p_LO
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			X_NLO = np.empty([1,12])
			avg_sq = (input[0,10] + input[0,11] + input[0,12] + input[0,13] + input[0,14] + input[0,15] + input[0,16] + input[0,17])/8
			X_NLO[0,0] = X[0,0]
			X_NLO[0,1] = X[0,1]
			X_NLO[0,2] = X[0,2]
			X_NLO[0,3] = X[0,3]
			X_NLO[0,4] = X[0,4]
			X_NLO[0,5] = X[0,5]
			X_NLO[0,6] = X[0,6]
			X_NLO[0,7] = X[0,7]
			X_NLO[0,8] = X[0,8]
			X_NLO[0,9] = X[0,9]
			X_NLO[0,10] = avg_sq
			X_NLO[0,11] = input[0,18]
			n2c1p_NLO = (X_NLO-K_mean)/K_std
			return n2c1p_LO, n2c1p_NLO
	elif array == 1:
		X = np.empty([input.shape[0],11])
		X[:,0] = input[:,0]
		X[:,1] = input[:,1]
		X[:,2] = input[:,2]
		X[:,3] = input[:,3]
		X[:,4] = input[:,4]
		X[:,5] = input[:,5]
		X[:,6] = input[:,6]
		X[:,7] = input[:,7]
		X[:,8] = input[:,8]
		X[:,9] = input[:,9]
		X[:,10] = input[:,10]
		n2c1p_LO = (X-mean_LO)/std_LO
		if LO == 1 and NLO == 0:
			return n2c1p_LO
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			X_NLO = np.empty([input.shape[0],12])
			avg_sq = (input[:,10] + input[:,11] + input[:,12] + input[:,13] + input[:,14] + input[:,15] + input[:,16] + input[:,17])/8
			X_NLO[:,0] = X[:,0]
			X_NLO[:,1] = X[:,1]
			X_NLO[:,2] = X[:,2]
			X_NLO[:,3] = X[:,3]
			X_NLO[:,4] = X[:,4]
			X_NLO[:,5] = X[:,5]
			X_NLO[:,6] = X[:,6]
			X_NLO[:,7] = X[:,7]
			X_NLO[:,8] = X[:,8]
			X_NLO[:,9] = X[:,9]
			X_NLO[:,10] = avg_sq
			X_NLO[:,11] = input[:,18]
			n2c1p_NLO = (X_NLO-K_mean)/K_std
			return n2c1p_LO, n2c1p_NLO
#n2c1-
def preprocess_n2c1m(input, LO=0, NLO=1, array=0):
	mean_LO = np.empty(11)
	mean_LO[0] = -4.755700840164661658e-01
	mean_LO[1] = 2.117738132998363276e-03
	mean_LO[2] = -5.205929753129339144e-01
	mean_LO[3] = 5.932704962568170481e-01
	mean_LO[4] = 1.493624710005907186e-02
	mean_LO[5] = 3.511309857754961244e-04
	mean_LO[6] = 5.143425461411699727e-03
	mean_LO[7] = 9.235123896347514905e-02
	mean_LO[8] = 1.321578801033057289e+02
	mean_LO[9] = 5.899694535525778747e+02
	mean_LO[10] = 2.779667372338677978e+03
	std_LO = np.empty(11)
	std_LO[0] = 4.563328835747698098e-01
	std_LO[1] = 7.520564538263386778e-01
	std_LO[2] = 4.331496327629039134e-01
	std_LO[3] = 4.351947471685814750e-01
	std_LO[4] = 3.585866735089053292e-01
	std_LO[5] = 4.348928306944435662e-01
	std_LO[6] = 5.840130928472564431e-01
	std_LO[7] = 5.765709928893349989e-01
	std_LO[8] = 8.043075775955236395e+02
	std_LO[9] = 4.061052148714400687e+02
	std_LO[10] = 1.281859813936049477e+03
	K_mean = np.empty(12) 
	K_std = np.empty(12)
	K_mean[0] = -4.752316561512869297e-01
	K_mean[1] = 1.008745410198923350e-03
	K_mean[2] = -5.193984414324636090e-01
	K_mean[3] = 5.944630700360548081e-01
	K_mean[4] = 1.524512519239599150e-02
	K_mean[5] = 8.229392973366991696e-04
	K_mean[6] = 4.899823033873955923e-03
	K_mean[7] = 9.410886163038903462e-02
	K_mean[8] = 1.328307981346624445e+02
	K_mean[9] = 5.962538120471474485e+02
	K_mean[10] = 2.810137797573902844e+03
	K_mean[11] = 2.104149157937155451e+03
	K_std[0] = 4.562051660259190955e-01
	K_std[1] = 7.523501192367527679e-01
	K_std[2] = 4.329109238549984640e-01
	K_std[3] = 4.352321787348275572e-01
	K_std[4] = 3.585819410276261943e-01
	K_std[5] = 4.359291130283679538e-01
	K_std[6] = 5.837564838028922454e-01
	K_std[7] = 5.757600111385091646e-01
	K_std[8] = 8.078324535461806590e+02
	K_std[9] = 4.052541040852241281e+02
	K_std[10] = 7.842787973080340862e+02
	K_std[11] = 9.370524988492832108e+02
	if array == 0:
		X = np.empty([1,11])
		X[0,0] = input[0,0]
		X[0,1] = input[0,1]
		X[0,2] = input[0,2]
		X[0,3] = input[0,3]
		X[0,4] = input[0,4]
		X[0,5] = input[0,5]
		X[0,6] = input[0,6]
		X[0,7] = input[0,7]
		X[0,8] = input[0,8]
		X[0,9] = input[0,9]
		X[0,10] = input[0,10]
		n2c1m_LO = (X-mean_LO)/std_LO
		if LO == 1 and NLO == 0:
			return n2c1m_LO
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			X_NLO = np.empty([1,12])
			avg_sq = (input[0,10] + input[0,11] + input[0,12] + input[0,13] + input[0,14] + input[0,15] + input[0,16] + input[0,17])/8
			X_NLO[0,0] = X[0,0]
			X_NLO[0,1] = X[0,1]
			X_NLO[0,2] = X[0,2]
			X_NLO[0,3] = X[0,3]
			X_NLO[0,4] = X[0,4]
			X_NLO[0,5] = X[0,5]
			X_NLO[0,6] = X[0,6]
			X_NLO[0,7] = X[0,7]
			X_NLO[0,8] = X[0,8]
			X_NLO[0,9] = X[0,9]
			X_NLO[0,10] = avg_sq
			X_NLO[0,11] = input[0,18]
			n2c1m_NLO = (X_NLO-K_mean)/K_std
			return n2c1m_LO, n2c1m_NLO
	elif array == 1:
		X = np.empty([input.shape[0],11])
		X[:,0] = input[:,0]
		X[:,1] = input[:,1]
		X[:,2] = input[:,2]
		X[:,3] = input[:,3]
		X[:,4] = input[:,4]
		X[:,5] = input[:,5]
		X[:,6] = input[:,6]
		X[:,7] = input[:,7]
		X[:,8] = input[:,8]
		X[:,9] = input[:,9]
		X[:,10] = input[:,10]
		n2c1m_LO = (X-mean_LO)/std_LO
		if LO == 1 and NLO == 0:
			return n2c1m_LO
		if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
			X_NLO = np.empty([input.shape[0],12])
			avg_sq = (input[:,10] + input[:,11] + input[:,12] + input[:,13] + input[:,14] + input[:,15] + input[:,16] + input[:,17])/8
			X_NLO[:,0] = X[:,0]
			X_NLO[:,1] = X[:,1]
			X_NLO[:,2] = X[:,2]
			X_NLO[:,3] = X[:,3]
			X_NLO[:,4] = X[:,4]
			X_NLO[:,5] = X[:,5]
			X_NLO[:,6] = X[:,6]
			X_NLO[:,7] = X[:,7]
			X_NLO[:,8] = X[:,8]
			X_NLO[:,9] = X[:,9]
			X_NLO[:,10] = avg_sq
			X_NLO[:,11] = input[:,18]
			n2c1m_NLO = (X_NLO-K_mean)/K_std
			return n2c1m_LO, n2c1m_NLO