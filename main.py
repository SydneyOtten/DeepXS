import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K
import pyslha
import argparse

"""Essential Functions"""
#argparse for indicating the pairs to predict

parser = argparse.ArgumentParser(description='DeepXs - predicting cross-sections since 2018')
parser.add_argument('-pairs', dest='integers', type=int, default=0,
                    help='Defines the pair(s). Check readme.txt for the specification.')
parser.add_argument('-return', dest='order', action='store', default='both',
                    help='Specifies if you want the cross-sections at leading (LO) or next-to-leading order (NLO). If you want both, specify -return both')
parser.add_argument('-stream', dest='stream', action='store', default=1, help='Specifies if you want the tensorflow session to continue. You will then have the possibility to continuously feed input data as DeepXs waits for new input.')
parser.add_argument('-Array', dest='Array', action='store', default=0, help='Declares if you want to give the input as an array (1) from a .txt file or not (0) and use an SLHA-file instead.')
parser.add_argument('-fn', dest='fn', action='store', type=str, help='Specifies the .txt file that DeepXs reads to predict the corresponding cross-section(s). The required layout is described in the readme.txt.')
args = parser.parse_args()

if int(args.integers) < 15 and int(args.integers) >= 0:
	setting = int(args.integers)
	print('You have specified to return the cross-section for pair combination ' + str(args.integers) + '.')
else:
	print('Please specify an integer that is between 0 and 14.')
	exit()
if args.order == 'LO':
	print('Summoning the AI to predict the leading order cross-section.')
	LO=1
	NLO=0
elif args.order == 'NLO':
	print('Summoning the AI to predict the next-to-leading order cross-section.')
	LO=0
	NLO=1
elif args.order == 'both':
	print('Summoning the AI to predict the leading and next-to-leading order cross-section.')
	LO=1
	NLO=1
else:
	print('Please specify either if you want me to return the LO, NLO or both cross-sections. You can do this by running the script via "python DeepXs.py PAIR_NUMBER -return LO/NLO/both". E.g. if you want to predict all pairs at NLO, do "python DeepXs.py 0 -return both".')
	exit()
if int(args.Array) == 0:
	print('Reading input from SLHA file')
	Array=0
elif int(args.Array) == 1:
	print('Reading input from an Array')
	Array=1
else:
	print('Please either specify -Array 0 or -Array 1.')
	exit()
if int(args.stream) == 0:
	active = False
elif int(args.stream) == 1:
	print('Opening a continuous portal to our AI for continuous streaming of input and output.')
	active = True
else:
	print('Please specify either -stream 0 or -stream 1.')
	exit()
fn = str(args.fn)
print('\n Welcome to DeepXs version 1.0\n')

#read input

def read_input(filename, Array = 0, fn = fn):
	if Array == 0:
		SLHA_FILE=pyslha.read(filename,  ignoreblocks=['SPINFO'])
		SLHA_FILE.blocks
		#parameters we're interested in
		mixing1=float(SLHA_FILE.blocks["UMIX"][1,1])
		mixing2=float(SLHA_FILE.blocks["UMIX"][1,2])
		mixing3=float(SLHA_FILE.blocks["VMIX"][1,1])
		mixing4=float(SLHA_FILE.blocks["VMIX"][1,2])
		mixing5=float(SLHA_FILE.blocks["NMIX"][2,1])
		mixing6=float(SLHA_FILE.blocks["NMIX"][2,2])
		mixing7=float(SLHA_FILE.blocks["NMIX"][2,3])
		mixing8=float(SLHA_FILE.blocks["NMIX"][2,4])
		mneut1=float(SLHA_FILE.blocks["MASS"][1000022])
		mneut2=float(SLHA_FILE.blocks["MASS"][1000023])
		mchar1=float(SLHA_FILE.blocks["MASS"][1000024])
		mchar2=float(SLHA_FILE.blocks["MASS"][1000037])
		msq1=float(SLHA_FILE.blocks["MASS"][1000001])
		msq2=float(SLHA_FILE.blocks["MASS"][2000001])
		msq3=float(SLHA_FILE.blocks["MASS"][2000002])
		msq4=float(SLHA_FILE.blocks["MASS"][1000002])
		msq5=float(SLHA_FILE.blocks["MASS"][1000003])
		msq6=float(SLHA_FILE.blocks["MASS"][2000003])
		msq7=float(SLHA_FILE.blocks["MASS"][1000004])
		msq8=float(SLHA_FILE.blocks["MASS"][2000004])
		mgluino=float(SLHA_FILE.blocks["MASS"][1000021])
		#other masses for making sure that n1 is the LSP
		lsp1=float(SLHA_FILE.blocks["MASS"][1000023])
		lsp2=float(SLHA_FILE.blocks["MASS"][1000025])
		lsp3=float(SLHA_FILE.blocks["MASS"][1000035])
		lsp4=float(SLHA_FILE.blocks["MASS"][1000024])
		lsp5=float(SLHA_FILE.blocks["MASS"][1000037])
		lsp6=float(SLHA_FILE.blocks["MASS"][1000002])
		lsp7=float(SLHA_FILE.blocks["MASS"][1000003])
		lsp8=float(SLHA_FILE.blocks["MASS"][2000003])
		lsp9=float(SLHA_FILE.blocks["MASS"][1000004])
		lsp10=float(SLHA_FILE.blocks["MASS"][2000004])
		lsp11=float(SLHA_FILE.blocks["MASS"][1000005])
		lsp12=float(SLHA_FILE.blocks["MASS"][2000005])
		lsp13=float(SLHA_FILE.blocks["MASS"][1000006])
		lsp14=float(SLHA_FILE.blocks["MASS"][2000006])
		lsp15=float(SLHA_FILE.blocks["MASS"][1000011])
		lsp16=float(SLHA_FILE.blocks["MASS"][2000011])
		lsp17=float(SLHA_FILE.blocks["MASS"][1000012])
		lsp18=float(SLHA_FILE.blocks["MASS"][1000013])
		lsp19=float(SLHA_FILE.blocks["MASS"][2000013])
		lsp20=float(SLHA_FILE.blocks["MASS"][1000014])
		lsp21=float(SLHA_FILE.blocks["MASS"][1000015])
		lsp22=float(SLHA_FILE.blocks["MASS"][2000015])
		lsp23=float(SLHA_FILE.blocks["MASS"][1000016])
		lsp24=float(SLHA_FILE.blocks["MASS"][1000021])
		if abs(mneut1) < abs(msq1) and abs(mneut1) < abs(msq2) and abs(mneut1) < abs(msq3) and abs(mneut1) < abs(lsp1) and abs(mneut1) < abs(lsp2) and abs(mneut1) < abs(lsp3) and abs(mneut1) < abs(lsp4) and abs(mneut1) < abs(lsp5) and abs(mneut1) < abs(lsp6) and abs(mneut1) < abs(lsp7) and abs(mneut1) < abs(lsp8) and abs(mneut1) < abs(lsp9) and abs(mneut1) < abs(lsp10) and abs(mneut1) < abs(lsp11) and abs(mneut1) < abs(lsp12) and abs(mneut1) < abs(lsp13) and abs(mneut1) < abs(lsp14) and abs(mneut1) < abs(lsp15) and abs(mneut1) < abs(lsp16) and abs(mneut1) < abs(lsp17) and abs(mneut1) < abs(lsp18) and abs(mneut1) < abs(lsp19) and abs(mneut1) < abs(lsp20) and abs(mneut1) < abs(lsp21) and abs(mneut1) < abs(lsp22) and abs(mneut1) < abs(lsp23) and abs(mneut1) < abs(lsp24): 
			pass
		else:
			print('WARNING: Neutralino 1 is not the LSP. Our AIs are specialised on cross-sections for SUSY models in which Neutralino 1 is the lightest supersymmetric particle. Predictions in this region of the parameter space have not yet been tested for reliability.')
		if msq1 < 500 or msq2 < 500 or msq3 < 500 or msq4 < 500 or msq5 < 500 or msq6 < 500 or msq7 < 500 or msq8 < 500:
			print('WARNING: One of the squarks is lighter than what our AIs are trained on. Predictions in this region of the parameter space have not yet been tested for reliability.')
		if abs(mchar1) < 100:
			print('WARNING: The chargino1 has a mass lower than 100 GeV and is excluded.')
		input = np.empty([1,19])
		input[0,0] = mixing1
		input[0,1] = mixing2
		input[0,2] = mixing3
		input[0,3] = mixing4
		input[0,4] = mixing5
		input[0,5] = mixing6
		input[0,6] = mixing7
		input[0,7] = mixing8
		input[0,8] = mneut2
		input[0,9] = mchar1
		input[0,10] = msq1
		input[0,11] = msq2
		input[0,12] = msq3
		input[0,13] = msq4
		input[0,14] = msq5
		input[0,15] = msq6
		input[0,16] = msq7
		input[0,17] = msq8
		input[0,18] = mgluino
	elif Array == 1:
		raw_input = pd.read_csv(fn, sep=' ', header=None)
		input = raw_input.values
		input = input.astype(np.float)
	else:
		exit()
	return input
#transform input
#c1c1
def preprocess_c1c1(input, LO=0, NLO=1, Array = 0):
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
	if Array == 0:
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
	elif Array == 1:
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
def preprocess_n2n2(input, LO=0, NLO=1, Array=0):
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
	if Array == 0:
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
	elif Array == 1:
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
def preprocess_n2c1p(input, LO=0, NLO=1, Array=0):
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
	if Array == 0:
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
	elif Array == 1:
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
def preprocess_n2c1m(input, LO=0, NLO=1, Array=0):
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
	if Array == 0:
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
	elif Array == 1:
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
#build models and load weights
#c1c1
def build_c1c1_AI(LO_weightfile='./c1+c1-/c1c1_LO.hdf5', K_weightsfile='./c1+c1-/c1c1_K.hdf5', LO=0, NLO=1):
	c1c1_LO = Sequential()
	for i in range(0,8):
		c1c1_LO.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.1), input_dim=6))
		c1c1_LO.add(Activation('selu'))
	c1c1_LO.add(Dense(1, kernel_initializer='uniform',activation='linear'))
	c1c1_LO.add(Dense(1, kernel_initializer='uniform',activation='linear'))
	c1c1_LO.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
	c1c1_LO.load_weights(LO_weightfile)
	if LO == 1 and NLO == 0:
		return c1c1_LO
	if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
		c1c1_NLO = Sequential()
		for i in range(0,8):
			c1c1_NLO.add(Dense(32, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.176177), input_dim=7))
			c1c1_NLO.add(Activation('selu'))
		c1c1_NLO.add(Dense(1, kernel_initializer='uniform',activation='linear'))
		c1c1_NLO.load_weights(K_weightsfile)
		return c1c1_LO, c1c1_NLO

#n2n2
def build_n2n2_AI(LO_weightfile='./n2n2/n2n2_LO.hdf5', K_weightsfile='./n2n2/n2n2_K.hdf5', LO=0, NLO=1):
	n2n2_LO = Sequential()
	for i in range(0,8):
		n2n2_LO.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.1), input_dim=7))
		n2n2_LO.add(Activation('selu'))
	n2n2_LO.add(Dense(1, kernel_initializer='uniform',activation='linear'))
	n2n2_LO.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
	n2n2_LO.load_weights(LO_weightfile)
	if LO == 1 and NLO == 0:
		return n2n2_LO
	if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
		n2n2_NLO = Sequential()
		for i in range(0,8):
			n2n2_NLO.add(Dense(32, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.176177), input_dim=6))
			n2n2_NLO.add(Activation('selu'))
		n2n2_NLO.add(Dense(1, kernel_initializer='uniform',activation='linear'))
		n2n2_NLO.load_weights(K_weightsfile)
		return n2n2_LO, n2n2_NLO
#n2c1+
def build_n2c1p_AI(LO_gen_weightfile='./n2c1+/n2c1+_LO_gen.hdf5', LO_spec_weightfile='./n2c1+/n2c1+_LO_spec.hdf5', K_gen_weightsfile='./n2c1+/n2c1+_K_gen.hdf5', K_spec_weightsfile='./n2c1+/n2c1+_K_spec.hdf5', LO=0, NLO=1):
	n2c1p_LO_gen = Sequential()
	for i in range(0,8):
		n2c1p_LO_gen.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.1), input_dim=11))
		n2c1p_LO_gen.add(Activation('selu'))
	n2c1p_LO_gen.add(Dense(1, kernel_initializer='uniform',activation='linear'))
	n2c1p_LO_gen.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
	n2c1p_LO_gen.load_weights(LO_gen_weightfile)
	n2c1p_LO_spec = Sequential()
	for i in range(0,8):
		n2c1p_LO_spec.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.1), input_dim=11))
		n2c1p_LO_spec.add(Activation('selu'))
	n2c1p_LO_spec.add(Dense(1, kernel_initializer='uniform',activation='linear'))
	n2c1p_LO_spec.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
	n2c1p_LO_spec.load_weights(LO_spec_weightfile)
	if LO == 1 and NLO == 0:
		return n2c1p_LO_gen, n2c1p_LO_spec
	if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
		n2c1p_NLO_gen = Sequential()
		for i in range(0,8):
			n2c1p_NLO_gen.add(Dense(32, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.176177), input_dim=12))
			n2c1p_NLO_gen.add(Activation('selu'))
		n2c1p_NLO_gen.add(Dense(1, kernel_initializer='uniform',activation='linear'))
		n2c1p_NLO_gen.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
		n2c1p_NLO_gen.load_weights(K_gen_weightsfile)
		n2c1p_NLO_spec = Sequential()
		for i in range(0,8):
			n2c1p_NLO_spec.add(Dense(32, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.176177), input_dim=12))
			n2c1p_NLO_spec.add(Activation('selu'))
		n2c1p_NLO_spec.add(Dense(1, kernel_initializer='uniform',activation='linear'))
		n2c1p_NLO_spec.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
		n2c1p_NLO_spec.load_weights(K_spec_weightsfile)
		return n2c1p_LO_gen, n2c1p_LO_spec, n2c1p_NLO_gen, n2c1p_NLO_spec
#n2c1-
def build_n2c1m_AI(LO_gen_weightfile='./n2c1-/n2c1-_LO_gen.hdf5',LO_spec_weightfile='./n2c1-/n2c1-_LO_spec.hdf5', K_weightsfile='./n2c1-/n2c1-_K.hdf5', LO=0, NLO=1):
	n2c1m_LO_gen = Sequential()
	for i in range(0,8):
		n2c1m_LO_gen.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.1), input_dim=11))
		n2c1m_LO_gen.add(Activation('selu'))
	n2c1m_LO_gen.add(Dense(1, kernel_initializer='uniform',activation='linear'))
	n2c1m_LO_gen.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
	n2c1m_LO_gen.load_weights(LO_gen_weightfile)
	n2c1m_LO_spec = Sequential()
	for i in range(0,8):
		n2c1m_LO_spec.add(Dense(100, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.1), input_dim=11))
		n2c1m_LO_spec.add(Activation('selu'))
	n2c1m_LO_spec.add(Dense(1, kernel_initializer='uniform',activation='linear'))
	n2c1m_LO_spec.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
	n2c1m_LO_spec.load_weights(LO_spec_weightfile)
	if LO == 1 and NLO == 0:
		return n2c1m_LO_gen,n2c1m_LO_spec 
	if (LO == 1 and NLO == 1) or (LO == 0 and NLO == 1):
		n2c1m_NLO = Sequential()
		for i in range(0,8):
			n2c1m_NLO.add(Dense(32, kernel_initializer=initializers.random_normal(mean=0.0, stddev=0.176177), input_dim=12))
			n2c1m_NLO.add(Activation('selu'))
		n2c1m_NLO.add(Dense(1, kernel_initializer='uniform',activation='linear'))
		n2c1m_NLO.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0), loss='mape')
		n2c1m_NLO.load_weights(K_weightsfile)
		return n2c1m_LO_gen, n2c1m_LO_spec, n2c1m_NLO
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
				K_pred=NLO_model_spec.predict(NLO_data)
				K_pred=np.float(K_pred*4)
				K_pred = K_pred.astype(np.float)
				NLO_pred=LO_pred*K_pred
			else:
				K_pred=NLO_model_gen.predict(NLO_data)
				K_pred = K_pred*4
				K_pred = K_pred.astype(np.float)
				NLO_pred=LO_pred*K_pred
			print("With a K-factor of: " + str(K_pred) + " \n")
			print("Therefore, at next-to-leading order: " + str(NLO_pred) + " pb\n")
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

""" MAIN """
#read input
input_data = read_input(fn, Array=Array)
#pair settings
c1c1_occurences = [0,1,5,6,7,11,12,13]
n2n2_occurences = [0,2,5,8,9,11,12,14]
n2c1p_occurences = [0,3,6,8,10,11,13,14]
n2c1m_occurences = [0,4,7,9,10,12,13,14]

if int(args.stream) == 0:
	#c1c1
	for i in c1c1_occurences:
		if i == setting:
			c1c1_LO_data, c1c1_NLO_data = preprocess_c1c1(input_data, LO=1, NLO=1, Array=Array)
			c1c1_LO_model, c1c1_NLO_model = build_c1c1_AI(LO_weightfile='./c1+c1-/c1c1_LO.hdf5', K_weightsfile='./c1+c1-/c1c1_K.hdf5', LO=1, NLO=1)
			c1c1_LO, c1c1_K, c1c1_NLO = predict_c1c1(LO_data=c1c1_LO_data,NLO_data=c1c1_NLO_data,LO_model=c1c1_LO_model,NLO_model=c1c1_NLO_model,LO=1,NLO=1)
	#n2n2
	for i in n2n2_occurences:
		if i == setting:
			n2n2_LO_data, n2n2_NLO_data = preprocess_n2n2(input_data, LO=1, NLO=1, Array=Array)
			n2n2_LO_model, n2n2_NLO_model = build_n2n2_AI(LO_weightfile='./n2n2/n2n2_LO.hdf5', K_weightsfile='./n2n2/n2n2_K.hdf5', LO=0, NLO=1)
			n2n2_LO, n2n2_K, n2n2_NLO = predict_n2n2(LO_data=n2n2_LO_data,NLO_data=n2n2_NLO_data,LO_model=n2n2_LO_model,NLO_model=n2n2_NLO_model,LO=0,NLO=1)
	#n2c1p
	for i in n2c1p_occurences:
		if i == setting:
			n2c1p_LO_data, n2c1p_NLO_data = preprocess_n2c1p(input_data, LO=1, NLO=1, Array=Array)
			n2c1p_LO_model_general, n2c1p_LO_model_specialised, n2c1p_NLO_model_general, n2c1p_NLO_model_specialised = build_n2c1p_AI(LO_gen_weightfile='./n2c1+/n2c1+_LO_gen.hdf5', LO_spec_weightfile='./n2c1+/n2c1+_LO_spec.hdf5', K_gen_weightsfile='./n2c1+/n2c1+_K_gen.hdf5', K_spec_weightsfile='./n2c1+/n2c1+_K_spec.hdf5', LO=1, NLO=1)
			n2c1p_LO, n2c1p_K, n2c1p_NLO = predict_n2c1p(LO_data=n2c1p_LO_data,NLO_data=n2c1p_NLO_data,LO_model_gen=n2c1p_LO_model_general, LO_model_spec=n2c1p_LO_model_specialised, NLO_model_gen=n2c1p_NLO_model_general, NLO_model_spec = n2c1p_NLO_model_specialised, LO=1,NLO=1)
	#n2c1m
	for i in n2c1m_occurences:
		if i == setting:
			n2c1m_LO_data, n2c1m_NLO_data = preprocess_n2c1m(input_data, LO=1, NLO=1, Array=Array)
			n2c1m_LO_model_gen, n2c1m_LO_model_spec, n2c1m_NLO_model = build_n2c1m_AI(LO_gen_weightfile='./n2c1-/n2c1-_LO_gen.hdf5', LO_spec_weightfile='./n2c1-/n2c1-_LO_spec.hdf5', K_weightsfile='./n2c1-/n2c1-_K.hdf5', LO=1, NLO=1)
			n2c1m_LO, n2c1m_K, n2c1m_NLO = predict_n2c1m(LO_data=n2c1m_LO_data,NLO_data=n2c1m_NLO_data,LO_model_gen=n2c1m_LO_model_gen,LO_model_spec=n2c1m_LO_model_spec,NLO_model=n2c1m_NLO_model,LO=1,NLO=1)
			
if int(args.stream) == 1:
	while active == True:
		#c1c1
		for i in c1c1_occurences:
			if i == setting:
				c1c1_LO_data, c1c1_NLO_data = preprocess_c1c1(input_data, LO=1, NLO=1, Array=Array)
				c1c1_LO_model, c1c1_NLO_model = build_c1c1_AI(LO_weightfile='./c1+c1-/c1c1_LO.hdf5', K_weightsfile='./c1+c1-/c1c1_K.hdf5', LO=1, NLO=1)
				c1c1_LO, c1c1_K, c1c1_NLO = predict_c1c1(LO_data=c1c1_LO_data,NLO_data=c1c1_NLO_data,LO_model=c1c1_LO_model,NLO_model=c1c1_NLO_model,LO=1,NLO=1)
		#n2n2
		for i in n2n2_occurences:
			if i == setting:
				n2n2_LO_data, n2n2_NLO_data = preprocess_n2n2(input_data, LO=1, NLO=1, Array=Array)
				n2n2_LO_model, n2n2_NLO_model = build_n2n2_AI(LO_weightfile='./n2n2/n2n2_LO.hdf5', K_weightsfile='./n2n2/n2n2_K.hdf5', LO=0, NLO=1)
				n2n2_LO, n2n2_K, n2n2_NLO = predict_n2n2(LO_data=n2n2_LO_data,NLO_data=n2n2_NLO_data,LO_model=n2n2_LO_model,NLO_model=n2n2_NLO_model,LO=0,NLO=1)
		#n2c1p
		for i in n2c1p_occurences:
			if i == setting:
				n2c1p_LO_data, n2c1p_NLO_data = preprocess_n2c1p(input_data, LO=1, NLO=1, Array=Array)
				n2c1p_LO_model_general, n2c1p_LO_model_specialised, n2c1p_NLO_model_general, n2c1p_NLO_model_specialised = build_n2c1p_AI(LO_gen_weightfile='./n2c1+/n2c1+_LO_gen.hdf5', LO_spec_weightfile='./n2c1+/n2c1+_LO_spec.hdf5', K_gen_weightsfile='./n2c1+/n2c1+_K_gen.hdf5', K_spec_weightsfile='./n2c1+/n2c1+_K_spec.hdf5', LO=1, NLO=1)
				n2c1p_LO, n2c1p_K, n2c1p_NLO = predict_n2c1p(LO_data=n2c1p_LO_data,NLO_data=n2c1p_NLO_data,LO_model_gen=n2c1p_LO_model_general, LO_model_spec=n2c1p_LO_model_specialised, NLO_model_gen=n2c1p_NLO_model_general, NLO_model_spec = n2c1p_NLO_model_specialised, LO=1,NLO=1)
		#n2c1m
		for i in n2c1m_occurences:
			if i == setting:
				n2c1m_LO_data, n2c1m_NLO_data = preprocess_n2c1m(input_data, LO=1, NLO=1, Array=Array)
				n2c1m_LO_model_gen, n2c1m_LO_model_spec, n2c1m_NLO_model = build_n2c1m_AI(LO_gen_weightfile='./n2c1-/n2c1-_LO_gen.hdf5', LO_spec_weightfile='./n2c1-/n2c1-_LO_spec.hdf5', K_weightsfile='./n2c1-/n2c1-_K.hdf5', LO=1, NLO=1)
				n2c1m_LO, n2c1m_K, n2c1m_NLO = predict_n2c1m(LO_data=n2c1m_LO_data,NLO_data=n2c1m_NLO_data,LO_model_gen=n2c1m_LO_model_gen,LO_model_spec=n2c1m_LO_model_spec,NLO_model=n2c1m_NLO_model,LO=1,NLO=1)
		#request new file
		new_SLHA_file = input("Type 'exit' to exit.\n Otherwise, specify the file to process next: ")
		if new_SLHA_file == 'exit':
			active = False
		else:
			input_data = read_input(new_SLHA_file, Array=0)
K.clear_session()