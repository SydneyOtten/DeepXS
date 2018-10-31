import pyslha
import numpy as np
import pandas as pd
#read input

def read_input(filename, array = 0, fn = 'sample_1.spc'):
	if array == 0:
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
	elif array == 1:
		raw_input = pd.read_csv(fn, sep=' ', header=None)
		input = raw_input.values
		input = input.astype(np.float)
	else:
		exit()
	return input