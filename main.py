import argparse
import os, sys
from preprocessing import preprocess_c1c1, preprocess_n2n2, preprocess_n2c1p, preprocess_n2c1m
from AI_builder import build_c1c1_AI, build_n2n2_AI, build_n2c1p_AI, build_n2c1m_AI
from predictors import predict_c1c1, predict_n2n2, predict_n2c1p, predict_n2c1m
from reader import read_input
from keras import backend as K
import numpy as np

"""Essential Functions"""
#argparse for indicating the pairs to predict

parser = argparse.ArgumentParser(description='DeepXS - predicting cross-sections since 2018')
parser.add_argument('-pairs', dest='integers', type=int, default=0,
                    help='Defines the pair(s). Check readme.txt for the specification.')
parser.add_argument('-return', dest='order', action='store', default='both',
                    help='Specifies if you want the cross-sections at leading (LO) or next-to-leading order (NLO). If you want both, specify -return both')
parser.add_argument('-stream', dest='stream', action='store', default=1, help='Specifies if you want the tensorflow session to continue. 0: no streaming. 1: You will then have the possibility to continuously feed input data as DeepXS waits for new input. 2: DeepXS will predict the cross-sections for every model in slha_dump.')
parser.add_argument('-array', dest='array', action='store', default=0, help='Declares if you want to give the input as an array (1) from a .txt file or not (0) and use an SLHA-file instead.')
parser.add_argument('-fn', dest='fn', action='store', type=str, help='Specifies the .txt file that DeepXS reads to predict the corresponding cross-section(s). The required layout is described in the readme.txt.')
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
	print('Please specify either if you want me to return the LO, NLO or both cross-sections. You can do this by running the script via "python DeepXS.py PAIR_NUMBER -return LO/NLO/both". E.g. if you want to predict all pairs at NLO, do "python DeepXS.py 0 -return both".')
	exit()
if int(args.array) == 0:
	print('Reading input from SLHA file')
	array=0
elif int(args.array) == 1:
	print('Reading input from an array')
	array=1
else:
	print('Please either specify -array 0 or -array 1.')
	exit()
if int(args.stream) == 0:
	active = False
elif int(args.stream) == 1:
	print('Opening a continuous portal to our AI for continuous streaming of input and output.')
	active = True
elif int(args.stream) == 2:
	print('Opening a continuous portal to our AI for continuous streaming of input and output.')
else:
	print('Please specify either -stream 0, -stream 1 or -stream 2.')
	exit()
fn = str(args.fn)
print('\n Welcome to DeepXS version Alpha 2\n')

""" MAIN """
#read input
input_data = read_input(fn, array=array)
#pair settings
c1c1_occurences = [0,1,5,6,7,11,12,13]
n2n2_occurences = [0,2,5,8,9,11,12,14]
n2c1p_occurences = [0,3,6,8,10,11,13,14]
n2c1m_occurences = [0,4,7,9,10,12,13,14]

if int(args.stream) == 0:
	#c1c1
	for i in c1c1_occurences:
		if i == setting:
			c1c1_LO_data, c1c1_NLO_data = preprocess_c1c1(input_data, LO=1, NLO=1, array=array)
			c1c1_LO_model, c1c1_NLO_model = build_c1c1_AI(LO_weightfile='./c1c1/c1c1_LO.hdf5', K_weightsfile='./c1c1/c1c1_K.hdf5', LO=1, NLO=1)
			c1c1_LO, c1c1_K, c1c1_NLO = predict_c1c1(LO_data=c1c1_LO_data,NLO_data=c1c1_NLO_data,LO_model=c1c1_LO_model,NLO_model=c1c1_NLO_model,LO=1,NLO=1)
	#n2n2
	for i in n2n2_occurences:
		if i == setting:
			n2n2_LO_data, n2n2_NLO_data = preprocess_n2n2(input_data, LO=1, NLO=1, array=array)
			n2n2_LO_model, n2n2_NLO_model = build_n2n2_AI(LO_weightfile='./n2n2/n2n2_LO.hdf5', K_weightsfile='./n2n2/n2n2_K.hdf5', LO=0, NLO=1)
			n2n2_LO, n2n2_K, n2n2_NLO = predict_n2n2(LO_data=n2n2_LO_data,NLO_data=n2n2_NLO_data,LO_model=n2n2_LO_model,NLO_model=n2n2_NLO_model,LO=0,NLO=1)
	#n2c1p
	for i in n2c1p_occurences:
		if i == setting:
			n2c1p_LO_data, n2c1p_NLO_data = preprocess_n2c1p(input_data, LO=1, NLO=1, array=array)
			n2c1p_LO_model_general, n2c1p_LO_model_specialised, n2c1p_NLO_model_general, n2c1p_NLO_model_specialised = build_n2c1p_AI(LO_gen_weightfile='./n2c1p/n2c1+_LO_gen.hdf5', LO_spec_weightfile='./n2c1p/n2c1+_LO_spec.hdf5', K_gen_weightsfile='./n2c1p/n2c1+_K_gen.hdf5', K_spec_weightsfile='./n2c1p/n2c1+_K_spec.hdf5', LO=1, NLO=1)
			n2c1p_LO, n2c1p_K, n2c1p_NLO = predict_n2c1p(LO_data=n2c1p_LO_data,NLO_data=n2c1p_NLO_data,LO_model_gen=n2c1p_LO_model_general, LO_model_spec=n2c1p_LO_model_specialised, NLO_model_gen=n2c1p_NLO_model_general, NLO_model_spec = n2c1p_NLO_model_specialised, LO=1,NLO=1)
	#n2c1m
	for i in n2c1m_occurences:
		if i == setting:
			n2c1m_LO_data, n2c1m_NLO_data = preprocess_n2c1m(input_data, LO=1, NLO=1, array=array)
			n2c1m_LO_model_gen, n2c1m_LO_model_spec, n2c1m_NLO_model = build_n2c1m_AI(LO_gen_weightfile='./n2c1m/n2c1-_LO_gen.hdf5', LO_spec_weightfile='./n2c1m/n2c1-_LO_spec.hdf5', K_weightsfile='./n2c1m/n2c1-_K.hdf5', LO=1, NLO=1)
			n2c1m_LO, n2c1m_K, n2c1m_NLO = predict_n2c1m(LO_data=n2c1m_LO_data,NLO_data=n2c1m_NLO_data,LO_model_gen=n2c1m_LO_model_gen,LO_model_spec=n2c1m_LO_model_spec,NLO_model=n2c1m_NLO_model,LO=1,NLO=1)
			
if int(args.stream) == 1:
	while active == True:
		#c1c1
		for i in c1c1_occurences:
			if i == setting:
				c1c1_LO_data, c1c1_NLO_data = preprocess_c1c1(input_data, LO=1, NLO=1, array=array)
				c1c1_LO_model, c1c1_NLO_model = build_c1c1_AI(LO_weightfile='./c1c1/c1c1_LO.hdf5', K_weightsfile='./c1c1/c1c1_K.hdf5', LO=1, NLO=1)
				c1c1_LO, c1c1_K, c1c1_NLO = predict_c1c1(LO_data=c1c1_LO_data,NLO_data=c1c1_NLO_data,LO_model=c1c1_LO_model,NLO_model=c1c1_NLO_model,LO=1,NLO=1)
		#n2n2
		for i in n2n2_occurences:
			if i == setting:
				n2n2_LO_data, n2n2_NLO_data = preprocess_n2n2(input_data, LO=1, NLO=1, array=array)
				n2n2_LO_model, n2n2_NLO_model = build_n2n2_AI(LO_weightfile='./n2n2/n2n2_LO.hdf5', K_weightsfile='./n2n2/n2n2_K.hdf5', LO=0, NLO=1)
				n2n2_LO, n2n2_K, n2n2_NLO = predict_n2n2(LO_data=n2n2_LO_data,NLO_data=n2n2_NLO_data,LO_model=n2n2_LO_model,NLO_model=n2n2_NLO_model,LO=0,NLO=1)
		#n2c1p
		for i in n2c1p_occurences:
			if i == setting:
				n2c1p_LO_data, n2c1p_NLO_data = preprocess_n2c1p(input_data, LO=1, NLO=1, array=array)
				n2c1p_LO_model_general, n2c1p_LO_model_specialised, n2c1p_NLO_model_general, n2c1p_NLO_model_specialised = build_n2c1p_AI(LO_gen_weightfile='./n2c1p/n2c1+_LO_gen.hdf5', LO_spec_weightfile='./n2c1p/n2c1+_LO_spec.hdf5', K_gen_weightsfile='./n2c1p/n2c1+_K_gen.hdf5', K_spec_weightsfile='./n2c1p/n2c1+_K_spec.hdf5', LO=1, NLO=1)
				n2c1p_LO, n2c1p_K, n2c1p_NLO = predict_n2c1p(LO_data=n2c1p_LO_data,NLO_data=n2c1p_NLO_data,LO_model_gen=n2c1p_LO_model_general, LO_model_spec=n2c1p_LO_model_specialised, NLO_model_gen=n2c1p_NLO_model_general, NLO_model_spec = n2c1p_NLO_model_specialised, LO=1,NLO=1)
		#n2c1m
		for i in n2c1m_occurences:
			if i == setting:
				n2c1m_LO_data, n2c1m_NLO_data = preprocess_n2c1m(input_data, LO=1, NLO=1, array=array)
				n2c1m_LO_model_gen, n2c1m_LO_model_spec, n2c1m_NLO_model = build_n2c1m_AI(LO_gen_weightfile='./n2c1m/n2c1-_LO_gen.hdf5', LO_spec_weightfile='./n2c1m/n2c1-_LO_spec.hdf5', K_weightsfile='./n2c1m/n2c1-_K.hdf5', LO=1, NLO=1)
				n2c1m_LO, n2c1m_K, n2c1m_NLO = predict_n2c1m(LO_data=n2c1m_LO_data,NLO_data=n2c1m_NLO_data,LO_model_gen=n2c1m_LO_model_gen,LO_model_spec=n2c1m_LO_model_spec,NLO_model=n2c1m_NLO_model,LO=1,NLO=1)
		#request new file
		new_SLHA_file = input("Type 'exit' to exit.\n Otherwise, specify the file to process next: ")
		if new_SLHA_file == 'exit':
			active = False
		else:
			input_data = read_input(new_SLHA_file, array=0)
			
if int(args.stream) == 2:
	#in the next version convert list of SLHA inputs into array and then use NNs only once. The current way is unnecessarily slow.
	SLHA_file_list = []
	for x in os.listdir('./SLHA_dump'):
		z = './SLHA_dump/' + str(x)
		SLHA_file_list.append(z)
	print(SLHA_file_list)
	print(np.shape(SLHA_file_list))
	for i in range(0, len(SLHA_file_list)):
		input_data = read_input(SLHA_file_list[i], array=0)
		#c1c1
		for i in c1c1_occurences:
			if i == setting:
				c1c1_LO_data, c1c1_NLO_data = preprocess_c1c1(input_data, LO=1, NLO=1, array=array)
				c1c1_LO_model, c1c1_NLO_model = build_c1c1_AI(LO_weightfile='./c1c1/c1c1_LO.hdf5', K_weightsfile='./c1c1/c1c1_K.hdf5', LO=1, NLO=1)
				c1c1_LO, c1c1_K, c1c1_NLO = predict_c1c1(LO_data=c1c1_LO_data,NLO_data=c1c1_NLO_data,LO_model=c1c1_LO_model,NLO_model=c1c1_NLO_model,LO=1,NLO=1)
		#n2n2
		for i in n2n2_occurences:
			if i == setting:
				n2n2_LO_data, n2n2_NLO_data = preprocess_n2n2(input_data, LO=1, NLO=1, array=array)
				n2n2_LO_model, n2n2_NLO_model = build_n2n2_AI(LO_weightfile='./n2n2/n2n2_LO.hdf5', K_weightsfile='./n2n2/n2n2_K.hdf5', LO=0, NLO=1)
				n2n2_LO, n2n2_K, n2n2_NLO = predict_n2n2(LO_data=n2n2_LO_data,NLO_data=n2n2_NLO_data,LO_model=n2n2_LO_model,NLO_model=n2n2_NLO_model,LO=0,NLO=1)
		#n2c1p
		for i in n2c1p_occurences:
			if i == setting:
				n2c1p_LO_data, n2c1p_NLO_data = preprocess_n2c1p(input_data, LO=1, NLO=1, array=array)
				n2c1p_LO_model_general, n2c1p_LO_model_specialised, n2c1p_NLO_model_general, n2c1p_NLO_model_specialised = build_n2c1p_AI(LO_gen_weightfile='./n2c1p/n2c1+_LO_gen.hdf5', LO_spec_weightfile='./n2c1p/n2c1+_LO_spec.hdf5', K_gen_weightsfile='./n2c1p/n2c1+_K_gen.hdf5', K_spec_weightsfile='./n2c1p/n2c1+_K_spec.hdf5', LO=1, NLO=1)
				n2c1p_LO, n2c1p_K, n2c1p_NLO = predict_n2c1p(LO_data=n2c1p_LO_data,NLO_data=n2c1p_NLO_data,LO_model_gen=n2c1p_LO_model_general, LO_model_spec=n2c1p_LO_model_specialised, NLO_model_gen=n2c1p_NLO_model_general, NLO_model_spec = n2c1p_NLO_model_specialised, LO=1,NLO=1)
		#n2c1m
		for i in n2c1m_occurences:
			if i == setting:
				n2c1m_LO_data, n2c1m_NLO_data = preprocess_n2c1m(input_data, LO=1, NLO=1, array=array)
				n2c1m_LO_model_gen, n2c1m_LO_model_spec, n2c1m_NLO_model = build_n2c1m_AI(LO_gen_weightfile='./n2c1m/n2c1-_LO_gen.hdf5', LO_spec_weightfile='./n2c1m/n2c1-_LO_spec.hdf5', K_weightsfile='./n2c1m/n2c1-_K.hdf5', LO=1, NLO=1)
				n2c1m_LO, n2c1m_K, n2c1m_NLO = predict_n2c1m(LO_data=n2c1m_LO_data,NLO_data=n2c1m_NLO_data,LO_model_gen=n2c1m_LO_model_gen,LO_model_spec=n2c1m_LO_model_spec,NLO_model=n2c1m_NLO_model,LO=1,NLO=1)
K.clear_session()