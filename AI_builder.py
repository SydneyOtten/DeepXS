from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras import initializers
from keras import backend as K

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