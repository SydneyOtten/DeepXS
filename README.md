# DeepXs
Deep Learning applied to Quantum Field Theory. An AI that predicts LHC pair production cross-sections for supersymmetric particles at next-to-leading order. 

Requirements:
Recent versions of Tensorflow, Keras, Pandas and pyslha.

Instructions:

Run from command line with "python main.py -pairs N -return LO/NLO/both -stream 0/1 -Array 0/1 -fn filename"

N must be specified as follows:

0 - all pairs
1 - c1c1
2 - n2n2
3 - n2c1+
4 - n2c1-
5 - 1&2
6 - 1&3
7 - 1&4
8 - 2&3
9 - 2&4
10 - 3&4
11 - 1,2&3
12 - 1,2&4
13 - 1,3&4
14 - 2,3&4

The default value of N is 0. The other default values are: -return both -stream 1 -Array 0.

Streaming allows to continuously feed SLHA files to the AI without having to reload the backend each time. 

Using -Array 1 allows to feed .txt files with the following format to the AI to predict many points at once:

