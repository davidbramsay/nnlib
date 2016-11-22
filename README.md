# nnlib
Coding Neural Nets in C from scratch-- for learning and for embedded implementation

This is my repo for Neural Net implementations in C.  To start, I've built out a 2D matrix library (matrix.h for function descriptions and structs) as well as a basic perceptron/feed-forward nueral network in nn.h.

The matrix library implements many basic matrix manipulations, including Gaussian elimination.

The neural network implementation initializes a fully connected feed-forward network of any number and size of layers.  It initializes the network, reads/parses text files for training data, implements backprop, and does recall.  Parameters include learning_rate, a cooling_rate (which reduces the learning_rate in between batches), and batch/mini-batch processing.  Training can be specified by maximum error per epoch, or max # of epochs.

Next steps include preprocessing and binary data file support, as well as basic implementations of RNNs with backprop through time, LSTMs, and Convolutional NNs.  I'd also like to do a word-embedding word2vec style tool for lower-dimensional representations for word-embeddings.   
