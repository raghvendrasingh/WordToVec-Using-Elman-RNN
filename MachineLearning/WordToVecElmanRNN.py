import os
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

theano.config.exception_verbosity = 'high'


class WordToVecElmanRNN(object):
    # elman neural net model to learn word-vector representations.

    def __init__(self, context_window_size, word_vector_dimension, num_classes, hidden_layer_dimension, vocab_size):
        """
          contextWindowSize - Size of the context window
          wordVectorDimension - Size of word vector
          inputHiddenWeight - weight matrix between input and hidden layer
          hiddenHiddenWeight - weight matrix between hidden layer of previoys timestep  and hidden layer of current time step
          hiddenOutputWeight - weight matrix between hidden and output layer
          hiddenBias - bias vector for the hidden layer
          outputBias - bias vector for the output layer
          initialContextVector - initial context vector
          numClasses - number of classes
          hiddenLayerDimension - dimension of the hidden layer
          vocab_size - size of the vocab
        """

        # rnn parameters
        self.embeddings = theano.shared(
            0.2 * np.random.uniform(-1.0, 1.0, (vocab_size + 1, word_vector_dimension)).astype(theano.config.floatX))
        self.inputHiddenWeight = theano.shared(name='inputHiddenWeight', value=0.2 * np.random.uniform(-1.0, 1.0, (
            context_window_size * word_vector_dimension, hidden_layer_dimension)).astype(theano.config.floatX))
        self.hiddenHiddenWeight = theano.shared(name='hiddenHiddenWeight', value=0.2 * np.random.uniform(-1.0, 1.0, (
            hidden_layer_dimension, hidden_layer_dimension)).astype(theano.config.floatX))
        self.hiddenOutputWeight = theano.shared(name='hiddenOutputWeight', value=0.2 * np.random.uniform(-1.0, 1.0, (
            hidden_layer_dimension, num_classes)).astype(theano.config.floatX))
        self.hiddenBias = theano.shared(name='hiddenBias',
                                        value=np.zeros(hidden_layer_dimension, dtype=theano.config.floatX))
        self.outputBias = theano.shared(name='outputBias', value=np.zeros(num_classes, dtype=theano.config.floatX))
        self.h_previous = theano.shared(name='h_previous', value=np.zeros(hidden_layer_dimension, dtype=theano.config.floatX))

        self.params = [self.embeddings, self.inputHiddenWeight, self.hiddenHiddenWeight, self.hiddenOutputWeight,
                       self.hiddenBias, self.outputBias, self.h_previous]
        self.names = ['embeddings', 'Wx', 'Wh', 'W', 'bh', 'b', 'h0']


def load_data(training_file):
    vocabulary = {}
    word_index = {}
    index = 0

    with open(training_file) as f:
        for sentence in f:
            words = sentence.split(',')
            for word in words:
                if word in vocabulary:
                    vocabulary[word] += 1
                else:
                    vocabulary[word] = 1
                    word_index[word] = index
                    index += 1
    f.close()
    return vocabulary, word_index


def reduce_vocab_and_word_index(vocab, word_index, threshold_frequency):
    # remove all the words from the vocab and word_index whose frequency is less than threshold_frequency
    new_vocab = dict(vocab)
    new_word_index = dict(word_index)
    for key, value in vocab.iteritems():
        if value < threshold_frequency:
            del new_vocab[key]
            del new_word_index[key]
    return new_vocab, new_word_index


def context_window(sentence, window_size):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence

    l :: array containing the word indexes

    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (window_size % 2) == 1
    assert window_size >= 1
    l = list(sentence)

    lpadded = window_size // 2 * [-1] + l + window_size // 2 * [-1]
    out = [lpadded[i:(i + window_size)] for i in range(len(l))]

    assert len(out) == len(l)
    return out


def return_index_array_for_word_present_in_word_index(words, word_index):
    new_words = []
    for word in words:
        if word in word_index:
            new_words.append(word_index[word])
    return new_words


def train(threshold_frequency=5, context_window_size=5, word_vector_dimension=50, hidden_layer_dimension=100, max_iter=10):

    training_file = 'cleaned_training_data1.txt'
    vocab, word_index = load_data(training_file)
    vocab, word_index = reduce_vocab_and_word_index(vocab, word_index, threshold_frequency)
    print "vocab = ", vocab
    print "word_index = ", word_index
    num_classes = len(vocab)
    vocab_size = len(vocab)
    word_to_vec = WordToVecElmanRNN(context_window_size, word_vector_dimension, num_classes, hidden_layer_dimension, vocab_size)

    ind_vec = T.ivector()
    # alp = theano.function([ind_vec],word_to_vec.embeddings[ind_vec])
    inp = word_to_vec.embeddings[ind_vec].reshape((1, context_window_size * word_vector_dimension))[0]
    # input_for_context = theano.function([ind_vec], inp)

    hidden_output = T.nnet.sigmoid(T.dot(inp, word_to_vec.inputHiddenWeight) + T.dot(word_to_vec.h_previous, word_to_vec.hiddenHiddenWeight) + word_to_vec.hiddenBias)
    softmax_output = T.nnet.softmax(T.dot(hidden_output, word_to_vec.hiddenOutputWeight) + word_to_vec.outputBias)

    p_y_given_x_context = softmax_output[0, :]

    # softmax_out = theano.function(inputs=[h_previous, inp], outputs=softmax_output)
    # y_pred = T.argmax(p_y_given_x_context, axis=1)

    lr = T.dscalar('lr')
    y = T.iscalar('y')  # target, presented as scalar
    nll_cost = -T.log(p_y_given_x_context)[y]  # the cost we minimize during training is the negative log likelihood
                                                  # of the model in symbolic format
    # cost = theano.function([h_previous,inp,y],context_nll)

    param_gradients = T.grad(nll_cost, word_to_vec.params)
    # gradients = theano.function([h_previous,inp,y,ind_vec],param_gradients)

    param_updates = OrderedDict((p, p - lr * g) for p, g in zip(word_to_vec.params, param_gradients))

    # compiling a Theano function `train_model` that returns the cost and hidden output, but at
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[y, lr, ind_vec],
        outputs=[nll_cost, hidden_output],
        updates=param_updates
        )

    emb = theano.function([], word_to_vec.embeddings)

    lr = 0.1
    curr_iter = 0
    folder = "LearnedParams"
    while curr_iter < max_iter:
        print "iteration = ", curr_iter + 1
        num_lines = 0
        k = 1
        with open(training_file) as f:
            for sentence in f:
                num_lines += 1
                if num_lines == k*10000:
                    print "Lines Processed = ", num_lines
                    k += 1
                words = sentence.split(',')
                index_arr = return_index_array_for_word_present_in_word_index(words, word_index)
                contexts = context_window(index_arr, context_window_size)
                word_to_vec.h_previous = theano.shared(name='h_previous', value=np.zeros(hidden_layer_dimension, dtype=theano.config.floatX))
                for i in range(len(contexts)):
                    print contexts[i]
                    target = contexts[i][context_window_size/2]
                    # update network for each context.
                    cost, word_to_vec.h_previous = train_model(target, lr, contexts[i])
                    # print "context = ", contexts[i]
                    # print "emdeddings  = ", emb()
                    #print "context_vec = ", word_to_vec.h_previous

        f.close()
        # save all the model parameters to separate files in
        # /home/raghvendra.singh/PycharmProjects/WordToVecRNN/MachineLearning/LearnedParams
        save(word_to_vec, folder)
        curr_iter += 1


def save(self, folder):
    for param, name in zip(self.params, self.names):
        np.save(os.path.join(folder, name + '.npy'), param.get_value())


if __name__ == '__main__':
    train()

