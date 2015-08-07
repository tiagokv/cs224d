from numpy import *
from nn.base import NNBase
from nn.math import make_onehot, sigmoid
from misc import random_weight_matrix, sigmoid_grad, softmax
##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))

##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed

        |V| = Size of vocabulary
        n   = length of our word vectors
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension

        print "input size:  %d" % dims[0]
        print "hidden size: %d" % dims[1]
        print "output size: %d" % dims[2]

        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!
        #### YOUR CODE HERE ####

        # any other initialization you need

        self.sparams.wv = wv.copy()
        self.params.W  = random_weight_matrix(param_dims["W"][0],param_dims["W"][1])
        self.params.U  = random_weight_matrix(param_dims["U"][0],param_dims["U"][1])

        #done automatically
        #self.params.b1 = zeros(param_dims["b1"])
        #self.params.b2 = zeros(param_dims["b2"])
        
        #print "W  shape: %s" % (self.params.W.shape,)
        #print "b1 shape: %s" % (self.params.b1.shape,)
        #print "U  shape: %s" % (self.params.U.shape,)
        #print "b2 shape: %s" % (self.params.b2.shape,)

        #### END YOUR CODE ####

    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        """
        #### YOUR CODE HERE ####

        onehot_vecs = expand_dims(self.sparams.L[window,:].flatten(),axis=0)

        #print "onehot_vecs.shape: %s " % (onehot_vecs.shape,)

        ##
        # Forward propagation
        a1 = self.params.W.dot(onehot_vecs.T).T + self.params.b1
        s  = sigmoid( 2.0 * a1 )
        h  = 2.0 * s - 1.0
        a2 = self.params.U.dot(h.T).T + self.params.b2
        y_hat = softmax( a2 ) 

        ##
        # Backpropagation
        t = zeros( y_hat.shape )
        t[:,label] = 1

        delta_out = y_hat - t

        self.grads.U  += h.T.dot(delta_out).T + self.lreg * self.params.U

        #print "delta_out  shape: %s" % (delta_out.shape,)

        self.grads.b2 += delta_out.flatten()
        #print "self.grads.b2.shape: %s " % (self.grads.b2.shape,)

        delta_hidden = delta_out.dot(self.params.U) * 4.0 * sigmoid_grad( s )
        
        self.grads.W  += delta_hidden.T.dot(onehot_vecs) + self.lreg * self.params.W
        self.grads.b1 += delta_hidden.flatten()

        #print "self.grads.b2.shape: %s " % (self.grads.b1.shape,)

        grad_xs = delta_hidden.dot(self.params.W).T
        #print "grad_xs.shape: %s " % (grad_xs.shape,)

        self.sgrads.L[window[0]] = grad_xs[range(0,50)].flatten()
        self.sgrads.L[window[1]] = grad_xs[range(50,100)].flatten()
        self.sgrads.L[window[2]] = grad_xs[range(100,150)].flatten()

        #### END YOUR CODE ####


    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####
        onehot_vecs = asarray( [ self.sparams.L[windows[i],:].flatten() for i in range(len(windows)) ] )

        a1 = self.params.W.dot(onehot_vecs.T).T + self.params.b1
        h  = tanh( a1 )
        a2 = self.params.U.dot(h.T).T + self.params.b2
        P  = softmax( a2 ) #y_hat

        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####
        proba = self.predict_proba(windows)
        c = argmax(proba,axis=1)

        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####
        """ 
        windows = array (n x windowsize), each row is a window of indices
        """

        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        if not hasattr(labels, "__iter__"):
            labels = [labels]

        proba = self.predict_proba(windows)

        J = sum( - log( proba[range(len(labels)),labels] ) )

        J += (self.lreg / 2.0) *  ( sum( self.params.W ** 2 ) + sum( self.params.U ** 2 ) )

        #### END YOUR CODE ####
        return J