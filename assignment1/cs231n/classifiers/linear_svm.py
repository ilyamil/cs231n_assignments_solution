from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        counter = 0
        for j in range(num_classes):
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if j == y[i]:
                continue    
            if margin > 0:
                loss += margin
                dW[:,j] += X[i,:]
                counter += 1
        dW[:,y[i]] += -counter * X[i,:]
            
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    # get a matrix of scores, where i'th row represents scores for each of C 
    # classes for i'th training example
    scores = np.matmul(X, W)
    
    # get scores for correct label for i'th training example
    correct_scores = scores[np.arange(num_train), y]
    
    # expand an array of correct scores of shape (N,) into shape (N, C), where
    # columns in a given row has same values (for example, [2.5,2.5,2.,5,..])
    correct_scores = np.tile(correct_scores, (num_classes, 1))
    
    # compute margins for the whole matrix 
    margin = scores - correct_scores.T + np.ones(scores.shape)
    # we have not to take into account margins for correct classes
    margin[np.arange(num_train), y] = 0
    
    # determine positive margins
    pos_margin = (margin > 0) * 1
    
    # loss is a sum over positive margins devided by the total num of examples
    loss = np.sum((pos_margin * margin)) / num_train + reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # count number of positive margins in each row
    counter = np.sum(pos_margin, axis = 1)
    pos_margin[np.arange(num_train), y] -= counter
    dW = (X.T).dot(pos_margin) / num_train + 2 * reg * W
                  

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return loss, dW