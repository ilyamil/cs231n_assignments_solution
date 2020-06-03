from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    for i in range(num_train):
        scores = X[i,:].dot(W)
        scores -= np.max(scores)
        prob = np.exp(scores)/np.sum(np.exp(scores))
        loss += -np.log(prob[y[i]])
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] += (prob[j] - 1) * X[i,:]
            else:
                dW[:,j] += prob[j] * X[i,:]
    
    # Compute average error and grad
    loss /= num_train
    dW /= num_train
    
    # Add regularization term
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    # get a matrix of scores, where i'th row represents scores for each of C 
    # classes for i'th training example
    scores = np.matmul(X, W)
    
    # prevent numerical instability and expinentiate scores to get unnormalized
    # log scores
    scores = np.exp(scores - np.max(scores))
    
    # recalculate scores for correct label for i'th training example 
    # then exponentiate and normalize them to get "probabilities"
    prob_matrix = scores / np.sum(scores, axis = 1)[:, None]
    # probabilities for true labels
    correct_prob = prob_matrix[np.arange(num_train), y]
   
    # compute average loss and add regularization
    loss = np.sum(-np.log(correct_prob))
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    # compute average gradient and add regularization
    dW = np.matmul(X.T, correct_prob - 1)
    dW /= num_train
    dW += 2 * reg * W
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
