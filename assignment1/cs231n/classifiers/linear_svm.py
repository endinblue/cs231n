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
      that X[i] has label ctogk9128
      , where 0 <= c < C.
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
        # y=w*x
        correct_class_score = scores[y[i]]
        #target class score. 
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
           
            if margin > 0:
                loss += margin
                #dW(margin) = dW ((X[i].dot(W))[j]) - dW((X[i].dot(W))[y[i]])
                dW[:,j] += X[i]
                dW[:,y[i]] -=X[i]
                
                
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 0.5*reg * np.sum(W*W)
    dW += reg*W
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    #2D there are no diffrence between matmul and dot. and ndarray(numpy) does not have matmul
    scores = X.dot(W)
    num_train = X.shape[0]
    
  
    correct_class_scores = scores[np.arange(num_train),y].reshape(num_train,1)
    lossfunction = np.maximum(0,scores - correct_class_scores +1)
    
    # correct scores set to 0
    lossfunction[np.arange(num_train),y] = 0 
    
    loss = lossfunction.sum() / num_train
    loss += reg*np.sum(W*W)
   
    
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    
    #D = maximum (0, scores - correct_class_scores +1)
    
    
    dLdD = np.zeros(lossfunction.shape)
    #if lossfunction > 0  dLdD = dscores(dscores) =1
    dLdD[lossfunction>0] = 1
    
    #d(correct_class_socres)/d(scores) =>dLdD[range(num_train),y] - 1
    valid = dLdD.sum(axis=1)
    dLdD[range(num_train),y] -=valid
   

    #dLdW = dLdD.dot(X.T)
    dW = (X.T).dot(dLdD)
    dW /= num_train
    dW += reg*2*W
    
    
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

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
