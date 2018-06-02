import numpy as np
from random import shuffle

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

  # My notes: We want to calculate the gradient of the lost function
  #           with respect to W. The loss function looks like this:
  #           Li = sum_j ( max( 0, W_j * X_i - W_yi * X_i + 1 ) ) + alpha * sum( W_j*W_j )

  # If we call margin = (Wj-Xi - Wyi*Xi + 1)

  # grad_Li_Wyi = (1/N) * sum_j( -Xi if margin > 0; 0 otherwise )
  # grad_Li_Wj = (1/N) * sum_j( Xi if margin > 0; 0 otherwise ) + alpha * 2 * W_j

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i]
        dW[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
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
  N = X.shape[0]
  C = W.shape[1]
  scores = np.dot(X,W) # N,C
  k = np.arange(N)
  correct_scores = scores[k,y].reshape(N,1) # (N,1)
  margins = np.maximum( scores - correct_scores + 1, 0 ) # N,C
  # Excluding the margin on the correct class (the +1 would influence the loss otherwise)
  assert margins.shape == (N,C)
  margins[k,y] = 0
  loss = (1/N) * np.sum( margins )
  # Regularization term
  loss += reg * np.sum(W*W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  aux = np.zeros_like(margins)
  aux[ margins > 0 ] = 1
  aux[ k, y ] -= np.sum( aux, axis=1 )
  dW = (1/N) * np.dot(X.T, aux)
  dW += 2*reg*W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
