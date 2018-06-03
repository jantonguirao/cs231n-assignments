import numpy as np
from random import shuffle

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
  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]
  for i in range(N):
    X_i = X[i,:]
    assert X_i.shape == (D,)
    f_i = X_i.dot( W )
    assert f_i.shape == (C,)
    # Normalization trick to avoid numeric instability: See http://cs231n.github.io/linear-classify/#softmax
    logC = -np.max( f_i )
    f_i += logC
    exp_f_i = np.exp(f_i)
    assert exp_f_i.shape == (C,)
    sum_exp_f_i = np.sum(exp_f_i)
    loss += -f_i[ y[i] ] + np.log( sum_exp_f_i )

    for j in range(C):
      dW[:,j] += ( exp_f_i[ j ] / sum_exp_f_i ) * X_i
      if j == y[i]:
        dW[:,j] += -X_i

  loss /= N
  loss += reg * np.sum( W*W )
  dW /= N
  dW += 2 * reg * W

  # Note Look at this to understand the derivative:
  # https://stats.stackexchange.com/questions/265905/derivative-of-softmax-with-respect-to-weights
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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

  N = X.shape[0]
  D = X.shape[1]
  C = W.shape[1]

  f = X.dot(W)
  assert f.shape == (N,C)
  # Normalization trick to avoid numeric instability: See http://cs231n.github.io/linear-classify/#softmax
  logC = -np.max( f, axis=1, keepdims=True )
  f += logC
  assert f.shape == (N,C)

  exp_f = np.exp(f)
  assert exp_f.shape == (N,C)

  sum_exp_f = np.sum(exp_f, axis=1, keepdims=True)
  assert sum_exp_f.shape == (N,1)

  p = exp_f / sum_exp_f
  assert p.shape == (N,C)

  loss = np.sum( -np.log( p[ np.arange(N), y] ) )

  ind = np.zeros_like(p)
  ind[ np.arange(N), y ] = 1

  # (D,N) * (N,C) -> (D,C)
  dW += X.T.dot( p - ind )

  loss /= N
  loss += reg * np.sum( W*W )
  dW /= N
  dW += 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
