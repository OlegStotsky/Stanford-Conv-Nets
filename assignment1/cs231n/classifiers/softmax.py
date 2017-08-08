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
  num_examples = X.shape[0]
  num_dims = W.shape[1]
  num_classes = W.shape[1]

  D = W.shape[0]
  C = W.shape[1]
  N = X.shape[0]

  for i in range(num_examples):
    scores = X[i].dot(W)
    max_score = np.max(scores)
    scores -= max_score
    #Value of softmax doesn't change if we subtract the same number from all the coordinates
    #So we subtract the maximum score for the numerical stability
    #Calculate the loss
    loss += -scores[y[i]] + np.log(np.sum(np.exp(scores)))  
    #Calculate the updates
    for j in range(num_classes):
      dW[:, j] += X[i]*(-1*(j==y[i]) + np.exp(scores[j])/np.sum(np.exp(scores)) )

    # #Average the scores and loss
    # dW /= num_examples
    # loss /= num_examples
    # #Add regularization
    # loss += reg*np.sum(W**2)/2
    # dW += reg*W

  loss /= N # we need to average out the sample
  dW /= N
  loss += reg*np.sum(W**2)/2  # we need to add the regularization terms
  dW += reg*W
    
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
  num_examples = X.shape[0]
  num_dims = X.shape[1]
  num_classes = W.shape[1]

  f = X.dot(W)
  f_max = np.max(f).reshape(-1, 1)
  f -= f_max
  scores = np.exp(f)

  scores_sums = np.sum(scores, axis=1)
  scores_correct = scores[np.arange(num_examples), y]
  f_correct = f[np.arange(num_examples), y]
  loss = np.sum(-f_correct + np.log(scores_sums))

  summ = scores / (scores_sums.reshape(-1, 1))
  bi_matrix = np.zeros_like(scores)
  bi_matrix[np.arange(num_examples), y] = -1

  summ += bi_matrix

  dW = (X.T).dot(summ)

  loss /= num_examples
  loss += reg*np.sum(W**2) / 2
  dW /= num_examples
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

