import numpy as np

class Affine:
  """
  Fully Connected Layer(Affine Transformation)
  入力データに対して重みとの内積を計算してバイアスを加算する全結合層
  """
  def __init__(self, W, b):
    # W shape       : (input dim, output dim)
    # b shape       : (output dim, )
    self.W = W      # weight
    self.b = b      # bias
    self.x = None   # forwardの入力データ記憶
    self.dW = None  # weightの勾配
    self.db = None  # biasの勾配

  def forward(self, x):
    # x shape       : (batch size, input dim)
    # output shape  : (batth size, output dim)
    self.x = x
    output = np.dot(x, self.W) + self.b

    return output

  def backward(self, dout):
    dx = np.dot(dout, self.W.T)
    self.dW = np.dot(self.x.T, dout)
    self.db = np.sum(dout, axis=0)

    return dx