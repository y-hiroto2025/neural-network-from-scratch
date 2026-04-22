import numpy as np

class Affine:
  """
  Fully Connected Layer(Affine Transformation)
  入力データに対して重みとの内積を計算してバイアスを加算する全結合層
  """
  def __init__(self, W, b):
    # W shape       : (input dim, output dim)
    # b shape       : (output dim, )
    self.W = W 
    self.b = b

  def forword(self, x):
    # x shape       : (batch size, input dim)
    # output shape  : (batth size, output dim)
    output = np.dot(x, self.W) + self.b
    return output