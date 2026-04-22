import numpy as np

class ReLU:
  """
  ReLU(Rectified Linear Unit)
  入力データが0以下ならば0、そうでないならそのままの値を出力するReLU層
  """
  def __init__(self):
    pass

  def forward(self, x):
    output = np.maximum(0, x)
    return output