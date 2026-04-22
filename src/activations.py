import numpy as np

class ReLU:
  """
  ReLU(Rectified Linear Unit)
  入力データが0以下ならば0、そうでないならそのままの値を出力するReLU層
  """
  def __init__(self):
    self.mask = None        # 順伝播の時に0以下だった箇所の記憶

  def forward(self, x):
    self.mask = (x <= 0)    # 0以下の場所をFalse

    output = x.copy()
    output[self.mask] = 0   # 0以下の場所を0

    return output

  def backward(self, dout):
    dx = dout.copy()
    dx[self.mask] = 0       # 順伝播で0以下だった箇所を0

    return dx