import numpy as np

def softmax(x):
    """
    前層からの出力を確率に変換する
    """
    c = np.max(x)
    exp_a = np.exp(x - c)     # オーバーフロー対策

    sum_exp_a = np.sum(exp_a)

    output = exp_a / sum_exp_a
    return output

def cross_entropy_error(output, label):
    """
    予測との誤差を計算する
    """
    delta = 1e-7      # オーバーフロー対策

    loss = -np.sum(label * np.log(output + delta))
    return loss


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 損失
        self.y = None     # softmax出力
        self.label = None # 教師データ

    def forward(self, x, label):
        self.label = label
        self.y = softmax(x)
        
        self.loss = cross_entropy_error(self.y, self.label)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.label.shape[0]
        
        dx = (self.y - self.label) / batch_size
        return dx