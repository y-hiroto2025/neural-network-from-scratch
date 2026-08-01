# Neural Network From Scratch with Numpy

## Overview
　本リポジトリはPythonでNumpyを用いてニューラルネットワークを実装し、各種アルゴリズムの挙動を比較・検証した学習リポジトリです。
 
　基本的な構造は 斎藤 康毅 著「ゼロから作るDeep Learning Pythonで学ぶディープラーニングの理論と実装（オライリー・ジャパン, 公式リポジトリ: https://github.com/oreilly-japan/deep-learning-from-scratch）」を参考にして実装しつつ、独自で比較検証や、Google Colabで動かしやすい形へのコードの整理を行っています。
 
　実装において、学習の効率化のため、要件定義、ディレクトリ構造の設計、およびおよびCI/CD（GitHub Actions）の下書き作成において、LLM(Gemini 3.1 Pro)の出力を使用しています。アルゴリズムの実装や比較の考察については、書籍を参考に自身の理解に基づいて行っています。

## Environment
- Python 3.12.10
- Numpy 1.26.0
- matplotlib 3.8.0
- pytest 7.4.0
- ruff 0.3.0

## Implementations
以下の機能を実装しました。
- **基礎モジュール:** パーセプトロン, 多層ニューラルネットワークの順伝播と逆伝播
- **活性化関数:** Step, Sigmoid, ReLU, SoftMax, Tanh
- **損失関数:** MSE, Cross-entropy Loss
- **最適化手法:** SGD, Momentum, AdaGrad, Adam
- **学習テクニック:** Batch Normalization, Weight decay, Dropout
- **データセット学習:** MNISTデータセットを用いた画像分類モデルの学習

## Articles
実装プロセスや比較の結果はQiitaにて連載記事として公開しています。
1. [Numpyでニューラルネットワークを実装して学習させてみた（誤差逆伝播法）]()　(準備中)
2. [活性化関数と損失関数でニューラルネットワークの学習はどう変わる？]()　(未公開)
3. [最適化手法で誤差逆伝播法による学習の収束スピードを比較してみた]()　(未公開)
4. [ニューラルネットワークの精度を上げる学習テクニックを実装してみた]()　(未公開)

## Directory Structure
```text
neural-network-from-scratch/
├── README.md
├── requirements.txt
├── .gitignore
├── .github/
│   └── workflows/
│       └── tests.yml
├── src/
│   ├── __init__.py
│   ├── layers.py                                # レイヤー実装
│   ├── activations.py                           # 活性化関数
│   ├── loss_functions.py                        # 損失関数
│   ├── optimizers.py                            # 最適化手法
│   ├── model.py                                 # NNのクラス
│   ├── utils.py                                 # 数値微分など
│   └── datasets.py                              # データローダ
├── notebooks/
│   ├── 01_basic_perceptron.ipynb                # パーセプトロン
│   ├── 02_activation_functions.ipynb            # 活性化関数
│   ├── 03_mlp_mnist.ipynb                       # 多層パーセプトロンでのMNISTデータセットの学習
│   ├── 04_backpropagation_visualization.ipynb   # 勾配確認と計算速度可視化
│   └── 05_optimization_comparison.ipynb         # 最適化手法の比較
├── tests/                                       # テスト実行のためのコード
│   ├── __init__.py
│   ├── test_layers.py
│   ├── test_activations.py
│   ├── test_loss_functions.py
│   ├── test_optimizers.py
│   └── test_model.py
├── examples/
│   ├── train_mnist.py                           # MNISTデータセットを使った学習
│   ├── train_iris.py                            # irisデータセットを使った学習
│   └── visualize_training.py                    # グラフ作成
├── data/
│   └── mnist/
└── docs/
    ├── architecture.md
    ├── backpropagation.md                       # 誤差逆伝播法の仕組み
    ├── optimization.md
    ├── activations_and_losses.md
    └── implementation_notes.md                  # 実装中のメモ
```
