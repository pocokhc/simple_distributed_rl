# 外部環境を元にした学習のサンプル

kaggle等、既にある外部環境やこちらからは操作できない外部環境に対して、本フレームワークで学習し、動かすサンプルコードとなります。

以下の手順で実装します。

1. 外部環境を用意（[env_external.py](./env_external.py)）
2. 外部環境をラップし、SRLで認識できる環境を作成（[env_srl.py](./env_srl.py)）
3. SRL環境を元に学習（[srl_model.py](./srl_model.py), [srl_train.py](./srl_train.py)）
4. 外部環境で指定されたフォーマットのエージェントを作成（[srl_agent.py](./srl_agent.py)）
5. 外部環境を元に学習されたエージェントを動作（[main.py](./main.py)）

実行手順は以下です。

1. [srl_train.py](./srl_train.py) を実行し、学習したファイル "_parameter.dat" を作成
2. [main.py](./main.py) を実行し、学習されたエージェントを動作

