.. _yamltraining:

===============
Yaml Training
===============

Yamlで学習内容を記述して学習する方法となります。


1. Yaml format
=====================

.. code-block:: yaml

    # --- EnvConfigの内容を記載します。（keyは[env, envs]のどちらか）
    # id にenvのIDを記載（必須）
    # その他オプションは必要に応じて記載（※1）
    env:
      id: Grid

    # --- RLConfigの内容を記載します。（keyは[rl,algorithm,algorithms]のどれか）
    # _target_ にアルゴリズムのConfigへのパスを記載（必須）
    # その他オプションは必要に応じて記載（※2）
    rl:
      _target_: srl.algorithms.ql.Config

    # --- 学習内容を記載します。（keyは[runner,runners,context,contexts]のどれか）
    # play_mode に学習内容を、追加で学習条件を記載（必須）
    # その他オプションは必要に応じて記載（※3）
    context: 
      play_mode: train  # Literal["train", "train_mp", "rollout", "train_only", "evaluate"]
      max_train_count: 10000


| ※1 envに設定できる項目はこちら :ref:`env_config`
| ※2 RLConfigに共通で設定できる項目はこちら :ref:`rl_config`、それ以外は各アルゴリズムを参照
| ※3 Contextに設定できる項目はこちら :ref:`run_context`


2. Run
=====================

yamlファイルをloadして実行する例です。


.. literalinclude:: yaml_training.yaml
   :caption: yaml_training.yaml
   :language: yaml

.. literalinclude:: yaml_training.py
   :caption: yaml_training.py
   :language: python


3. hydra
=====================

設定管理ライブラリの hydra(https://hydra.cc/) から作られた設定や OmegaConf(https://github.com/omry/omegaconf) からでもロードできます。
コード例は以下です。

.. literalinclude:: yaml_training_hydra.py


| その他のサンプルは以下の examples を見てください。
| - examples/sample_template
| - examples/sample_template_adv

また、yamlのサンプルは configs フォルダ配下を見てください。

