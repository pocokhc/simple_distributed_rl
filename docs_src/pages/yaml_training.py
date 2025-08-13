import os

import srl

yaml_path = os.path.join(os.path.dirname(__file__), "yaml_training.yaml")

# yamlファイルから読み込む
runner = srl.load(yaml_path)

# loadした設定情報を表示
runner.summary(show_changed_only=True)

# yamlに設定されている内容で実行する
runner.play()

# 結果の簡易評価
print(runner.evaluate())
