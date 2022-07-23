

# v0.5.4
+ AlphaZero追加
+ othello追加
+ get_valid_actions を追加
+ sequenceに経験収集だけ、学習だけをする仕組みを追加
+ パラメータの読み込みをRLConfig側に変更(それに伴って set_parameter_path を削除)
+ sequenceでWorker作成時にRLConfigが指定された場合のバグ修正
+ 細かい修正
  + 自作層をkl.Layerからkeras.Modelに変更
  + pygameのViewerを修正
  + model_summaryの引数を修正
  + mp train callback の on_trainer_train_skip を削除
  + on_trainer_train_endをon_trainer_trainに変更
  + experience_memoryにてbatchサイズ未満の処理を追加
  + invalid actionsのチェック機構を追加

# v0.5.3

+ history作成
+ 複数プレイ時の1ターンで実行できるプレイヤー数を、複数人から一人固定に変更。
  + 実装を複雑にしているだけでメリットがほぼなさそうだったので
  + 1ターンに複数人実行したい場合も、環境側で複数ターンで各プレイヤーのアクションを収集すれば実現できるため
  + これに伴い EnvBase のIFを一部変更
+ MCTSを実装
  + 実装に伴い rl.algorithms.modelbase を追加
+ runner.renderの引数を一部変更
  + terminal,GUI,animationの描画をそれぞれ設定できるように変更
+ OXにGUIを実装
+ modelbaseのシミュレーションstep方法を修正(RL側メインに変更)
