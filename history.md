# v0.6.0
+ MuZero追加
+ StochasticMuZero追加
+ AlphaZeroのシミュレーションの伝播をrootまでに変更
+ 割引率の変数名をgammaからdiscountに変更
+ 必須パッケージに tensorflow_probability を追加
+ render方法を全体的に見直し
  作成側は terminal or rgb_arrayに統一
  ユーザは terminal or window or notebookに統一
  + EnvBaseからrender_guiを削除
  + RenderTypeを削除
  + WorkerBaseの継承先からcall_renderを削除し、WorkerBaseのrender_terminalで統一
  + WorkerBaseのrenderをrender_terminalはに変更
  + WorkerBaseのrender_terminalはオプション化
  + animationのfpsの引数を削除し、intervalを追加
+ stateの数フレームスタック化を個別実装からフレームワーク側に実装
  + これに伴いWorkerBaseを見直し
    + 引数のEnvRun/WorkerRunは補助とし、プリミティブな引数も追加
    + player_indexを必須に
  + Spaceのobservation情報はdiscreteとcontinuousで統一
  + Spaceのobservation情報でlow,highを使う場面がいまだないので削除
  + RLConfigのSpaceに関するプロパティ名を変更
+ sequence.Configにsave/loadを追加（テスト導入）
+ animation周りを見直して修正、情報も描画できるように追加（テスト導入）
+ 細かい修正
  + loggerを少し追加
  + print progressの表示を修正
  + dtypeのfloatをfloat32に統一
  + その他いろいろ

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
