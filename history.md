# v0.6.2
1. フレームワーク関係
   1. 学習などの実行の窓口をrunnerに統一
      1. __init__を記載し、アクセスを簡略化
      1. 以前のアクセスも可能
      1. それに伴ってサンプルを修正
   1. font関係を見直し


# v0.6.1
1. フレームワーク関係
   1. Processorを見直し
      1. RLParameterとRLRemoteMemoryのrestore/backupをcall_restore/call_backupに変更
          1. 圧縮関係など、任意の前後処理をフレームワークが追加できるように変更
          1. restore/backupをrlテストに追加
      1. RLConfig側でprocessorsを追加できるように変更
          1. RLConfigにset_processorを追加
          1. imageが決まっているアルゴリズムに対して追加
          [agent57,agent57_light,alphazero,C51,ddpg,dqn,muzero,r2d2,rainbow,sac,stochastic_muzero]
      1. ImageProcessorを変更
          1. どの環境にも適用できるように変更（assertからif文に変更）
          1. 引数grayからimage_typeにして指定できるように変更
   1. RLConfigの初期化を見直し
      1. RLConfigが書き変わった場合にreset_configが再度必要になるように変更
      1. _set_config_by_envをset_config_by_envに変更しoption化(set_config_by_actorと同じ立ち位置に)
   1. mpでも経験収集のみできるように変更(disable_trainerを追加）
   1. Renderを見直し
      1. viewerでrgb_arrayを取得する際にpygameのwindowを非表示にするように変更
      1. renderingの画像作成を見直し、fontも指定できるように変更
      1. RL側もrender_rgb_arrayを追加
      1. WorkerRunにrender_rgb_arrayとrender_windowを追加
   1. Env の render_rgb_array をRLの状態にできる仕組みを追加
      1. RLConfigにchange_observation_render_imageを追加
      1. RenderImageProcessor を作成
   1. 細かい修正
      1. RLのregisterで名前がかぶっていた場合の処理を例外に変更
      1. train(学習時)にassert文を追加
      1. mp_print_progressの進捗表示を一部変更
      1. 暫定で使用メモリ率を表示
1. アルゴリズム関係
   1. r2d2をstatelessでも実装できたので修正
   1. agent57もstatelessで実装
   1. stochastic_muzeroのバグ修正
   1. stochastic_muzeroとmuzeroのLayerを見直して修正
   1. オリジナルアルゴリズムSearchDynaQを追加(テスト導入)
1. Env関係
   1. pendulum_image削除(Env の render_rgb_array から学習できる仕組みを追加したので)
   1. Gridにrender_rgb_arrayを追加
1. その他
   1. 自作アルゴリズム作成手順(notebook)を大幅に更新
   1. READMEにバージョンも追記
   1. このhistoryをリファクタリング
   1. diagramsを更新


# v0.6.0
1. フレームワーク関係
   1. 必須パッケージに tensorflow_probability を追加
   1. render方法を全体的に見直し
      作成側は terminal or rgb_arrayに統一
      ユーザは terminal or window or notebookに統一
      1. EnvBaseからrender_guiを削除
      1. RenderTypeを削除
      1. WorkerBaseの継承先からcall_renderを削除し、WorkerBaseのrender_terminalで統一
      1. WorkerBaseのrenderをrender_terminalはに変更
      1. WorkerBaseのrender_terminalはオプション化
      1. animationのfpsの引数を削除し、intervalを追加
      1. animation周りを見直して修正、情報も描画できるように追加（テスト導入）
    1. stateの数フレームスタック化を個別実装からフレームワーク側に実装
       1. WorkerBaseを見直し、引数のEnvRun/WorkerRunは補助とし、プリミティブな引数も追加
       1. player_indexを必須に
       1. Spaceのobservation情報はdiscreteとcontinuousで統一
       1. Spaceのobservation情報でlow,highを使う場面がいまだないので削除
       1. RLConfigのSpaceに関するプロパティ名を変更
1. アルゴリズム関係
   1. MuZero追加
   1. StochasticMuZero追加
   1. AlphaZeroのシミュレーションの伝播をrootまでに変更
   1. 割引率の変数名をgammaからdiscountに変更
1. その他
   1. sequence.Configにsave/loadを追加（テスト導入）
   1. loggerを少し追加
   1. print progressの表示を修正
   1. dtypeのfloatをnp.float32に統一
   1. その他いろいろ

# v0.5.4
1. フレームワーク関係
   1. get_valid_actions を追加
   1. sequenceに経験収集だけ、学習だけをする仕組みを追加
   1. パラメータの読み込みをRLConfig側に変更(それに伴って set_parameter_path を削除)
   1. sequenceでWorker作成時にRLConfigが指定された場合のバグ修正
   1. mp train callback の on_trainer_train_skip を削除
   1. mp_callbacksにて、on_trainer_train_endをon_trainer_trainに変更
   1. invalid actionsのチェック機構を追加
1. アルゴリズム関係
   1. AlphaZero追加
   1. 自作層をkl.Layerからkeras.Modelに変更
   1. model_summaryの引数を修正
   1. experience_memoryにてbatchサイズ未満の処理を追加
1. Env関係
   1. othello追加
   1. pygameのViewerを修正


# v0.5.3
1. フレームワーク関係
   1. 複数プレイ時の1ターンで実行できるプレイヤー数を、複数人から一人固定に変更。
      ・実装を複雑にしているだけでメリットがほぼなさそうだったので
      ・1ターンに複数人実行したい場合も、環境側で複数ターンで各プレイヤーのアクションを収集すれば実現できるため
      1. これに伴い EnvBase のIFを一部変更
   1. runner.renderの引数を一部変更
      1. terminal,GUI,animationの描画をそれぞれ設定できるように変更
   1. modelbaseのシミュレーションstep方法を修正(RL側メインに変更)
1. アルゴリズム関係
   1. MCTSを実装
      1. 実装に伴い rl.algorithms.modelbase を追加
1. Env関係
   1. OXにGUIを実装
1. その他
   1. history作成
