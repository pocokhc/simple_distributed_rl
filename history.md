# TODO list

1. BizHawkのenv作成
1. tensorboard
1. (IMPALA)

# v0.10.1

**MainUpdates**

1. runner配下を大幅にリファクタリング
   1. progress等のoptionの指定を引数からクラス形式に変更、使い方が変わります（examplesを参照）
   1. examples配下を合わせて整理
1. srl.rl.models配下を再整理し、もう1つ階層を追加（importのpathも変わります）
1. PriorityMemoryの設定方法をmodelと同様にConfigクラスに変更
   1. BestEpisodeMemory と DemoMemory を追加
   1. env_play に人間が操作した結果を memory に保存する仕組みを追加(R2D3みたいな)
   1. PriorityMemory にエピソード終了処理を入れたかったので on_step を追加
   1. この修正に合わせて PriorityMemory を使っているアルゴリズムも修正
1. VSCode上でpylanceによるエラーをなくすように修正
   1. define.py の型アノテーションを全部Typeで終わる名前に変更
1. remote（分散コンピューティングによる学習）をテスト導入
1. spaces をリファクタリング
   1. "srl.base.env.spaces" を "srl.base.spaces" に移動

**Algorims**

1. DQNを最適化
1. Rainbowをtorchに対応し、最適化も実施

**OtherUpdates**

1. Gridにparameter追加(動作に変化なし)
1. RLConfigに使ってないparameterが追加されたら警告を出すように変更
1. 速度計測を一部実施し修正(batchとrandom.choice)
1. dockerfileにblackとflake8を追加

# v0.10.0

GPU,PyTorch,Dockerの環境を見直して、それに伴い大幅なリファクタリングを行いました。

**MainUpdates**

1. 下記更新に合わせて一部のユーザ側のI/Fを変更
   1. enable_file_logger のデフォルトを False に変更
   1. exampleにQiitaで書いていた各アルゴリズムのコードを移動
   1. READMEを分かりやすく更新
   1. ハイパーパラメータの Model の指定方法を変更(Configクラスを作りました)
   1. EnvBaseを一部変更（主にTurnBase2Playerに影響）
   1. EnvRunに値チェック機構を導入(Configにcheck_action/check_valを追加)
1. Tensorflow の multiprocessing + GPU の環境を見直し
   1. tensorflow の初期化タイミングを用意し、GPU関係の初期化を実装
   1. set_memory_growthオプションをデフォルトTrueで追加（TFではデフォルトFalse）
   1. MpConfigとConfigを統合し、MpConfigを削除
1. PyTorchを追加（暫定）
   1. `srl.rl.models`を整理
      1. 既存コードを `srl.rl.models.tf` に移動
      1. TFのmodelを'FunctionalAPI'から'SubclassingAPI'に変更(仮導入)
      1. srl.rl.models.torch ディレクトリを追加
      1. modelのrl_configの指定をクラスで指定し、tf/torchに依存しない構造に変更
   1. torchのdocker環境を追加
   1. DQNをtf/torch両方に対応（仮導入）
   1. RLConfigのregister方法の仕様を変更
      1. Algorithmにて例えばtf/torchで使い分けたい場合の仕組みを取り入れました
      1. RLConfigのgetNameをstaticmethodをやめてただの関数に変更
      1. registerのConfigの引数をクラスからインスタンスに変更
1. 上記に関連して、GPU環境を見直し
   1. configをallocateからdeviceに変数名変更
   1. rl_configとenv_configにもdevice情報を追加(runner.configから継承される)
1. Dockerファイルをリファクタリング
   1. dockersディレクトリを作成しその配下に必要なファイルをまとめました
   1. Dockerで常駐しない場合があるのでcommand追加
   1. TF2.10で一旦Dockerファイルを固定
   1. torch用Dockerファイルを作成
   1. 最新環境用のlatestファイルを追加
1. gym の space を Graph と Sequence 以外に対応
   1. これに伴って、gym の space は画像を除き、すべて１次元に変換されます。
   1. GymWrapper の引数に画像判定を行うかどうかの check_image を追加
1. WorkerRunとEnvRunのI/Fを見直し
   1. set_render_mode を reset で実施するように引数を追加(set_render_modeは削除)
   1. set_seed も reset で実施（set_seedを削除）
   1. EnvBaseのstepにて、player_indexを削除し、next_player_index の propertyを追加
   1. kaggle環境に適用できるようにdirect機構を見直し
   1. next_player_index を戻り値からプロパティに変更
1. Test framework を unittest から pytest に変更
   1. カバレッジを計測するcreate_coverage.pyを追加(VSCodeに埋め込むと重すぎた)
   1. 乱数固定に。

**Algorims**

1. PlanetをWorldModelsのコードに合わせて修正

**OtherUpdates**

1. env関係
   1. envのcloseで例外時に処理を続けるように変更
   1. gym の状態判定シミュレーションで回数を指定できる引数 prediction_step を追加
   1. env.baseにreward_infoを導入（仮）
1. game_windowにてKeyを押す判定を厳密に定義
1. Configのenv指定を文字列でもできるように変更
1. sub versionの表記でアルファベットを使うと一部のライブラリでエラーが出たので数字に変更
1. RLConfigのreset_configをresetに名前変更

**Bug Fixes**

1. ArrayDiscreteSpace にて discrete を返す時、最大値(high)を含めていなかった不具合修正
1. TF2.11.0から一部のoptimizerでエラーが出るようになったのでlegacyを追加
1. TFのloss計算にて正則化項が加味されるように修正
1. Py3.11でrandom.choiceにnumpy配列を渡すとエラーが出る不具合修正
1. print_progressで残り時間が不明な場合、0と表示される場合を-に変更

# v0.9.1

**Big changes**

Env を手動操作のみで実行できる env_play の追加と、実行結果を window(pygame)上で見れる test_play を追加しました。
これに伴い、Runner と Render 関係を大幅見直し、runner 関係の引数が一部変わっています。

1. runner関係の変更（主に引数）
   1. enable_profiling を追加（CPU/GPU情報の取得を変更可能に）
   1. eval_env_sharing を追加（評価用に使うenvを学習用と共有するかどうか）
   1. file_logger_interval -> file_logger_enable_train_log
   1. enable_checkpoint -> file_logger_enable_checkpoint
   1. checkpoint_interval -> file_logger_checkpoint_interval
   1. file_logger_enable_episode_log を追加（エピソード情報を別途保存します）
   1. render_terminal, render_window, enable_animation を廃止し、render_mode に統一
   1. render の戻り値から Render を廃止（報酬のみ返ります）
1. UpdateDetails
   1. sequence の train,evaluate,render,animation を play_facade に統合し、リファクタリング
   1. sequence_play.py, trainer.py を play_sequence.py, play_trainer.py に名前変更
   1. file_logger.py を file_log_reader.py と file_log_writer.py に分割
   1. file_log_writer.py にてエピソード情報も記録できるように追加（これに伴いlogディレクトリ構成が変わっています）
   1. 評価(eval)を play_sequence 側に組み込みではなくcallback側に移動(evaluate.pyを追加)
   1. render の初期化を play_sequence 側に組み込み
   1. callbacksの on_step_begin の位置をactionの後に変更し、actionの前に on_step_action_before を追加
   1. pynvml(nvidia)の初期化終了処理を play_sequence 側に実装
   1. runner配下にimgフォルダを追加

**MainUpdates**

1. Runner
   1. test_play を追加
   1. env_play を追加
1. RL
   1. RLConfigをdataclass化
      + 継承していたRLConfigで__init__を予備必要があったが、逆に呼んではダメになりました
      + これによりRLConfigインスタンス時に継承元のハイパーパラメータも（VSCode上で）候補一覧に表示されるようになりました
1. Docker環境を追加

**OtherUpdates**

1. Env
   1. 環境のrender_rgb_arrayで使われる想定のViewerをpygame_wrapperに変更（クラスベースではなく関数ベースに）
   1. EnvRun に render_interval を追加
   1. env_play用に、get_key_bind を追加（暫定導入）
1. RL
   1. RL側でdiscreteとcontinuousの両方に対応できる場合の書き方を整理
      + base.Space に rl_action_type を追加
      + rl.algorithms に any_action.py を作成
      + サンプルとして、vanilla_policy_discrete.py と vanilla_policy_continuous.py をvanilla_policy.py に統合
1. Callbacks
   1. 問題なさそうだったので Callback と TrainerCallback を統合
   1. GPU使用率を print_progress に追加
   1. CPUの数取得を psutil.Process().cpu_affinity() から multiprocessing.cpu_count() に変更
   1. 統計情報でエラーがでても止まらないように変更
1. Utils
   1. common に is_package_imported と compare_equal_version を追加
1. Other
   1. 実行時に必要なimportのみを読み込むようにできる限り制限（特に外部ライブラリ）（暫定導入）
   1. font のパスをどこからでも参照できるように修正
   1. examples の minimum_raw でフローが見えやすいように学習と評価の関数を別に記載
   1. 確認しているtensorflowの最低versionを2.1.0から2.2.1に変更
   1. 確認しているgymの最低versionを0.21.0からpygameを導入した0.22.0に変更
   1. pygameの描画をdummyにし、画面がない環境でも動作するように変更
   1. examplesにsample_pendulum.pyを追加

**Bug Fixes**

1. render で2episode以降に初期画像が初期化されない不具合修正
1. gym_wrapper で追加の引数が反映されていなかった不具合修正
1. image_processor の trimming でwとhが逆になっていた不具合修正
1. linux/mac環境でtensorflow+mpを実行するとフリーズする不具合を修正（詳細はrunner.mpの修正箇所を参照）

# v0.9.0

**Big changes**

1. algorithmsとenvsをパッケージ内に移動

**Updates**

1. Config
   1. EnvConfigの skip_frames を frameskip に変更
   1. Configから eval_player を削除し、eval_reward を eval_rewards に変更
   1. eval用Configをリファクタリング
1. Runner
   1. sequence.pyが大きくなったのでファイル分割
   1. callbacksのon_step_beginの位置をactionの前に変更
1. rendering
   1. 日本語の文字化け修正
   1. renderingで、状態が画像の場合、rlに入力される画像も表示できるように追加
   1. render時にstep内は画像情報などが保持されるように修正
1. processor
   1. image_processorにtrimming機能を追加
1. RL
   1. RLWorkerにrecent_statesを追加(追加に伴い実装側で変数名は使えなくなりました)
   1. humanの挙動を改善
1. Env
   1. GymWrapperを更新
1. Other
   1. examplesの sample_atari.py を更新
   1. examplesに minimum_env.py を追加
   1. unittestにpackageのskip追加

**Bug Fixes**

1. ExperienceReplayBufferをmp環境で実行するとmemory上限を認識しない不具合修正
   + ExperienceReplayBuffer/ReplayMemoryのrestore/backupにて、下位versionと互換がありません

# v0.8.0

1. gym v0.26.0 に合わせて大幅修正
   1. (カスタマイズ側のI/F変更)EnvのresetとWorkerのon_reset/policyの戻り値にInfoを追加
      1. 合わせて全環境を見直し
      1. 合わせて全アルゴリズムを見直し
      1. 一応Baseを直接継承していないInfoは省略可能に(DeprecationWarning)
   1. 合わせてEnvBaseのI/Fを一部変更
   1. gym.makeでrender_modeを指定するようで、合わせてrender関係を見直し
      1. 1episodeの最初にrender_modeによって動作を変更できるように変更(set_render_mode関数を追加)
      1. render_windowのintervalのデフォルト値を環境側で指定できるように render_interval プロパティを追加
      1. renderでrgbがない場合、terminal情報を画像にするが、それをbase側(env/rl共に)に組み込み
         1. baseにIRender/Renderクラスを作成し、ここで管理
      1. パラメータにfont_nameとfont_sizeを追加（Renderingからは削除）
      1. renderの未定義をNotImplementedErrorからNoneに変更
1. フレームワーク関係
   1. Infoに文字列も許可し、数値以外は文字列にする処理を追加
   1. print_progressの表示方法を変更
   1. runnerに過去のhistoryを読み込む関数を作成(load_history)
   1. WorkerとRLConfigを別ファイルに(比較的大きくなったので)
1. アルゴリズム関係
   1. WorldModelsで温度パラメータのサンプル計算がおかしいbug fix
   1. PlaNet追加
   1. PR #5 より、float_category_encode/decodeを変更
   1. common_tfにgaussian_kl_divergence 追加
1. Env関係
   1. Envのチェック機構を強化し、ArrayDiscreteSpaceを見直して改修
      1. spaceにcheck_val関数を追加
      1. 合わせて全環境を見直し
      1. test.envをupdate
   1. spaceにconvert関数を追加（値をできる限りspaceの型に変換する）
   1. EnvConfigを整理し、runner.Config にあったパラメータをEnvConfig側に移動
1. その他
   1. READMEのインストール方法を更新
   1. READMEのサンプルコードを更新
   1. tests配下にrequirementsを追加
      1. 合わせてそのversionはできる限り動作するように修正

# v0.7.0

1. 大きな変更
   1. フレームワークと実際の実装を分離
      1. envsとalgorithmsのフォルダを作成し、実装はそちらで管理
      1. 実行方法が変更され、env,algorithmをimportする必要あり
      1. setupを見直し、フレームワークに必要なパッケージだけに変更（各envs/algorithmsで必要なパッケージはそちらで管理）
      1. フレームワークを tensorflow に依存しないように変更
1. フレームワーク関係
   1. runnerを大幅修正
      1. 学習などの実行の窓口をrunnerに統一
         1. __init__を記載し、アクセスを簡略化
         1. 以前のアクセスも可能
         1. それに伴ってサンプルを修正
      1. logの管理方法を見直して大幅に修正
      1. parameter/remote_memoryのload方法を見直して修正
      1. runnerにて、trainerを整備し、mpと統合
      1. callbacksの引数を展開する形から辞書に変更
      1. logとprintの紐づけを強くしてconfigに追加(テスト導入)
      1. validationをevaluateに変更
   1. Workerのtrainingとdistributedの変数をコンストラクタで指定するように変更
      1. srl.rl.makeを廃止、workerとtrainerは毎回作るように変更(example/minimumを参照)
   1. font関係を見直し(フリーフォントを組み込み)
   1. RLWorkerにenvプロパティを追加
   1. invalid_actions を見直し
      1. spaceのboxのaction_discrete_encodeを実装
      1. Env側で使えるのはaction_spaceにdiscreteを使った場合のみに変更
      1. RL側はRLActionTypeがdiscreteの場合のみに変更
      1. DiscreteSpaceTypeを削除しintに変更
      1. RLWorkerのsample_actionを離散値の場合直接出すように変更
   1. sequenceに初期状態を取得できる get_env_init_state を追加
   1. plotの引数でylimを指定できるように変更
1. アルゴリズム関係
   1. WorldModels 追加
1. Env関係
   1. cartpole_continuous を削除(gym updateに伴うメンテが厳しいため)
1. その他
   1. max_stepsとtimeoutが反映されない不具合修正
   1. intervalの初期値を60fpsに変更
   1. README修正
   1. TODO listを追加(将来的にはRoadMapへ？)

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
