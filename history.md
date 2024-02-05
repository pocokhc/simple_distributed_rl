# TODO list

// (tensorboard) SRL上でいいI/Fの作成方法が思い浮かばず保留
// (SEED RL) 大量のActor向けなのでいったん見送り
// (MARL) マルコフ過程みたいなモデルがある？Actor同士の通信方法の定義が見当たらずに保留
// (jax) batch数(32)ぐらいの量だとnumpyの方が早かったので見送り
1. (tf/torchの互換パラメータの作成)
1. Async-SGD
1. (distribution)memoryの仕様決め
1. (distribution)オリジナルrl/env対応
1. RLの定義でrl_configからmakeしたほうが素直？結構変更が入るので保留
1. spaceの複数入力（画像+値の入力など）

# v0.14.1

**MainUpdates**

1. [base.rl.worker_run] add: prev_state, prev_action, prev_invalid_actionを追加
1. [algorithms.search_dynaq] update: 更新
1. [runner.callbacks] add: historyのintervalにinterval_modeを追加し、間隔を選べるように変更

**OtherUpdates**

1. [envs.grid] update: print関係を修正

**Bug Fixes**

1. [runner.core_mp] fix: mp.QueueにてMACでqsizeが例外を出す不具合修正


# v0.14.0

**MainUpdates**

1. [base]配下
   1. リファクタリング
      1. [base.run] change: RunStateをRunStateActorとRunStateTrainerに分割、合わせてCallbacksの引数も変更
   1. [base.run] new: yieldを追加し、step毎に制御できる機能を追加
   1. [base.run] change: core.pyが大きくなったのでcore_play.pyとcore_train_only.pyに分割
   1. [base.env] new: エピソード最初に乱数を追加する random_noop_max を追加
   1. [base] new: RLMemoryにRLMemoryTypesを追加
   1. [base.processor] new: preprocess_doneを追加し、Processをリファクタリング
   1. [base.define] change: SHAPE2,SHAPE3を削除し、IMAGEを追加
   1. [base.define] change: RLTypesをRLTypesとRLBaseTypesに分割
   1. [base.define] new: DoneTypesを追加し、終了状態を判定できるように変更
   1. [base.rl.RLTrainer] change: RLTrainerのtrain_on_batchsを廃止し、trainに戻しました
   1. [base.rl.RLTrainer] add: RLTrainerにdistributedとtrain_onlyのプロパティを追加
   1. [base.rl.RLTrainer] new: train_startとtrain_end関数を追加
   1. [base.rl.RLWorker] new: on_startとon_end関数を追加
   1. [basel.rl.RLConfig] update/new: setup後にパラメータの変更で警告が出るように変更し、変更可能パラメータを指定できるようにget_changeable_parametersを追加
1. [runner] : リファクタリング
    1. delete: setup_wkdirを削除
    1. add: set_checkpointにis_loadを追加し、過去のディレクトリから最新のパラメータをloadするコードを追加
    1. add: parameter,memory,trainer,workersを参照できるように変更
    1. add: model_summaryの引数にexpand_nestedを追加
    1. fix: create_eval_runnerのバグ修正
1. [tensorflow] : リファクタリング
    1. [rl.models.tf] new: distributionsを追加、主に確率を扱うNNを一元管理することが目的(ベルヌーイ分布,カテゴリカル分布,正規分布等)
    1. [rl.models.tf] update: InputBlockを見直し
    1. [rl.models] change: DuelingNetworkBlockを使わない場合の名前をNoDuelingNetworkBlockに変更
    1. [rl.models] update: その他exceptionに伴う細かい修正
1. [algorithms] 確率モデルの更新に伴って再整理
    1. PPO
    1. DDPG
    1. SAC
1. [algorithms] new: dreamer_v3追加、合わせてv1とv2を統合
1. [runner.core_mp] fix: 最適化
1. [rl.processors] new: AtariProcessor追加

**OtherUpdates**

1. [runner.callbacks] fix: evalで時間がかかる場合に待機時間を経過し、連続で実行される状態を回避
1. [runner.callbacks] fix: 2重にcallbackが実行される不具合修正
1. [base.env.gymnasium_wrapper] fix: metadataがない環境に対応
1. [base.env.gymnasium_wrapper] fix: render_modeが消えてたバグ修正
1. [base.rl.RLTrainer] update: 毎step、train_info={}を追加
1. [rl.functions.common] fix: get_random_max_indexで元の値が変わる不具合を修正
1. [runner.game_window] fix: -+で例外が発生する不具合修正
1. [algorithms.dqn] change: MemoryをPriorityをやめてただのReplayBufferに（シンプルな実装に修正）
1. [algorithms.search_dynaq] update: 100%に
1. [base.exception] new: フレームワーク用のExceptionを追加
1. [render] update: render周りを見直してリファクタリング
1. [render] update: font sizeのデフォルトを12から18に変更
1. [examples.baseline] update: gym.frozen_lake update
1. [examples.baseline] new: gym.taxi追加
1. [examples.baseline] new: env.grid追加

# v0.13.3

**MainUpdates**

1. [runner.callbacks.PrintProgress] update
    1. 表示単位を1秒間の回数に変更
    1. 表示を複数行できるオプションを追加
    1. runnerにset_progress_optionsを追加し、引数を分割
    1. SendQとRecvQを作成し、速度が分かるように
1. [runner] change: dist系のtrainの終了条件をtrainとtimeoutのみに限定
1. [runner] new: load_checkpointを追加し、フォルダから読み込みえるように
1. [runner.distribution] new: history_on_fileを追加
1. [runner.distribution] update: 長時間学習用に、checkpoint/historyの環境を整備

**OtherUpdates**

1. [runner.distribution] fix: Checkpointのバグ修正
1. [runner.callbacks] update: evaluateに例外対策を追加

# v0.13.2

**MainUpdates**

1. [distribution] update: リファクタリング
   1. Queueの基本動作をRabbitMQではなくRedisに変更
   1. Task,Parameter,Memory周りのinterfaceを定義
   1. (Task,Parameterも分離できるように作成はしたけどRedisで両方賄えるので分離はしていない)
   1. Taskの死活監視をUTC時間を基準に厳密化
   1. TaskManagerを外に出してasyncの動作を明確化
   1. 副産物として内部動作改善
1. [callbacks] change: リファクタリング
   1. baseとRunnerのcallbackを明確に分割
   1. 合わせてcallbacksの引数をrunnerからcontext,stateに変更
1. [examples.baseline] new: ベースラインを試験的に導入

**OtherUpdates**

1. [distribution] new: MQTTを追加
1. [runner.mp] update: queue,remote_boardのやりとりを最適化
1. [rl.processors] new: spaceを正規化するNormalizeProcessorを追加
1. [runner] change: timeout引数の型をintからfloatに変更
1. [runner.mp_debug] delete: 更新が追い付かないのでmp_debugを一時的に削除

# v0.13.1

クラウドサービスを利用した分散学習の環境を整備

**MainUpdates**

1. [runner] new: wkdirによる連続学習環境を整備
   1. 合わせてmpとdistributionも更新
   1. deviceの初期化方法を変更
   1. サンプル追加(examples/sample_long_training.py)
1. [distribution] change: QueueのやりとりにRabbitMQを使えるように変更
1. [distribution] new: QueueのやりとりにGoogleCloud Pub/Subを追加
1. [distribution] change: TaskIDを廃止し、基本1学習のみとする
1. [distribution] new: 非同期にタスクを実行できる仕組みを追加
1. [k8s] new: sampleコード追加

**OtherUpdates**

1. [base.rl] new: RLTrainerにbatchに関係なく実行するtrain_no_batchsを追加
1. [callbacks] change: on_trainer_train_end -> on_trainer_loop に変更
1. [core.trainer] change: trainの戻り値をboolにし、train_on_batchsを実行したかを返すように変更
1. [torch] fix: GPUのbackup/restoreの割り当てを改善(並列処理でTrainerとrestoreでCPU/GPUが競合した)、to_cpu/from_cpuの引数を追加
1. [runner] fix: psutilの実行方法を改善
1. [exception] new: SRL用のExceptionを追加（とりあえずクラスだけ）
1. [dockers] change: 構成を見直し全体的にリファクタリング
1. [algorithm.search_dynaq] update: 更新
1. [runner.mp] fix: torch+mpでboardとのやりとりで強制終了するバグ修正
1. [base.rl] new: input_is_image追加
1. [rl.functions] new: symlog追加

# v0.13.0

Redisを用いた分散コンピューティングによる分散学習を実装しました。
それに伴いアーキテクチャを見直しています。
RLTrainer（とRLMemory）の変更が一番大きいので、自作でアルゴリズムを作成している場合はドキュメントを参照してください。
(trainの実装がなくなり、代わりにtrain_on_batchsを実装する必要があります)

また、次のアップデートで長時間の学習（途中復旧など）の実装を予定しています。

**MainUpdates**

1. 分散コンピューティングによる学習のためにアーキテクチャ見直し
   1. [base] update: RLTrainerのtrainの動作を定義、batch処理だけを train_on_batchs として抜き出し。(自作方法も変わるのでドキュメントも更新)
   1. [base] update: RLTrainerに合わせてRLMemoryを全体的に見直し
      1. [base] rename: RLRemoteMemory -> RLMemoryに名前変更
      1. capacity,memory.warmup_size,batch_size等をmemoryのハイパーパラメータとして実装
      1. is_memory_warmup_needed等train側の処理で必要になる関数を定義
      1. [base] update: Worker側のRLMemoryはIRLMemoryWorkerに変更（addのみを実行）
   1. [base] update: RLWorker周りを見直してアップデート（大きくWorkerBaseをなくしてRLWorkerに統合しました）
   1. [base] update: runner.coreの一部をbase.runに移動し、base内で動作に関してもある程度保証
      1. コードが長いので、runnerの窓口関数たちをrunner_facade.pyに移動
      1. 終了条件にmax_memoryを追加
      1. [base.render] update: render関係を見直して更新
1. [runner.distribution] new: redisによる分散学習を追加
1. [runner.mp] update: プロセスによる通信ではなくスレッドによる通信に変更
1. [docs] update: 更新に合わせてドキュメントを全体的に見直し

**OtherUpdates**
1. [algorithm] del: agent57_stateful, r2d2_statefulを削除(メンテをしていないので)
1. [runner.remote] del: redisに置き換えるので削除
1. [rl.memory] del: best_episode_memory,demo_memoryが実装と相性が悪いので削除(将来的に再実装する可能性はあり)
1. [callbacks] change: TrainerCallbackのon_trainer_train->on_trainer_train_endに変更
1. [callbacks] change: 途中停止のintermediate_stopを廃止し、stepなどの戻り値で制御に変更
1. [rl.scheduler] new: updateを追加し、更新があるか分かるように変更
1. [render] update: gifの生成をArtistAnimationからPIL.imageに変更


# v0.12.2

**MainUpdates**

1. [runner.remote] new: 複数PCの分散学習をとりあえず実装(multiprocessingによるIP通信)
1. [utils.serialize] new: runner.Config, runner.Context, RLConfig, EnvConfigにてjsonに変換できるdictを生成する to_json_dict を utils.serialize.py にまとめてリファクタリング
1. [algorithms] new: DreamerV2を追加
1. [docs] new: 各アルゴリズムのハイパーパラメータの説明を追加（テスト導入）（とりあえずql,dqn）

**OtherUpdates**

1. [runner.mp_debug] new: mpのdebug用に逐次でmpと似た動作をするmp_debugを追加（テスト導入）
1. [tests] change: simple_checkをpytestだけにし、またパラメータ指定もできるようにリファクタ
1. [vscode] dev: VSCodeの最新の状態に見直し
1. [docker] update: versionを最新の状態に更新

**Bug Fixes**

1. [algorithms] fix: target_modelを使うアルゴリズムで初回にonline側とweightsを同期していなかった不具合を修正
1. [algorithms.stochastic_muzero] fix: 学習が安定しない場合があったので一部lossの計算をcross_entropy_lossからmse_lossに変更

# v0.12.1

**MainUpdates**

1. [rl.schedulers] new: 特定のハイパーパラメータを動的に変更できる schedulers 機能を追加しました。
1. [rl.algorithms] update: agent57を改善し、torch にも対応できるように大幅に修正しました。
1. [rl.algorithms] update: sacの動作を修正し、いくつかのバグを修正しました。
1. [rl.algorithms] update: ppoも合わせて修正しました。
1. [rl.models]
   1. move: ファイルの場所を見直し、いくつかのファイルを移動しました。
   1. new: activateをtf/torchで共通にし、区別なく使えるようにしました。
   1. new: mlp_blockの一部パラメータをtf/torchで共通に
   1. change: tf/torchを決めるパラメータ'framework'をクラスに変更
   1. change: DuelingNetworkをMLPと統合し、ハイパーパラメータから指定しやすくなるように変更
   1. change: NoisyDenseを実装し、tensorflow_addonsを使用しないように変更
   1. update: torchのNoisyDenseを実装
1. [doc] big update: 構成を見直して大幅に更新。また、docstringを追加して説明を追加。

**OtherUpdates**

1. [rl.common_tf] fix: compute_logprobの計算方法が間違っていたので修正
1. [register] change: envのIDが被ると上書きされる動作から例外を出す動作に変更
1. [env.gym_wrapper] fix: gym_prediction_by_simulationが反映されていない不具合修正
1. [docker] update: latestを最新に更新
1. [runner] fix: psutilがmpでNanになるバグ修正
1. [runner] delete: remote学習はいったん整理するために一時的に削除
1. [runner.callbacks.eval] fix: evalでseedを変えると上書きされるので無効に変更

# v0.12.0

RunnerとWorkerを大きく変更しました。
これにより実行方法とアルゴリズムの実装方法に大きな変更があります。

・実行方法の変更

``` python
from srl import runner
from srl.algorithms import ql

config = runner.Config("Grid", ql.Config())

# train
parameter, _, _ = runner.train(config, timeout=10)

# evaluate
rewards = runner.evaluate(config, parameter, max_episodes=10)
print(f"evaluate episodes: {rewards}")
```

↓

``` python
import srl
from srl.algorithms import ql

runner = srl.Runner("Grid", ql.Config())

# train
runner.train(timeout=10)

# evaluate
rewards = runner.evaluate(max_episodes=10)
print(f"evaluate episodes: {rewards}")
```

ParameterやRemoteMemory等をRUnnerクラス内で管理するようにしました。  
  
・RLConfigの実装の変更
action_type,observation_type の名前の変更と、新しく get_use_framework の実装が必要になりました。

``` python
@dataclass
class RLConfig(ABC):
   # change name: action_type -> base_action_type
   @property
   @abstractmethod
   def action_type(self) -> RLTypes:
      raise NotImplementedError()

   # change name: observation_type -> base_observation_type
   @property
   @abstractmethod
   def base_observation_type(self) -> RLBaseTypes:
      raise NotImplementedError()

   # new abstractmethod
   @abstractmethod
   def get_use_framework(self) -> str:
      raise NotImplementedError()
```

・memory/NNのハイパーパラメータの指定方法の変更（及び実装方法の変更）
実装方法は各algorithmのコードを見てください。

DQNの例は以下です。

``` python
from srl.algorithms import dqn
from srl.rl import memories
from srl.rl.models import dqn as dqn_model
from srl.rl.models import mlp

rl_config = dqn.Config(
   memory=memories.ProportionalMemoryConfig(capacity=10_000, beta_steps=10000),
   image_block_config=dqn_model.R2D3ImageBlockConfig(),
   hidden_block_config=mlp.MLPBlockConfig(layer_sizes=(512,)),
)
```

↓

``` python
from srl.algorithms import dqn
# import が不要になりました

rl_config = dqn.Config()
# 関数を呼び出す形で指定
rl_config.memory.capacity = 10_000  # capacityのみ特別で直接代入(もしかしたら今後関数に変更するかも)
rl_config.memory.set_proportional_memory(beta_steps=10000)
rl_config.image_block.set_r2d3_image()
rl_config.hidden_block.set_mlp(layer_sizes=(512,))
```

**MainUpdates**

1. Runnerのクラス化
   1. パラメータや状態をクラス化し、Runnerクラス内部で管理
      - Config  : 実行前に設定されるパラメータをまとめたクラス
      - Context : 実行直前に決定されるパラメータをまとめたクラス
      - State   : 実行中に変動する変数をまとめたクラス
   1. callbackをTrainer,MP,Gameで分割
   1. callbackの引数をRunnerクラスに統一
   1. 実行方法が変わったので関係ある場所を全体的に見直し
   1. GPUの動作を安定させるため、RLConfigに "get_use_framework" を追加
1. RLConfigのパラメータ指定方法を改善
   1. base.remote_memory,base.modelを削除し、rl.memories,rl.modelsにこれ関係の実装を集約
   1. memoryのアルゴリズム実装側はRLConfigに継承させる形で実装
      継承したRLRemoteMemoryからConfigが参照できるようにし、実装側のコードを削減（基本passだけでよくなった）
   1. NNのアルゴリズム実装側はインスタンスとして生成し、関数でパラメータを指定できるように変更
1. Worker周りを大幅にリファクタリング
　主な変更は以下です。
　・space/typeの指定方法の改善
　・RLWorkerのencoder/decoderの処理をWorkerRunに変更し、状態をWorkerRunに集約
　・WorkerBaseでRLConfig,RLParameter,RLRemoteMemoryを保持するように変更し、Workerの区別をしなくなるように変更
   1. RLConfigのresetにて、space/type の決め方を改善  
      以下はRLConfigから見たプロパティ
      action_space     : RLから見たaction_space(実態はenv.action_spaceと同じ)
      base_action_type : RLが想定するaction_type(RLTypes)
      action_type      : RLの実際のaction_type(RLTypes)
      observation_space    : RLから見たobservation_space
      base_observation_type: RLが想定するobservation_type(RLTypes)
      observation_type     : RLの実際のobservation_type(RLTypes)
      env_observation_type : RLから見たenv側のobservation_type(EnvObservationTypes)
   1. 上記に合わせてRLConfigのプロパティを'action_type''observation_type'を'base_action_type''base_observation_type'に変更
   1. RLConfigのset_config_by_envの責務をRLConfig自体に実装し標準装備に。
      代わりにset_config_by_envはオプションとして残す。(引数は変更)
      またこれにより、RLConfigを分ける必要が実質なくなるので、'DiscreteActionConfig'等は不要（一応残しておきます）
   1. 元RLWorkerで行っていたencoder/decoderをWorkerRunに移し、encode後の状態を保持する実装に変更。
      これにより継承先の引数をまとめることができるので引数を WorkerRun のみに統一。
      （この影響で一部の継承用Workerで引数が変更になっています）
   1. WorkerBaseでRLConfig,RLParameter,RLRemoteMemoryを保持するように変更。
      - RLConfigが指定されてない場合はDummyConfigを保持
      - RLParameterとRLRemoteMemoryはNoneを許容
      - これにより、ExtendWorkerとRuleBaseWorkerを区別なく扱えるようにしました
   1. RuleBaseWorkerの命名をEnvWorkerに変更
   1. 合わせてdocument周りを整理（diagrams,examples,docsを更新）

**OtherUpdates**

1. print_progressを改善
   1. info_types を追加し、表示方法を指定できるようにして文字を削減（例えばintなら小数以下は表示しない）
   1. evalのタイミングを表示のタイミングのみにし高速化
   1. 出来る限り状態を保持しないアルゴリズムに変更し高速化
1. rl_configのlogの出力をOn/Offできるように変更
1. env_runの'sample'を'sample_action'と'sample_observation'に変更
1. testsのalgorithms配下をリファクタリング、tfとtorchで分けるように変更
1. 値のチェック機構を強化(EnvとRLのconfigにenable_sanitize_valueとenable_assertion_valueを追加)
1. gym-retroのdockerファイルを追加

# v0.11.2

**MainUpdates**

1. replay_windowを最初に全エピソード実行するのではなく、1エピソード毎に実行する方法に変更
1. feat[algorithms.dreamer]: Dreamer追加

**OtherUpdates**

1. クラス図を最新の状況に合わせて更新
1. feat[runner.core_simple]: multiplayerに対応
1. test[runner_core_simple]:テスト追加
1. update[algorithms.dqn]:tf.function追加

**Bug Fixes**

1. runnerのmpでユーザ指定callbacksが伝わっていない不具合を修正
1. fix[runner.core]: infoにactor_idを追加

# v0.11.1

**MainUpdates**

1. Env環境のロード方法を改善
   1. gymnasiumに対応
   1. gym/gymnasium に対応した環境を汎用的にロードできる仕組みを追加
   1. gym/gymnasium の act/obs/reward/done に wrapper を適用できる仕組みを追加
1. render_windowの描画方法をmatplotlibからpygameに変更
1. runnerの変更
   1. runnerに最低限の実行しかしないtrain_simpleを追加（テスト導入）
   1. runner直下の実行方法を整理、以下の項目でまとめました
      - 描画
        - render_terminal
        - render_window
        - animation
      - 後から内容を描画
        - replay_window
      - 手動プレイ
        - play_terminal
        - play_window
1. 更新に合わせてドキュメントを更新

**Algorims**

1. Agent57_lightをtorchに対応

**OtherUpdates**

1. PreProcessorクラス周りをリファクタリング
1. change_observation_render_imageをuse_render_image_for_observationに名前を変更
1. SRLで実装されているenvはimportなしで読めるように修正
1. render_discrete_actionを修正
1. print_progressの残り時間の計算方法を改善
1. いくつかのファイルを移動
   1. EnvRunとWorkerRunを別ファイルに
   1. RLWorkerを別ファイルに
   1. runner直下の窓口ファイルのファイル名を facade_ に変更
   1. processorの実装フォルダの場所をbaseからrlへ移動
1. train_mpのデバッグ用にcore_mp_debugを追加
1. 細かいbug fix

# v0.11.0

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
   1. algorithm配下はまだ保留（tensorflow関係のエラーが直しづらいため）
1. remote（分散コンピューティングによる学習）をテスト導入
1. spaces をリファクタリング
   1. "srl.base.env.spaces" を "srl.base.spaces" に移動
   1. 関数名を変えて、encode/decodeを明確化
   1. 合わせて関係あるdefineを整理、不要なものを削除
1. Documentを作成

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
      - 継承していたRLConfigで__init__を予備必要があったが、逆に呼んではダメになりました
      - これによりRLConfigインスタンス時に継承元のハイパーパラメータも（VSCode上で）候補一覧に表示されるようになりました
1. Docker環境を追加

**OtherUpdates**

1. Env
   1. 環境のrender_rgb_arrayで使われる想定のViewerをpygame_wrapperに変更（クラスベースではなく関数ベースに）
   1. EnvRun に render_interval を追加
   1. env_play用に、get_key_bind を追加（暫定導入）
1. RL
   1. RL側でdiscreteとcontinuousの両方に対応できる場合の書き方を整理
      - base.Space に rl_action_type を追加
      - rl.algorithms に any_action.py を作成
      - サンプルとして、vanilla_policy_discrete.py と vanilla_policy_continuous.py をvanilla_policy.py に統合
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
   - ExperienceReplayBuffer/ReplayMemoryのrestore/backupにて、下位versionと互換がありません

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
