import hydra
from omegaconf import OmegaConf

import srl


@hydra.main(version_base=None, config_path=".", config_name="yaml_training")
def main(cfg):
    OmegaConf.resolve(cfg)

    # dict形式からrunnerを作成
    runner = srl.load(cfg)

    # loadした設定情報を表示
    runner.summary(show_changed_only=True)

    # 設定されている内容で実行する
    runner.play()

    # 結果の簡易評価
    print(runner.evaluate())


if __name__ == "__main__":
    main()
