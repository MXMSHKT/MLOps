import hydra
from omegaconf import DictConfig

from dog_mlops.dog_infer import infer
from dog_mlops.dog_train import train


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(cfg)
    train(cfg)
    infer(cfg)


if __name__ == "__main__":
    main()
