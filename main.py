import argparse

from src.pipeline import FullPipeline
from src.utils import Config, Device, Env


class CLI:
    @staticmethod
    def parse():
        p = argparse.ArgumentParser()
        p.add_argument("--config", default="config/config.yaml")
        p.add_argument("--dataset", choices=["ag_news", "dbpedia_14", "both"], default="both")
        p.add_argument("--skip-training", action="store_true", dest="skip_training")
        p.add_argument("--runs", type=int, default=None)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--no-hp", action="store_true", dest="no_hp")
        p.add_argument("--no-ablation", action="store_true", dest="no_ablation")
        p.add_argument("--no-scale", action="store_true", dest="no_scale")
        p.add_argument("--output", default="results")
        return p.parse_args()


def main():
    options = CLI.parse()
    Env().load().hf_login()
    Device.describe()
    config = Config.from_yaml(options.config)
    FullPipeline(config, options).run()


if __name__ == "__main__":
    main()
