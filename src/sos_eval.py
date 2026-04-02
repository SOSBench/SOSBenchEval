import argparse
import json

from config import add_shared_args, load_run_config
from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="SOSBench v2 generation/evaluation pipeline.")
    add_shared_args(parser)
    args = parser.parse_args()
    config = load_run_config(args)
    result = run_pipeline(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
