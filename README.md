# SOSBench

[Project Page](https://sosbench.github.io/) | [OpenReview](https://openreview.net/forum?id=2Td8r7KYK2) | [Dataset](https://huggingface.co/datasets/SOSBench/SOSBench) | [Legacy V1](./v1/README.md)

SOSBench is a benchmark for evaluating safety overrefusal in LLMs. This repository is the main codebase for the paper project and provides the public evaluation harness, benchmark data access, and leaderboard utilities.

## News

- [26/04] We release the v2 public evaluation stack with a LiteLLM-based pipeline, Hugging Face dataset loading, resumable generation and evaluation, and leaderboard summaries.
- [26/02] Our paper is accepted to ICLR 2026.
- [25/06] SOSBench is integrated into [Microsoft PyRIT](https://github.com/microsoft/PyRIT).
- [25/05] We release the SOSBench project and public codebase.

## Overview

The default public workflow is the v2 harness under [`src/`](./src), which runs generation and evaluation in two stages on the public `SOSBench/SOSBench` dataset. The previous public implementation is preserved under [`v1/`](./v1).

Note: v2 updates both the evaluation prompt setting and the default judge-model setting. The legacy v1 stack is deprecated for new runs, but it is still preserved for backward compatibility, experimental reproducibility, and reference to the earlier configuration in `v1/src/config.py`.

For full harness setup, environment configuration, local serving, and advanced usage, see [`AGENTS.md`](./AGENTS.md).

## Quick Start

```bash
cp .env.example .env
cd src/scripts
bash build_env_uv.sh
cd ../..
python src/sos_eval.py --stage all --dataset sosbench --gen-model gpt-5 --eval-model gpt-5
```

SOSBench is distributed through Hugging Face parquet files. Use a compatible Arrow stack when creating the environment; the public harness pins `pyarrow>=20.0.0` in [`src/scripts/requirements.txt`](./src/scripts/requirements.txt) to avoid parquet read failures on older installs.

To aggregate run summaries into a leaderboard-style table:

```bash
python src/sos_leaderboard.py --run-root runs --format table
```

## Citation

If you use SOSBench in your work, please cite our paper and refer to the OpenReview page above for the current manuscript.
```
@inproceedings{jiangsosbench,
  title={SoSBench: Benchmarking Safety Alignment on Six Scientific Domains},
  author={Jiang, Fengqing and Ma, Fengbo and Xu, Zhangchen and Li, Yuetai and Rao, Zixin and Ramasubramanian, Bhaskar and Niu, Luyao and Li, Bo and Chen, Xianyan and Xiang, Zhen and others},
  booktitle={The Fourteenth International Conference on Learning Representations}
}
```
## Acknowledgment

The public v2 refactor was completed with assistance from Codex. If you encounter issues with the release, please contact fqjiang@uw.edu.
