## SOSBench V1

The legacy public implementation now lives under `v1/src`.

### Usage

Install the legacy environment:

```bash
cd v1/src/scripts
bash build_env.sh
```

Generate responses:

```bash
cd v1/src
python resp_gen.py --model model_name_in_config
```

Evaluate an existing run:

```bash
cd v1/src
python resp_eval.py --exp_dir path_to_run
```
