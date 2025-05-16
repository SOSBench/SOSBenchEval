## SOSBench

### Usage

#### Install environment

Go to `src/scripts`, then run the following:
```
bash  build_env.sh
```


#### Run eval

Before running, ensure you set up the API in `src/config.py` for models to be evaluated.

Then under `src`

```
python resp_gen.py --model  model_name_in_config
```

we also support the following argument
`--n`: number of evaluate example, if you do not want to run full set
`--start_idx`: the start idx of load dataset
`--dataset`: dataset evaluate, follow setup in `src/config.py`

Then for model generated response, use the output dir (e.g., `./exp_output/sosbench/gpt-4.1-mini/250516-110929`)

```
python resp_eval.py --exp_dir  exp_dir_path
```
