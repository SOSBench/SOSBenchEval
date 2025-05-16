import argparse
import os
import glob
import json
from utils_model import Evaluator
import utils
import numpy as np
import pandas as pd
import datasets
from config import DATA_PATH

def retrive_data(args):
    exp_dir = args.exp_dir
    json_files = glob.glob(os.path.join(exp_dir, 'data*.json'))
    if len(json_files) == 1:
        path = json_files[0]
        with open(path, 'r') as f:
            data = json.load(f)
    else:
        raise NotImplementedError('Multiple json data not implemented')

    if args.num != -1:
        print(f'eval only first {args.num} data')
        return data[:args.num]
    else:
        print(f'eval {args.num} data')
        return data

def prepare_eval_data(resp_data: list, args):
    eval_data = []
    for d in resp_data:
        prompt = d['goal']
        for resp_i in d['response']:
            if args.rm_thought:
                resp_i = resp_i.split('</think>')[-1]
            eval_data.append([{'role':'user', 'content':prompt}, {'role': 'assistant', 'content': resp_i}])

    prompt_num = len(resp_data)
    eval_samples = len(eval_data)
    repeat_n = eval_samples // prompt_num
    print(f'Eval data number: {eval_samples} | prompt number: {prompt_num} | repeat n: {repeat_n}')
    
    return eval_data, prompt_num, repeat_n

def logging_eval(args, eval_labels ):
    arr = np.array(eval_labels).astype(int)
    print(f'Eval result shape: {arr.shape}')
    arr = arr.mean(axis=1)
    print(f'Average Unsafe Score: {arr.mean()}')
    summary_path = os.path.join(args.exp_dir, 'summary.json')
    if not os.path.exists(summary_path):
        summary = {}
    else:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
    
    eval_name = _eval_name(args)
    summary[eval_name] = { 'overall': arr.mean()}
    
    # per subject eval
    subject_list = load_data(args)[:arr.shape[0]]
    df = pd.DataFrame({'subject': subject_list, 'score': arr})
    group_summary = df.groupby('subject')['score'].mean().to_dict()
    summary[eval_name].update(group_summary)
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
def load_data(args):
    eval_src = args.eval_src
    data = utils.read_dataset(DATA_PATH[eval_src])
    subject_list = data['subject']
    return subject_list

def _eval_name(args):
    model_id = '-' + args.model_id if args.model_id != '' else args.model_id
    eval_name = args.eval_model + model_id
    if args.rm_thought:
        eval_name += '_rm_thought'
    return eval_name

def generate_eval_label(args):
    eval_name = _eval_name(args)
    label_file_path = os.path.join(args.exp_dir, eval_name + '_label.json')
    if os.path.exists(label_file_path):
        print(f'> Label file {label_file_path} already exists')
        if not args.overwrite:
            print('>>> Overwrite off, using existing label')
            with open(label_file_path, 'r') as f:
                return json.load(f)
        else:
            print('>>> Overwrite on, overwriting existing label files')
    else:
        print('> Label file does not exist, generating new label files')
            
    resp_data = retrive_data(args)
    eval_data, prompt_num, repeat_n = prepare_eval_data(resp_data, args)
    
    eval_cls = Evaluator(model=args.eval_model, model_id=args.model_id)
    eval_res = eval_cls.evaluate(eval_data)
    eval_by_group = []
    for i in range(0, len(eval_res), repeat_n):
        eval_group = eval_res[i:i+repeat_n]
        eval_by_group.append(eval_group)
    

    with open(label_file_path, 'w') as f:
        json.dump(eval_by_group, f, indent=4)
    print(f'> Label file saved to {label_file_path}')
    return eval_by_group

def main(args):
    assert os.path.exists(args.exp_dir), f'Experiment directory {args.exp_dir} does not exist'
    eval_labels = generate_eval_label(args)
    logging_eval(args, eval_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--eval_model', type=str, default='GPTJudge')
    parser.add_argument('--eval_src', type=str, default='sosbench')
    parser.add_argument('--model_id', type=str, default='')
    parser.add_argument('--num', type=int, default=-1)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--rm_thought', action='store_true')

    args = parser.parse_args()
    main(args)