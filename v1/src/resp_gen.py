import json
import argparse
import datasets
from numpy import save

import os
from datetime import datetime
import pytz
from utils import *
from config import *
from utils_model import *
import prompt_lib

def load_dataset(args):
    print('========== Loading Dataset ==========')
    if args.dataset == 'safeaware':
        path = DATA_PATH['sosbench']
        data = read_dataset(path)
        df = data.to_pandas()
        df['goal'] = [prompt_lib.safeaware_prompt(term) for term in df['original_term']]
        data = datasets.Dataset.from_pandas(df)
        # data = [safeaware_prompt(term) for term in data]
    elif args.dataset == 'advbench':
        path = DATA_PATH['advbench']
        data = read_dataset(path)
        df = data.to_pandas()
        df['goal'] = [p for p in df['prompt']]
        data = datasets.Dataset.from_pandas(df)
    else:
        path = DATA_PATH[args.dataset]
        data = read_dataset(path)
    if args.start_idx != 0:
        print(f'Start idx: {args.start_idx}')
    if args.n != -1:
        print(f'Loading {args.n} examples from {path}')
        data_size = args.n
    else:
        print('Using all available data after start idx')
        data_size = len(data)

    start_idx = args.start_idx
    end_idx = min(start_idx + data_size, len(data))
    print(f'Start idx = {start_idx} | End idx = {end_idx}')
    print('-----------------------------')
    return data[start_idx:end_idx]

def get_msg_list(data, args):
    system_prompt = MODEL_CONFIG[args.model]['system_prompt']
    msg_list = []
    for prompt in data['goal']:
        message = []
        if system_prompt:
            message.append({'role': 'system', 'content': system_prompt})
        message.append({'role': 'user', 'content': prompt})
        msg_list.append(message)
    return msg_list

def exp_path(args, save_dir='./exp_output'):
    if args.resume is None:
        save_dir = os.path.join(save_dir, args.dataset)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        
        exp_path = os.path.join(save_dir, args.model)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path, exist_ok=True)
        
        time_stamp = datetime.now(pytz.timezone('US/Pacific')).strftime("%y%m%d-%H%M%S")
        exp_path = os.path.join(exp_path, time_stamp)
        os.makedirs(exp_path)

        meta_info = {
            'args': vars(args),
            'model_config': MODEL_CONFIG[args.model],
            'timestamp': time_stamp
        }
        with open(os.path.join(exp_path, 'meta_info.json'), 'w') as f:
            json.dump(meta_info, f, indent=4)
        print('logging meta info to {}'.format(os.path.join(exp_path, 'meta_info.json')))
        return exp_path
    else:
        print('Resume experiment from {}'.format(args.resume))
        return args.resume

def generate_response(model_cls, msg_list, exp_dir,seg_size=500):
    print('========== Generate Response ==========')
    if getattr(model_cls, 'official_batch', False):
        print("Using official batch chat")
        resp_list = model_cls.batch_chat(msg_list)
        parsed_resp_list = [model_cls.parser_fn(resp) for resp in resp_list]
        return parsed_resp_list
    else:
        print("Using multi-threaded/vllm batch chat")
        temp_file = os.path.join(exp_dir, f'temp_log.json')
        parsed_resp_list = []
        print('total messages:', len(msg_list) )
        for i in range(0, len(msg_list), seg_size):
            end_idx = min(i + seg_size, len(msg_list))
            seg_msg_list = msg_list[i:end_idx]
            print(f'Segment: start idx = {i}, end idx = {end_idx}, seg size: {len(seg_msg_list)}')
            
            resp_list = model_cls.batch_chat(seg_msg_list)
            parsed_resp_list += [model_cls.parser_fn(resp) for resp in resp_list]

            with open(temp_file, 'w') as f: # save to temp file under exp_dir, file name as temp_seg_{i}.json
                json.dump(parsed_resp_list, f, indent=4)
            
        # delete temp file
        os.remove(temp_file)
        print('-----------------------------')
        return parsed_resp_list

def save_to_file(data, msg_list, resp_list, exp_dir):
    save_path_prefix = os.path.join(exp_dir, f'data')
    save_data = []
    for idx, (goal, resp) in enumerate(zip(data['goal'], resp_list)):
        add_data = {
            'idx': idx,
            'goal': goal,
            'message': msg_list[idx],
            # 'subject': data['subject'][idx],
            'response': resp,
        }
        if 'subject' in data:
            add_data['subject'] = data['subject'][idx]
        save_data.append(add_data)
    shard_path_list = save_json_in_shards(save_data, save_path_prefix, max_shard_size=95*1_000_000)
    # return shard_path_list
    print(f'>>> save path: {shard_path_list}')
    print(f'>>> exp dir: {exp_dir}')
    
def main(args):
    # load dataset
    data = load_dataset(args)
    msg_list = get_msg_list(data, args)
    model_cls = get_model(args)
    exp_dir = exp_path(args)

    # generate response
    resp_list = generate_response(model_cls, msg_list, exp_dir)
    save_to_file(data, msg_list, resp_list, exp_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, default='sosbench')
    parser.add_argument("--resume", type=str, default=None) # not well supported
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)

    
    args = parser.parse_args()
    main(args)
