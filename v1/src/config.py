from collections import defaultdict

# Eval Running
MAX_WORKERS = 50
MAX_RETRIES = 4
VLLM_GPU_UTILS=0.9


API_DICT = {
    'openai': {
        'api_key': '',
        'base_url': "https://api.openai.com/v1"
    },
    'openai-moderation': {
        'api_key': '',
        'base_url': "https://api.openai.com/v1"
    },
    'grok': {
        'api_key': '',
        'base_url': "https://api.x.ai/v1",
    },
    'together': {
        'api_key': '',
        'base_url': 'https://api.together.xyz/v1'
    },
    'anthropic': {
        'api_key': '',  
    },
    'gemini': {'api_key':''},
}

MAX_TOKENS=512
REASONING_MAX=MAX_TOKENS*10
THINKING_BUDGET=REASONING_MAX - MAX_TOKENS

MODEL_CONFIG = {
    'gpt-4o-mini': {
        'system_prompt': False,
        'model_id': 'gpt-4o-mini-2024-07-18',
        'endpoint': 'openai',
        'official_batch': False,
        'max_workers': 150,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS}
    },
    'gpt-4o-2024-08-06': {
        'system_prompt': False,
        'model_id': 'gpt-4o-2024-08-06',
        'endpoint': 'openai',
        'official_batch': True,
        'max_workers': 150,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS}
    },
    'gpt-4o-2024-11-20': {
        'system_prompt': False,
        'model_id': 'gpt-4o-2024-11-20',
        'endpoint': 'openai',
        'official_batch': False,
        'max_workers': 100,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS}
    },
    'gpt-4.1-2025-04-14': {
        'system_prompt': False,
        'model_id': 'gpt-4.1-2025-04-14',
        'endpoint': 'openai',
        'max_workers': 100,
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS}
    },
    'gpt-4.1-mini': {
        'system_prompt': False,
        'model_id': 'gpt-4.1-mini-2025-04-14',
        'endpoint': 'openai',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS}
    },
    'gpt-4.1-nano': {
        'system_prompt': False,
        'model_id': 'gpt-4.1-nano-2025-04-14',
        'endpoint': 'openai',
        'official_batch': False,
        'max_workers': 100,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS}
    },
    'o3-2025-04-16': {
        'system_prompt': False,
        'model_id': 'o3-2025-04-16',
        'endpoint': 'openai',
        'official_batch': False,
        'gen_args': {
            'max_tokens': REASONING_MAX}
    },
    'o4-mini-2025-04-16': {
        'system_prompt': False,
        'model_id': 'o4-mini-2025-04-16',
        'endpoint': 'openai',
        'official_batch': False,
        'gen_args': {
            'max_tokens': REASONING_MAX
        }
    },
    'o4-mini-2025-04-16-low': {
        'system_prompt': False,
        'model_id': 'o4-mini-2025-04-16',
        'endpoint': 'openai',
        'official_batch': False,
        'gen_args': {
            'max_tokens': REASONING_MAX,
            'reasoning_effort': 'low'
        }
    },
    'o4-mini-2025-04-16-high': {
        'system_prompt': False,
        'model_id': 'o4-mini-2025-04-16',
        'endpoint': 'openai',
        'official_batch': False,
        'gen_args': {
            'max_tokens': 20000,
            'reasoning_effort': 'high'
        }
    },
    # test only
    'gpt-3.5-turbo-0125': {
        'system_prompt': False,
        'model_id': 'gpt-3.5-turbo-0125',
        'endpoint': 'openai',
        'official_batch': True,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }
    },
    'claude-3-7-sonnet-20250219': {
        'system_prompt': False,
        'model_id': 'claude-3-7-sonnet-20250219',
        'endpoint': 'anthropic',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        },
        'max_workers': 10,
        'max_retries': 15,
    },
    'claude-3-5-sonnet-20241022': {
        'system_prompt': False,
        'model_id': 'claude-3-5-sonnet-20241022',
        'endpoint': 'anthropic',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        },
        'max_workers': 10,
        'max_retries': 15,
    },
    'claude-3-7-sonnet-20250219+thinking': {
        'system_prompt': False,
        'model_id': 'claude-3-7-sonnet-20250219',
        'endpoint': 'anthropic',
        'official_batch': False,
        'gen_args': {
            'temperature': 1, # `temperature` may only be set to 1 when thinking is enabled
            'max_tokens': REASONING_MAX,
            'thinking': {
                'type': 'enabled',
                'budget_tokens': THINKING_BUDGET
            }
        },
        'max_workers': 20,
        'max_retries': 15,
    },
    'claude-3-7-sonnet-20250219+thinking+1024': {
        'system_prompt': False,
        'model_id': 'claude-3-7-sonnet-20250219',
        'endpoint': 'anthropic',
        'official_batch': True,
        'gen_args': {
            'temperature': 1, # `temperature` may only be set to 1 when thinking is enabled
            'max_tokens': 1024 + MAX_TOKENS,
            'thinking': {
                'type': 'enabled',
                'budget_tokens': 1024
            }
        },
        'max_workers': 5,
        'max_retries': 15,
    },
    'claude-3-7-sonnet-20250219+thinking+16384': {
        'system_prompt': False,
        'model_id': 'claude-3-7-sonnet-20250219',
        'endpoint': 'anthropic',
        'official_batch': True,
        'gen_args': {
            'temperature': 1, # `temperature` may only be set to 1 when thinking is enabled
            'max_tokens': 16384 + MAX_TOKENS,
            'thinking': {
                'type': 'enabled',
                'budget_tokens': 16384
            }
        },
        'max_workers': 5,
        'max_retries': 15,
    },


    # together
    'Qwen3-235B-A22B': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen3-235B-A22B-fp8-tput',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'Llama-4-Maverick-17B-128E-Instruct': {
        'system_prompt': False,
        'model_id': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Llama-4-Scout-17B-16E-Instruct': {
        'system_prompt': False,
        'model_id': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Llama-3.1-405B-Instruct': {
        'system_prompt': False,
        'model_id': 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Deepseek-R1': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'Deepseek-V3-0324': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-V3',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Llama-3.3-70B-Instruct': {
        'system_prompt': False,
        'model_id': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen2.5-72B-Instruct': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-72B-Instruct-Turbo',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen2.5-7B-Instruct': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-7B-Instruct-Turbo',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'DeepSeek-R1-Distill-Llama-70B': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'endpoint': 'together',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    # grok
    'grok-3-mini-beta': {
        'system_prompt': False,
        'model_id': 'grok-3-mini-beta',
        'endpoint': 'grok',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'reasoning_effort': 'low',
            'max_tokens': REASONING_MAX
        }
    },
    'grok-3-mini-beta-high': {
        'system_prompt': False,
        'model_id': 'grok-3-mini-beta',
        'endpoint': 'grok',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'reasoning_effort': 'high',
            'max_tokens': REASONING_MAX*4
        }
    },
    'grok-3-beta': {
        'system_prompt': False,
        'model_id': 'grok-3-beta',
        'endpoint': 'grok',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'max_tokens': MAX_TOKENS
        }
    },
    # vllm
    'Qwen2.5-7B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-7B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},

    # vllm
    'wildguard':{
        'system_prompt': False,
        'model_id': 'allenai/wildguard',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 2,
        'gen_args': {
            'max_tokens': 50,
            'temperature': 0.0,
        }
    },
    'Mistral-7B-Instruct-v0.1': {
        'system_prompt': False,
        'model_id': 'mistralai/Mistral-7B-Instruct-v0.1',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }
    },
    'Llama-3-8B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'meta-llama/Meta-Llama-3-8B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen2.5-72B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-72B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen2.5-32B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-32B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen2.5-14B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-14B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 2,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen2.5-7B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-7B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen2.5-1.5B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen2.5-0.5B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen2.5-0.5B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Llama-3.3-70B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'meta-llama/Llama-3.3-70B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Llama-3.1-70B-Instruct-VLLM': {
        'system_prompt': False,
        'model_id': 'meta-llama/Llama-3.1-70B-Instruct',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Qwen3-32B-Thinking-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen3-32B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 2,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'Qwen3-14B-Thinking-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen3-14B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 2,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'Qwen3-8B-Thinking-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen3-8B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 2,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'Qwen3-4B-Thinking-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen3-4B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'Qwen3-1.7B-Thinking-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen3-1.7B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'Qwen3-0.6B-Thinking-VLLM': {
        'system_prompt': False,
        'model_id': 'Qwen/Qwen3-0.6B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'R1-Distill-Llama-70B-VLLM': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'R1-Distill-Llama-70B-VLLM': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'R1-Distill-Qwen-32B-VLLM': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 2,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'R1-Distill-Qwen-14B-VLLM': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-14B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 2,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'R1-Distill-Llama-8B-VLLM': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'R1-Distill-Qwen-7B-VLLM': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'R1-Distill-Qwen-1.5B-VLLM': {
        'system_prompt': False,
        'model_id': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'gemma-3-27b-it': {
        'system_prompt': False,
        'model_id': 'google/gemma-3-27b-it',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'gemma-3-12b-it': {
        'system_prompt': False,
        'model_id': 'google/gemma-3-12b-it',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 2,
        'vllm_args': {
            'dtype': 'bfloat16'
        },
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'gemma-3-4b-it': {
        'system_prompt': False,
        'model_id': 'google/gemma-3-4b-it',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'vllm_args': {
            'dtype': 'bfloat16'
        },
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'gemma-3-1b-it': {
        'system_prompt': False,
        'model_id': 'google/gemma-3-1b-it',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},

    'ChemDFM-v1.5-8B': {
        'system_prompt': False,
        'model_id': 'OpenDFM/ChemDFM-v1.5-8B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'ChemLLM-7B-Chat-1_5-DPO': {
        'system_prompt': False,
        'model_id': 'AI4Chem/ChemLLM-7B-Chat-1_5-DPO', 
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Med-LLaMA3-8B': {
        'system_prompt': False,
        'model_id': 'YBXL/Med-LLaMA3-8B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS,
            'chat_template': '''{%- for m in messages %}{%- if m.role == 'user' %} User: {{ m.content }}{%- elif m.role == 'assistant' %}Assistant: {{ m.content }}{%- endif %}{%- endfor %}{%- if add_generation_prompt %}Assistant: {% endif %}'''
        }},
    'BioMistral-7B-SLERP': {
        'system_prompt': False,
        'model_id': 'BioMistral/BioMistral-7B-SLERP',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'PsychoCounsel-Llama3-8B': {
        'system_prompt': False,
        'model_id': 'Psychotherapy-LLM/PsychoCounsel-Llama3-8B',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 1,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Llama3.1-70B-ShiningValiant2': {
        'system_prompt': False,
        'model_id': 'ValiantLabs/Llama3.1-70B-ShiningValiant2',
        'endpoint': 'vllm',
        'official_batch': False,
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }}, 
    'Mixtral-8x7B-Instruct-v0.1': {
        'system_prompt': False,
        'model_id': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'endpoint': 'vllm',
        'official_batch': False,
        'vllm_args': {
            'dtype': 'bfloat16'
        },
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Mixtral-8x7B-Instruct_RMU': {
        'system_prompt': False,
        'model_id': 'cais/Mixtral-8x7B-Instruct_RMU',
        'endpoint': 'vllm',
        'official_batch': False,
        'vllm_args': {
            'dtype': 'bfloat16'
        },
        'num_gpus': 4,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'zephyr-7b-beta': {
        'system_prompt': False,
        'model_id': 'HuggingFaceH4/zephyr-7b-beta',
        'endpoint': 'vllm',
        'official_batch': False,
        'vllm_args': {
            'dtype': 'bfloat16'
        },
        'num_gpus': 2,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'Zephyr_RMU': {
        'system_prompt': False,
        'model_id': 'cais/Zephyr_RMU',
        'endpoint': 'vllm',
        'official_batch': False,
        'vllm_args': {
            'dtype': 'bfloat16'
        },
        'num_gpus': 2,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': MAX_TOKENS
        }},
    'gemini-2.5-pro': {
        'system_prompt': False,
        'model_id': 'gemini-2.5-pro-preview-05-06',
        'endpoint': 'gemini',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'gemini-2.5-flash': {
        'system_prompt': False,
        'model_id': 'gemini-2.5-flash-preview-04-17',
        'endpoint': 'gemini',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},    
    'gemini-2.5-pro-05-06': {
        'system_prompt': False,
        'model_id': 'gemini-2.5-pro-preview-05-06',
        'endpoint': 'gemini',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'max_tokens': REASONING_MAX
        }},
    'gemini-2.5-flash-04-17': {
        'system_prompt': False,
        'model_id': 'gemini-2.5-flash-preview-04-17',
        'endpoint': 'gemini',
        'official_batch': False,
        'max_workers': 10,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'thinking_budget': THINKING_BUDGET,
            'max_tokens': REASONING_MAX
        }},
    'gemini-2.5-flash-04-17-1024': {
        'system_prompt': False,
        'model_id': 'gemini-2.5-flash-preview-04-17',
        'endpoint': 'gemini',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'thinking_budget': 1024,
            'max_tokens': 1024+MAX_TOKENS + 100
        }},
    'gemini-2.5-flash-04-17-4096': {
        'system_prompt': False,
        'model_id': 'gemini-2.5-flash-preview-04-17',
        'endpoint': 'gemini',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'thinking_budget': 4096,
            'max_tokens': 4096+MAX_TOKENS + 100
        }},
    'gemini-2.5-flash-04-17-16384': {
        'system_prompt': False,
        'model_id': 'gemini-2.5-flash-preview-04-17',
        'endpoint': 'gemini',
        'official_batch': False,
        'gen_args': {
            'temperature': 0.0,
            'top_p': 1.0,
            'thinking_budget': 16384,
            'max_tokens': 16384+MAX_TOKENS + 100
        }},
}

DATA_PATH = {
    'sosbench': 'SOSBench/SOSBench',
    'sosbench-lite': 'SOSBench/SOSBench-Lite',
}