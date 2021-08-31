# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 index,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.index=index
        self.label=label

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def convert_examples_to_features(js,tokenizer,args):
    #source
    code=' '.join(js['code'].split())
    code_tokens=tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids+=[tokenizer.pad_token_id]*padding_length
    return InputFeatures(source_tokens,source_ids,js['index'],int(js['label']))

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        data = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                data.append(js)
        for js in data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        self.label_examples = {}
        for e in self.examples:
            if e.label not in self.label_examples:
                self.label_examples[e.label] = []
            self.label_examples[e.label].append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        label = self.examples[i].label
        index = self.examples[i].index
        labels = list(self.label_examples)
        labels.remove(label)
        while True:
            shuffle_example = random.sample(self.label_examples[label], 1)[0]
            if shuffle_example.index != index:
                p_example = shuffle_example  # different example with same label
                break
        n_example = random.sample(self.label_examples[random.sample(labels, 1)[0]], 1)[
            0]  # label has removed, n_example with different label

        return (torch.tensor(self.examples[i].input_ids), torch.tensor(p_example.input_ids),
                torch.tensor(n_example.input_ids), torch.tensor(label))

'''
eval_dataset = None
def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
        eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        p_inputs = batch[1].to(args.device)
        n_inputs = batch[2].to(args.device)
        label = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss, vec = model(inputs, p_inputs, n_inputs, label)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    pre_vecs = np.load(args.vec_data_file)
    pre_labels = np.load(args.label_data_file)

    vecs = np.concatenate(vecs, 0)
    labels = np.concatenate(labels, 0)

    scores = np.matmul(vecs, pre_vecs.T)
    print(scores)
    print(vecs.shape)
    print(labels.shape)
    print(scores.shape)
    indexs = []
    for example in eval_dataset.examples:
        indexs.append(example.index)
    print(len(indexs))
    dic = {}  # func 0 --> sum_0 - 1(self)
    for i in range(pre_labels.shape[0]):
        scores[i, i] = -1000000
        if int(pre_labels[i]) not in dic:
            dic[int(pre_labels[i])] = -1
        dic[int(pre_labels[i])] += 1

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    cnt = 0
    with open(os.path.join(args.output_dir,args.output_data_file),'w') as f:
        for index,sort_id in zip(indexs,sort_ids):
            js={}
            js['index']=index

            label = int(labels[cnt])
            js['label'] = label

            cont = 0
            for j in range(dic[label]):
                index = sort_ids[cnt, j]
                if int(pre_labels[index]) == label:
                    cont += 1

            cnt += 1
            js['score'] = cont/dic[label]
            f.write(json.dumps(js)+'\n')
'''

def test(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_dataset = TextDataset(tokenizer, args,args.test_data_file)


    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    vecs=[]
    labels=[]
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        p_inputs = batch[1].to(args.device)
        n_inputs = batch[2].to(args.device)
        label = batch[3].to(args.device)
        with torch.no_grad():
            lm_loss,vec = model(inputs,p_inputs,n_inputs,label)
            eval_loss += lm_loss.mean().item()
            vecs.append(vec.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    vecs=np.concatenate(vecs,0)

    labels=np.concatenate(labels,0)
    cluster = {}
    for i in range(len(labels)):
        label = labels[i]
        if label not in cluster:
            cluster[label] = 0
        cluster[label] += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    scores=np.matmul(vecs,vecs.T)
    for i in range(scores.shape[0]):
        scores[i,i]=-1000000
    sort_ids=np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]
    indexs=[]
    for example in eval_dataset.examples:
        indexs.append(example.index)
    with open(os.path.join(args.output_dir,args.output_data_file),'w') as f:
        cnt=0
        for index,sort_id in zip(indexs,sort_ids):
            js={}
            js['index']=index
            js['answers']=[]
            for idx in sort_id[:cluster[labels[cnt]] - 1]:
                js['answers'].append(indexs[int(idx)])
            f.write(json.dumps(js)+'\n')
            cnt += 1


parser = argparse.ArgumentParser()

# dir for bin
parser.add_argument("--bin_data_file", default=None, type=str, required=True)
# dir for cpp
parser.add_argument("--test_data_file", default=None, type=str, required=True)

# output dir
parser.add_argument("--output_data_file", default=None, type=str, required=True)

# from original
parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
parser.add_argument("--config_name", default="", type=str,
                    help="Optional pretrained config name or path if not the same as model_name_or_path")
parser.add_argument("--tokenizer_name", default="", type=str,
                    help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument("--no_cuda", action='store_true',
                    help="Avoid using CUDA when available")
parser.add_argument('--epoch', type=int, default=42,
                    help="random seed for initialization")
parser.add_argument("--cache_dir", default="", type=str,
                    help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
parser.add_argument("--block_size", default=-1, type=int,
                    help="Optional input sequence length after tokenization."
                         "The training dataset will be truncated in block of this size for training."
                         "Default to the model max input length for single sentence inputs (take into account special tokens).")
parser.add_argument("--eval_batch_size", default=4, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.device = device
args.n_gpu = torch.cuda.device_count()
args.per_gpu_eval_batch_size=args.eval_batch_size//args.n_gpu
set_seed(args.seed)

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)

config.num_labels=1
tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
if args.block_size <= 0:
    args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)

if args.model_name_or_path:
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
else:
    model = model_class(config)

model=Model(model,config,tokenizer,args)

logger.info("Training/evaluation parameters %s", args)

# Evaluation
results = {}
checkpoint_prefix = os.path.join('checkpoint-best-map', args.bin_data_file)
output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
model.load_state_dict(torch.load(output_dir))
model.to(args.device)
test(args, model, tokenizer)
