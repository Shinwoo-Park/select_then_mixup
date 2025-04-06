import logging
import os
from easydict import EasyDict as edict
import random
import torch
import json
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from transformers import (get_linear_schedule_with_warmup, RobertaModel, RobertaTokenizer)
import data_monitoring_config as data_monitoring_config
from util import iter_product
from model import Model
logger = logging.getLogger(__name__)
import pandas as pd
from typing import List
from torch.nn import Softmax

def log_training_dynamics(save_home: str,
                          epoch: int,
                          train_ids: List[int],
                          train_probs: List[float]): 
  """
  Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
  """
  td_df = pd.DataFrame({"guid": train_ids,
                        f"prob_epoch_{epoch}": train_probs,
                        })

  logging_dir = os.path.join(save_home, f"training_dynamics")
  # Create directory for logging training dynamics, if it doesn't already exist.
  if not os.path.exists(logging_dir):
    os.makedirs(logging_dir)
  epoch_file_name = os.path.join(logging_dir, f"dynamics_epoch_{epoch}.jsonl")
  td_df.to_json(epoch_file_name, lines=True, orient="records")
  logger.info(f"Training Dynamics logged to {epoch_file_name}")
  
class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 code_tokens,
                 code_ids,
                 nl_tokens,
                 nl_ids,
                 url,
                 idx,
                 label
    ):
        self.code_tokens = code_tokens
        self.code_ids = code_ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids
        self.url = url
        self.idx = idx
        self.label = label

def convert_examples_to_features(js,tokenizer,log):
    """convert examples to token ids"""

    code = ' '.join(js['code_tokens']) if type(js['code_tokens']) is list else ' '.join(js['code'].split())
    code_tokens = tokenizer.tokenize(code)[:256-4]
    code_tokens =[tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = 256 - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length
    
    nl = ' '.join(js['docstring_tokens']) if type(js['docstring_tokens']) is list else ' '.join(js['doc'].split())
    nl_tokens = tokenizer.tokenize(nl)[:128-4]
    nl_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = 128 - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length  

    return InputFeatures(code_tokens,code_ids,nl_tokens,nl_ids,js['url'] if "url" in js else js["retrieval_idx"], js["idx"] if "idx" in js else 1, js["label"] if "label" in js else 1)

class TextDataset(Dataset):
    def __init__(self, tokenizer, log, file_path):
        self.examples = []
        data = []
        with open(file_path) as f:
            if "jsonl" in file_path:
                for line in f:
                    line = line.strip()
                    js = json.loads(line)
                    if 'function_tokens' in js:
                        js['code_tokens'] = js['function_tokens']
                    data.append(js)
            elif "codebase"in file_path or "code_idx_map" in file_path:
                js = json.load(f)
                for key in js:
                    temp = {}
                    temp['code_tokens'] = key.split()
                    temp["retrieval_idx"] = js[key]
                    temp['doc'] = ""
                    temp['docstring_tokens'] = ""
                    data.append(temp)
            elif "json" in file_path:
                for js in json.load(f):
                    data.append(js) 

        for js in data:
            self.examples.append(convert_examples_to_features(js,tokenizer,log))
                
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("# {}".format(idx))
                logger.info("idx: {}".format(example.idx))
                logger.info("label: {}".format(str(example.label)))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))                              
        
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):   
        return (torch.tensor(self.examples[i].code_ids),torch.tensor(self.examples[i].nl_ids), self.examples[i].url, self.examples[i].label)
 
def train(log, model, tokenizer, save_home):
    
    train_dataset = TextDataset(tokenizer, log, 'dataset/' + log.param.train_dataset + '/' + 'train.jsonl')
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=log.param.train_batch_size,num_workers=4)
    
    #get optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=log.param.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, num_training_steps = len(train_dataloader) * log.param.epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", log.param.epoch)
    logger.info("  Total optimization steps = %d", len(train_dataloader)*log.param.epoch)
    
    model.zero_grad()
    model.train()
    tr_num,tr_loss,best_mrr = 0,0,0 
    for idx in range(log.param.epoch): 
        softmax = Softmax(dim=1)
        train_ids = None
        train_logits = None
        for step,batch in enumerate(train_dataloader):
            #get inputs
            code_inputs = batch[0].to(log.param.device)    
            nl_inputs = batch[1].to(log.param.device)

            #get code and nl vectors
            code_vec = model(code_inputs=code_inputs)
            nl_vec = model(nl_inputs=nl_inputs)
            
            #calculate scores and loss
            scores = torch.einsum("ab,cb->ac",nl_vec,code_vec)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(scores*20, torch.arange(code_inputs.size(0), device=scores.device))

            #report loss
            tr_loss += loss.item()
            tr_num += 1
            if (step+1)%100 == 0:
                logger.info("epoch {} step {} loss {}".format(idx,step+1,round(tr_loss/tr_num,5)))
                tr_loss = 0
                tr_num = 0
            
            #backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step() 

            logits = softmax(scores*20)
            logits = torch.diagonal(logits)
            logits = logits.tolist() 
            
            ### record values
            if train_logits is None:
                train_ids = list(map(str, list(batch[2])))
                train_logits = logits
            else:
                train_ids.extend(list(map(str, list(batch[2]))))
                train_logits.extend(logits)

        log_training_dynamics(save_home=save_home,
                            epoch=idx,
                            train_ids=train_ids,
                            train_probs=train_logits,
                            )
        
        #evaluate    
        results = evaluate(log, model, tokenizer, log.param.train_dataset, data_type="valid")
        for key, value in results.items():
            logger.info("  %s = %s", key, round(value,4))    
        #save best model
        if results['eval_mrr']>best_mrr:
            best_mrr = results['eval_mrr']
            logger.info("  "+"*"*20)  
            logger.info("  Best mrr:%s", round(best_mrr,4))
            logger.info("  "+"*"*20)                          
            model_to_save = model.module if hasattr(model,'module') else model
            output_dir = os.path.join(save_home, '{}'.format('model.bin')) 
            torch.save(model_to_save.state_dict(), output_dir)
            logger.info("Saving model checkpoint to %s", output_dir)

def evaluate(log, model, tokenizer, eval_data, data_type):
    if data_type == "test":
        if eval_data == "csn":
            query_file = 'dataset/' + eval_data + '/' + 'test.jsonl'
            code_file = 'dataset/' + eval_data + '/' + 'codebase.jsonl'
        elif eval_data == "advtest":
            query_file = 'dataset/' + eval_data + '/' + 'test.jsonl'
            code_file = 'dataset/' + eval_data + '/' + 'test.jsonl'
        elif eval_data == "cosqa":
            query_file = 'dataset/' + eval_data + '/' + 'test.json'
            code_file = 'dataset/' + eval_data + '/' + 'code_idx_map.txt'
        elif eval_data == "xlcost":
            query_file = 'dataset/' + eval_data + '/' + 'test.jsonl'
            code_file = 'dataset/' + eval_data + '/' + 'test.jsonl'
    else: 
        query_file = 'dataset/' + eval_data + '/' + 'valid.jsonl'
        code_file = 'dataset/' + eval_data + '/' + 'codebase.jsonl'

    query_dataset = TextDataset(tokenizer, log, query_file)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=log.param.eval_batch_size, num_workers=4)
    code_dataset = TextDataset(tokenizer, log, code_file)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=log.param.eval_batch_size, num_workers=4)    

    model.eval()
    code_vecs = [] 
    nl_vecs = []
    for batch in query_dataloader:  
        nl_inputs = batch[1].to(log.param.device)
        with torch.no_grad():
            nl_vec = model(nl_inputs=nl_inputs) 
            nl_vecs.append(nl_vec.cpu().numpy()) 
    for batch in code_dataloader:
        code_inputs = batch[0].to(log.param.device)    
        with torch.no_grad():
            code_vec = model(code_inputs=code_inputs)
            code_vecs.append(code_vec.cpu().numpy())  
    model.train()    
    code_vecs = np.concatenate(code_vecs,0)
    nl_vecs = np.concatenate(nl_vecs,0)
    
    scores = np.matmul(nl_vecs,code_vecs.T)
    
    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:,::-1]    
    
    nl_urls = []
    code_urls = []
    for example in query_dataset.examples:
        nl_urls.append(example.url)
        
    for example in code_dataset.examples:
        code_urls.append(example.url)

    ranks = []
    for url, sort_id in zip(nl_urls,sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1/rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr":float(np.mean(ranks))
    }

    return result

def run(log): 
    np.random.seed(log.param.seed)
    random.seed(log.param.seed)
    torch.manual_seed(log.param.seed)
    torch.cuda.manual_seed(log.param.seed)
    torch.cuda.manual_seed_all(log.param.seed)
    os.environ['PYHTONHASHSEED'] = str(log.param.seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 

    log.param.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.param.n_gpu = torch.cuda.device_count()

    if log.param.model == "CodeBERT":
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaModel.from_pretrained("microsoft/codebert-base")
        model = Model(model)
    elif log.param.model == "GraphCodeBERT":
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
        model = RobertaModel.from_pretrained("microsoft/graphcodebert-base")
        model = Model(model)
    elif log.param.model == "UniXcoder":
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        model = RobertaModel.from_pretrained("microsoft/unixcoder-base")
        model = Model(model)

    model.to(log.param.device)
    if log.param.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    test_data = ["csn", "advtest", "cosqa", "xlcost"] 

    if log.param.epoch == 1:
        # Zero-shot evaluation
        print("Zero-shot evaluation")
        zero_shot_eval_results = {}
        for td in test_data:
            result = evaluate(log, model, tokenizer, td, data_type="test")
            for key in sorted(result.keys()):
                zero_shot_eval_results[td] = str(round(result[key],4))
        with open(log.param.model + "_zero_shot_eval_results.jsonl", "w") as f:
            json.dump(zero_shot_eval_results, f)

    save_home = "./save/" + log.param.model + "/" + str(log.param.seed) + "/"
    os.makedirs(save_home, exist_ok=True)

    # Zero-shot evaluation
    if log.param.seed == 42:
        print("Zero-shot evaluation")
        zero_shot_eval_results = {}
        for td in test_data:
            result = evaluate(log, model, tokenizer, td, data_type="test")
            for key in sorted(result.keys()):
                zero_shot_eval_results[td] = str(round(result[key],4))
        with open(save_home + "zero_shot_eval_results.jsonl", "w") as f:
            json.dump(zero_shot_eval_results, f)

    # training 
    print("Training")
    train(log, model, tokenizer, save_home)            

    # evaluation
    print("Evaluation")
    checkpoint_prefix = 'model.bin' 
    model_to_load = model.module if hasattr(model, 'module') else model  
    model_to_load.load_state_dict(torch.load(save_home + checkpoint_prefix))      
    model.to(log.param.device)
    eval_results = {}
    for td in test_data:
        result = evaluate(log, model, tokenizer, td, data_type="test")
        for key in sorted(result.keys()):
            eval_results[td] = str(round(result[key],4))
    with open(save_home + "eval_results.jsonl", "w") as f:
        json.dump(eval_results, f)

if __name__ == '__main__':

    tuning_param = data_monitoring_config.tuning_param

    param_list = [data_monitoring_config.param[i] for i in tuning_param]
    param_list = [tuple(tuning_param)] + list(iter_product(*param_list)) ## [(param_name),(param combinations)]

    for param_com in param_list[1:]: # as first element is just name

        log = edict()
        log.param = data_monitoring_config.param

        for num,val in enumerate(param_com):
            log.param[param_list[0][num]] = val

        run(log)