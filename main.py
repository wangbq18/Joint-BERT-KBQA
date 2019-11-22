import os
import torch
import random
import logging
import pickle

from tqdm import tqdm, trange
from random import shuffle
import numpy as np

import utils
from model import KBQA
from batch_loader import BatchLoader
from bert.optimization import BertAdam
from config import args

if not os.path.isdir(args.model_dir):
    os.makedirs(args.model_dir)


def eval(model, ner_dev_data, dev_data, dev_bl, graph, entity_linking, args):
    ## Evaluate
    model.eval()
    dev_iter = iter(dev_bl)
    num = cor_num_s = cor_num_p = cor_num_o = 0
    t = trange(len(dev_bl), desc='Eval')
    for i in t:
        batch_data = next(dev_iter)
        batch_samples = dev_data[args.batch_size*i: args.batch_size*i+batch_data[0].size(0)]
        batch_ners = ner_dev_data[args.batch_size*i: args.batch_size*i+batch_data[0].size(0)]
        batch_data = tuple(tmp.to(args.device) for tmp in batch_data)
        head_logits, tail_logits = (tmp.cpu() for tmp in model(batch_data, 0, False)) # (bz, seqlen)
        heads = head_logits.argmax(dim=-1).tolist()
        for j, head in enumerate(heads):
            tail = tail_logits[j][head:].argmax().item()+head
            tokens = batch_ners[j]['tokens']
            subject = ''.join(tokens[head: tail+1]).replace('##', '').replace('□', ' ')
            if subject[0] == '《':
                subject = subject[1:]
            if subject[-1] == '》':
                subject = subject[:-1]
            question = batch_samples[j]['question']
            gold_spo = batch_samples[j]['triple']
            spos = graph.get(subject, [])
            ons = entity_linking.get(subject, [])
            for on in ons:
                spos += graph.get(on, [])
            pres = list(set([spo[1] for spo in spos]))
            objs = set()
            pre = ''
            if pres:
                sub_re_data = bl.build_re_data(question, pres)
                sub_re_bl = bl.batch_loader(None, sub_re_data, args.ner_max_len, args.re_max_len, args.batch_size, is_train=False)
                sub_labels = []
                for batch_data in sub_re_bl:
                    batch_data = tuple(tmp.to(args.device) for tmp in batch_data)
                    label_logits = model(batch_data, 1, False).cpu().tolist()
                    sub_labels += label_logits
                index_pre = np.argmax(sub_labels)
                pre = pres[index_pre].replace(' ', '')
                for spo in spos:
                    s, p, o = spo
                    if subject in s and p == pre:
                        objs.add(o)
            num += 1
            cor_num_s += 1 if subject == gold_spo[0] else 0
            cor_num_p += 1 if pre == gold_spo[1] else 0
            cor_num_o += 1 if gold_spo[-1] in objs else 0
            # if gold_spo[-1] not in objs:
            #     print('XXXXXXXXXXXX')
            # print(question)
            # print(gold_spo)
            # print(subject, pre)
            # print(objs)
            # print(pres)
            # input()
            t.set_postfix(acc_s='{:.2f}'.format(cor_num_s/num*100), 
                          acc_p='{:.2f}'.format(cor_num_p/num*100), 
                          acc_o='{:.2f}'.format(cor_num_o/num*100))
    return cor_num_o/num

if __name__ == '__main__':
    # Use GPUs if available
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the random seed for reproducible experiments
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))
    logging.info('device: {}'.format(args.device))
    logging.info('Hyper params:%r'%args.__dict__)

    # Create the input data pipeline
    logging.info('Loading the datasets...')
    bl = BatchLoader(args)
    ## Load train and dev data
    train_data = bl.load_data('train.json')
    dev_data = bl.load_data('dev.json')
    ## Train data
    ner_train_data, re_train_data = bl.build_data(train_data, is_train=True)
    train_bls = bl.batch_loader(ner_train_data, re_train_data, args.ner_max_len, args.re_max_len, args.batch_size, is_train=True)
    num_batchs_per_task = [len(train_bl) for train_bl in train_bls]
    logging.info('num of batch per task for train: {}'.format(num_batchs_per_task))
    train_task_ids = sum([[i]*num_batchs_per_task[i] for i in range(len(num_batchs_per_task))], [])
    shuffle(train_task_ids)
    ## Dev data
    ner_dev_data, _ = bl.build_data(dev_data, is_train=False)
    dev_bl = bl.batch_loader(ner_dev_data, None, args.ner_max_len, args.re_max_len, args.batch_size, is_train=False)
    logging.info('num of batch for dev: {}'.format(len(dev_bl)))

    # Model
    model = KBQA.from_pretrained(args.bert_model_dir)
    model.to(args.device)

    # Optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'names': [n for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
            'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'names': [n for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
            'weight_decay_rate': 0.0}
    ]
    args.steps_per_epoch = sum(num_batchs_per_task)
    args.total_steps = args.steps_per_epoch * args.epoch_num
    optimizer = BertAdam(params=optimizer_grouped_parameters, 
                         lr=args.learning_rate, 
                         warmup=args.warmup, 
                         t_total=args.total_steps, 
                         max_grad_norm=args.clip_grad, 
                         schedule=args.schedule)

    logging.info('Loading graph and entity linking...')
    graph = pickle.load(open('graph/graph.pkl', 'rb'))
    entity_linking = pickle.load(open('graph/entity_linking.pkl', 'rb'))

    if args.do_train_and_eval:
        # Train and evaluate
        best_acc = 0
        for epoch in range(args.epoch_num):
            ## Train
            model.train()
            t = trange(args.steps_per_epoch, desc='Epoch {} -Train'.format(epoch))
            loss_avg = utils.RunningAverage()
            train_iters = [iter(tmp) for tmp in train_bls] # to use next and reset the iterator
            for i in t:
                task_id = train_task_ids[i]
                batch_data = next(train_iters[task_id])
                batch_data = tuple(tmp.to(args.device) for tmp in batch_data)
                loss = model(batch_data, task_id, True)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                loss_avg.update(loss.item())
                t.set_postfix(loss='{:5.4f}'.format(loss.item()), avg_loss='{:5.4f}'.format(loss_avg()))
            acc = eval(model, ner_dev_data, dev_data, dev_bl, graph, entity_linking, args)
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'state_dict': model.state_dict(),
                                   'optim_dict': optimizer.state_dict()},
                                   is_best=acc>best_acc,
                                   checkpoint=args.model_dir)
            best_acc = max(best_acc, acc)
    
    if args.do_eval:
        logging.info('num of batch for dev: {}'.format(len(dev_bl)))
        utils.load_checkpoint(os.path.join(args.model_dir, 'best.pth.tar'), model)
        eval(model, ner_dev_data, dev_data, dev_bl, graph, entity_linking, args)
    
    if args.do_predict:
        utils.load_checkpoint(os.path.join(args.model_dir, 'best.pth.tar'), model)
        model.eval()
        logging.info('Loading graph and entity linking...')
        graph = pickle.load(open('graph/graph.pkl', 'rb'))
        entity_linking = pickle.load(open('graph/entity_linking.pkl', 'rb'))
        while True:
            try:
                logging.info('请输入问题：')
                question = input()
                ner_data = bl.build_ner_data(question)
                ner_bl = bl.batch_loader(ner_data, None, args.ner_max_len, args.re_max_len, args.batch_size, is_train=False)
                for batch_data in ner_bl:
                    batch_data = tuple(tmp.to(args.device) for tmp in batch_data)
                    head_logits, tail_logits = (tmp.cpu() for tmp in model(batch_data, 0, False)) # (bz, seqlen)
                    head = head_logits[0].argmax().item()
                    tail = tail_logits[0][head:].argmax().item()+head
                    tokens = ner_data[0]['tokens']
                    subject = ''.join(tokens[head: tail+1]).replace('##', '').replace('□', ' ')
                    if subject[0] == '《':
                        subject = subject[1:]
                    if subject[-1] == '》':
                        subject = subject[:-1]
                    logging.info('抽到的主语为：{}'.format(subject))
                    spos = []
                    spos += graph.get(subject, [])
                    ons = entity_linking.get(subject, [])
                    for on in ons:
                        spos += graph.get(on, [])
                    spos = set(spos)
                    pres = list(set([spo[1] for spo in spos]))
                    # logging.info('候选关系为：{}'.format(pres))
                    if pres:
                        sub_re_data = bl.build_re_data(question, pres)
                        sub_re_bl = bl.batch_loader(None, sub_re_data, args.ner_max_len, args.re_max_len, args.batch_size, is_train=False)
                        sub_labels = []
                        for batch_data in sub_re_bl:
                            batch_data = tuple(tmp.to(args.device) for tmp in batch_data)
                            label_logits = model(batch_data, 1, False).cpu().tolist()
                            sub_labels += label_logits
                        index_pre = np.argmax(sub_labels)
                        pre = pres[index_pre]
                        logging.info('最可能的关系为：{}'.format(pre))
                        for spo in spos:
                            s, p, o = spo
                            if subject in s and p == pre:
                                logging.info('答案为：{}'.format(spo))
                logging.info('\n')
            except:
                logging.info('出错了！是否继续？y/n')
                cmd = input()
                if cmd == 'y':
                    pass
                else:
                    break
