import os
import sys

sys.path.append('../')
import numpy as np
import ujson as json
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import time
import re
from utils import get_logger
from dataset import Dataset
from rouge import Rouge
from model import BERT_rank, TripletLoss
import random
import csv
import pandas as pd
from transformers import AdamW


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(config, device):
    set_seed(config)
    rouge = Rouge()
    logger = get_logger(config.train_log)
    logger.info("Building model...")
    # data reading
    train_dataset = Dataset(config.train_dataset_file, config, 'train')
    train_it_num = len(train_dataset) // config.batch_size
    dev_dataset = Dataset(config.dev_dataset_file, config, 'dev')
    dev_it_num = len(dev_dataset) // config.val_batch_size
    with open(config.train_title, 'r') as f:
        template_txt = f.readlines()
    with open(config.dev_title, 'r') as f:
        dev_title = f.readlines()

    # define and import model
    # loss_function = TripletLoss(config.margin)
    model = BERT_rank(1, dropout=config.dropout).to(device)

    # setting parameter, optimizer and loss function
    bert_model = list(map(id, model.bert.parameters()))
    other_params = filter(lambda p: id(p) not in bert_model, model.parameters())
    # no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": model.bert.parameters(),
            "weight_decay": 0.03,
            'lr': 1e-5
        },
        {
            "params": other_params,
            "weight_decay": 0.03,
            'lr': 1e-2
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, eps=1e-8
    )
    # optimizer = optim.SGD(params=param_optimizer, lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)
    loss_func = torch.nn.MSELoss()
    if config.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    if config.model:
        model.load_state_dict(torch.load(os.path.join(config.save_dir, config.model)))
    model.train()

    # training parameter
    steps = 0
    patience = 0
    losses = 0
    min_loss = 0
    start_time = time.time()
    valid_losses = 0

    # begin training
    for epoch in range(config.epochs):
        batches = train_dataset.gen_batches(config.batch_size, shuffle=True)
        for batch in tqdm(batches, total=train_it_num):
            optimizer.zero_grad()

            (input_ids,
             token_type_ids,
             attention_mask,
             tp_idx, scores) = batch
            # article_input_ids, article_input_mask = article_input_ids.to(device), article_input_mask.to(device)
            # title_input_ids, title_input_mask = title_input_ids.to(device), title_input_mask.to(device)
            # template_input_ids, template_input_mask = template_input_ids.to(device), template_input_mask.to(device)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            scores = scores.to(device)

            # predicting score
            # gold_score = model(article_input_ids, article_input_mask, title_input_ids, title_input_mask)
            # neg_score = model(article_input_ids, article_input_mask, template_input_ids, template_input_mask)
            # y = torch.ones_like(gold_score)
            # print(scores.shape)
            pred = model((input_ids, token_type_ids, attention_mask))
            # print(loss.shape)
            # print(pred.shape)

            loss = loss_func(scores, pred)
            if config.n_gpu > 1:
                loss = loss.mean()
            losses += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            # testing in validation every few steps
            if (steps + 1) % config.checkpoint == 0:
                losses = losses / config.checkpoint
                logger.info(f"Iteration {steps} train loss {losses}")
                losses = 0

            steps += 1
        batches = dev_dataset.gen_batches(config.val_batch_size, shuffle=False)
        template = []
        if isinstance(model, torch.nn.DataParallel):
            net = model.module
        net = net.to(device)
        net.eval()
        with torch.no_grad():
            for batch in tqdm(batches, total=dev_it_num):
                (input_ids,
                 token_type_ids,
                 attention_mask,
                 tp_idx, scores) = batch
                # article_input_ids, article_input_mask = article_input_ids.to(device), article_input_mask.to(device)
                # title_input_ids, title_input_mask = title_input_ids.to(device), title_input_mask.to(device)
                # template_input_ids, template_input_mask = template_input_ids.to(device), template_input_mask.to(device)
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                attention_mask = attention_mask.to(device)
                scores = scores.to(device)

                # predicting score
                # gold_score = model(article_input_ids, article_input_mask, title_input_ids, title_input_mask)
                # neg_score = model(article_input_ids, article_input_mask, template_input_ids,
                #                   template_input_mask)
                # y = torch.ones_like(gold_score)
                pair_CLS = net.encode((input_ids, token_type_ids, attention_mask))
                pred = net.scoring(pair_CLS)

                # loss = loss_func(pred, scores)
                # if config.n_gpu > 1:
                #     loss = loss.mean()
                # valid_losses += loss.item()

                temp_scores = pred.view(-1, config.template_num)
                # print(temp_scores.shape)
                _, index = torch.max(temp_scores, dim=1)
                for i in range(len(index)):
                    idx = index[i] + config.template_num * i
                    tid = int(tp_idx[idx])
                    template.append(template_txt[tid])
        # valid_losses /= dev_it_num
        accuracy = rouge.get_scores(template, dev_title, avg=True)['rouge-2']['f']
        logger.info(f'epcohs {epoch} dev rouge-2 f {accuracy}')

        # early stop
        if accuracy > min_loss:
            patience = 0
            min_loss = accuracy
            fn = os.path.join(config.save_dir, f"model_final.pkl")
            torch.save(model.state_dict(), fn)
        else:
            print(f"patience is {patience}")
            patience += 1
            if patience > config.early_stop:
                logger.info('early stop because val loss is continue increasing!')
                end_time = time.time()
                logger.info(f"total training time {end_time - start_time}")
                exit()
        # valid_losses = 0
    fn = os.path.join(config.save_dir, "model_final.pkl")
    torch.save(model.state_dict(), fn)


def test(config, device):
    rouge = Rouge()
    keyword = config.keyword
    logger = get_logger(config.dev_log)
    if keyword == 'test':
        test_dataset = Dataset(config.test_dataset_file, config)
        with open(config.train_title, 'r') as f:
            template_txt = f.readlines()
        with open(config.test_title, 'r') as f:
            test_title = f.readlines()
    dev_it_num = len(test_dataset) // config.val_batch_size
    batches = test_dataset.gen_batches(config.val_batch_size, shuffle=False)

    model = BERT_rank(1, config.dropout).to(device)
    if not config.model:
        raise Exception('Empty parameter of --model')
    model.load_state_dict(torch.load(os.path.join(config.save_dir, config.model)))
    model.eval()

    template = []
    # template_save = open(config.template_save, 'w')

    with torch.no_grad():
        for batch in tqdm(batches, total=dev_it_num):
            (input_ids,
             token_type_ids,
             attention_mask,
             tp_idx, scores) = batch
            # article_input_ids, article_input_mask = article_input_ids.to(device), article_input_mask.to(device)
            # title_input_ids, title_input_mask = title_input_ids.to(device), title_input_mask.to(device)
            # template_input_ids, template_input_mask = template_input_ids.to(device), template_input_mask.to(device)
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            attention_mask = attention_mask.to(device)
            pred = model(input_ids, token_type_ids, attention_mask)
            temp_scores = pred.view(-1, config.template_num)
            _, index = torch.max(temp_scores, dim=1)
            for i in range(len(index)):
                idx = index[i] + config.template_num * i
                tid = int(tp_idx[idx])
                template.append(template_txt[tid])
                # template_save.write(template_txt[tid])
        acc = rouge.get_scores(template, test_title, avg=False)
        result_acc = []
        for accuracy in acc:
            result_acc.append(accuracy['rouge-2']['f'])
        save_file = pd.DataFrame({'score': result_acc, 'templates': template, 'title': test_title})
        save_file.to_csv(config.template_save, index=False, sep='\t')
        accuracy = rouge.get_scores(template, test_title, avg=True)['rouge-2']['f']
        print(f"rouge2 result is {accuracy}")
