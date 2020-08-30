import sys

sys.path.append('../')
from transformers import BertTokenizer
import os
import torch
import numpy as np
from tqdm import tqdm
import json
import random
from rouge import Rouge


def sample_index(train_index, sample_num):
    indexs = np.load(train_index)
    return indexs[:sample_num, :]


class InputFeatures(object):
    def __init__(self, text_a_input_ids, text_a_input_mask, text_b_input_ids, text_b_input_mask, text_c_input_ids,
                 text_c_input_mask, tp_ids):
        self.text_a_input_ids = text_a_input_ids
        self.text_b_input_ids = text_b_input_ids
        self.text_c_input_ids = text_c_input_ids
        self.tp_ids = tp_ids


def convert_text_to_example(data_type, article, title_file, target, index_file, sample_num, article_len,
                            title_len, tokenizer: BertTokenizer, save_file):
    print(f"building {data_type} examples....")
    max_len = article_len + title_len
    with open(article, 'r') as f:
        articles = f.readlines()
    with open(title_file, 'r') as f:
        titles = f.readlines()
    with open(target, 'r') as f:
        templates = f.readlines()
    dataset_index = np.load(index_file)
    rouge = Rouge()

    def filter_func(example, limit):
        if data_type == 'train':
            return len(example) > limit
        else:
            return False

    examples = []
    if sample_num == 0:
        sample = range(len(articles))
    else:
        test_index = np.load('../data/test_index.npy')
        all_index = test_index.reshape(test_index.shape[0] * test_index.shape[1], )
        all_index = set(list(all_index))
        sample = list(all_index)
    for i in tqdm(sample):
        article = articles[i]
        article = article.replace("''", '" ').replace("``", '" ')
        title = titles[i]
        title = title.replace("''", '" ').replace("``", '" ')
        if data_type == 'train':
            if len(tokenizer.tokenize(article)) + len(tokenizer.tokenize(title)) > max_len:
                continue

        # all_template = []
        # for index in dataset_index[i]:
        #     template = templates[int(index)]
        #     template = template.replace("''", '" ').replace("``", '" ')
        #     all_template.append(template)
        # scores = rouge.get_scores([title] * len(dataset_index[i]), all_template, avg=False)
        # if data_type == 'train' and len(list(filter(lambda x: x['rouge-1']['r'] < 1.0, scores))) < 15:
        #     continue

        # article_tokens = tokenizer.tokenize(article)
        # article_ids = tokenizer.convert_tokens_to_ids(article_tokens)
        # if len(article_ids) > article_len:
        #     article_ids = article_ids[:article_len]
        # article_input_ids = tokenizer.build_inputs_with_special_tokens(article_ids)

        # title_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(title))
        # if len(title_ids) > title_len:
        #     title_ids = title_ids[:title_len]
        # title_input_ids = tokenizer.build_inputs_with_special_tokens(title_ids)
        select_num = 0
        if data_type == 'train':
            # select_num = 3
            # select_index = []
            # sep = len(dataset_index[i])//select_num
            # for j in range(0, len(dataset_index[i]), sep):
            #     select_index.append(dataset_index[i][j])
            select_num = len(dataset_index[i])
            select_index = dataset_index[i][:select_num]
        else:
            select_num = len(dataset_index[i])
            select_index = dataset_index[i][:select_num]
        for index in select_index:
            template = templates[int(index)]
            template = template.replace("''", '" ').replace("``", '" ')
            # if data_type == 'train':
            #     score = 1.0
            #     AT_sentence = [(article, title), (article, template)]
            #     AT_feature = tokenizer(
            #         [example for example in AT_sentence],
            #         max_length=max_len,
            #         padding='max_length',
            #         truncation=True
            #     )
            #     examples.append({
            #         'input_ids': AT_feature['input_ids'],
            #         'token_type_ids': AT_feature['token_type_ids'],
            #         'attention_mask': AT_feature['attention_mask'],
            #         'art_idx': str(i),
            #         'tp_idx': str(index),
            #         'score': score
            #     })
            if True:
                score = rouge.get_scores(title, template)[0]['rouge-1']['r']
                pair_sentence = [(article, template)]
                feature = tokenizer(
                    [example for example in pair_sentence],
                    max_length=max_len,
                    padding='max_length',
                    truncation=True
                )
                examples.append({
                    'input_ids': feature['input_ids'][0],
                    'token_type_ids': feature['token_type_ids'][0],
                    'attention_mask': feature['attention_mask'][0],
                    'art_idx': str(i),
                    'tp_idx': str(index),
                    'score': score
                })
            # template_tokens = tokenizer.tokenize(template)
            # template_ids = tokenizer.convert_tokens_to_ids(template_tokens)
            # if len(template_ids) > title_len:
            #     template_ids = template_ids[:title_len]
            # template_input_ids = tokenizer.build_inputs_with_special_tokens(template_ids)

            # while True:
            #     random_temp = random.sample(templates, 1)[0]
            #     if random_temp != template:
            #         break
            # random_temp = random_temp.replace("''", '" ').replace("``", '" ')
            # random_tokens = tokenizer.tokenize(random_temp)
            # random_ids = tokenizer.convert_tokens_to_ids(random_tokens)
            # if len(random_ids) > title_len:
            #     random_ids = random_ids[:title_len]
            # random_input_ids = tokenizer.build_inputs_with_special_tokens(random_ids)

            # examples.append({
            #     'article_input_ids': article_input_ids,
            #     'title_input_ids': title_input_ids,
            #     'template_input_ids': template_input_ids,
            #     'random_input_ids': random_input_ids,
            #     'art_idx': str(i),
            #     'tp_idx': str(index),
            #     'score': score
            # })
    print(f"example num is :{len(examples)}")
    print('saving file in {}'.format(save_file))
    with open(save_file, 'w') as f:
        json.dump(examples, f)
    print('finish saving')


class Dataset:
    def __init__(self, data_file, config, data_type):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.data_size = len(self.data)
        self.indices = list(range(self.data_size))
        self.config = config
        self.data_type = data_type

    def gen_batches(self, batch_size, shuffle=True, pad_id=0):
        if shuffle:
            np.random.shuffle(self.indices)
        for batch_start in np.arange(0, self.data_size, batch_size):
            batch_indices = self.indices[batch_start: batch_start + batch_size]
            yield self._one_mini_batch(batch_indices, pad_id)

    def _one_mini_batch(self, batch_indices, pad_id):
        # article_input_ids, article_input_mask = self.dynamic_padding('article_input_ids', batch_indices, pad_id)
        # title_input_ids, title_input_mask = self.dynamic_padding('title_input_ids', batch_indices, pad_id)
        # template_input_ids, template_input_mask = self.dynamic_padding('template_input_ids', batch_indices, pad_id)
        # random_input_ids, random_input_mask = self.dynamic_padding('random_input_ids', batch_indices, pad_id)
        # if self.data_type == 'train':
        #     ab_input_ids = [self.data[i]['input_ids'][0] for i in batch_indices]
        #     ab_token_type_ids = [self.data[i]['token_type_ids'][0] for i in batch_indices]
        #     ab_attention_mask = [self.data[i]['attention_mask'][0] for i in batch_indices]
        #     ab = (torch.LongTensor(ab_input_ids), torch.LongTensor(ab_token_type_ids),
        #              torch.LongTensor(ab_attention_mask))
        #     ac_input_ids = [self.data[i]['input_ids'][1] for i in batch_indices]
        #     ac_token_type_ids = [self.data[i]['token_type_ids'][1] for i in batch_indices]
        #     ac_attention_mask = [self.data[i]['attention_mask'][1] for i in batch_indices]
        #     ac = (torch.LongTensor(ac_input_ids), torch.LongTensor(ac_token_type_ids),
        #           torch.LongTensor(ac_attention_mask))
        #     template_idx = [self.data[i]['tp_idx'] for i in batch_indices]
        #     scores = [self.data[i]['score'] for i in batch_indices]
        #     return ab, ac, template_idx, torch.FloatTensor(scores)
        if True:
            input_ids = [self.data[i]['input_ids'] for i in batch_indices]
            token_type_ids = [self.data[i]['token_type_ids'] for i in batch_indices]
            attention_mask = [self.data[i]['attention_mask'] for i in batch_indices]
            template_idx = [self.data[i]['tp_idx'] for i in batch_indices]
            scores = [self.data[i]['score'] for i in batch_indices]
            return (torch.LongTensor(input_ids),
                    torch.LongTensor(token_type_ids),
                    torch.LongTensor(attention_mask),
                    template_idx,
                    torch.FloatTensor(scores))
        # return (torch.LongTensor(article_input_ids), torch.LongTensor(article_input_mask),
        #         torch.LongTensor(title_input_ids), torch.LongTensor(title_input_mask),
        #         torch.LongTensor(template_input_ids), torch.LongTensor(template_input_mask),
        #         template_idx, torch.FloatTensor(scores))

    def dynamic_padding(self, key_word, indices, pad_id):
        sample = []
        max_len = 0
        for i in indices:
            sample.append(self.data[i][key_word])
            l = len(self.data[i][key_word])
            max_len = max(max_len, l)
        pad_sample = [ids + [pad_id] * (max_len - len(ids)) for ids in sample]
        mask = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in sample]
        return pad_sample, mask

    def sentence_pair_padding(self):
        pass

    def __len__(self):
        return self.data_size
