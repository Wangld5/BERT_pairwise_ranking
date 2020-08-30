import torch
import os
import sys
import numpy as np
import argparse
from train import train, test
from dataset import convert_text_to_example
from transformers import BertTokenizer

parser = argparse.ArgumentParser()
parser.add_argument('-train_log', type=str)
parser.add_argument('-train_dataset_file', type=str)
parser.add_argument('-dev_dataset_file', type=str)
parser.add_argument('-test_dataset_file', type=str)
parser.add_argument('-batch_size', type=int, default=36)
parser.add_argument('-val_batch_size', type=int, default=45)
parser.add_argument('-train_article', type=str)
parser.add_argument('-train_title', type=str)
parser.add_argument('-train_index_file', type=str)
parser.add_argument('-dev_article', type=str)
parser.add_argument('-dev_title', type=str)
parser.add_argument('-dev_index_file', type=str)
parser.add_argument('-test_article', type=str)
parser.add_argument('-test_title', type=str)
parser.add_argument('-test_index_file', type=str)
parser.add_argument('-sample_num', type=int, default=100000)
parser.add_argument('-article_limit', type=int, default=100)
parser.add_argument('-title_limit', type=int, default=50)
parser.add_argument('-out_dim', type=int, default=128)
parser.add_argument('-dropout', type=float, default=0.2)
parser.add_argument('-model', type=str, default=None)
parser.add_argument('-save_dir', type=str)
parser.add_argument('-L2_norm', type=float, default=1e-2)
parser.add_argument('-learning_rate', type=float, default=1e-2)
parser.add_argument('-margin', type=float, default=0.001)
parser.add_argument('-epochs', type=int, default=10)
parser.add_argument('-grad_clip', type=float, default=5.0)
parser.add_argument('-checkpoint', type=int, default=3000)
parser.add_argument('-template_num', type=int, default=15)
parser.add_argument('-early_stop', type=int, default=5)
parser.add_argument('-dev_log', type=str)
parser.add_argument('-template_save', type=str)
parser.add_argument('-keyword', type=str, default='test')
parser.add_argument('-mode', type=str)
parser.add_argument('-data_type', type=str)
parser.add_argument('-seed', type=int, default=1001)
config = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.n_gpu = torch.cuda.device_count()

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased/bert-base-uncased-vocab.txt')
    if config.mode == 'preprocess':
        # convert_text_to_example('test', config.test_article, config.test_title, config.train_title,
        #                         config.test_index_file, 0, config.article_limit, config.title_limit,
        #                         tokenizer, config.test_dataset_file)
        # convert_text_to_example('dev', config.dev_article, config.dev_title, config.train_title,
        #                         config.dev_index_file, 0, config.article_limit, config.title_limit,
        #                         tokenizer, config.dev_dataset_file)
        convert_text_to_example('train', config.train_article, config.train_title, config.train_title,
                                config.train_index_file, config.sample_num, config.article_limit, config.title_limit,
                                tokenizer, config.train_dataset_file)
    if config.mode == 'train':
        train(config, device)
    if config.mode == 'test':
        test(config, device)


