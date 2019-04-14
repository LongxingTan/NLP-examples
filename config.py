import json
import argparse
from collections import defaultdict


parser = argparse.ArgumentParser()
parser.add_argument('--n_epoch', type=int, default=5, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate for training')
parser.add_argument('--model_dir',type=str, default='./result/checkpoint',help='Tensorflow checkpoint directory')
parser.add_argument('--words_file',type=str, default='./data/source.txt',help='words file')
parser.add_argument('--tags_file',type=str, default='./data/target.txt',help='tags file')
parser.add_argument('--train_examples',type=str, default='./data/example.train',help='train file')
parser.add_argument('--dev_examples',type=str, default='./data/example.dev',help='dev file')
parser.add_argument('--test_examples',type=str, default='./data/example.test',help='test file')
parser.add_argument('--word_vocab_file',type=str, default='./data/vocab_dict.txt',help='vocab dict from words')
parser.add_argument('--char_vocab_file',type=str, default='./data/char_dict.txt',help='char vocab dict from words')
parser.add_argument('--tag_lookup_file',type=str, default='./data/tag_dict.txt',help='tag dict from tags')
parser.add_argument('--seq_length',type=int,default=150,help='sequence length for one example')
parser.add_argument('--embed_size',type=int, default=100,help='embedding dim')
parser.add_argument('--lstm_hidden_size',type=int, default=100,help='lstm hidden size')
parser.add_argument('--dropout_keep',type=float, default=0.95,help='dropout keep prob')
parser.add_argument('--if_ema',type=bool, default=False,help='if use ema or not')
parser.add_argument('--if_char',type=bool, default=True,help='if use char embedding or word embedding')
parser.add_argument('--do_train',type=bool, default=False,help='if train the model or not')
parser.add_argument('--do_eval',type=bool, default=True,help='if evaluate the model or not')
parser.add_argument('--do_predict',type=bool, default=True,help='if predict the model or not')
parser.add_argument('--bert_config_file',type=str, default='./pretrained/bert_chn/bert_config.json',help='bert config')
parser.add_argument('--bert_init_checkpoint',type=str, default='./pretrained/bert_chn/bert_model.ckpt',help='bert ckpt')
parser.add_argument('--bert_char_vocab_file',type=str, default='./pretrained/bert_chn/vocab.txt',help='bert vocab')
parser.add_argument('--output_dir',type=str, default='./result',help='output dir')


args = parser.parse_args()
params = vars(args)


class Config(object):
    def __init__(self):
        self.params=defaultdict()

    def from_json_file(self,json_file):
        with open(json_file, 'r') as f:
            self.params = json.load(f)

    def to_json_string(self,json_file,params):
        with open(json_file, 'w') as f:
            json.dump(params, f)


if __name__=='__main__':
    config = Config()
    config.to_json_string('./config.json',params)
    #config.from_json_file('./config.json')