import tensorflow as tf
import functools
import codecs
import os
import jieba
from create_lookup_dict import *
from config import params
import itertools

# option1: feed the tf.data by generator
def generator_fn(words_file,tags_file):
    with open(words_file,'r',encoding='utf-8') as words, open(tags_file,'r',encoding='utf-8') as tags:
        for line_words, line_tags in zip(words, tags):
            words = [w for w in line_words.strip().split()]
            tags = [t for t in line_tags.strip().split()]
            assert len(words) == len(tags), "Words and tags lengths don't match"

            chars = [[c for c in w] for w in line_words.strip().split()]
            lengths = [len(c) for c in chars]
            max_len = max(lengths)
            chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
            yield ((words, len(words)), (chars, lengths)), tags


def input_fn_builder(params,training):
    # initialise a Dataset from a generator, this is useful when we have an array of different elements length
    def input_fn():
        shapes = ((([None], ()),  # (words, nwords)
                   ([None, None], [None])),  # (chars, nchars)
                  [None])
        types = (((tf.string, tf.int32),
                  (tf.string, tf.int32)),
                 tf.string)
        defaults = ((('<pad>', 0),
                     ('<pad>', 0)),
                    'O')
        dataset=tf.data.Dataset.from_generator(functools.partial(generator_fn, params['words_file'], params['tags_file']),
                                               output_shapes=shapes, output_types=types)

        if training:
            dataset=dataset.shuffle(buffer_size=100).repeat(params['n_epoch'])
        dataset = (dataset.padded_batch(params.get('batch_size', 20), shapes, defaults).prefetch(1))
        return dataset
    return input_fn


# option 2: feed the data by tensor for bert
class Bert_preprocessor(object):
    def __init__(self,params):
        self.params=params
        self._create_or_load_map_file()

    def get_train_data(self,data_dir):
        train=self._load_sentences(data_dir)
        train_features=self._convert_examples_to_features(train,set_type='train')
        return train_features

    def get_dev_data(self,data_dir):
        dev=self._load_sentences(data_dir)
        dev_features=self._convert_examples_to_features(dev,set_type='dev')
        return dev_features

    def get_test_data(self,data_dir):
        test=self._load_sentences(data_dir)
        test_features=self._convert_examples_to_features(test,set_type='test')
        return test_features

    def _load_sentences(self,data_dir):
        sentence,sentences=[],[]
        for line in codecs.open(data_dir,'r','utf8'):
            line = line.rstrip()
            # print(list(line))
            if not line:
                if len(sentence) > 0:
                    if 'DOCSTART' not in sentence[0][0]:
                        sentences.append(sentence)
                    sentence = []
            else:
                if line[0] == " ":
                    line = "$" + line[1:]
                    word = line.split()
                else:
                    word = line.split()
                assert len(word) >= 2, print([word[0]])
                sentence.append(word)
        if len(sentence) > 0:
            if 'DOCSTART' not in sentence[0][0]:
                sentences.append(sentence)
        return sentences

    def _create_or_load_map_file(self):
        if not os.path.exists(self.params['word_vocab_file']):
            create_lookup_dict(self.params['words_file'], self.params['word_vocab_file'])
            create_char_lookup_dict(self.params['words_file'], self.params['char_vocab_file'])
            create_lookup_dict(self.params['tags_file'], self.params['tag_lookup_file'])
        self.vocab,index2vocab=load_lookup_dict(self.params['word_vocab_file'])
        self.char_vocab,index2char=load_bert_char(self.params['bert_char_vocab_file'],params)
        self.tag,index2tag=load_lookup_dict(self.params['tag_lookup_file'])
        self.params['n_tags']=len(self.tag)

    def _convert_examples_to_features(self,examples,set_type):
        features=[]
        for example in examples:
            string=[w[0] for w in example]
            chars=[self.char_vocab[w.lower() if w.lower() in self.char_vocab else '<UNK>'] for w in string]
            segs=self._create_word_features("".join(string))

            if set_type=='train':
                tags=[self.tag[w[-1]] for w in example]
            else:
                tags=[0 for _ in chars]

            if len(string)<self.params['seq_length']:
                padding = [0] * (self.params['seq_length'] - len(string))
                chars = chars + padding
                segs = segs + padding
                tags = tags + padding
            else:
                chars=chars[:self.params['seq_length']]
                segs=segs[:self.params['seq_length']]
                tags=tags[:self.params['seq_length']]
            features.append([chars, segs, tags])
            if len(features)==1:
                print('string',string)
                print('chars',chars)
                print('tags',tags)
        return features

    def _create_word_features(self,example):
        seg_features=[]
        for word in jieba.cut(example):
            if len(word)==1:
                seg_features.append(0)
            else:
                tmp=[2]*len(word)
                tmp[0]=1
                tmp[-1]=3
                seg_features.extend(tmp)
        return seg_features


def bert_input_fn_builder(features,batch_size,is_training):
    char_ids,seg_ids,tag_ids=[],[],[]
    for feature in features:
        char,seg,tag=feature
        char_ids.append(char)
        seg_ids.append(seg)
        tag_ids.append(tag)

    def input_fn():
        num_examples = len(features)
        d=tf.data.Dataset.from_tensor_slices({
            "char_ids":tf.constant(char_ids,shape=[num_examples,params['seq_length']],dtype=tf.int32),
            "seg_ids": tf.constant(seg_ids, shape=[num_examples,params['seq_length']], dtype=tf.int32),
            "tag_ids":tf.constant(tag_ids,shape=[num_examples,params['seq_length']],dtype=tf.int32)})
        if is_training:
            d=d.repeat()
            d=d.shuffle(buffer_size=100)
        d=d.batch(batch_size=batch_size)
        return d
    return input_fn

# test
if __name__=='__main__':
    processor=Bert_preprocessor(params)
    processor.get_train_data('./data/example.train')
    #generator_fn(words_file='./data/source.txt',tags_file='./data/target.txt')
