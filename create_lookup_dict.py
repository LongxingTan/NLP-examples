import collections
import itertools
from config import params


def create_lookup_dict(file,target_file,type='dict'):
    vocab_set=set()
    char_set=set()
    vocab_list=[]
    f=open(file, 'r', encoding='utf-8')
    for line in f:
        vocab_set.update([i for i in line.strip().split()])

    vocab_list=list(vocab_set)
    if type=='dict':
        vocab_dict = {}
        for i, word in enumerate(vocab_list):
            vocab_dict[word] = i
        with open(target_file, 'w') as f:
            for key in vocab_dict.keys():
                if key != '':
                    f.write("%s,%s\n" % (key, vocab_dict[key]))

    else:
        with open(target_file, 'w') as f:
            for word in vocab_list:
                if word != '':
                    f.write("%s\n" % (word))


def create_char_lookup_dict(file,target_file,type='dict'):
    vocab_set = set(['<UNK>'])
    f = open(file, 'r', encoding='utf-8')
    for line in f:
        vocab_list=[c for w in line.strip().split() for c in w ]
        #vocab_list=list(itertools.chain(*[[c for c in w] for w in line.strip().split()]))
        vocab_set.update(vocab_list)

    vocab_list = list(vocab_set)
    if type == 'dict':
        vocab_dict = {}
        for i, word in enumerate(vocab_list):
            vocab_dict[word] = i
        with open(target_file, 'w') as f:
            for key in vocab_dict.keys():
                if key != '':
                    f.write("%s,%s\n" % (key, vocab_dict[key]))
    else:
        with open(target_file, 'w') as f:
            for word in vocab_list:
                if word != '':
                    f.write("%s\n" % (word))



def load_lookup_dict(file):
    vocab=collections.OrderedDict()
    index_vocab=collections.OrderedDict()

    with open(file,'r') as reader:
        for line in reader:
            word,index=line.strip().split(",")
            vocab[word]=int(index)
            index_vocab[int(index)]=word
    return vocab,index_vocab

def load_bert_char(vocab_file, params):
    vocab = collections.OrderedDict()
    index_vocab = collections.OrderedDict()
    index = 0

    with open(vocab_file, 'rb') as reader:
        while True:
            tmp = reader.readline()
            token = convert_to_unicode(tmp)
            if not token:
                break

            token = token.strip()
            vocab[token] = index
            index_vocab[index] = token
            index += 1
    params.update(vocab_size=len(vocab))
    vocab.update({'<UNK>':len(vocab)})
    return vocab, index_vocab

def convert_to_unicode(text):
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode('utf-8', 'ignore')
    else:
        raise ValueError("unsupported text string type: %s" % (type(text)))

if __name__=='__main__':
    #create_char_lookup_dict(params['words_file'],params['char_vocab_file'])
    #create_lookup_dict(params['words_file'], params['word_vocab_file'])
    #create_lookup_dict(params['tags_file'], params['tag_lookup_file'])
    tag,index2tag=load_lookup_dict(params['tag_lookup_file'])