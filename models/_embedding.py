import tensorflow as tf

class Embedding_layer():
    def __init__(self,vocab_size,embed_size,params=None,embedding_type='random',vocab=None):
        self.vocab_size=vocab_size
        self.embed_size=embed_size
        self.embedding_type=embedding_type
        self.vocab=vocab
        self.params=params

    def create_embedding_table(self,embedding_type):
        if embedding_type=='random':
            embedding_table = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1.0, 1.0),name='embed_w')
            return embedding_table

    def __call__(self, x):
        if self.embedding_type != 'multi-channel':
            embedding_table = self.create_embedding_table(embedding_type=self.embedding_type)
            with tf.name_scope("embedding"):
                embeddings = tf.nn.embedding_lookup(embedding_table, x)
                return embeddings
