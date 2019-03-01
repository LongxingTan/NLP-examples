import tensorflow as tf
from models._embedding import Embedding_layer

class Bi_LSTM(object):
    def __init__(self,params,training):
        self.params=params
        self.training=training
        self.embedding_layer=Embedding_layer(params['vocab_size'],params['embed_size'])

    def build(self,inputs):
        embedding_output=self.embedding_layer(inputs)
        if self.training:
            embedding_output=tf.nn.dropout(embedding_output,keep_prob=0.95)

        with tf.variable_scope('lstm'):
            cell_fw=tf.nn.rnn_cell.LSTMCell(self.params['lstm_hidden_size'])
            cell_bw=tf.nn.rnn_cell.LSTMCell(self.params['lstm_hidden_size'])
            lstm_outputs,_=tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,embedding_output,
                                                           sequence_length=None,dtype=tf.float32)
            lstm_out=tf.concat(lstm_outputs,axis=-1) #shape [batch_size,seq_length,2*lstm_hidden_size]

            if self.training:
                lstm_out=tf.nn.dropout(lstm_out,keep_prob=0.95)

        with tf.variable_scope('dense'):
            #lstm_out=tf.transpose(lstm_out,[1,0,2])
            logits=tf.layers.dense(lstm_out,self.params['n_tags'])
        return logits

    def __call__(self,inputs):
        logits=self.build(inputs)
        return logits