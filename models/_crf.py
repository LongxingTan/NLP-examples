import tensorflow as tf

class CRF_layer():
    def __init__(self,n_tags,params):
        self.n_tags=n_tags
        self.params=params


    def __call__(self,logits, tags,lengths):
        batch_size=tf.shape(logits)[0]
        with tf.variable_scope("crf_loss"):
            trans = tf.get_variable(
                "transitions",
                shape=[self.n_tags, self.n_tags],
                initializer=tf.contrib.layers.xavier_initializer())
            if self.n_tags is None:
                return None, trans
            else:
                log_likelihood, trans = tf.contrib.crf.crf_log_likelihood(
                    inputs=logits, #=> batch_size*seq_length,n_tags
                    tag_indices=tags,#=> batch_size*seq_length
                    transition_params=trans,
                    sequence_lengths=lengths) #=> [batch_size]
                return tf.reduce_mean(-log_likelihood), trans
