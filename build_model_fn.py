import tensorflow as tf
from models.bert import get_assignment_map_from_checkpoint
from models._embedding import Embedding_layer
from models._crf import CRF_layer
import functools

def model_fn_builder(ner_model,params):
    def model_fn(features,labels,mode):
        if isinstance(features, dict):
            features = ((features['words'], features['nwords']),
                        (features['chars'], features['nchars']))
        (words, nwords), (chars, nchars) = features

        with open(params['tag_lookup_file'],'r') as f:
            indices=[idx for idx,tag in enumerate(f) if tag.strip()!='']
            num_tags=len(indices)+1
            params['n_tags']=num_tags

        if params['if_char']:
            with open(params['char_vocab_file'], 'r') as f:
                char_vocabs = [idx for idx, tag in enumerate(f) if tag.strip() != '']
                params['vocab_size'] = len(char_vocabs) + 2
            vocab_chars = tf.contrib.lookup.index_table_from_file(params['char_vocab_file'], num_oov_buckets=1)
            input_ids = vocab_chars.lookup(chars)
        else:
            with open(params['word_vocab_file'], 'r') as f:
                word_vocabs = [idx for idx, tag in enumerate(f) if tag.strip() != '']
                params['vocab_size'] = len(word_vocabs) + 2

            vocab_words = tf.contrib.lookup.index_table_from_file(params['word_vocab_file'], num_oov_buckets=1)
            input_ids = vocab_words.lookup(words)
        # char_embedding=Embedding_layer(params['vocab_size'],params['embed_size'])(char_ids)
        # word_embedding=Embedding_layer(params['vocab_size'],params['embed_size'])(seg_ids)

        training=(mode==tf.estimator.ModeKeys.TRAIN)
        model=ner_model(params,training)
        logits=model(input_ids)
        crf_params = tf.get_variable('crf', [params['n_tags'], params['n_tags']], dtype=tf.float32)
        pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"predict_tag_id": pred_ids}
            return tf.estimator.EstimatorSpec(mode,predictions=predictions)

        else:
            vocab_tags=tf.contrib.lookup.index_table_from_file(params['tag_lookup_file'])
            tags=vocab_tags.lookup(labels)

            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, tags, nwords, crf_params)
            loss = tf.reduce_mean(-log_likelihood)

            weights = tf.sequence_mask(params['seq_length'])
            metrics = {'acc': tf.metrics.accuracy(tags, pred_ids, weights),}
            for metric_name, op in metrics.items():
                tf.summary.scalar(metric_name, op[1])

            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

            elif mode==tf.estimator.ModeKeys.TRAIN:
                train_op=tf.train.AdamOptimizer(params['learning_rate']).minimize(loss,global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)
    return model_fn


def bert_model_fn_builder(ner_model,params,init_checkpoint=None):
    def model_fn(features,labels,mode):
        char_ids=features['char_ids']
        seg_ids=features['seg_ids']
        tags=tag_ids=features['tag_ids']
        lengths = tf.reduce_sum(tf.sign(tf.abs(char_ids)), reduction_indices=1)

        model = ner_model(training=(mode == tf.estimator.ModeKeys.TRAIN), params=params)
        logits=model(char_ids) #=> batch_size*seq_length*n_tags

        if init_checkpoint:
            tvars = tf.trainable_variables()
            (assignment_map,initialized_variable_names)=get_assignment_map_from_checkpoint(tvars,init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint,assignment_map)

        crf_layer = CRF_layer(params['n_tags'], params)
        loss, trans = crf_layer(logits, tags, lengths)
        pred_ids, _ = tf.contrib.crf.crf_decode(logits,trans,lengths)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"predict_tag_id":pred_ids}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        else:
            if mode == tf.estimator.ModeKeys.EVAL:
                weights = tf.sequence_mask(params['seq_length'])
                metrics = {'acc': tf.metrics.accuracy(tags, pred_ids, weights), }
                for metric_name, op in metrics.items():
                    tf.summary.scalar(metric_name, op[1])
                return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

            elif mode == tf.estimator.ModeKeys.TRAIN:
                train_op = tf.train.AdamOptimizer(learning_rate=params['learning_rate']).minimize(loss, global_step=tf.train.get_or_create_global_step())
                return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    return model_fn
