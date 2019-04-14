import logging
from config import params
import tensorflow as tf
import os
from build_input_fn import input_fn_builder,bert_input_fn_builder,Bert_preprocessor
from build_model_fn import model_fn_builder,bert_model_fn_builder
from models.bi_lstm import Bi_LSTM
from models.bert import Bert

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def run_ner(ner_model):
    # estimator
    model_fn=model_fn_builder(ner_model=ner_model,params=params)
    run_config=tf.estimator.RunConfig(save_checkpoints_secs=180)
    estimator=tf.estimator.Estimator(model_fn=model_fn,model_dir=params["model_dir"],config=run_config)
    # input function for estimator
    train_input_fn=input_fn_builder(params,training=True)
    eval_input_fn=input_fn_builder(params,training=False)
    # train
    estimator.train(input_fn=train_input_fn,max_steps=1500)

def run_ner_bert(ner_model):
    model_fn=bert_model_fn_builder(ner_model,params,init_checkpoint=params['bert_init_checkpoint'])
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=180)
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=params["model_dir"], config=run_config)

    processor=Bert_preprocessor(params)
    if params['do_train']:
        train_examples = processor.get_train_data(params['train_examples'])
        num_examples=len(train_examples)
        train_steps=int(num_examples//params['batch_size']*params['n_epoch'])
        tf.logging.info("**** start to train ****")
        tf.logging.info("Num examples %d",num_examples)
        tf.logging.info("Num steps %d", train_steps)
        train_input_fn=bert_input_fn_builder(train_examples,batch_size=params['batch_size'],is_training=True)
        estimator.train(input_fn=train_input_fn,max_steps=train_steps)

        dev_examples=processor.get_dev_data(params['dev_examples'])
        num_dev_examples=len(dev_examples)
        dev_steps=int(num_dev_examples//params['batch_size'])
        tf.logging.info("**** running eval ****")
        tf.logging.info("Num examples %d", num_dev_examples)
        dev_input_fn=bert_input_fn_builder(dev_examples,batch_size=params['batch_size'],is_training=False)
        estimator.evaluate(input_fn=dev_input_fn,steps=dev_steps)

    if params['do_predict']:
        predict_examples=processor.get_test_data(params['test_examples'])
        num_test_examples=len(predict_examples)
        tf.logging.info("**** running predict ****")
        tf.logging.info("Num examples %d", num_test_examples)
        test_input_fn=bert_input_fn_builder(predict_examples,batch_size=params['batch_size'],is_training=False)
        result=estimator.predict(input_fn=test_input_fn)
        output_file = os.path.join(params['output_dir'], "label_test.txt")

        with open(output_file, "w", encoding='utf-8') as writer:
            tf.logging.info("***** Predict results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__=='__main__':
    run_ner_bert(ner_model=Bert)
    #run_ner(ner_model=Bi_LSTM)
