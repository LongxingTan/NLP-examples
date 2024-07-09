"""
huggingface: https://huggingface.co/learn/nlp-course/chapter3/1?fw=pt
onnx: https://github.com/huggingface/notebooks/blob/c267eeecdbdd708f454c59405e69d7e657310ddf/examples/text_classification_quantization_ort.ipynb#L561
"""

import re
from pathlib import Path
import os
import shutil
from functools import partial
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


class CFG:
    MODEL_NAME_OR_PATH = "hfl/chinese-roberta-wwm-ext"
    NUM_LABELS = 2
    MAX_LENGTH = 64
    LOGGING_DIR = "logging_dir"
    MODEL_DIR = "model_result"


def cleanquestion(x: str) -> str:
    """去除文本中的符号 仅保留中文、英文、数字等"""
    str_text = re.sub(
        u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", x)
    return str_text


def preprocess(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=CFG.MAX_LENGTH)


def build_data(tokenizer):
    text_dataset = load_dataset('csv', data_files={
        'train': ['data_all/data/train_data.csv'],
        'test': ['data_all/data/test_data.csv']})

    # 在实际工程中，会先使用`Tokenizer`把所有的文本转换成`input_ids`,`token_type_ids`,`attention_mask`，然后在训练的时候，这步就不再做了，目的是减少训练过程中cpu处理数据的时间，不给显卡休息时间。
    tokenized_text = text_dataset.map(partial(preprocess, tokenizer=tokenizer), batched=True)
    return tokenized_text


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train():
    shutil.rmtree(CFG.LOGGING_DIR, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME_OR_PATH)

    dataset = build_data(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, max_length=CFG.MAX_LENGTH)

    model = AutoModelForSequenceClassification.from_pretrained(CFG.MODEL_NAME_OR_PATH, num_labels=CFG.NUM_LABELS)

    training_args = TrainingArguments(
        output_dir=CFG.MODEL_DIR,
        overwrite_output_dir=True,
        logging_dir=CFG.LOGGING_DIR,
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        do_eval=True,
        evaluation_strategy="steps",
        eval_accumulation_steps=50,
        eval_steps=50,
        logging_steps=50,
        save_steps=100,
        num_train_epochs=4,
        weight_decay=0.01,
        save_total_limit=3,
        jit_mode_eval=True,
        fp16=True,
        fp16_opt_level='O3',
        load_best_model_at_end=True,  # 最后，加载效果最好的模型
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()


if __name__ == '__main__':
    train()
