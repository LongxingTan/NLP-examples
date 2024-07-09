from transformers import AutoTokenizer, AutoModel
import torch
import onnxruntime


from optimum.onnxruntime.configuration import OptimizationConfig
from optimum.onnxruntime import ORTOptimizer, ORTModelForFeatureExtraction

model_id = "model/bge-base-zh-v1.5"
onnx_path = "model/bge_base_zh_v1_5_onnx"

model = ORTModelForFeatureExtraction.from_pretrained(model_id=model_id, from_transformers=True)
optimizer = ORTOptimizer.from_pretrained(model)

optimizer_config = OptimizationConfig(
    optimization_level=2,
    optimize_for_gpu=True,
    fp16=True
)
optimizer.optimize(save_dir=onnx_path, optimization_config=optimizer_config)


from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer,AutoModel
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from typing import List
import time
import numpy as np
import torch

model_id = "model/bge-base-zh-v1.5"
onnx_path = "model/bge_base_zh_v1_5_onnx"


def load_model_raw(model_id):
    model_raw = AutoModel.from_pretrained(model_id, device_map="cuda:0")
    # model_raw = model_raw.to(torch.float16)
    model_raw.eval()
    # model_raw.half()
    return model_raw


def load_model_ort(model_path):
    model = ORTModelForFeatureExtraction.from_pretrained(
        model_id=model_path,
        file_name="model_optimized.onnx",
        provider="CUDAExecutionProvider",
    )
    return model


model_raw = load_model_raw(model_id=model_id)
model_ort = load_model_ort(model_path=onnx_path)
tokenizer = AutoTokenizer.from_pretrained(model_id)


sentence1 = ["哈哈"]

tokenizer_output = tokenizer(sentence1, padding=True, truncation=True, return_tensors="pt")
# tokenizer_output

for k in tokenizer_output.keys():
    tokenizer_output[k] = tokenizer_output[k].cuda()

raw_o1 = model_raw(**tokenizer_output)
ort_o1 = model_ort(**tokenizer_output)
