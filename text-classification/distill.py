import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from transformers.trainer import Trainer


class Teacher(object):
    def __init__(self, model_path, max_len):
        self.tokenizer = AutoTokenizer()
        self.model = AutoModel()
        self.model.eval()

    def predict(self):
        return


class DistilTrainer(Trainer):
    def __init__(self, teacher_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model

    def compute_loss(self, model, inputs, return_outputs=False):
        query, passage, pair = inputs
        student_scores = model(query=query, passage=passage).scores
        with torch.no_grad():
            teacher_scores = self.teacher_model(pair=pair).scores

        teacher_mat = torch.zeros(student_scores.shape, dtype=student_scores.dtype, device=teacher_scores.device)
        index = torch.arange(teacher_scores.size(0), device=teacher_scores.device)
        teacher_scores = torch.softmax(teacher_scores.view(student_scores.size(0), -1) / self.args.teacher_temp, dim=1,
                                       dtype=student_scores.dtype)
        teacher_mat = torch.scatter(teacher_mat,
                                    dim=-1,
                                    index=index.view(student_scores.size(0), -1),
                                    src=teacher_scores)
        student_scores = nn.functional.log_softmax(student_scores / self.args.student_temp, dim=1)
        loss = nn.functional.kl_div(student_scores, teacher_mat, reduction='batchmean')
        return loss
