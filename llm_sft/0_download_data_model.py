from modelscope.msdatasets import MsDataset
from modelscope import snapshot_download, AutoModel, AutoTokenizer

# 数据
ds = MsDataset.load('AI-ModelScope/DISC-Law-SFT', subset_name='default', split='train', cache_dir='../inputs')
train_dataset = ds.to_hf_dataset()
train_dataset.to_json('../inputs/train_data.json')

# 模型
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-xcomposer-7b', cache_dir='/root/model', revision='master')


def gen_batches_train():
    # https://plainenglish.io/community/fine-tuning-mistral-7b-model-with-your-custom-data-010eb6
    ds = MsDataset.load('AI-ModelScope/DISC-Law-SFT', subset_name='default', split='train', cache_dir='../inputs')
    total_samples = 10000
    val_pct = 0.1
    train_limit = int(total_samples * (1 - val_pct))
    counter = 0
    trainbatch=[]

    for sample in iter(ds):
        if counter >= train_limit:
            break

        original_prompt = sample['prompt'].replace("### Input:\n", '').replace('# Python code\n', '')
        instruction_start = original_prompt.find("### Instruction:") + len("### Instruction:")
        # prompt has ### Input\n which i want to remove
        instruction_end = original_prompt.find("### Output:")

        instruction = original_prompt[instruction_start:instruction_end].strip()
        content_start = original_prompt.find("### Output:") + len("### Output:")
        content = original_prompt[content_start:].strip()
        new_text_format = f'<s>[INST] {instruction} [/INST] ```python\n{content}```</s>'

        tokenized_output = tokenizer(new_text_format)
        #  yield {''text'': new_text_format}
        trainbatch.append({'text': new_text_format})

        counter += 1

    return trainbatch


def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": schema},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }
