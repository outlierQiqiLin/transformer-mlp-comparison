# data/sst2_loader.py

from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader

# 建议使用 BERT base uncased 的 tokenizer，因为它是一个高效的基线
TOKENIZER_NAME = 'bert-base-uncased'
MAX_SEQ_LENGTH = 128 # 序列最大长度，用于padding/truncation

def get_sst2_tokenizer():
    """返回用于 SST-2 的分词器。"""
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)

def tokenize_function(examples, tokenizer):
    """对数据集中的文本进行分词和编码。"""
    # 'sentence' 是 SST-2 数据集中的文本列名
    return tokenizer(
        examples['sentence'], 
        truncation=True, 
        padding='max_length', 
        max_length=MAX_SEQ_LENGTH
    )

def load_sst2_data(batch_size, num_workers=0):
    """
    加载、预处理 SST-2 数据集，并返回 PyTorch DataLoaders。
    """
    # 1. 加载数据集和 Tokenizer
    dataset = load_dataset('glue', 'sst2')
    tokenizer = get_sst2_tokenizer()

    # 2. 对所有 split 应用分词函数
    # 使用 map 函数并行处理，并删除原始文本列
    tokenized_datasets = dataset.map(
        lambda x: tokenize_function(x, tokenizer), 
        batched=True,
        remove_columns=["sentence", "idx"]
    )
    
    # 3. 格式化为 PyTorch Tensor
    tokenized_datasets.set_format("torch")
    
    # 4. 创建 DataLoaders
    # 训练集需要打乱 (shuffle=True)
    train_dataloader = DataLoader(
        tokenized_datasets["train"], 
        shuffle=True, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 验证集/测试集不需要打乱
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], 
        shuffle=False, 
        batch_size=batch_size,
        num_workers=num_workers
    )

    # 5. 返回 DataLoader 和 Tokenizer
    return train_dataloader, eval_dataloader, tokenizer