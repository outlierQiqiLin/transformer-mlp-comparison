# utils/data_loader_text.py
"""
用于文本任务 (如 SST-2) 的数据加载器
"""
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

def get_dataloader_sst2(config):
    """为 SST-2 创建数据加载器"""
    
    dataset_name = config.get('dataset.name')
    if dataset_name != 'sst2':
        raise ValueError("此加载器仅适用于 SST-2")

    tokenizer_name = config.get('dataset.tokenizer_name')
    max_seq_len = config.get('dataset.max_seq_len')
    batch_size = config.get('training.batch_size')
    num_workers = config.get('training.num_workers', 4)

    # 1. 加载分词器
    print(f"📊 加载分词器: {tokenizer_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as e:
        print(f"❌ 自动下载分词器失败: {e}")
        raise
        
    vocab_size = tokenizer.vocab_size

    # 2. 加载数据集
    print("📊 加载 SST-2 数据集...")
    try:
        dataset = load_dataset('glue', 'sst2')
    except Exception as e:
        print(f"❌ 自动下载 SST-2 失败: {e}")
        raise

    # 3. 定义预处理函数
    def preprocess_function(examples):
        # Hugging Face 'datasets' 会自动处理 'sentence' 列
        tokenized = tokenizer(
            examples['sentence'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_seq_len
        )
        # 将 'label' 列重命名为 'labels' 以便与 PyTorch 兼容
        tokenized['labels'] = examples['label']
        return tokenized

    print("🤖 预处理数据集...")
    processed_dataset = dataset.map(
        preprocess_function, 
        batched=True,
        remove_columns=['sentence', 'idx', 'label'] # 移除原始列
    )

    # 4. 设置 PyTorch 格式
    processed_dataset.set_format(
        type='torch', 
        columns=['input_ids', 'attention_mask', 'labels']
    )

    # 5. 创建 DataLoaders
    train_dataset = processed_dataset['train']
    test_dataset = processed_dataset['validation'] # SST-2 的测试集没有标签，我们用验证集

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"   词汇表大小 (Vocab Size): {vocab_size}")
    
    # 返回 vocab_size，模型创建时需要
    return train_loader, test_loader, vocab_size