import numpy as np
from torchdata.datapipes.iter import IterableWrapper
import torch
from torch.utils.data import DataLoader
import datasets
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from constants import *


def get_dataset(cache_path=DATA_CACHE_PATH, year='2016'):
    """
    Download and/or load the IWSLT Ted Talks English/German dataset from the HuggingFace repository.

    Args:
        cache_path: Path of directory to write/read the dataset

    Returns: loaded dataset as a HuggingFace DatasetDict object, which contains the dataset splits (each is a PyArrow Dataset object)

    """
    dataset = datasets.load_dataset(path="ted_talks_iwslt", cache_dir=cache_path, language_pair=("en", "de"), year=year)
    return dataset


def initialize_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN))

    # padding and truncation
    tokenizer.enable_padding(pad_id=3, pad_token=PAD_TOKEN)
    tokenizer.enable_truncation(max_length=MAX_TOKEN_LEN)

    # normalization
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    # pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # post-processing not needed for training, only for finalizing the tokenizer pipeline in the dataloader
    # tokenizer.post_processor = TemplateProcessing(single=BOS_TOKEN + " $0 " + EOS_TOKEN,
    #                                               special_tokens=[(BOS_TOKEN, 1), (EOS_TOKEN, 2)])
    return tokenizer


def train_bpe_tokenizer(tokenizer, tokenizer_path=TOKENIZER_PATH, cache_path=DATA_CACHE_PATH):
    trainer = BpeTrainer(special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN])

    sentences = []
    for year in ['2015', '2016']:
        dataset = get_dataset(cache_path=cache_path, year=year)
        for pair in dataset['train']['translation']:
            sentences.append(pair['en'])
            sentences.append(pair['de'])

    def batch_iterator(batch_size=1000):
        for i in range(0, len(sentences), batch_size):
            yield sentences[i: i + batch_size]

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(sentences))

    tokenizer.save(str(Path(tokenizer_path / 'tokenizer.json')))


def load_tokenizer(tokenizer_path):
    return Tokenizer.from_file(str(Path(tokenizer_path / 'tokenizer.json')))


def collate_fn(batch):
    # batch[0] because the datapipe already provides the dataloader batched sentences
    # so the dataloader is set to batch_size=1 which has a batch inside of it
    source = [pair[0] for pair in batch[0]]
    target = [pair[1] for pair in batch[0]]

    # source encodings - not using BOS for source language (is there a benefit to using it?)
    tokenizer.post_processor = TemplateProcessing(single="$0 [EOS]", special_tokens=[("[EOS]", 2)])
    source_encodings = tokenizer.encode_batch(source)
    src_ids = [src_enc.ids for src_enc in source_encodings]
    # src_tokens = [src_enc.tokens for src_enc in source_encodings]
    src_mask = [src_enc.attention_mask for src_enc in source_encodings]

    # target encodings
    tokenizer.post_processor = TemplateProcessing(single="[BOS] $0 [EOS]", special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
    target_encodings = tokenizer.encode_batch(target)
    tgt_ids = [tgt_enc.ids for tgt_enc in target_encodings]
    # tgt_tokens = [tgt_enc.tokens for tgt_enc in target_encodings]
    tgt_pad_mask = [tgt_enc.attention_mask for tgt_enc in target_encodings]

    src_ids, src_mask, tgt_ids, tgt_pad_mask = torch.tensor(src_ids), torch.tensor(src_mask), torch.tensor(
        tgt_ids), torch.tensor(tgt_pad_mask)

    # reshape masks for attention calculations
    batch_size, src_seq_length = src_ids.shape
    tgt_seq_length = tgt_ids.shape[1]

    src_mask = src_mask.reshape(batch_size, 1, 1, src_seq_length) == 1
    tgt_pad_mask = tgt_pad_mask.reshape(batch_size, 1, 1, tgt_seq_length) == 1
    tgt_future_mask = torch.ones((1, 1, tgt_seq_length, tgt_seq_length)).tril() == 1
    tgt_mask = tgt_pad_mask & tgt_future_mask

    return src_ids, tgt_ids, src_mask, tgt_mask


def get_data_loaders(cache_path=DATA_CACHE_PATH, batch_size=10):
    training_data = pd.DataFrame()

    for year in ['2015', '2016']:
        df = get_dataset(cache_path=cache_path, year=year)['train'].flatten().to_pandas()
        training_data = pd.concat([training_data, df])

    dp = IterableWrapper(zip(training_data['translation.en'].values.tolist(), training_data['translation.de'].values.tolist()))

    def sort_key(bucket):
        return [bucket[i] for i in np.argsort([len(pair[0]) for pair in bucket])]

    batch_dp = dp.bucketbatch(batch_size=batch_size, drop_last=False, batch_num=100, bucket_num=100, sort_key=sort_key)

    loader = DataLoader(dataset=batch_dp, shuffle=True, collate_fn=collate_fn)

    return loader


if __name__ == "__main__":

    # # train new tokenizer on iwslt 2015,2016
    tokenizer = initialize_tokenizer()
    train_bpe_tokenizer(tokenizer)

    # load trained tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # test dataloader
    train_loader = get_data_loaders()

    print(next(iter(train_loader)))