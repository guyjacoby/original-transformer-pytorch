import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
import datasets
from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
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

    return tokenizer


def train_bpe_tokenizer(tokenizer_path=TOKENIZER_PATH, cache_path=DATA_CACHE_PATH):
    tokenizer = initialize_tokenizer()
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


def tokenize_batch(tokenizer, batch, is_source, is_pretokenized=False):
    if is_source:
        tokenizer.post_processor = TemplateProcessing(single="$0 [EOS]", special_tokens=[("[EOS]", 2)])
    else:
        tokenizer.post_processor = TemplateProcessing(single="[BOS] $0 [EOS]",
                                                      special_tokens=[("[BOS]", 1), ("[EOS]", 2)])
    encodings = tokenizer.encode_batch(batch, is_pretokenized=is_pretokenized)
    ids = [enc.ids for enc in encodings]
    mask = [enc.attention_mask for enc in encodings]

    return torch.tensor(ids, dtype=torch.short), torch.tensor(mask, dtype=torch.bool)


def collate_fn(batch):
    # load tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # batch[0] because the datapipe already provides the dataloader batched sentences
    # so the dataloader is set to batch_size=1 which has a batch inside it
    source = [pair[0] for pair in batch[0]]
    target = [pair[1] for pair in batch[0]]

    # encode source - not using BOS for source language (is there a benefit to using it?)
    src_ids, src_mask = tokenize_batch(tokenizer, source, True)

    # encode target - using both BOS and EOS
    tgt_ids, tgt_pad_mask = tokenize_batch(tokenizer, target, False)

    # target ids for input are shifted by one using BOS, compared to target ids as labels without BOS
    tgt_ids_input = tgt_ids[:, :-1]
    tgt_pad_mask = tgt_pad_mask[:, :-1]
    tgt_ids_output = tgt_ids[:, 1:]

    # reshape masks for attention calculations
    batch_size, src_seq_length = src_ids.shape
    tgt_seq_length = tgt_ids_input.shape[1]
    src_mask = src_mask.reshape(batch_size, 1, 1, src_seq_length) == 1
    tgt_pad_mask = tgt_pad_mask.reshape(batch_size, 1, 1, tgt_seq_length) == 1
    tgt_future_mask = torch.ones((1, 1, tgt_seq_length, tgt_seq_length)).tril() == 1
    tgt_mask = tgt_pad_mask & tgt_future_mask

    return src_ids, tgt_ids_input, tgt_ids_output, src_mask, tgt_mask


def sort_key(bucket):
    return [bucket[i] for i in np.argsort([len(pair[0]) for pair in bucket])]


def get_data_loaders(cache_path=DATA_CACHE_PATH, batch_size=10):
    if Path(TOKENIZER_PATH / 'tokenizer.json').is_file():

        # create train data pipeline and dataloader from 2015/2016
        training_data = pd.DataFrame()
        for year in ['2015', '2016']:
            df = get_dataset(cache_path=cache_path, year=year)['train'].flatten().to_pandas()
            training_data = pd.concat([training_data, df])

        train_dp = IterableWrapper(
            zip(training_data['translation.en'].values.tolist(), training_data['translation.de'].values.tolist()))
        train_batch_dp = train_dp.bucketbatch(batch_size=batch_size, drop_last=False, batch_num=100, bucket_num=100,
                                              sort_key=sort_key)
        train_loader = DataLoader(dataset=train_batch_dp, shuffle=True, collate_fn=collate_fn)

        # create eval data pipeline and dataloader from 2014
        eval_data = get_dataset(cache_path=cache_path, year='2014')['train'].flatten().to_pandas()
        eval_dp = IterableWrapper(eval_data['translation.en'].values.tolist(),
                                  eval_data['translation.de'].values.tolist())
        eval_batch_dp = eval_dp.bucketbatch(batch_size=batch_size, drop_last=False, batch_num=100, bucket_num=100,
                                            sort_key=sort_key)
        eval_loader = DataLoader(dataset=eval_batch_dp, shuffle=True, collate_fn=collate_fn)

        return train_loader, eval_loader

    else:
        raise Exception(f'No tokenizer found, please train one first by running {Path(__file__)}.')


if __name__ == "__main__":

    # train new tokenizer on iwslt 2015,2016
    train_bpe_tokenizer()
    print(f'Trained and saved the BPE tokenizer.')

    # train, eval = get_data_loaders()
    # a = next(iter(train))