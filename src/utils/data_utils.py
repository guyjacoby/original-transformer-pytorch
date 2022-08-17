import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
from datasets import load_dataset
from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from typing import Optional
from datargs import parse, arg
from dataclasses import dataclass
from loguru import logger

from src.utils.constants import *


def get_dataset(split):
    return load_dataset(path="wmt14", name='de-en', cache_dir=DATA_CACHE_PATH, split=split)


def initialize_tokenizer(max_token_length: int):
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN, end_of_word_suffix=SUFFIX))

    # padding and truncation
    tokenizer.enable_padding(pad_id=3, pad_token=PAD_TOKEN)
    tokenizer.enable_truncation(max_length=max_token_length)

    # normalization
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    # pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # decoder
    tokenizer.decoder = decoders.BPEDecoder(suffix=SUFFIX)

    return tokenizer


def train_bpe_tokenizer(train_size: int, vocab_size: int, max_token_length: int):
    logger.info('Initializing BPE tokenizer...')
    logger.info(f'train size = {train_size} | vocab size = {vocab_size}  max token length = {max_token_length}')
    tokenizer = initialize_tokenizer(max_token_length)
    trainer = BpeTrainer(vocab_size=vocab_size,
                         show_progress=True,
                         special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN],
                         end_of_word_suffix=SUFFIX)
    train_set = get_dataset(split='train')
    train_set = train_set.flatten()[:train_size]

    def batch_iterator(batch_size=5000):
        for i in range(0, len(train_set), batch_size):
            yield train_set[i: i + batch_size]['translation.en'] + train_set[i: i + batch_size]['translation.de']

    logger.info('Training tokenizer...')
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(train_set))
    logger.info('Finished training tokenizer')
    return tokenizer


def save_tokenizer(tokenizer, tokenizer_path: Path = TOKENIZER_PATH):
    tokenizer.save(str(tokenizer_path / 'tokenizer.json'))
    logger.info(f'Saved tokenizer to {tokenizer_path.absolute()}')


def load_tokenizer(tokenizer_path: Path = TOKENIZER_PATH):
    return Tokenizer.from_file(str(tokenizer_path / 'tokenizer.json'))


def tokenize_batch(tokenizer, batch, is_source, is_pretokenized=False, add_special_tokens=True):
    if is_source:
        tokenizer.post_processor = TemplateProcessing(single="$0 " + EOS_TOKEN,
                                                      pair=None,
                                                      special_tokens=[(EOS_TOKEN, 2)])
    else:
        tokenizer.post_processor = TemplateProcessing(single=BOS_TOKEN + " $0 " + EOS_TOKEN,
                                                      pair=None,
                                                      special_tokens=[(BOS_TOKEN, 1), (EOS_TOKEN, 2)])

    encodings = tokenizer.encode_batch(batch, is_pretokenized=is_pretokenized, add_special_tokens=add_special_tokens)
    ids = [enc.ids for enc in encodings]
    mask = [enc.attention_mask for enc in encodings]

    return torch.tensor(ids, dtype=torch.int), torch.tensor(mask, dtype=torch.bool)


def create_target_mask(tgt_pad_mask):
    batch_size, tgt_seq_length = tgt_pad_mask.shape
    tgt_pad_mask = tgt_pad_mask.reshape(batch_size, 1, 1, tgt_seq_length) == 1
    tgt_future_mask = torch.ones((1, 1, tgt_seq_length, tgt_seq_length)).tril() == 1
    return tgt_pad_mask & tgt_future_mask


def collate_fn(batch):
    # load tokenizer
    tokenizer = load_tokenizer(TOKENIZER_PATH)

    # batch[0] because the datapipe already provides the dataloader batched sentences
    # so the dataloader is set to batch_size=1 which has a batch inside it
    source = [pair[0] for pair in batch[0]]
    target = [pair[1] for pair in batch[0]]

    # encode source - not using BOS for source language (is there a benefit to using it?)
    src_ids, src_mask = tokenize_batch(tokenizer, source, is_source=True)

    # encode target - using both BOS and EOS
    tgt_ids, tgt_pad_mask = tokenize_batch(tokenizer, target, is_source=False)

    # target ids for input are shifted by one using BOS, compared to target ids as labels without BOS
    # target ids for label are reshaped to accommodate loss fn that expects (sample, vocab)
    tgt_ids_input = tgt_ids[:, :-1]
    tgt_pad_mask = tgt_pad_mask[:, :-1]
    tgt_ids_label = tgt_ids[:, 1:].reshape(-1, 1)

    # create pad and future mask for target
    tgt_mask = create_target_mask(tgt_pad_mask)

    # reshape src masks for attention calculations
    batch_size, src_seq_length = src_ids.shape
    src_mask = src_mask.reshape(batch_size, 1, 1, src_seq_length) == 1

    return src_ids, tgt_ids_input, tgt_ids_label, src_mask, tgt_mask


def sort_key(bucket):
    return [bucket[i] for i in np.argsort([len(pair[0]) for pair in bucket])]


def _get_dataloader(dataset_type: str, dataset_size: Optional[int], batch_size: int):
    if Path(TOKENIZER_PATH / 'tokenizer.json').exists():
        dataset = get_dataset(split=dataset_type)
        flat_dataset = dataset.flatten()

        # -1 for all elements in list of samples
        if dataset_size is None:
            dataset_size = -1

        # create train data pipeline and dataloader
        dp = IterableWrapper(zip(flat_dataset['translation.en'][:dataset_size],
                                 flat_dataset['translation.de'][:dataset_size]))
        batch_dp = dp.bucketbatch(batch_size=batch_size, drop_last=False, batch_num=100, bucket_num=100,
                                  sort_key=sort_key)
        dataloader = DataLoader(dataset=batch_dp, shuffle=True, collate_fn=collate_fn)
        return dataloader

    else:
        raise Exception(f'No tokenizer found, please train one first by running {Path(__file__)}.')


def get_dataloaders(train_size: int, val_size: Optional[int], batch_size: int):
    train_dataloader = _get_dataloader(dataset_type='train', dataset_size=train_size, batch_size=batch_size)
    val_dataloader = _get_dataloader(dataset_type='validation', dataset_size=val_size, batch_size=batch_size)
    return train_dataloader, val_dataloader


@dataclass
class TrainTokenizerParams:
    train_size: int = arg(default=400_000, help="number of sentence pairs to train tokenizer on, default is 400K")
    vocab_size: int = arg(default=TOKENIZER_VOCAB_SIZE, help=f"tokenizer vocabulary size, default is "
                                                             f"{TOKENIZER_VOCAB_SIZE}")
    max_token_length: int = arg(default=MAX_TOKEN_LEN, help=f"maximum number of tokens in tokenized sentence, default "
                                                            f"is {MAX_TOKEN_LEN}")


def main():
    # train new tokenizer on WMT16
    train_tokenizer_params = parse(TrainTokenizerParams)
    tokenizer = train_bpe_tokenizer(train_size=train_tokenizer_params.train_size,
                                    vocab_size=train_tokenizer_params.vocab_size,
                                    max_token_length=train_tokenizer_params.max_token_length)
    save_tokenizer(tokenizer)


if __name__ == "__main__":
    main()
