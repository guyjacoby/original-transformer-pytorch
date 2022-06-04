import numpy as np
import torch
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import IterableWrapper
import datasets
from tokenizers import Tokenizer, normalizers, decoders
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from .constants import *

TOKENIZER_VOCAB_SIZE = 37_000


def get_dataset(split, cache_path=DATA_CACHE_PATH):
    """
    Download and/or load the WMT16 English/German dataset from the HuggingFace repository.

    Args:
        cache_path: Path of directory to write/read the dataset

    Returns: Tuple of flattened dataset splits (each is a PyArrow Dataset object)

    """
    return datasets.load_dataset(path="wmt16", name='de-en', cache_dir=cache_path, split=split)


def initialize_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token=UNK_TOKEN, end_of_word_suffix=SUFFIX))

    # padding and truncation
    tokenizer.enable_padding(pad_id=3, pad_token=PAD_TOKEN)
    tokenizer.enable_truncation(max_length=MAX_TOKEN_LEN)

    # normalization
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    # pre-tokenization
    tokenizer.pre_tokenizer = Whitespace()

    # decoder
    tokenizer.decoder = decoders.BPEDecoder(suffix=SUFFIX)

    return tokenizer


def train_bpe_tokenizer(tokenizer_path=TOKENIZER_PATH, cache_path=DATA_CACHE_PATH):
    tokenizer = initialize_tokenizer()
    trainer = BpeTrainer(vocab_size=TOKENIZER_VOCAB_SIZE,
                         show_progress=True,
                         special_tokens=[UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN],
                         end_of_word_suffix=SUFFIX)

    train_set = get_dataset(split=('train'), cache_path=cache_path)
    train_set = train_set.flatten()

    # noinspection PyTypeChecker
    def batch_iterator(batch_size=5000):
        for i in range(0, len(train_set), batch_size):
            yield train_set[i: i + batch_size]['translation.en'] + train_set[i: i + batch_size]['translation.de']

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(train_set))

    tokenizer.save(str(Path(tokenizer_path / 'tokenizer.json')))


def load_tokenizer(tokenizer_path):
    return Tokenizer.from_file(str(Path(tokenizer_path / 'tokenizer.json')))


def tokenize_batch(tokenizer , batch, is_source, is_pretokenized=False, add_special_tokens=True):
    if is_source:
        tokenizer.post_processor = TemplateProcessing(single="$0 " + EOS_TOKEN, special_tokens=(EOS_TOKEN, 2))
    else:
        tokenizer.post_processor = TemplateProcessing(single=BOS_TOKEN + " $0 " + EOS_TOKEN,
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


def get_data_loaders(batch_size, cache_path=DATA_CACHE_PATH):
    if Path(TOKENIZER_PATH / 'tokenizer.json').is_file():

        train_set, eval_set = get_dataset(split=('train', 'test'), cache_path=DATA_CACHE_PATH)
        train_set, eval_set = train_set.flatten(), eval_set.flatten()
        # create train data pipeline and dataloader
        train_dp = IterableWrapper(zip(train_set['translation.en'], train_set['translation.de']))
        train_batch_dp = train_dp.bucketbatch(batch_size=batch_size,
                                              drop_last=False,
                                              batch_num=100,
                                              bucket_num=100,
                                              sort_key=sort_key)
        train_loader = DataLoader(dataset=train_batch_dp, shuffle=True, collate_fn=collate_fn)

        # create eval data pipeline and dataloader
        eval_dp = IterableWrapper(zip(eval_set['translation.en'], eval_set['translation.de']))
        eval_batch_dp = eval_dp.bucketbatch(batch_size=batch_size,
                                            drop_last=False,
                                            batch_num=100,
                                            bucket_num=100,
                                            sort_key=sort_key)
        eval_loader = DataLoader(dataset=eval_batch_dp, shuffle=True, collate_fn=collate_fn)

        return train_loader, eval_loader

    else:
        raise Exception(f'No tokenizer found, please train one first by running {Path(__file__)}.')


if __name__ == "__main__":
    # train new tokenizer on iwslt 2015,2016
    train_bpe_tokenizer()
    print(f'Trained and saved the BPE tokenizer.')
