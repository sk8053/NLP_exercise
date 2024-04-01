######## https://github.com/bentrevett/pytorch-seq2seq
import torch
import torch.nn as nn
#import torch.optim as optim
import random
import numpy as np
import spacy
import datasets # Hugging face datasets
from datasets import Dataset as h_dataset
import torchtext

#import tqdm
#import evaluate
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

### spacy
### datasets
### nn.utils.rnn.pad_sequence
### torchtext.vocab.build_vocab_from_iterator

### Huggingface datasets
### https://huggingface.co/docs/datasets/en/process

## set random seeds
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

def get_all_data(batch_size=128):
    ## load dataset

    dataset = datasets.load_dataset("bentrevett/multi30k")

    ## splite data as train, test and validation sets.

    train_data, valid_data, test_data = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )


    ## load English and German languate with spacy ##
    en_nlp = spacy.load("en_core_web_sm")
    de_nlp = spacy.load("de_core_news_sm")


    ## tokenizer function ##
    def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):

        en_tokens = [token.text for token in en_nlp.tokenizer(example["en"])][:max_length]
        de_tokens = [token.text for token in de_nlp.tokenizer(example["de"])][:max_length]
        if lower:
            en_tokens = [token.lower() for token in en_tokens]
            de_tokens = [token.lower() for token in de_tokens]
        en_tokens = [sos_token] + en_tokens + [eos_token]
        de_tokens = [sos_token] + de_tokens + [eos_token]
        return {"en_tokens": en_tokens, "de_tokens": de_tokens}

    ### obtain train, valid, test data using tokenizer function ###
    max_length = 1_000
    lower = True
    sos_token = "<sos>"
    eos_token = "<eos>"

    fn_kwargs = {
        "en_nlp": en_nlp,
        "de_nlp": de_nlp,
        "max_length": max_length,
        "lower": lower,
        "sos_token": sos_token,
        "eos_token": eos_token,
    }

    train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)


    ##### Create vocabulary set ####
    min_freq = 2
    unk_token = "<unk>"
    pad_token = "<pad>"

    special_tokens = [
        unk_token,
        pad_token,
        sos_token,
        eos_token,
    ]

    en_vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["en_tokens"],
        min_freq=min_freq,
        specials=special_tokens,
    )

    de_vocab = torchtext.vocab.build_vocab_from_iterator(
        train_data["de_tokens"],
        min_freq=min_freq,
        specials=special_tokens,
    )

    assert en_vocab[unk_token] == de_vocab[unk_token]
    assert en_vocab[pad_token] == de_vocab[pad_token]

    unk_index = en_vocab[unk_token]
    pad_index = en_vocab[pad_token]
    en_vocab.set_default_index(unk_index)
    de_vocab.set_default_index(unk_index)

    ######### find the index of each word per sentence using vocab-dictionary #####
    def numericalize_example(example, en_vocab, de_vocab):
        en_ids = en_vocab.lookup_indices(example["en_tokens"])
        de_ids = de_vocab.lookup_indices(example["de_tokens"])
        return {"en_ids": en_ids, "de_ids": de_ids}
    fn_kwargs = {"en_vocab": en_vocab, "de_vocab": de_vocab}

    train_data = train_data.map(numericalize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(numericalize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(numericalize_example, fn_kwargs=fn_kwargs)

    ### create data format #####
    data_type = "torch"
    format_columns = ["en_ids", "de_ids"]

    train_data = train_data.with_format(
        type=data_type, columns=format_columns, output_all_columns=True
    )

    valid_data = valid_data.with_format(
        type=data_type,
        columns=format_columns,
        output_all_columns=True,
    )

    test_data = test_data.with_format(
        type=data_type,
        columns=format_columns,
        output_all_columns=True,
    )

    ########### get collate_function ####
    def get_collate_fn(pad_index):
        def collate_fn(batch):
            batch_en_ids = [example["en_ids"] for example in batch]
            batch_de_ids = [example["de_ids"] for example in batch]
            batch_en_ids = nn.utils.rnn.pad_sequence(batch_en_ids, padding_value=pad_index)
            batch_de_ids = nn.utils.rnn.pad_sequence(batch_de_ids, padding_value=pad_index)
            batch = {
                "en_ids": batch_en_ids,
                "de_ids": batch_de_ids,
            }
            return batch

        return collate_fn

    ### create dataloader ####
    def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
        collate_fn = get_collate_fn(pad_index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )
        return data_loader


    
    train_data_loader = get_data_loader(train_data, batch_size, pad_index, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, batch_size, pad_index)
    test_data_loader = get_data_loader(test_data, batch_size, pad_index)
    
    input_dim = len(de_vocab)
    output_dim = len(en_vocab)
    
    #src_pad_idx = de_vocab.vocab.get_stoi()
    
    extra_info = dict(input_dim = input_dim, 
                    output_dim = output_dim, 
                    pad_index = pad_index,
                    test_data = test_data,
                    en_nlp = en_nlp,
                    de_nlp = de_nlp,
                    en_vocab = en_vocab,
                    de_vocab = de_vocab,
                    lower = lower,
                    sos_token = sos_token,
                    eos_token = eos_token
                    )
    
    return train_data_loader, valid_data_loader, test_data_loader, extra_info