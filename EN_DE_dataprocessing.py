import torch
import torch.nn as nn
import random
import spacy
import datasets # Hugging face datasets
from datasets import Dataset as h_dataset
import torchtext
from tqdm import tqdm
#from torch.utils.data import DataLoader, Dataset
seed = 1234
random.seed(seed)
#np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True


    
class EN_DE_proccessing:

    def __init__(self, dataset, batch_size = 128, 
                      unk_token = '<unk>', 
                     pad_token = '<pad>', sos_token = '<sos>', 
                     eos_token = '<eos>',
                     lower = True):
        
        '''
        print(len(dataset["train"].data['en']))
        print(dataset["train"]['en'][:10])
        print(dataset["train"]['de'][:10])
        '''
        train_data, test_data, valid_data = dataset['train'], dataset['test'], dataset['validation']

        self.unk_token =unk_token
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token

        ## load English and German language processing tools from spacy ##
        ### spacy is used to process the whole sentence
        self.en_nlp = spacy.load("en_core_web_sm")
        self.de_nlp = spacy.load("de_core_news_sm")

        # 1. tokenize sentences
        nlp_process_tools = {'en_nlp':self.en_nlp, 'de_nlp':self.de_nlp} # NLP processing spacy tools
        print('tokenizing all sentences ... \n')
        train_data_tokens = self.tokenize_sentences(train_data, **nlp_process_tools)
        test_data_tokens = self.tokenize_sentences(test_data, **nlp_process_tools)
        valid_data_tokens = self.tokenize_sentences(valid_data, **nlp_process_tools)

        # 2. create the vocabulary set
        print('creating vocabulary set ... \n')
        total_data = dict()
        total_data['en_tokens'] = train_data_tokens['en_tokens'] + test_data_tokens['en_tokens'] + valid_data_tokens['en_tokens']
        total_data['de_tokens'] = train_data_tokens['de_tokens'] + test_data_tokens['de_tokens'] + valid_data_tokens['de_tokens']
        self.en_vocab_dict, self.de_vocab_dict, pad_index = self.create_vocab(data = total_data, min_freq = 2)
        

        # 3. map tokens in sentences to the corresponding indices in vocabulary dictionary
        vocabulary_dictionaries = {'en_vocab_dict':self.en_vocab_dict, 'de_vocab_dict':self.de_vocab_dict}
        ## update train, test, valid data by mapping tokens to indices 
        print('mapping all tokens to indices  ... \n')
        self.train_data_ids = self.map_tokens_to_indices(train_data_tokens, **vocabulary_dictionaries)
        self.test_data_ids = self.map_tokens_to_indices(test_data_tokens, **vocabulary_dictionaries)
        self.valid_data_ids = self.map_tokens_to_indices(valid_data_tokens, **vocabulary_dictionaries)

        # 4. add pads to all the sentences and create dataloader with batch size
        self.train_dataloader = self.get_data_loader(self.train_data_ids, batch_size = batch_size, pad_index = pad_index, shuffle=True)
        self.test_dataloader = self.get_data_loader(self.test_data_ids, batch_size = batch_size, pad_index = pad_index)
        self.valid_dataloader = self.get_data_loader(self.valid_data_ids, batch_size = batch_size, pad_index = pad_index)

        ## extra info
        self.input_dim, self.output_dim = len(self.de_vocab_dict), len(self.en_vocab_dict)
        self.lower = lower
        self.pad_index = pad_index

    def tokenize_sentences(self, data, en_nlp, de_nlp, lower_ch = True, 
                           max_length=1000, sos_token='<sos>', eos_token = '<eos>'):
        
        new_data_with_tokens = {'en_tokens':[], 'de_tokens':[]}

        for data_en_i, data_de_i in tqdm(zip(data['en'], data['de']), total = len(data['en']), ascii=True, desc = 'number of sentences'):

            en_tokens = [token.text for token in en_nlp.tokenizer(str(data_en_i))][:max_length]
            de_tokens = [token.text for token in de_nlp.tokenizer(str(data_de_i))][:max_length]
            if lower_ch is True:
                en_tokens = [token.lower() for token in en_tokens]
                de_tokens = [token.lower() for token in de_tokens]
                
            en_tokens = [sos_token] + en_tokens + [eos_token] # append start and end tokens
            de_tokens = [sos_token] + de_tokens + [eos_token]

            new_data_with_tokens['en_tokens'].append(en_tokens)
            new_data_with_tokens['de_tokens'].append(de_tokens)

        return new_data_with_tokens
    
    def create_vocab(self, data, min_freq = 2):
        
        # unk_token unknown token, which would be used if we find any word that is out of vocabulary set. 
        # pad_token  pad token, to make the length of each setence the same
        # sos_token: token indicating 'start' of setence
        # eos_token: token indicating 'end' of setence

        special_tokens = [
            self.unk_token,
            self.pad_token,
            self.sos_token,
            self.eos_token,
           ]

            # https://pytorch.org/text/stable/vocab.html
            # torchtext.vocab.build_vocab_from_iterator 
            # Build a Vocab from an iterator
            # iterator – Iterator used to build Vocab. Must yield list or iterator of tokens.
            #min_freq – The minimum frequency needed to include a token in the vocabulary.
            #specials – Special symbols to add. The order of supplied tokens will be preserved.
            #special_first – Indicates whether to insert symbols at the beginning or at the end.
            #max_tokens – If provided, creates the vocab from the max_tokens - len(specials) most frequent 

            ## return : A Vocab object

        en_vocab = torchtext.vocab.build_vocab_from_iterator(
            data["en_tokens"], # list
            min_freq=min_freq,
            specials=special_tokens,
        )

        de_vocab = torchtext.vocab.build_vocab_from_iterator(
            data["de_tokens"], # list
            min_freq=min_freq,
            specials=special_tokens,
        )

        assert en_vocab[self.unk_token] == de_vocab[self.unk_token]
        assert en_vocab[self.pad_token] == de_vocab[self.pad_token]

        unk_index = en_vocab[self.unk_token]
        pad_index = en_vocab[self.pad_token]

        ### torchtext.vocab.set_default_index 
        ## Value of default index. This index will be returned when OOV token is queried.
        ## OOV: Out of Vocabulary tokens
        en_vocab.set_default_index(unk_index)
        de_vocab.set_default_index(unk_index)

        return en_vocab, de_vocab, pad_index
    
    def map_tokens_to_indices(self, token_sentences, en_vocab_dict, de_vocab_dict):
        # token_sentences: setences that are tokenized: the set of lists having different length
        # en_vocab_dict: dictionary collecting all the Enlish words from the given data
        # de_vocab_dict: dictionary collecting all the German words from the given data

        # map tokens of sentences to the corresponding indices in vocabulary dictionary
        vocab_indices = dict(en_ids = [], de_ids = [])
        
        for en_vocab, de_vocab in tqdm(zip(token_sentences['en_tokens'], token_sentences['de_tokens']), 
                                       desc = 'number of setences', ascii= True, total = len(token_sentences['en_tokens'])):
            en_ids = en_vocab_dict.lookup_indices(en_vocab)
            de_ids = de_vocab_dict.lookup_indices(de_vocab)

            vocab_indices['en_ids'].append(en_ids)
            vocab_indices['de_ids'].append(de_ids)

        return vocab_indices
    
    def add_pads_to_setences(self, data, pad_value = 1):
        # add padd to tokenized setences so that each setence has the same lenght of tokens. 
        L = len(data['en_ids'])
        
        assert L == len(data['de_ids'])
        
        _en_ids = [torch.tensor(data['en_ids'][i]) for i in range(L)]
        _de_ids = [torch.tensor(data['de_ids'][i]) for i in range(L)]

        en_ids_with_pads = nn.utils.rnn.pad_sequence(_en_ids, padding_value = pad_value)
        de_ids_with_pads = nn.utils.rnn.pad_sequence(_de_ids, padding_value = pad_value)

        return {'en_ids':en_ids_with_pads.T, 'de_ids':de_ids_with_pads.T}
    

    ########### get collate_function ####
    def get_collate_fn(self, pad_index):
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
    def get_data_loader(self, dataset, batch_size, pad_index, shuffle=False):
        # converting dictionary to huggingface data and formating 
        data_type = "torch"
        format_columns = ["en_ids", "de_ids"]
        dataset = h_dataset.from_dict(dataset)

        dataset = dataset.with_format(
            type=data_type, columns=format_columns, output_all_columns=True)
        
        collate_fn = self.get_collate_fn(pad_index)
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
        )

        return data_loader

if __name__  == '__main__':
     ## load dataset
    dataset = datasets.load_dataset("bentrevett/multi30k")
    en_de_proc = EN_DE_proccessing(dataset=dataset, batch_size=128)
    train_dataloader = en_de_proc.train_dataloader
    print(len(train_dataloader))
    for i, train_data in enumerate(train_dataloader):
        print(train_data['en_ids'].shape)# shape = [sentence length, batch size]
