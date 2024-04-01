######## https://github.com/bentrevett/pytorch-seq2seq
import torch
import torch.nn as nn
#import random
import numpy as np
import datasets # Hugging face datasets
#import torchtext
import torch.optim as optim
from EN_DE_dataprocessing import EN_DE_proccessing
from seq2seq_transformer import Seq2Seq
import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

### spacy
### datasets
### nn.utils.rnn.pad_sequence
### torchtext.vocab.build_vocab_from_iterator

### Huggingface datasets
### https://huggingface.co/docs/datasets/en/process
batch_size = 128
dataset = datasets.load_dataset("bentrevett/multi30k")
en_de_proc = EN_DE_proccessing(dataset=dataset, batch_size=batch_size)

train_data_loader, test_data_loader, valid_data_loader = en_de_proc.train_dataloader, en_de_proc.test_dataloader, en_de_proc.valid_dataloader
input_dim, output_dim = en_de_proc.input_dim, en_de_proc.output_dim
print(f'input dimension is {input_dim}')
print(f'output dimension is {output_dim}')

pad_index = en_de_proc.pad_index
test_data = en_de_proc.test_data_ids


hidden_dim = 260
enc_layers = 5
dec_layers = 5
enc_heads = 10
dec_heads = 10
enc_pf_dim = 512
dec_pf_dim = 512
enc_dropout = 0.15
dec_dropout = 0.15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

model_params = dict(input_dim  = input_dim, 
                    output_dim = output_dim, 
                    hidden_dim = hidden_dim, 
                    enc_layers = enc_layers,
                    dec_layers = dec_layers,
                    dec_heads = dec_heads,
                    enc_heads = enc_heads,
                    enc_pf_dim = enc_pf_dim,
                    dec_pf_dim = dec_pf_dim,
                    enc_dropout = enc_dropout,
                    dec_dropout = dec_dropout,
                    trg_pad_idx = pad_index, 
                    src_pad_idx = pad_index, 
                    device = device)

model = Seq2Seq(**model_params).to(device)
print (f'total number of model parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
print (f'total number of encoder-model parameters is {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}')
print (f'total number of decoder-model parameters is {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}')
#print (f'total number of attention-model parameters is {sum(p.numel() for p in model.attention.parameters() if p.requires_grad)}')

lr = 1e-3
n_epochs = 50
clip = 1.0

is_train = True
best_valid_loss = float("inf")

optimizer = optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

def train_fn(model, data_loader, optimizer, criterion, clip,  device):
    model.train()
    epoch_loss = 0
    for batch in data_loader:

        src = batch["de_ids"].to(device)    
        trg = batch["en_ids"].to(device)
        src, trg = src.transpose(0,1), trg.transpose(0,1)
        # src = [batch size, src length]
        # trg = [batch size, trg length]
       
        optimizer.zero_grad()
        output, _ = model(src, trg[:,:-1])
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[:,1:].contiguous().view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            src = batch["de_ids"].to(device)
            
            trg = batch["en_ids"].to(device)
            src, trg = src.transpose(0,1), trg.transpose(0,1)
            # src = [batch size, src length]
            # trg = [batch size, trg length]
            output, _ = model(src, trg[:,:-1])  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[:,1:].contiguous().view(-1)
            # trg = [(trg length - 1) * batch size]
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


if is_train is True:
    for epoch in tqdm.tqdm(range(n_epochs)):
        train_loss = train_fn(
            model,
            train_data_loader,
            optimizer,
            criterion,
            clip,
            device,
        )
        valid_loss = evaluate_fn(
            model,
            valid_data_loader,
            criterion,
            device,
        )
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            #torch.save(model.state_dict(), "saved_models/tut3-model.pt")
            print('-----model is saved ---')
            torch.save({
                'model':model.state_dict(),
                'model_params': model_params,
                'lr':lr,
                'optimizer': optimizer.state_dict()
                }, "save_models/simple_translation.pt"
            )

        print(f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}")
        print(f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}")

model.load_state_dict(torch.load("save_models/simple_translation.pt")['model'])
test_loss = evaluate_fn(model, test_data_loader, criterion, device)
print(f"| Test Loss: {test_loss:.3f} | Test PPL: {np.exp(test_loss):7.3f} |")

def translate_sentence(
    sentence,
    model,
    en_nlp,
    de_nlp,
    en_vocab,
    de_vocab,
    lower,
    sos_token,
    eos_token,
    device,
    max_output_length=25,
):
    model.eval()
    with torch.no_grad():
        if isinstance(sentence, str):
            de_tokens = [token.text for token in de_nlp.tokenizer(sentence)]
        else:
            de_tokens = [token for token in sentence]

        if lower:
            de_tokens = [token.lower() for token in de_tokens]

        de_tokens = [sos_token] + de_tokens + [eos_token]
        src_index = de_vocab.lookup_indices(de_tokens)
        src_tensor = torch.LongTensor(src_index).unsqueeze(-1).to(device)
        
        src_tensor = src_tensor.transpose(0,1)

        src_mask = model.make_src_mask(src_tensor)
        with torch.no_grad():
            enc_src =  model.encoder(src_tensor, src_mask)
        
        trg_indexes = en_vocab.lookup_indices([sos_token])
        for i in range(max_output_length):
            
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            
            #trg_tensor = trg_tensor.transpose(0,1)

            trg_mask = model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)

            if pred_token == en_vocab[eos_token]:
                break

        trg_tokens = en_vocab.lookup_tokens(trg_indexes)
    return trg_tokens[1:], de_tokens, attention

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


sentence = dataset['test']['de'][0]
expected_translation = dataset["test"]['en'][0]
print(sentence, expected_translation)

translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp = en_de_proc.en_nlp,
    de_nlp = en_de_proc.de_nlp,
    en_vocab = en_de_proc.en_vocab_dict,
    de_vocab = en_de_proc.de_vocab_dict,
    lower = en_de_proc.lower,
    sos_token = en_de_proc.sos_token,
    eos_token = en_de_proc.eos_token,
    device = device)

#display_attention(sentence, translation, attention)

print(sentence_tokens)
print(translation)
print('-------------------------------------')

###################
sentence = "Ein Mann sieht sich einen Film an."
translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp = en_de_proc.en_nlp,
    de_nlp = en_de_proc.de_nlp,
    en_vocab = en_de_proc.en_vocab_dict,
    de_vocab = en_de_proc.de_vocab_dict,
    lower = en_de_proc.lower,
    sos_token = en_de_proc.sos_token,
    eos_token = en_de_proc.eos_token,
    device = device
)
print(sentence)
print(translation)
