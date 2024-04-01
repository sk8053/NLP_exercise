######## https://github.com/bentrevett/pytorch-seq2seq
import torch
import torch.nn as nn
#import torch.optim as optim
#import random
import numpy as np
#import spacy
#import datasets # Hugging face datasets
#import torchtext
import torch.optim as optim
from get_data import get_all_data
from seq2seq import Seq2Seq
import tqdm
#import evaluate
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

### spacy
### datasets
### nn.utils.rnn.pad_sequence
### torchtext.vocab.build_vocab_from_iterator

### Huggingface datasets
### https://huggingface.co/docs/datasets/en/process

train_data_loader, valid_data_loader, test_data_loader, extra_info = get_all_data()

input_dim, output_dim = extra_info['input_dim'], extra_info['output_dim']
print(f'input dimension is {input_dim}')
print(f'output dimension is {output_dim}')

pad_index = extra_info['pad_index']
test_data = extra_info['test_data']

encoder_embedding_dim = 256
decoder_embedding_dim = 256
encoder_hidden_dim = 512
decoder_hidden_dim = 512
encoder_dropout = 0.5
decoder_dropout = 0.5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_params = dict(input_dim  = input_dim, 
                    output_dim = output_dim, 
                    encoder_hidden_dim = encoder_hidden_dim, 
                    decoder_hidden_dim = decoder_hidden_dim,
                    encoder_embedding_dim = encoder_embedding_dim,  
                    decoder_embedding_dim = decoder_hidden_dim,
                    encoder_dropout = encoder_dropout, 
                    decoder_dropout =decoder_dropout,
                    src_pad_idx = pad_index, 
                    device = device)


model = Seq2Seq(**model_params).to(device)
print (f'total number of model parameters is {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
print (f'total number of encoder-model parameters is {sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)}')
print (f'total number of decoder-model parameters is {sum(p.numel() for p in model.decoder.parameters() if p.requires_grad)}')
print (f'total number of attention-model parameters is {sum(p.numel() for p in model.attention.parameters() if p.requires_grad)}')

lr = 5e-4
n_epochs = 30
clip = 1.2
teacher_forcing_ratio = 0.5
is_train = True
best_valid_loss = float("inf")

optimizer = optim.Adam(model.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss(ignore_index=pad_index)

def train_fn(model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["de_ids"].to(device)
        trg = batch["en_ids"].to(device)
        
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
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
        for i, batch in enumerate(data_loader):
            src = batch["de_ids"].to(device)
            trg = batch["en_ids"].to(device)
            # src = [src length, batch size]
            # trg = [trg length, batch size]
            output = model(src, trg, 0)  # turn off teacher forcing
            # output = [trg length, batch size, trg vocab size]
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            # output = [(trg length - 1) * batch size, trg vocab size]
            trg = trg[1:].view(-1)
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
            teacher_forcing_ratio,
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
                'teacher_forcing_ratio': teacher_forcing_ratio,
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
        ids = de_vocab.lookup_indices(de_tokens)
        src_tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        encoder_outputs, hidden, cell_state = model.encoder(src_tensor)
        inputs = en_vocab.lookup_indices([sos_token])
        attentions = torch.zeros(max_output_length, 1, len(ids))
        
        mask = model.create_mask(src_tensor)
        
        for i in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            output, hidden, cell_state,  attention = model.decoder(
                inputs_tensor, hidden,cell_state, encoder_outputs, mask)
            attentions[i] = attention
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if predicted_token == en_vocab[eos_token]:
                break
        en_tokens = en_vocab.lookup_tokens(inputs)
    return en_tokens, de_tokens, attentions[: len(en_tokens) - 1]

def plot_attention(sentence, translation, attention):
    fig, ax = plt.subplots(figsize=(10, 10))
    attention = attention.squeeze(1).numpy()
    cax = ax.matshow(attention, cmap="bone")
    ax.set_xticks(ticks=np.arange(len(sentence)), labels=sentence, rotation=90, size=15)
    translation = translation[1:]
    ax.set_yticks(ticks=np.arange(len(translation)), labels=translation, size=15)
    
    plt.savefig('plot_attention.png')
    plt.close()


sentence = test_data[0]["de"]
expected_translation = test_data[0]["en"]
print(sentence, expected_translation)

translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp = extra_info['en_nlp'],
    de_nlp = extra_info['de_nlp'],
    en_vocab = extra_info['en_vocab'],
    de_vocab = extra_info['de_vocab'],
    lower = extra_info['lower'],
    sos_token = extra_info['sos_token'],
    eos_token = extra_info['eos_token'],
    device = device)

plot_attention(sentence, translation, attention)

print(sentence_tokens)
print(translation)
print('-------------------------------------')

###################
sentence = "Ein Mann sieht sich einen Film an."
translation, sentence_tokens, attention = translate_sentence(
    sentence,
    model,
    en_nlp = extra_info['en_nlp'],
    de_nlp = extra_info['de_nlp'],
    en_vocab = extra_info['en_vocab'],
    de_vocab = extra_info['de_vocab'],
    lower = extra_info['lower'],
    sos_token = extra_info['sos_token'],
    eos_token = extra_info['eos_token'],
    device = device
)
print(sentence)
print(translation)
