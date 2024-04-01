from seq2seq import Seq2Seq
import torch

saved_data = torch.load("save_models/simple_translate.pt")
model_params = saved_data['model_params']

model = Seq2Seq(**model_params)
model.eval()
