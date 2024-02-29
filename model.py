import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from multiprocessing import Queue, Event
import numpy as np

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def forward(self, x):
        seq_len, d_model = x.size(1), self.d_model
        position = torch.arange(seq_len, dtype=torch.float, device=x.device).unsqueeze(1).to(device=x.device)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).to(device=x.device)

        pe = torch.zeros(seq_len, d_model, device=x.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).expand_as(x)
        x = x + pe

        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, lr: float = 0.001, output_seq_len=1, device: torch.device = None, background: bool = False, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.device = device
        self.output_seq_len = output_seq_len
        self.input_linear = nn.Linear(input_dim, model_dim, dtype=dtype)
        self.positional_encoding = PositionalEncoding(model_dim, dropout)
        #self.transformer = 
        self.layer_norm = nn.LayerNorm(model_dim, dtype=dtype)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(model_dim, num_heads, dim_feedforward, dropout, batch_first=True, dtype=dtype), num_encoder_layers)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(model_dim, num_heads, dim_feedforward, dropout, batch_first=True, dtype=dtype), num_decoder_layers)
        self.output_linear = nn.Linear(model_dim, output_dim, dtype=dtype)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.MSELoss()

        self.clip_threshold = 1.0

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src, tgt=None):
        src = self.input_linear(src)
        src = self.positional_encoding(src)
        src = self.layer_norm(src)
        memory = self.encoder(src)
        
        if self.training:
            assert tgt is not None, "Target sequence must be provided during training"
            
            # Create the square subsequent mask to prevent attention to future tokens
            tgt_seq_len = tgt.size(1)
            mask = self.generate_square_subsequent_mask(tgt_seq_len).to(self.device)

            tgt = self.input_linear(tgt)
            tgt = self.positional_encoding(tgt)
            tgt = self.layer_norm(tgt)

            # Pass the mask to the decoder
            output = self.decoder(tgt, memory, tgt_mask=mask)
        else:
            output = memory

        output = self.output_linear(output)
        return output
    
    def calculate_total_grad_norm(self, norm_type=2):
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type) for p in self.parameters() if p.grad is not None]), norm_type)
        return total_norm

    def dynamic_gradient_clip(self):
        alpha = 0.99
        lower_bound, upper_bound = 0.5, 2.0

        total_norm = self.calculate_total_grad_norm()
        
        self.clip_threshold = alpha * self.clip_threshold + (1 - alpha) * total_norm
        self.clip_threshold = max(min(self.clip_threshold, upper_bound), lower_bound)
        
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_threshold)
    
    def custom_loss(self, y_true, y_pred):
        mse = torch.mean((y_true - y_pred) ** 2)
        mae = torch.mean(torch.abs(y_true - y_pred))
        
        mse_weight = 0.5
        mae_weight = 0.5
        
        weighted_loss = (mse_weight * mse) + (mae_weight * mae)
        
        return weighted_loss


    def train_model(self, data_loader: DataLoader, epochs: int = 100, verbose: bool = True):
        for epoch in range(epochs):
            epoch_loss = 0

            for batch_index, (src, tgt) in enumerate(data_loader):
                self.train()

                src = src.to(self.device)
                tgt = tgt.to(self.device)

                self.optimizer.zero_grad()

                output = self(src, tgt)

                loss = self.custom_loss(output, tgt)
                #loss = self.criterion(output, tgt)
                loss.backward()

                self.dynamic_gradient_clip()

                self.optimizer.step()

                epoch_loss += loss.item()

                print(f'Epoch progress: {epoch + 1} / {epochs}, Batch: {batch_index + 1} / {len(data_loader)}, Loss: {loss}')

            yield epoch + 1, epoch_loss

    def evaluate(self, data_loader: DataLoader, auto_regression: bool = False):
        self.eval()
        total_loss = 0.0

        with torch.no_grad():
            outputs = []
            targets = []
            for batch_index, (input, target) in enumerate(data_loader):
                input, target = input.to(self.device), target.to(self.device)

                if auto_regression:

                    input_sequence = input
                    predictions = []

                    for _ in range(self.output_seq_len):
                        prediction = self.forward(input_sequence)[:, -1:, :]
                        predictions.append(prediction)

                        input_sequence = torch.cat((input_sequence[:, 1:, :], prediction), dim=1)

                    predictions = torch.cat(predictions, dim=1)
                else:
                    predictions = self(input)

                loss = self.criterion(predictions, target)
                total_loss += loss.item()

                output = predictions.detach().cpu().numpy()
                target = target.detach().cpu().numpy()

                outputs.append(output)
                targets.append(target)

                print(f'Model evaluation: Batch: {batch_index + 1} / {len(data_loader)}, Loss: {loss.item()}')

                #yield loss, output, target
            yield total_loss / len(data_loader), np.concatenate(outputs), np.concatenate(targets)