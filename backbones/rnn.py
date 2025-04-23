import torch 
import torch.nn as nn
import math
from backbones.model_tcn import TlioTcn

class RNNBlock(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=3,
                 rnn_type="GRU", bidirectional=False):
        super(RNNBlock, self).__init__()

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=n_layers, bidirectional=bidirectional,  batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=n_layers, bidirectional=bidirectional,  batch_first=True)
            
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
    
    def forward(self, x, hidden_state=None):
        out, hidden_state = self.rnn(x, hidden_state)
        out = self.fc(out)
        return out, hidden_state

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RNNConditionalNetwork(nn.Module):
    def __init__(self, input_dim=6, global_cond_dim=256, hidden_size=256, rnn_type="GRU",
                 n_layers=3, bidirectional=False, output_dim=6):
        super(RNNConditionalNetwork, self).__init__()

        #sinusoidal positional embedding for timestep
        self.diffusion_step_embed_dim = 256
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(self.diffusion_step_embed_dim),
            nn.Linear(self.diffusion_step_embed_dim, self.diffusion_step_embed_dim * 4),
            nn.ReLU(),
            nn.Linear(self.diffusion_step_embed_dim * 4, self.diffusion_step_embed_dim),
        )

        #global conditioning network
        self.global_cond_encoder = TlioTcn(
            input_size=6,
            output_size=global_cond_dim,
            num_channels=[64, 64, 64, 64, 128, 128, 128],
            kernel_size=2,
            dropout=0.2,
            activation="GELU"
        )

        # Concatenate the global condition and diffusion embedding
        self.fc_input = nn.Linear(input_dim + global_cond_dim + self.diffusion_step_embed_dim, hidden_size)

        # RNN block (GRU/LSTM)
        self.rnn_block_1 = RNNBlock(input_size=hidden_size, hidden_size=hidden_size, 
                                  output_size=hidden_size, n_layers=n_layers, 
                                  rnn_type=rnn_type, bidirectional=bidirectional)

        # self.rnn_block_2 = RNNBlock(input_size=hidden_size, hidden_size=hidden_size, 
        #                           output_size=hidden_size, n_layers=n_layers, 
        #                           rnn_type=rnn_type, bidirectional=bidirectional)

        # self.rnn_block_3 = RNNBlock(input_size=hidden_size, hidden_size=hidden_size, 
        #                           output_size=hidden_size, n_layers=n_layers, 
        #                           rnn_type=rnn_type, bidirectional=bidirectional)
        
        #final output layer
        self.output_layer = nn.Linear(hidden_size, output_dim)
    
    def forward(self, noisy_bias, timestep, global_cond_input):
        batch_size, seq_len, _ = noisy_bias.size()

        #Step 1 encode the timestep into the sinusoidal embeddings
        if not torch.is_tensor(timestep):
            timestep = torch.tensor([timestep], dtype=torch.long, device=noisy_bias.device)
        timestep = timestep.expand(batch_size)
        time_embedding = self.diffusion_step_encoder(timestep)

        #Step 2 encode the global condition
        global_feature = self.global_cond_encoder(global_cond_input)

        #step 3 Concatenate sample with global features and timestep embedding
        time_embedding = time_embedding.unsqueeze(1).expand(-1, seq_len, -1)
        global_feature = global_feature.unsqueeze(1).expand(-1, seq_len, -1)
        rnn_input = torch.cat([noisy_bias, global_feature, time_embedding], dim=-1)

        #step 4 project the input into the hidden space
        rnn_input = self.fc_input(rnn_input)

        #step 5 pass through the rnn block
        rnn_output, _ = self.rnn_block_1(rnn_input)
        # rnn_output, _ = self.rnn_block_2(rnn_output)
        # rnn_output, _ = self.rnn_block_3(rnn_output)

        #step 6 project the output into the final output space
        output = self.output_layer(rnn_output)

        return (output,)


        
    