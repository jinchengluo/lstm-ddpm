import torch
import torch.nn as nn

class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_f = torch.randn(hidden_size, input_size)
        self.W_i = torch.randn(hidden_size, input_size)
        self.W_C = torch.randn(hidden_size, input_size)
        self.W_o = torch.randn(hidden_size, input_size)
        self.U_f = torch.randn(hidden_size, hidden_size)
        self.U_i = torch.randn(hidden_size, hidden_size)
        self.U_C = torch.randn(hidden_size, hidden_size)
        self.U_o = torch.randn(hidden_size, hidden_size)

        self.b_f = torch.zeros((hidden_size, 1))
        self.b_i = torch.zeros((hidden_size, 1))
        self.b_C = torch.zeros((hidden_size, 1))
        self.b_o = torch.zeros((hidden_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def tanh(self, z):
        return torch.tanh(z)

    def forward(self, x, h_prev, C_prev):
        f_t = self.sigmoid(torch.matmul(self.W_f, x) + torch.matmul(self.U_f, h_prev) + self.b_f) # forget gate
        i_t = self.sigmoid(torch.matmul(self.W_i, x) + torch.matmul(self.U_i, h_prev) + self.b_i) # input gate

        C_tilda_t = self.tanh(torch.matmul(self.W_C, x) + torch.matmul(self.U_C, h_prev) + self.b_C) # candidate cell state
        C_t = f_t * C_prev + i_t * C_tilda_t # cell state update

        o_t = self.sigmoid(torch.matmul(self.W_o, x) + torch.matmul(self.U_o, h_prev) + self.b_o) # output gate
        h_t = o_t * self.tanh(C_t) # hidden state update

        dynamics = {"input": i_t, "forget": f_t, "cell_update": C_tilda_t}
        return h_t, C_t, dynamics
    

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.predictor = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # x shape: [Batch, Sequence_Len, 1]
        batch_size, seq_len, _ = x.size()
        h = torch.zeros(batch_size, 16).to(x.device)
        c = torch.zeros(batch_size, 16).to(x.device)
        
        outputs = []
        history = {"input": [], "forget": []}

        for t in range(seq_len):
            x_t = x[:, t, :]
            h, c, dynamics = self.cell(x_t, h, c)
            pred = self.predictor(h)
            outputs.append(pred)
            
            # Store metrics
            history["input"].append(dynamics["input"])
            history["forget"].append(dynamics["forget"])

        return torch.stack(outputs, dim=1), history