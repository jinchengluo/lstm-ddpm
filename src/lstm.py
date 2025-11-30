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

        self.b_f = torch.zeros((1, hidden_size))
        self.b_i = torch.zeros((1, hidden_size))
        self.b_C = torch.zeros((1, hidden_size))
        self.b_o = torch.zeros((1, hidden_size))


    def sigmoid(self, z):
        return 1 / (1 + torch.exp(-z))

    def tanh(self, z):
        return torch.tanh(z)

    def forward(self, x, h_prev, C_prev):
        f_t = self.sigmoid(torch.matmul(x, self.W_f.t()) + torch.matmul(h_prev, self.U_f.t()) + self.b_f) 
        
        i_t = self.sigmoid(torch.matmul(x, self.W_i.t()) + torch.matmul(h_prev, self.U_i.t()) + self.b_i) 
        
        C_tilda_t = self.tanh(torch.matmul(x, self.W_C.t()) + torch.matmul(h_prev, self.U_C.t()) + self.b_C) 
        
        C_t = f_t * C_prev + i_t * C_tilda_t 

        o_t = self.sigmoid(torch.matmul(x, self.W_o.t()) + torch.matmul(h_prev, self.U_o.t()) + self.b_o) 
        
        h_t = o_t * self.tanh(C_t)

        dynamics = {"input": i_t, "forget": f_t, "cell_update": C_tilda_t}
        return h_t, C_t, dynamics
    

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=16):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size=input_size, hidden_size=hidden_size)
        self.predictor = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        # x shape: [Batch, Sequence_Len, 1]
        batch_size, seq_len, _ = x.size()
        # print(batch_size, seq_len)
        h = torch.zeros(batch_size, self.hidden_size).to(x.device)
        c = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
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
    
    def generate_sequence(self, seed_data, future_steps=50):
        """
        Uses the trained model to generate future steps based on a seed sequence.

        Args:
            model: The trained LSTM model
            seed_data: Tensor of shape [Seq_Len, Input_Size] (a single sequence)
            future_steps: How many steps to generate
            
        Returns:
            generated_seq: Tensor of shape [Future_Steps, Input_Size]
        """
        self.eval()
        
        # Initialize internal state
        h = torch.zeros(1, self.hidden_size).to(seed_data.device)
        c = torch.zeros(1, self.hidden_size).to(seed_data.device)
        
        generated_values = []
        
        with torch.no_grad():
            # 1. Warm up the internal state (h, c) using the seed data
            seq_len = seed_data.size(1)
            for t in range(seq_len):
                x_t = seed_data[t, :] # Shape (1, Input_Size)
                h, c, _ = self.cell(x_t, h, c)
            
            # The model is now primed. The last 'h' contains the memory of the seed.
            # We make the first prediction.
            current_input = self.predictor(h) 
            generated_values.append(current_input)
            
            # 2. Autoregressive Generation Loop
            for _ in range(future_steps - 1):
                # Feed the LAST PREDICTION as the NEXT INPUT
                h, c, _ = self.cell(current_input, h, c)
                pred = self.predictor(h)
                
                generated_values.append(pred)
                current_input = pred # Update input for next step
                
        return torch.stack(generated_values, dim=1)