import numpy as np
import torch.nn as nn

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_f = np.random.randn(hidden_size, input_size)
        self.W_i = np.random.randn(hidden_size, input_size)
        self.W_C = np.random.randn(hidden_size, input_size)
        self.W_o = np.random.randn(hidden_size, input_size)
        self.U_f = np.random.randn(hidden_size, hidden_size)
        self.U_i = np.random.randn(hidden_size, hidden_size)
        self.U_C = np.random.randn(hidden_size, hidden_size)
        self.U_o = np.random.randn(hidden_size, hidden_size)

        self.b_f = np.zeros((hidden_size, 1))
        self.b_i = np.zeros((hidden_size, 1))
        self.b_C = np.zeros((hidden_size, 1))
        self.b_o = np.zeros((hidden_size, 1))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def tanh(self, z):
        return np.tanh(z)

    def forward(self, x, h_prev, C_prev):
        f_t = self.sigmoid(self.W_f @ x + self.U_f @ h_prev + self.b_f) # forget gate
        i_t = self.sigmoid(self.W_i @ x + self.U_i @ h_prev + self.b_i) # input gate

        C_tilda_t = self.tanh(self.W_C @ x + self.U_C @ h_prev + self.b_C) # candidate cell state
        C_t = f_t * C_prev + i_t * C_tilda_t # cell state update

        o_t = self.sigmoid(self.W_o @ x + self.U_o @ h_prev + self.b_o) # output gate
        h_t = o_t * self.tanh(C_t) # hidden state update

        return h_t, C_t