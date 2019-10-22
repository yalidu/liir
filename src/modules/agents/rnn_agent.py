import torch.nn as nn
import torch.nn.functional as F
  
class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
    
# class RNNAgent(nn.Module):
#     def __init__(self, input_shape, args):
#         super(RNNAgent, self).__init__()
#         self.args = args

#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
# #         self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.fc3 = nn.Linear(args.rnn_hidden_dim, int(args.rnn_hidden_dim/2))

#         self.fc4 = nn.Linear(int(args.rnn_hidden_dim/2), args.n_actions)

#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

#     def forward(self, inputs, hidden_state):
#         x = F.relu(self.fc1(inputs))
#         x1 = F.relu(self.fc2(x))
#         x2 = F.relu(self.fc3(x1))
#         q = self.fc4(x2)

#         hidden_state  = 0
#         h = hidden_state
#         return q, h
    