# utils.py
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3
MAX_LENGTH = 10
teacher_forcing_ratio = 0
print_every = 10
save_every = 50
clip = 50.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
model_name = 'cb_model'
attn_model = 'dot'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64
checkpoint = 0
num_episodes = 10