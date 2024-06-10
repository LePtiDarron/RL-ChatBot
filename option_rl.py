# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import loadLinesAndConversations, extractSentencePairs, readVocs, filterPairs
from models import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
from reinforcement import train_rl
from utils import *


def run(model_filename):
    corpus_name = "movie-corpus"
    corpus = os.path.join("data", corpus_name)
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")
    save_dir = os.path.join("data", "save")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lines, conversations = loadLinesAndConversations(os.path.join(corpus, "utterances.jsonl"))
    pairs = extractSentencePairs(conversations)
    print('Reading voc...')
    voc, pairs = readVocs(datafile, corpus_name)
    pairs = filterPairs(pairs)

    loadFilename = None
    loadFilename = os.path.join(save_dir, f'{model_filename}.tar')

    if os.path.exists(loadFilename):
        checkpoint = torch.load(loadFilename)
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']
        print(f'Loaded model <{model_filename}> trained during {checkpoint['iteration']} epochs.')

    print('Building encoder and decoder ...')
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if os.path.exists(loadFilename):
        embedding.load_state_dict(embedding_sd)
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if os.path.exists(loadFilename):
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    encoder.train()
    decoder.train()

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if os.path.exists(loadFilename):
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    print('Initializing ...')
    start_iteration = 1
    if os.path.exists(loadFilename):
        start_iteration = checkpoint['iteration'] + 1
        loss = checkpoint['loss'] + 1

    encoder.eval()
    decoder.eval()
    searcher = GreedySearchDecoder(encoder, decoder, device)
    print('RL training : Start')
    model_info = {
        'encoder_optimizer' : encoder_optimizer,
        'decoder_optimizer' : decoder_optimizer,   
        'iteration' : start_iteration,
        'loss' : loss,
        'embedding' : embedding.state_dict(),
    }
    train_rl(encoder, decoder, searcher, voc, device, model_info, model_filename)
    print('RL training : Done')
