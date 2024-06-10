# main.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import loadLinesAndConversations, extractSentencePairs, readVocs, filterPairs
from models import EncoderRNN, LuongAttnDecoderRNN, GreedySearchDecoder
from testchat import simulate_dialogue
from utils import *


def run(model_filename1, model_filename2):
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

    loadFilename1 = None
    loadFilename1 = os.path.join(save_dir, f'{model_filename1}.tar')

    if os.path.exists(loadFilename1):
        checkpoint1 = torch.load(loadFilename1)
        encoder_sd1 = checkpoint1['en']
        decoder_sd1 = checkpoint1['de']
        encoder_optimizer_sd1 = checkpoint1['en_opt']
        decoder_optimizer_sd1 = checkpoint1['de_opt']
        embedding_sd1 = checkpoint1['embedding']
        voc.__dict__ = checkpoint1['voc_dict']
        print(f'Loaded model <{model_filename1}> trained during {checkpoint1['iteration']} epochs.')

    print('Building encoder and decoder ...')
    embedding1 = nn.Embedding(voc.num_words, hidden_size)
    if os.path.exists(loadFilename1):
        embedding1.load_state_dict(embedding_sd1)
    encoder1 = EncoderRNN(hidden_size, embedding1, encoder_n_layers, dropout)
    decoder1 = LuongAttnDecoderRNN(attn_model, embedding1, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if os.path.exists(loadFilename1):
        encoder1.load_state_dict(encoder_sd1)
        decoder1.load_state_dict(decoder_sd1)
    encoder1 = encoder1.to(device)
    decoder1 = decoder1.to(device)
    print('Models built and ready to go!')

    encoder1.train()
    decoder1.train()

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder1.parameters(), lr=learning_rate * decoder_learning_ratio)
    if os.path.exists(loadFilename1):
        encoder_optimizer.load_state_dict(encoder_optimizer_sd1)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd1)
    
    encoder1.eval()
    decoder1.eval()
    searcher1 = GreedySearchDecoder(encoder1, decoder1, device)
    print('Bot 1 is ready.')
    
    
    loadFilename2 = None
    loadFilename2 = os.path.join(save_dir, f'{model_filename2}.tar')

    if os.path.exists(loadFilename2):
        checkpoint2 = torch.load(loadFilename2)
        encoder_sd2 = checkpoint2['en']
        decoder_sd2 = checkpoint2['de']
        encoder_optimizer_sd2 = checkpoint2['en_opt']
        decoder_optimizer_sd2 = checkpoint2['de_opt']
        embedding_sd2 = checkpoint2['embedding']
        voc.__dict__ = checkpoint2['voc_dict']
        print(f'Loaded model <{model_filename2}> trained during {checkpoint2['iteration']} epochs.')

    print('Building encoder and decoder ...')
    embedding2 = nn.Embedding(voc.num_words, hidden_size)
    if os.path.exists(loadFilename2):
        embedding2.load_state_dict(embedding_sd2)
    encoder2 = EncoderRNN(hidden_size, embedding2, encoder_n_layers, dropout)
    decoder2 = LuongAttnDecoderRNN(attn_model, embedding2, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if os.path.exists(loadFilename2):
        encoder2.load_state_dict(encoder_sd2)
        decoder2.load_state_dict(decoder_sd2)
    encoder2 = encoder2.to(device)
    decoder2 = decoder2.to(device)
    print('Models built and ready to go!')

    encoder2.train()
    decoder2.train()

    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder2.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder2.parameters(), lr=learning_rate * decoder_learning_ratio)
    if os.path.exists(loadFilename2):
        encoder_optimizer.load_state_dict(encoder_optimizer_sd2)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd2)
    
    encoder2.eval()
    decoder2.eval()
    searcher2 = GreedySearchDecoder(encoder2, decoder2, device)
    print('Bot 2 is ready.')
    
    simulate_dialogue(device, searcher1, searcher2, voc)
