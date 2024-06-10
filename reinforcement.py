# reinforcement.py
from colorama import Fore, Style
from data import normalizeString
from train import indexesFromSentence
import torch.optim as optim
import torch
from utils import *
import os

def saveModel(encoder, decoder, voc, model_info, model_filename):
    save_dir = os.path.join("data", "save")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save({
        'iteration': model_info['iteration'],
        'en': encoder.state_dict(),
        'de': decoder.state_dict(),
        'en_opt': model_info['encoder_optimizer'].state_dict(),
        'de_opt': model_info['decoder_optimizer'].state_dict(),
        'loss': model_info['loss'],
        'voc_dict': voc.__dict__,
        'embedding': model_info['embedding']
    }, os.path.join(save_dir, f'{model_filename}.tar'))
    print(f'Model saved as {model_filename}.tar')

def compute_reward(response, user_feedback):
    return 1 if user_feedback == "like" else -1

class PolicyGradientAgent:
    def __init__(self, encoder, decoder, learning_rate=0.001):
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)

    def update_policy(self, log_probs, rewards):
        discounted_rewards = self.discount_rewards(rewards)
        discounted_rewards = discounted_rewards.detach().requires_grad_(False)

        loss = -torch.sum(log_probs * discounted_rewards)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def discount_rewards(self, rewards, gamma=0.99):
        discounted_rewards = []
        running_add = 0
        for r in reversed(rewards):
            running_add = running_add * gamma + r
            discounted_rewards.insert(0, running_add)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
        if len(discounted_rewards) > 1:
            return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        else:
            return discounted_rewards

def generate_response(searcher, voc, input_sentence, device, max_length=MAX_LENGTH):
    indexes_batch = [indexesFromSentence(voc, input_sentence)]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    input_batch = input_batch.to(device)
    lengths = lengths.to("cpu")
    tokens, scores = searcher(input_batch, lengths, max_length)
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words, scores

def train_rl(encoder, decoder, searcher, voc, device, model_info, model_filename, num_episodes=10):
    agent = PolicyGradientAgent(encoder, decoder)
    for episode in range(num_episodes):
        input_sentence = input(Fore.BLUE + '[YOU]: ' + Style.RESET_ALL).strip()
        if input_sentence.lower() in ['q', 'quit']:
            break
        input_sentence = normalizeString(input_sentence)
        output_words, scores = generate_response(searcher, voc, input_sentence, device)
        output_sentence = ' '.join([x for x in output_words if x not in ['EOS', 'PAD']])
        print(Fore.MAGENTA + '[BOT]:' + Style.RESET_ALL, output_sentence)
        user_feedback = input(Fore.GREEN + '[FEEDBACK - like/dislike]: ' + Style.RESET_ALL).strip()
        reward = compute_reward(output_sentence, user_feedback)

        log_probs = torch.tensor(scores, dtype=torch.float32).detach().requires_grad_(True)
        loss = agent.update_policy(log_probs, [reward])

        print(f"Episode {episode + 1}, Loss: {loss}, Reward: {reward}")

        saveModel(encoder, decoder, voc, model_info, model_filename)
