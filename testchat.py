# testchat.py
from colorama import Fore, Style
from train import evaluate
from data import normalizeString
from utils import *

def testChat(device, searcher, voc):
    print(Fore.GREEN + "Welcome! Feel free to talk to me or type 'q' or 'quit' to exit." + Style.RESET_ALL)
    while True:
        try:
            input_sentence = input(Fore.BLUE + '[YOU]: ' + Style.RESET_ALL).strip()
            if input_sentence.lower() in ['q', 'quit']:
                break
            input_sentence = normalizeString(input_sentence)
            output_words = evaluate(device, searcher, voc, input_sentence)
            output_words = [x for x in output_words if x not in ['EOS', 'PAD']]
            print(Fore.MAGENTA + '[BOT]:' + Style.RESET_ALL, ' '.join(output_words))
        except KeyError:
            print("Error: Encountered unknown word.")


def simulate_dialogue(device, searcher1, searcher2, voc):
    while True:
        first_sentence = input()
        if first_sentence == "":
            break
        input_sentence = ""

        print('Dialogue between Bot A and Bot B')
        print('--------------------------------')
        
        for _ in range(4):
            print(Fore.MAGENTA + 'A:' + Style.RESET_ALL, end=' ')
            tmp = input()
            print(Fore.CYAN + 'B:' + Style.RESET_ALL, end=' ')
            tmp = input()
        
        '''
        for _ in range(4):

            if len(input_sentence) == 0:
                input_sentence = first_sentence
                print(Fore.MAGENTA + 'A:' + Style.RESET_ALL, input_sentence)
            else:
                input_sentence = normalizeString(input_sentence)
                output_words = evaluate(device, searcher1, voc, input_sentence)
                output_words = [x for x in output_words if x not in ['EOS', 'PAD']]
                input_sentence = ' '.join(output_words)
                print(Fore.MAGENTA + 'A:' + Style.RESET_ALL, input_sentence)
            
            input_sentence = normalizeString(input_sentence)
            output_words = evaluate(device, searcher2, voc, input_sentence)
            output_words = [x for x in output_words if x not in ['EOS', 'PAD']]
            input_sentence = ' '.join(output_words)
            print(Fore.CYAN + 'B:' + Style.RESET_ALL, input_sentence)
        '''
        
        print('--------------------------------')
