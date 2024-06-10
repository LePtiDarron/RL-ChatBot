# main.py
import os
from utils import *
import option_dialogue, option_rl, option_test, option_train
import argparse
import shutil


def main(option, model_filename, model_filename2, n_iteration):
    save_dir = os.path.join("data", "save")
    loadFilename = os.path.join(save_dir, f'{model_filename}.tar')
    if not os.path.exists(loadFilename):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sampleFileName = os.path.join("data", "samples", "SampleModel.tar")
        shutil.copyfile(sampleFileName, loadFilename)
        print(f"New Model <{loadFilename}> created from <{sampleFileName}>")
    
    if option == 'seq':
        option_train.run(model_filename, n_iteration)
        return
    if option == 'rl':
        option_rl.run(model_filename)
        return
    if option == 'test':
        option_test.run(model_filename)
        return
    if option == 'dialogue':
        if len(model_filename2) <= 0:
            print('Missing second model')
            return
        option_dialogue.run(model_filename, model_filename2)
        return
    print(f'Unknown option : {option}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL ChatBot")
    parser.add_argument('-o', '--option', type=str, required=True, help='seq / rl / test / dialogue')
    parser.add_argument('-n', '--name', type=str, required=True, help='model name')
    parser.add_argument('-N', '--name2', type=str, required=False, help='second model name')
    parser.add_argument('-e', '--epoch', type=str, required=False, help='number of epoch')
    args = parser.parse_args()
    main(args.option, args.name, args.name2, args.epoch)
