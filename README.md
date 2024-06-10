# RL ChatBot

This project implements a chatbot based on Seq2Seq and Reinforcement Learning (RL) architectures. The code allows for training, testing, and running dialogues between different models.

## Prerequisites

Ensure you have Python installed (version 3.6 or higher). Install the necessary dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Usage

Running the Main Script
The main script main.py allows you to choose different options for training, testing, or engaging in dialogues with the models.

Arguments

-o, --option: Choose an option from seq, rl, test, dialogue.

-n, --name: Specify the primary model name.

-N, --name2: (Optional) Specify the secondary model name (required for the dialogue option).

-e, --epoch: (Optional) Specify the number of iterations for training.

### Seq2Seq Training

```bash
python main.py -o seq -n <model_name> -e <number_of_iterations>
```

### RL Training

```bash
python main.py -o rl -n <model_name>
```

### Testing a Model

```bash
python main.py -o test -n <model_name>
```

### Running Dialogue Between Two Models

```bash
python main.py -o dialogue -n <model_name_1> -N <model_name_2>
```
