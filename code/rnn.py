import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'

class RNN(nn.Module):
    def __init__(self, input_dim, h):
        super(RNN, self).__init__()
        self.h = h
        self.numOfLayer = 1
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh')
        self.W = nn.Linear(h, 5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, inputs):
        # Obtain hidden representations from the RNN
        output, hidden = self.rnn(inputs)  # output shape: (seq_len, batch, hidden_dim)
        # Apply linear transformation to each time step's output
        outputs_linear = self.W(output)  # shape: (seq_len, batch, 5)
        # Sum over the time steps to get a single representation for the sequence
        summed_outputs = torch.sum(outputs_linear, dim=0)  # shape: (batch, 5)
        # Apply softmax to obtain the final probability distribution (log probabilities)
        predicted_vector = self.softmax(summed_outputs)
        return predicted_vector

def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(), int(elt["stars"]) - 1))
    for elt in validation:
        val.append((elt["text"].split(), int(elt["stars"]) - 1))
    return tra, val

def load_test_data(test_data):
    with open(test_data) as test_f:
        testing = json.load(test_f)
    tst = []
    for elt in testing:
        tst.append((elt["text"].split(), int(elt["stars"]) - 1))
    return tst

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required=True, help="hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required=True, help="num of epochs to train")
    parser.add_argument("--train_data", required=True, help="path to training data")
    parser.add_argument("--val_data", required=True, help="path to validation data")
    parser.add_argument("--test_data", default="to fill", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    # Set up model; here input_dim should match the embedding size in word_embedding.pkl (e.g., 50)
    model = RNN(50, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0
    last_train_accuracy = 0
    last_validation_accuracy = 0

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        correct = 0
        total = 0
        minibatch_size = 16
        N = len(train_data)
        loss_total = 0
        loss_count = 0
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                input_words = " ".join(input_words)
                # Remove punctuation and tokenize
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                # Lookup word embeddings
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
                # Create tensor with shape (seq_len, batch, embedding_dim)
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                example_loss = model.compute_Loss(output.view(1, -1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                loss = example_loss if loss is None else loss + example_loss
            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()
        print("Epoch {} training loss: {}".format(epoch + 1, loss_total / loss_count))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        train_accuracy = correct / total

        model.eval()
        correct = 0
        total = 0
        print("Validation started for epoch {}".format(epoch + 1))
        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct / total

        if validation_accuracy < last_validation_accuracy and train_accuracy > last_train_accuracy:
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
            torch.save(model.state_dict(), "rnn_model.pth")
            print("Model saved as rnn_model.pth")
            stopping_condition = True
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = train_accuracy

        epoch += 1

    if args.test_data != "to fill":
        print("========== Testing ==========")
        test_examples = load_test_data(args.test_data)
        correct = 0
        total = 0
        for input_words, gold_label in tqdm(test_examples):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1
        print("Test Accuracy: {}".format(correct / total))
