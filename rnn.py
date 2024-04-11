import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import time
from tqdm import tqdm
import json
import string
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser

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
        _, hidden = self.rnn(inputs)
        output_representation = self.W(hidden[-1])
        output_sum = torch.sum(output_representation, dim=0)
        predicted_vector = self.softmax(output_sum.view(1, -1))
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", default=5, type=int, required=False, help="hidden_dim")
    parser.add_argument("-e", "--epochs", default=20, type=int, required=False, help="num of epochs to train")
    parser.add_argument("--train_data", default="./Data_Embedding/training.json", required=False, help="path to training data")
    parser.add_argument("--val_data", default="./Data_Embedding/validation.json", required=False, help="path to validation data")
    parser.add_argument("--test_data", default="./Data_Embedding/test.json", help="path to test data")
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")

    args = parser.parse_args()

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)

    print("========== Vectorizing data ==========")
    model = RNN(50, args.hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    word_embedding = pickle.load(open('./Data_Embedding/word_embedding.pkl', 'rb'))

    stopping_condition = False
    epoch = 0

    last_train_accuracy = 0
    last_validation_accuracy = 0

    train_losses = []
    dev_accuracies = []

    while not stopping_condition:
        random.shuffle(train_data)
        model.train()
        print("Training started for epoch {}".format(epoch + 1))
        train_data = train_data
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
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words ]
                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss

            loss = loss / minibatch_size
            loss_total += loss.data
            loss_count += 1
            loss.backward()
            optimizer.step()

        train_losses.append(loss_total/loss_count)
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        trainning_accuracy = correct/total

        model.eval()
        correct = 0
        total = 0
        random.shuffle(valid_data)
        print("Validation started for epoch {}".format(epoch + 1))
        valid_data = valid_data

        for input_words, gold_label in tqdm(valid_data):
            input_words = " ".join(input_words)
            input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
            vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
            vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
            output = model(vectors)
            predicted_label = torch.argmax(output)
            correct += int(predicted_label == gold_label)
            total += 1

        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        validation_accuracy = correct/total
        dev_accuracies.append(validation_accuracy)

        if validation_accuracy < last_validation_accuracy and trainning_accuracy > last_train_accuracy:
            stopping_condition=True
            print("Training done to avoid overfitting!")
            print("Best validation accuracy is:", last_validation_accuracy)
        else:
            last_validation_accuracy = validation_accuracy
            last_train_accuracy = trainning_accuracy

        epoch += 1

    epochs = range(1, epoch + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, dev_accuracies, label='Development Set Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Development Set Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('rnn_lr {}.png'.format(args.learning_rate))

    plt.show()

    model.eval()

    error_examples = []
    num_error_examples = 5

    for input_words, gold_label in valid_data:
        input_words = " ".join(input_words)
        input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
        vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]
        vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
        output = model(vectors)
        predicted_label = torch.argmax(output)
        if predicted_label != gold_label:
            error_examples.append((input_words, predicted_label, gold_label))
            if len(error_examples) == num_error_examples:
                break

    for example in error_examples:
        input_words, predicted_label, gold_label = example

        print("Error Analysis:")
        print("Input Text:", ' '.join(input_words))
        print("Predicted Label:", predicted_label.item())
        print("Label:", gold_label)
        if predicted_label < 0:
            print("Error: Predicted label out of range.")
        else:
            predicted_class_prob = torch.exp(output[0][predicted_label]).item()
            print("Probability of Predicted Class:", predicted_class_prob)
