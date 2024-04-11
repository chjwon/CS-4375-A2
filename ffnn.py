import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import time
import json
from argparse import ArgumentParser
from tqdm import tqdm
import matplotlib.pyplot as plt

unk = '<UNK>'

class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)
        self.softmax = nn.LogSoftmax()
        self.loss = nn.NLLLoss()

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        hidden_representation = self.W1(input_vector)
        activation_result = self.activation(hidden_representation)
        output_representation = self.W2(activation_result)
        predicted_vector = self.softmax(output_representation)
        return predicted_vector

def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 

def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)
    return vocab, word2index, index2word 

def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))
    return vectorized_data

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
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate")
    
    parser.add_argument('--do_train', action='store_true')

    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)

    print("========== Loading data ==========")
    train_data, valid_data = load_data(args.train_data, args.val_data)
    vocab = make_vocab(train_data)
    vocab, word2index, index2word = make_indices(vocab)

    print("========== Vectorizing data ==========")
    train_data = convert_to_vector_representation(train_data, word2index)
    valid_data = convert_to_vector_representation(valid_data, word2index)

    model = FFNN(input_dim=len(vocab), h=args.hidden_dim)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    print("========== Training for {} epochs ==========".format(args.epochs))

    train_losses = []
    dev_accuracies = []

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = None
        correct = 0
        total = 0
        start_time = time.time()
        print("Training started for epoch {}".format(epoch + 1))
        random.shuffle(train_data)
        minibatch_size = 16
        N = len(train_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
            loss.backward()
            optimizer.step()
        print("Training completed for epoch {}".format(epoch + 1))
        print("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Training time for this epoch: {}".format(time.time() - start_time))

        train_losses.append(loss.item())

        correct = 0
        total = 0
        start_time = time.time()
        print("Validation started for epoch {}".format(epoch + 1))
        minibatch_size = 16
        N = len(valid_data)
        for minibatch_index in tqdm(range(N // minibatch_size)):
            optimizer.zero_grad()
            loss = None
            for example_index in range(minibatch_size):
                input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                predicted_vector = model(input_vector)
                predicted_label = torch.argmax(predicted_vector)
                correct += int(predicted_label == gold_label)
                total += 1
                example_loss = model.compute_Loss(predicted_vector.view(1, -1), torch.tensor([gold_label]))
                if loss is None:
                    loss = example_loss
                else:
                    loss += example_loss
            loss = loss / minibatch_size
        print("Validation completed for epoch {}".format(epoch + 1))
        print("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
        print("Validation time for this epoch: {}".format(time.time() - start_time))

        dev_accuracies.append(correct / total)

    epochs = range(1, args.epochs + 1)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss lr {}'.format(args.learning_rate))
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, dev_accuracies, label='Development Set Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Development Set Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('ffnn_lr {}.png'.format(args.learning_rate))
    plt.show()

    # Error analysis
    model.eval()
    error_examples = []
    num_error_examples = 5

    for input_vector, gold_label in valid_data:
        predicted_vector = model(input_vector)
        predicted_label = torch.argmax(predicted_vector)
        if predicted_label != gold_label:
            error_examples.append((input_vector, predicted_label, gold_label))
            if len(error_examples) == num_error_examples:
                break

    for example in error_examples:
        input_vector, predicted_label, gold_label = example
        
        input_words = [index2word[index] for index, count in enumerate(input_vector) if count != 0]
        
        print("Error Analysis:")
        print("Input Text:", ' '.join(input_words))
        print("Predicted Label:", predicted_label.item())
        print("Label:", gold_label)
        
        if predicted_label < 0 or predicted_label >= model.output_dim:
            print("Error: Predicted label out of range.")
        else:
            predicted_class_prob = torch.exp(predicted_vector[predicted_label]).item()
            print("Probability of Predicted Class:", predicted_class_prob)


