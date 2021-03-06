import sys
import numpy as np
import torch
import torch.nn as nn
import os
import random
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, N, D_in, D_in_prefix, D_in_suffix, E, H, D_out):
        """
        model compounds
        :param N: batch size
        :param D_in: input dim
        :param E: embedding size
        :param H: hidden layer dim
        :param D_out: output dim
        """
        super(MLP, self).__init__()
        self.embWords = torch.nn.Embedding(D_in, E)
        self.embWords.shape = torch.Tensor(N, E * 5)
        self.embPrefix = torch.nn.Embedding(D_in_prefix, E)
        self.embPrefix.shape = torch.Tensor(N, E * 5)
        self.embSuffix = torch.nn.Embedding(D_in_suffix, E)
        self.embSuffix.shape = torch.Tensor(N, E * 5)
        self.linear1 = torch.nn.Linear(E * 5, H)
        self.linear2 = torch.nn.Linear(H, D_out)
        self.dropout = nn.Dropout()

    def forward(self, xWords, xPrefix, xSuffix):
        """
        foward engine
        :param xWords: word
        :param xPrefix: prefix
        :param xSuffix: suffix
        :return: tag prediction
        """
        h1Words = self.embWords(xWords).view(self.embWords.shape.size())
        h1Prefix = self.embPrefix(xPrefix).view(self.embPrefix.shape.size())
        h1Suffix = self.embSuffix(xSuffix).view(self.embSuffix.shape.size())
        h2 = self.linear1(self.dropout(h1Words + h1Prefix + h1Suffix)).tanh()
        y_pred = self.linear2(h2)
        return y_pred


class mainTagger:
    def __init__(self, fileName, wordVectors=None, vocabInput=None):
        """
        main Tagger ctor
        :param fileName: input file
        :param wordVectors: pretrained word vectors
        :param vocabInput: pretrained word vocab
        """
        self.vecs = None
        # read file to words and tags
        sentencesList = []
        with open(fileName) as input_file:
            cur_sentence = []
            for line in input_file.read().split('\n'):
                if len(line) == 0:
                    sentencesList.append(
                        [("<s>", "S"), ("<s>", "S")] +
                        cur_sentence +
                        [("</s>", "S"), ("</s>", "S")])
                    cur_sentence = []
                    continue
                sp = line.split()
                if wordVectors:
                    cur_sentence.append((sp[0].lower(), sp[1]))
                else:
                    cur_sentence.append((sp[0], sp[1]))
        self.wordAndTag = [item for sublist in sentencesList for item in sublist]

        # get ngrams
        self.ngrams = []
        for i in range(len(self.wordAndTag)):
            if self.wordAndTag[i][1] != "S":
                self.ngrams.append(
                    ([self.wordAndTag[i - 2][0], self.wordAndTag[i - 1][0], self.wordAndTag[i][0],
                      self.wordAndTag[i + 1][0], self.wordAndTag[i + 2][0]], self.wordAndTag[i][1]))

        self.ngramsPref = []
        self.ngramsSuff = []
        for i in range(len(self.wordAndTag)):
            if self.wordAndTag[i][1] != "S":
                self.ngramsPref.append(
                    [self.wordAndTag[i - 2][0][:3], self.wordAndTag[i - 1][0][:3], self.wordAndTag[i][0][:3],
                     self.wordAndTag[i + 1][0][:3], self.wordAndTag[i + 2][0][:3]])
                self.ngramsSuff.append(
                    [self.wordAndTag[i - 2][0][-3:], self.wordAndTag[i - 1][0][-3:], self.wordAndTag[i][0][-3:],
                     self.wordAndTag[i + 1][0][-3:], self.wordAndTag[i + 2][0][-3:]])

        # words
        if wordVectors and vocabInput:
            self.vecs = np.loadtxt(wordVectors)
            vocabWords = np.loadtxt(vocabInput, dtype=str, comments=[])
            self.vocab = [word for word in vocabWords if word != '']
        else:
            self.vocab = set([b[0] for b in self.wordAndTag])
            self.vocab.add("UUUNKKK")
        self.prefix = set([b[:3] for b in self.vocab])
        self.prefix.add("UUUNKKK")
        self.suffix = set([b[-3:] for b in self.vocab])
        self.suffix.add("UUUNKKK")
        self.wordToIdx = {word: i for i, word in enumerate(self.vocab)}
        self.idxToWord = {i: word for i, word in enumerate(self.vocab)}
        self.preToIdx = {word: i for i, word in enumerate(self.prefix)}
        self.idxToPre = {i: word for i, word in enumerate(self.prefix)}
        self.sufToIdx = {word: i for i, word in enumerate(self.suffix)}
        self.idxToSuf = {i: word for i, word in enumerate(self.suffix)}

        # tags
        self.tags = set([b[1] for b in self.wordAndTag])
        self.tagToIdx = {word: i for i, word in enumerate(self.tags)}
        self.idxToTag = {i: word for i, word in enumerate(self.tags)}

    def getIdx(self, word, list):
        """
        if word has not been seen return UNK idx
        :param word: word
        :return: idx
        """
        if word in list:
            return list[word]
        else:
            return list["UUUNKKK"]

    def calcAccuracy(self, y_pred, y):
        """
        calc accuracy
        :param y_pred: predicted res
        :param y: true res
        :return: good and total counts
        """
        good = total = 0
        _, y_pred_idx = torch.max(y_pred, 1)
        for i in range(0, len(y) - 1):
            total += 1
            if y_pred_idx.data[i] == y.data[i]:
                if self.idxToTag[y.data[i]] == 'O':
                    total -= 1
                else:
                    good += 1
        return total, good

    def epoch(self, N, model, lossFunction, mainTagger, optimizer=None):
        """
        epochs
        :param N: batch size
        :param model: model
        :param lossFunction: loss function
        :param mainTagger: main tagger
        :param optimizer: optimizer (for train only)
        :return: loss and acc
        """
        random.shuffle(self.ngrams)

        total = good = 0
        totalLoss = torch.Tensor([0])

        for i in range(0, len(self.ngrams) - N, N):
            # Create random Tensors to hold inputs and outputs, and wrap them in Variables
            batch_words = self.ngrams[i:i + N]
            batch_prefix = self.ngramsPref[i:i+N]
            batch_suffix = self.ngramsSuff[i:i+N]
            batch_x_words = [[mainTagger.getIdx(w, mainTagger.wordToIdx) for w in x] for x, y in batch_words]
            batch_x_prefix = [[mainTagger.getIdx(w, mainTagger.preToIdx) for w in x] for x in batch_prefix]
            batch_x_suffix = [[mainTagger.getIdx(w, mainTagger.sufToIdx) for w in x] for x in batch_suffix]

            batch_y = [mainTagger.tagToIdx[y] for x, y in batch_words]

            x_words = torch.autograd.Variable(torch.LongTensor(batch_x_words))
            x_prefix = torch.autograd.Variable(torch.LongTensor(batch_x_prefix))
            x_suffix = torch.autograd.Variable(torch.LongTensor(batch_x_suffix))

            y = torch.autograd.Variable(torch.LongTensor(batch_y))

            # Forward pass: Compute predicted y by passing x to the model
            y_pred = model(x_words,x_prefix,x_suffix)

            # Compute loss
            loss = lossFunction(y_pred, y)
            totalLoss += loss.data
            addTotal, addGood = mainTagger.calcAccuracy(y_pred, y)
            total += addTotal
            good += addGood

            # backprop for train
            if optimizer:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return float(totalLoss[0]) / N, float(good) / total


def genPlot(folder, y, name):
    plt.figure()
    plt.plot(range(len(y)), y, linewidth=2.0)
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(name + " vs Epochs")
    plt.savefig(folder + '/' + name + ".png")


if __name__ == "__main__":
    # read inputs
    dir = sys.argv[1]
    type = sys.argv[2]
    wordVectors = sys.argv[3] if len(sys.argv) > 3 else False
    vocabInput = sys.argv[4] if len(sys.argv) > 4 else False
    out_dir = type + "_output_prefix_and_suffix" + ("_pre_trained" if wordVectors else "")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # batch size
    N = 1000

    # train and dev class
    if wordVectors and vocabInput:
        train = mainTagger(dir + "/train", wordVectors, vocabInput)
    else:
        train = mainTagger(dir + "/train")
    dev = mainTagger(dir + "/dev")
    # model
    D_in, D_in_prefix, D_in_suffix, E, H, D_out = len(train.vocab), len(train.prefix), len(train.suffix), 50, 100, len(train.tags)
    model = MLP(N, D_in, D_in_prefix, D_in_suffix, E, H, D_out)

    # load Embedding if exists
    if wordVectors:
        model.embWords.weight.data.copy_(torch.from_numpy(train.vecs))
    # loss Cross Entropy
    lossFunction = nn.CrossEntropyLoss()

    # Adam optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # EPOCH
    EPOCH_NUM = 25
    epochsDevAcc = []
    epochsDevLoss = []

    for epoch in range(EPOCH_NUM):
        # Train
        trainLoss, trainAccuracy = train.epoch(N, model, lossFunction, train, optimizer=optimizer)
        #learning_rate = max(learning_rate/2,1e-3)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Dev
        devLoss, devAccuracy = dev.epoch(N, model, lossFunction, train)

        # Print
        print "epoch:", epoch, " train loss:", trainLoss, " train acc:", trainAccuracy, " dev loss:", devLoss, " dev acc:", devAccuracy
        epochsDevLoss.append(devLoss)
        epochsDevAcc.append(devAccuracy)

    # save graph and model
    torch.save(model, out_dir + "/model_file.pt")
    np.save(out_dir + '/idx_to_tag.npy', train.idxToTag)
    np.save(out_dir + '/word_to_idx.npy', train.wordToIdx)
    np.save(out_dir + '/idx_to_pre.npy', train.idxToPre)
    np.save(out_dir + '/pre_to_idx.npy', train.preToIdx)
    np.save(out_dir + '/idx_to_suf.npy', train.idxToSuf)
    np.save(out_dir + '/suf_to_idx.npy', train.sufToIdx)
    genPlot(out_dir, epochsDevLoss, "dev_loss")
    genPlot(out_dir, epochsDevAcc, "dev_accuracy")

