import sys
import torch
import numpy as np
from tagger3 import MLP

def predictFile(nGRAMS, ngramsPref, ngramsSuff, model, N, getWordIdx, getPreIdx, getSufIdx, idxToTag, fileName):
    """
    greedy predict
    :param nGRAMS: test nGrams
    :param model: model from saved model file
    :param N: batch size
    :param getWordIdx: word to index dictionary from saved
    :param idxToTag: tag to index dictionart from saved
    :param fileName: output file name
    :return: N/A
    """
    file = []

    for i in range(0, len(nGRAMS), N):

        batch = nGRAMS[i:i + N]
        batch_pref = ngramsPref[i:i + N]
        batch_suff = ngramsSuff[i:i + N]

        batchLength = len(batch)
        while len(batch) < N:
            batch.append(batch[-1])


        batch_x = [[getWordIdx(w) for w in x] for x in batch]
        batch_x_prex = [[getPreIdx(w) for w in x] for x in batch]
        batch_x_suffx = [[getSufIdx(w) for w in x] for x in batch]

        x = torch.autograd.Variable(torch.LongTensor(batch_x))
        xpred = torch.autograd.Variable(torch.LongTensor(batch_x_prex))
        xsuff = torch.autograd.Variable(torch.LongTensor(batch_x_suffx))

        y_pred = model(x, xpred, xsuff)

        _, prediction = torch.max(y_pred, 1)

        for j in range(0, len(prediction[:batchLength]) - 1):
            word = nGRAMS[i + j][2]
            tag = idxToTag[prediction.data[j]]
            file.append(word + "\t" + tag)
            if nGRAMS[i + j][-2:] == ["</s>", "</s>"]:
                file.append('')
    f = open(fileName, 'w')
    f.write('\n'.join(file))
    f.close()

def getNGRAMS(fileName):
    """
    generate nGrams words only
    :param fileName: test file
    :return: word ngrams
    """
    sentencesList = []
    with open(fileName) as input_file:
        cur_sentence = []
        for line in input_file.read().split('\n'):
            if len(line) == 0:
                sentencesList.append(
                    ["<s>", "<s>"] +
                    cur_sentence +
                    ["</s>", "</s>"])
                cur_sentence = []
                continue
            cur_sentence.append(line)
    words = [item for sublist in sentencesList for item in sublist]

    ngrams = []
    for i in range(len(words)):
        if words[i] not in {"<s>","</s>"}:
            ngrams.append(
                [words[i - 2], words[i - 1], words[i], words[i + 1], words[i + 2]] )

    ngramsPref = []
    ngramsSuff = []
    for i in range(len(words)):
        if words[i] not in {"<s>","</s>"}:
            ngramsPref.append(
                [words[i - 2][:3], words[i - 1][:3], words[i][:3],
                 words[i + 1][:3], words[i + 2][:3]])
            ngramsSuff.append(
                [words[i - 2][-3:], words[i - 1][-3:], words[i][-3:],
                 words[i + 1][-3:], words[i + 2][-3:]])
    return ngrams, ngramsPref, ngramsSuff



        
if __name__ == "__main__":
    # read inputs
    test_input = sys.argv[1]
    model_path = sys.argv[2]
    wordToIdx = np.load(sys.argv[3]).item()
    idxToTag = np.load(sys.argv[4]).item()
    preToIdx = np.load(sys.argv[5]).item()
    sufToIdx = np.load(sys.argv[6]).item()
    output_file = sys.argv[7]

    def getWordIdx(word):
        if word in wordToIdx:
            return wordToIdx[word]
        else:
            return wordToIdx["UUUNKKK"]

    def getPreIdx(word):
        if word in preToIdx:
            return preToIdx[word]
        else:
            return preToIdx["UUUNKKK"]

    def getSufIdx(word):
        if word in sufToIdx:
            return sufToIdx[word]
        else:
            return sufToIdx["UUUNKKK"]


    # nGRAMS
    ngrams, ngramsPref, ngramsSuff = getNGRAMS(test_input)
    # batch size
    N = 1000
    # load model
    model = torch.load(model_path)
    # predict
    predictFile(ngrams, ngramsPref, ngramsSuff, model, N , getWordIdx, getPreIdx, getSufIdx, idxToTag, output_file)
