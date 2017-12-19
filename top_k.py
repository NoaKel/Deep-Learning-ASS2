import numpy as np
import heapq
import sys
vecs = np.loadtxt(sys.argv[1]).tolist()
vocab = np.loadtxt(sys.argv[2],dtype=str, comments=[]).tolist()

def most_similar(word, k):
    distances = {}
    for idx, v in enumerate(vecs):
        distances[distance(word, v)] = vocab[idx]
    values = heapq.nlargest(k+1, distances)[1:]
    words = [distances[key] for key in values]
    return words, values

def distance(u,v):
    return np.dot(u,v)/(np.sqrt(np.dot(u,u))*np.sqrt(np.dot(v,v)))

if __name__ == '__main__':
    print "dog:", most_similar(vecs[vocab.index("dog")],5)
    print "england:", most_similar(vecs[vocab.index("england")],5)
    print "john:", most_similar(vecs[vocab.index("john")],5)
    print "explode:", most_similar(vecs[vocab.index("explode")],5)
    print "office:", most_similar(vecs[vocab.index("office")],5)
