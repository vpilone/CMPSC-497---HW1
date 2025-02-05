# Vinny Pilone
# This is a very basic implementation of a logistic regression model using softmax

# import depedencies

from datasets import load_dataset
import tensorflow as tf
from sklearn.metrics import accuracy_score
import numpy as np

numSentancesToTrain = 1000

# load test and train datasets
train_dataset = load_dataset("batterydata/pos_tagging", split="train")
test_dataset = load_dataset("batterydata/pos_tagging", split="test")

# create empty dictionaries for tokenization and embeddings
vocab = {}
vocab.setdefault("<OOV>", 0)
vocabInverse = {}
vocabInverse.setdefault(0, "<OOV>")

classifications = {}
classifications.setdefault("<OOV", 0)
classificationsInverse = {}
classificationsInverse.setdefault(0, "<OOV>")

# fill vocab and classification dictionaries
for item in train_dataset["words"][0:numSentancesToTrain]:
    for word in item:
        if vocab.get(word) == None:
            vocab[word] = (int)(len(vocab))
            vocabInverse[(int)(len(vocab)) - 1] = word

for item in train_dataset["labels"][0:numSentancesToTrain]:
    for pos in item:
        if classifications.get(pos) == None:
            classifications[pos] = (int)(len(classifications))
            classificationsInverse[(int)(len(classifications)) - 1] = pos


# Create random embeddings and train them compared to one-hot ground truth
embeddingGen = np.random.default_rng()
embeddings = embeddingGen.standard_normal(size=(len(vocab), len(classifications)))
for epoch in range(1):
    for i in range(len(train_dataset["labels"][0:numSentancesToTrain])):
        for j in range(len(train_dataset["labels"][i])):
            index = vocab.get(train_dataset["words"][i][j])
            softmaxTensor = tf.math.softmax(embeddings[index])
            # print(softmaxTensor)
            # print(tf.math.argmax(softmaxTensor))
            temp = [0] * (len(classifications))
            temp[(int)(classifications[(train_dataset["labels"][i][j])])] = 1
            temp = tf.math.softmax(softmaxTensor + temp)
            embeddings[index] = temp

# create final predictions based on weights with test data
finalPred = []

for item in test_dataset["words"][0:]:
    pred = []
    for word in item:
        try:
            softmaxTensor = tf.math.softmax(embeddings[vocab.get(word)])
            pred.append(classificationsInverse[(int)(tf.math.argmax(softmaxTensor))])
        except:
            pred.append("<00V>")
    finalPred.append(pred)


# report accuracy
correct = 0
total = 0

for i in range(len(finalPred)):
    for j in range(len(finalPred[i])):
        total += 1
        if finalPred[i][j] == test_dataset["labels"][i][j]:
            correct += 1
    print("current loss:", correct / total)

print("final loss:", correct / total)
