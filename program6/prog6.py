# Program 6: Implementation of Naive Bayes (manual pages 34-37).
# Aim: predict a message's class and calculate classification accuracy.
# Description of the Data Set:
# Name of the Data Set: UCI SMS Spam Collection (sms.tsv beside this file).
# Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
# Mirror: https://github.com/justmarkham/pycon-2016-tutorial/blob/master/data/sms.tsv
# No. of Features: 1 raw text feature, message (converted to word counts).
# No. of Instances: 5574 in the supplied mirror (4827 ham and 747 spam).
# Remove 403 exact duplicates: use 5171 messages (4518 ham and 653 spam).
# No. of Numerical Data: 0 raw input columns; encoded word counts are numeric.
# No. of Categorical Data: 0 input columns (message is unstructured text);
# the target is 1 categorical column. Input counts exclude this target.
# Name of the Target Variable: label (spam or ham).
# Theory: multinomial Naive Bayes assumes words are independent given a class.
# P(class | message) is proportional to P(class) * product(P(word | class)).
# Algorithm (p35-36): count classes/words; normalize probabilities in training;
# score every test word with log probabilities; choose the highest class score.
# Seen word: P(word | class) = count / total words in that class.
# Absent word: use the manual's printed (1+1)/(class word total + vocabulary).
# Process all test words, including words absent from the training vocabulary.
# Repeated words count repeatedly. Vocabulary size comes from training only.
# Manual note: the printed absent-word rule is retained exactly as requested;
# it is not the standard, consistently normalized add-one smoothing formula.
# Advantage: simple and fast. Limits: word independence ignores context, and
# class imbalance means accuracy alone can hide missed spam messages.
# Input: bundled TSV. Remove exact duplicate messages before the split; messages
# differing only in punctuation/case may remain. Shuffle each class with seed 7.
# Use 80% train/20% test: 4136 training and 1035 test messages (rounded per class).
# Output: first 10 actual/predicted labels and accuracy across ALL test messages.
# Requires Python's standard library only. Run: python3 prog6.py

import math
import random
import re
from collections import Counter
from pathlib import Path


def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


def train(rows):
    classes, counts = Counter(), {}
    for text, label in rows:
        classes[label] += 1
        counts.setdefault(label, Counter()).update(tokenize(text))
    vocabulary = set().union(*(set(words) for words in counts.values()))
    totals = {label: sum(words.values()) for label, words in counts.items()}
    feature_probabilities = {
        label: {word: count / totals[label] for word, count in words.items()}
        for label, words in counts.items()}
    class_probabilities = {label: count / len(rows) for label, count in classes.items()}
    return class_probabilities, feature_probabilities, totals, len(vocabulary)


def predict(text, model):
    class_probabilities, feature_probabilities, totals, vocabulary_size = model
    words = tokenize(text)
    scores = {}
    for label in class_probabilities:
        score = math.log(class_probabilities[label])
        for word in words:
            if word not in feature_probabilities[label]:
                score += math.log((1 + 1) / (totals[label] + vocabulary_size))
            else:
                score += math.log(feature_probabilities[label][word])
        scores[label] = score
    return max(scores, key=scores.get)  # Equal scores use training class order.


if __name__ == "__main__":
    with (Path(__file__).parent / "sms.tsv").open(encoding="utf-8") as file:
        rows = []
        for line in file:
            label, text = line.rstrip("\n").split("\t", 1)
            rows.append((text, label))
    rows = list(dict.fromkeys(rows))  # Prevent identical messages crossing sets.
    training, testing = [], []
    rng = random.Random(7)
    for label in sorted({label for text, label in rows}):
        group = [row for row in rows if row[1] == label]
        rng.shuffle(group)
        split = int(0.8 * len(group))
        training.extend(group[:split])
        testing.extend(group[split:])
    rng.shuffle(testing)
    model = train(training)
    correct = 0
    print(f"Unique messages: {len(rows)} (exact duplicates removed)")
    print(f"Training: {len(training)} messages | Testing: {len(testing)} messages")
    print("First 10 predictions:")
    for i, (text, actual) in enumerate(testing):
        predicted = predict(text, model)
        correct += predicted == actual
        if i < 10:
            print(f"Actual: {actual:4} | Predicted: {predicted:4}")
    print(f"Accuracy: {correct / len(testing):.2%}")
