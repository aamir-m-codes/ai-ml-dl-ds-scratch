import pandas as pd
from collections import defaultdict
import math

df = pd.read_csv("spam_ham_dataset.csv")

spam_rows = df[df["label"] == "spam"]
ham_rows = df[df["label"] == "ham"]

vocab = defaultdict(int)
bow_spam = defaultdict(int)
bow_ham = defaultdict(int)
spam_cond = defaultdict(int)
ham_cond = defaultdict(int)

tp = tn = fp = fn = 0

spam_rows_total_len = len(spam_rows)
ham_rows_total_len = len(ham_rows)

# trim data into 50:50 ratio
if spam_rows_total_len < ham_rows_total_len:
    ham_rows = ham_rows[:spam_rows_total_len]
    ham_rows_total_len = spam_rows_total_len
else:
    spam_rows = spam_rows[:ham_rows_total_len]
    spam_rows_total_len = ham_rows_total_len

# test and train split (20|80)
train_size = int(spam_rows_total_len * 0.8)

spam_rows_test = spam_rows[train_size:]
ham_rows_test = ham_rows[train_size:]

spam_rows = spam_rows[:train_size]
ham_rows = ham_rows[:train_size]

# store vocabulary and bags of words
for row in spam_rows["text"]:
    for word in row.split(" "):
        vocab[word.lower()] += 1
        bow_spam[word.lower()] += 1

for row in ham_rows["text"]:
    for word in row.split(" "):
        vocab[word.lower()] += 1
        bow_ham[word.lower()] += 1

# prior probability
spam_prior = spam_rows_total_len / (spam_rows_total_len + ham_rows_total_len)
ham_prior = ham_rows_total_len / (spam_rows_total_len + ham_rows_total_len)

total_words_spam = sum(bow_spam.values())
total_words_ham = sum(bow_ham.values())
total_vocab = len(vocab)

# conditional probability
for word in vocab:
    # spam_cond[word] = float((bow_spam[word] + 1) / (len(bow_spam) + len(vocab)))
    spam_cond[word] = float((bow_spam[word] + 1) / (total_words_spam + total_vocab))
    # ham_cond[word] = float((bow_ham[word] + 1) / (len(bow_ham) + len(vocab)))
    ham_cond[word] = float((bow_ham[word] + 1) / (total_words_ham + total_vocab))

# test phase
for row in spam_rows_test["text"]:
    # temp_spam_probab = spam_prior
    temp_spam_probab = math.log(spam_prior)
    # temp_ham_probab = ham_prior
    temp_ham_probab = math.log(ham_prior)
    
    for word in row.split(" "):
        norm_word = word.lower()
        # if norm_word in bow_spam:
        # if norm_word in bow_ham:
        if norm_word in vocab:
            temp_spam_probab += math.log(spam_cond[norm_word])
            temp_ham_probab += math.log(ham_cond[norm_word])

    if temp_spam_probab >= temp_ham_probab:
        tp += 1
    else:
        fn += 1

for row in ham_rows_test["text"]:
    # temp_spam_probab = spam_prior
    temp_spam_probab = math.log(spam_prior)
    # temp_ham_probab = ham_prior
    temp_ham_probab = math.log(ham_prior)
    
    for word in row.split(" "):
        norm_word = word.lower()
        # if norm_word in bow_spam:
        # if norm_word in bow_ham:
        if norm_word in vocab:
            temp_spam_probab += math.log(spam_cond[norm_word])
            temp_ham_probab += math.log(ham_cond[norm_word])

    if temp_spam_probab <= temp_ham_probab:
        tn += 1
    else:
        fp += 1

print(f"tp -> {tp}")
print(f"fn -> {fn}")
print(f"tn -> {tn}")
print(f"fp -> {fp}")

acc = (tp + tn) / (tp + fn + tn + fp)
prec = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * ((prec * rec) / (prec + rec))

print(f"Accuracy: {acc*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")
print(f"F1 Score: {f1*100:.2f}%")
