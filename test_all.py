#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on the
IMDB dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path

import spacy
from spacy.util import minibatch, compounding

@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    category=("Category", "option", "c", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_texts=("Number of texts to train from", "option", "t", int),
    n_iter=("Number of training iterations", "option", "n", int))
def main(model=None, output_dir=None, n_iter=20, n_texts=2000, category=None):
    if category==None:
        print("Category requried parameter")
        exit
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank('en')  # create blank Language class
        print("Created blank 'en' model")

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'textcat' not in nlp.pipe_names:
        textcat = nlp.create_pipe('textcat')
        nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    else:
        textcat = nlp.get_pipe('textcat')

    # add label to text classifier
    textcat.add_label(category)

    # load the dataset
    print("Loading validate data...")
    (validate_texts, validate_cats) = load_data(category, limit=n_texts)
    print("Using {} examples ({} validate)"
          .format(n_texts, len(validate_texts)))
    validate_data = list(zip(validate_texts,
                          [{'cats': cats} for cats in validate_cats]))

    if output_dir is not None:
        output_dir = Path(output_dir)
        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # doc2 = nlp2(test_text)
        # print(test_text, doc2.cats)

        bag_validate_texts = [
        ]

        for test_text in validate_data[0][0]:
            doc2 = nlp2(test_text)
            print(doc2.cats['bag'], '\t', test_text.replace("\t", "").replace("\n", "").replace("\r", ""))


def read_data(category, data_dir, limit=0):
    examples = []
    for subdir, label in ((category, 1), ('other', 0)):
        for filename in (data_dir / subdir).iterdir():
            with filename.open('r', encoding='utf8') as file_:
                text = file_.read()
            text = text.replace('<br />', '\n\n')
            if text.strip():
                examples.append((text, label))
    random.shuffle(examples)
    if limit >= 1:
        examples = examples[:limit]
    return examples

def load_dataset(category, limit=0):
    validate_loc = Path('./' + category) / 'validate'
    return read_data(category, validate_loc, limit=limit)

def load_data(category, limit=0, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    train_data = load_dataset(category)
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    cats = [{category: bool(y)} for y in labels]
    split = int(len(train_data) * split)
    return (texts[:split], cats[:split]), (texts[split:], cats[split:])


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0   # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0   # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}


if __name__ == '__main__':
    plac.call(main)
