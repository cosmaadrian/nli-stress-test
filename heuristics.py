
import string
from spellchecker import SpellChecker
from nltk.corpus import wordnet


def word_overlap(text1, text2):
    """
     Computes the word overlap between two texts.
    Returns the percentage of words that overlap.
    """
    # remove punctuation
    text1 = text1.translate(str.maketrans('', '', string.punctuation))
    text2 = text2.translate(str.maketrans('', '', string.punctuation))

    text1 = text1.lower().split()
    text2 = text2.lower().split()
    overlap = set(text1).intersection(set(text2))

    if len(text2) == 0:
        return 0

    return len(overlap) / len(text2)


def contains_negation(text1, text2):
    """
        Returns True if the text contains a negation word.
    """
    text1 = text1.lower()
    text1 = text1.translate(str.maketrans('', '', string.punctuation))
    text1 = text1.split()

    text1_has_negation = False

    if 'no' in text1 or 'not' in text1 or 'never' in text1 or 'none' in text1:
        text1_has_negation = True

    text2 = text2.lower()
    text2 = text2.translate(str.maketrans('', '', string.punctuation))
    text2 = text2.split()

    text2_has_negation = False

    if 'no' in text2 or 'not' in text2 or 'never' in text2 or 'none' in text2:
        text2_has_negation = True

    return int(text1_has_negation or text2_has_negation)

def misspelled_words(text1, text2):
    text = text1.lower() + ' ' + text2.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    wordlist = text.split()

    spell = SpellChecker()
    amount_miss = len(list(spell.unknown(wordlist)))

    if len(wordlist) == 0:
        return 0

    return amount_miss / len(wordlist)

def number_of_antonyms(text1, text2):
    """
        Returns the number of antonyms of the words in text1 contained in text2.
    """

    text1 = text1.lower()
    text2 = text2.lower()

    text1 = text1.translate(str.maketrans('', '', string.punctuation))
    text2 = text2.translate(str.maketrans('', '', string.punctuation))

    text1 = text1.split()
    text2 = text2.split()

    antonyms = []
    for word_text1 in text1:
        for syn in wordnet.synsets(word_text1):
            for lm in syn.lemmas():
                if lm.antonyms():
                    antonyms.append(lm.antonyms()[0].name())

    antonyms = set(antonyms)
    count = 0
    for word_text2 in text2:
        if word_text2 in antonyms:
            count += 1

    if len(text2) == 0:
        return 0

    return count / len(text2)

def length_missmatch(text1, text2):
    """
        Returns the difference in length between text1 and text2.
    """
    text1 = text1.lower()
    text2 = text2.lower()

    text1 = text1.translate(str.maketrans('', '', string.punctuation))
    text2 = text2.translate(str.maketrans('', '', string.punctuation))

    text1 = text1.split()
    text2 = text2.split()

    if len(text2) == 0:
        return 0

    return (len(text1) - len(text2)) / len(text2)
