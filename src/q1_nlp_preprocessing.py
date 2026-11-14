"""
CS5720 Home Assignment 4 - Part II, Q1: NLP preprocessing pipeline.

Runs the three classic preprocessing steps from the Chapter 7 (NLP) lecture on
one sentence and prints the result after each step:

    1. tokenize the sentence into words and punctuation
    2. drop common English stop words
    3. stem every remaining token down to its root form

Everything here uses NLTK, the same library the lecture used, so the output can be
compared directly with the slides.
"""

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# The sentence the assignment asks us to process.
SENTENCE = "NLP techniques are used in virtual assistants like Alexa and Siri."


def ensure_nltk_data() -> None:
    """Download the NLTK resources this script needs, if they are not present.

    `punkt_tab` holds the sentence/word tokenizer model and `stopwords` holds the
    English stop word list. The downloads are small and only happen once; after that
    nltk.data.find() succeeds and nothing is fetched.
    """
    for resource, path in [("punkt_tab", "tokenizers/punkt_tab"),
                           ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def preprocess(sentence: str) -> tuple[list[str], list[str], list[str]]:
    """Tokenize, remove stop words, and stem a sentence.

    Returns the three intermediate lists so the caller can print or test each stage.
    The stop word check lowercases each token before looking it up, because the NLTK
    list is all lowercase and we do not want a capitalized "The" to slip through.
    Punctuation is left in on purpose: the assignment's expected output describes
    the first list as "all words and punctuation", and the lecture example kept the
    trailing "!" as well.
    """
    # Step 1 - tokenize. word_tokenize splits on whitespace and also peels
    # punctuation off into its own token, so the final "." becomes a separate item.
    tokens = word_tokenize(sentence)

    # Step 2 - remove stop words. These are high-frequency function words
    # ("are", "in", "and", "like") that carry little topical meaning on their own.
    stop_words = set(stopwords.words("english"))
    tokens_no_stop = [tok for tok in tokens if tok.lower() not in stop_words]

    # Step 3 - stem. The Porter stemmer strips suffixes with fixed rules, so
    # "techniques" -> "techniqu" and "assistants" -> "assist". The result is not
    # always a real dictionary word; that is the trade-off versus lemmatization.
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(tok) for tok in tokens_no_stop]

    return tokens, tokens_no_stop, stemmed


def main() -> None:
    ensure_nltk_data()
    tokens, tokens_no_stop, stemmed = preprocess(SENTENCE)

    print("Sentence:", SENTENCE)
    print()
    print("1. Original Tokens:          ", tokens)
    print("2. Tokens Without Stopwords: ", tokens_no_stop)
    print("3. Stemmed Words:            ", stemmed)

    # Show which words the stop word filter actually removed, so the effect of
    # step 2 is visible rather than implied.
    removed = [tok for tok in tokens if tok not in tokens_no_stop]
    print()
    print("Stop words removed:", removed)


if __name__ == "__main__":
    main()
