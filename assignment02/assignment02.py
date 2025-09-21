import nltk
from nltk.corpus import wordnet, stopwords
import spacy
import pandas as pd
from collections import Counter
import string

# Load spacy model
nlp = spacy.load('en_core_web_sm')


def part01():
    consume_synset = wordnet.synset('consume.v.02')
    all_hyponyms = set(consume_synset.closure(lambda x: x.hyponyms()))
    all_hyponyms.add(consume_synset)
    return all_hyponyms


def part02(synset_collection):
    lemma_names = set()
    for synset in synset_collection:
        for lemma_name in synset.lemma_names():
            if '_' not in lemma_name:
                lemma_names.add(lemma_name)
    return lemma_names


def part03(tweet_texts, consumption_words):

    stop_words = set(stopwords.words('english'))
    punctuation = set(string.punctuation)
    
    bigram_counter = Counter()
    
    for tweet in tweet_texts:

        doc = nlp(tweet.lower())
        
        # Extract lemmatized tokens, excluding stopwords and punctuation
        tokens = []
        for token in doc:
            if (token.lemma_ not in stop_words and 
                token.text not in punctuation and 
                not token.is_punct and 
                not token.is_space and
                token.lemma_.strip() != ''):
                tokens.append(token.lemma_)
        
        # Generate bigrams and count those starting with consumption words
        for i in range(len(tokens) - 1):
            first_word = tokens[i]
            second_word = tokens[i + 1]
            
            if first_word in consumption_words:
                bigram = (first_word, second_word)
                bigram_counter[bigram] += 1
    
    return bigram_counter


if __name__ == "__main__":
    tweets_df = pd.read_excel('Data/1557-tweets.xlsx')

    synsets = part01()
    print(len(synsets))
    print(list(synsets)[:20])

    lemmas = part02(synsets)
    print(len(lemmas))
    print(sorted(list(lemmas))[:10])    
    
    bigrams = part03(tweets_df.text, lemmas)
    print(len(bigrams))
    print(bigrams.most_common(20))