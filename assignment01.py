import pandas as pd
import re
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import bigrams
from nltk.corpus import stopwords
import string

def part01(df):
    results = []
    pattern = re.compile(r'\b(Italy|Italian)\b', re.IGNORECASE)
    
    for abstract in df['Abstract']:
        if pd.isna(abstract) or not isinstance(abstract, str):
            continue
            
        matches = pattern.findall(abstract)
        if matches:
            results.append((matches, len(matches), abstract))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:10]

def part02(df):
    italy_pattern = re.compile(r'\b(Italy|Italian|Italians)\b', re.IGNORECASE)
    china_pattern = re.compile(r'\b(China|Chinese)\b', re.IGNORECASE)
    
    italy_bigrams = []
    china_bigrams = []
    
    for abstract in df['Abstract']:
        if pd.isna(abstract) or not isinstance(abstract, str):
            continue
            
        tokens = word_tokenize(abstract.lower())
        abstract_bigrams = list(bigrams(tokens))
        
        for bigram in abstract_bigrams:
            first_word, second_word = bigram
            
            if italy_pattern.match(first_word):
                italy_bigrams.append(bigram)
            
            if china_pattern.match(first_word):
                china_bigrams.append(bigram)
    
    italy_counter = Counter(italy_bigrams)
    china_counter = Counter(china_bigrams)
    
    return (italy_counter.most_common(20), china_counter.most_common(20))

def part03(df):
    stop_words = set(stopwords.words('english'))
    italy_pattern = re.compile(r'\b(italy|italian|italians)\b', re.IGNORECASE)
    china_pattern = re.compile(r'\b(china|chinese)\b', re.IGNORECASE)
    
    italy_bigrams = []
    china_bigrams = []
    
    for abstract in df['Abstract']:
        if pd.isna(abstract) or not isinstance(abstract, str):
            continue
            
        tokens = word_tokenize(abstract.lower())
        
        filtered_tokens = []
        for token in tokens:
            if (token not in string.punctuation and 
                token not in stop_words and 
                not all(c in string.punctuation for c in token)):
                filtered_tokens.append(token)
        
        abstract_bigrams = list(bigrams(filtered_tokens))
        
        for bigram in abstract_bigrams:
            first_word, second_word = bigram
            
            if italy_pattern.match(first_word):
                italy_bigrams.append(bigram)
            
            if china_pattern.match(first_word):
                china_bigrams.append(bigram)
    
    italy_counter = Counter(italy_bigrams)
    china_counter = Counter(china_bigrams)
    
    return (italy_counter.most_common(20), china_counter.most_common(20))




def main(df):
    # PART 01
    part01_results = part01(df)
    for i, (matches, count, abstract) in enumerate(part01_results, 1):
        print(f"{i}. Count: {count}, Matches: {matches}")
        print(f"   Abstract: {abstract[:100]}...")
        print()

    # PART 01
    italy_bigrams_02, china_bigrams_02 = part02(df)

    print("Top 20 Italy/Italian bigrams:")
    for bigram, count in italy_bigrams_02:
        print(f"  {bigram}: {count}")

    print("\nTop 20 China/Chinese bigrams:")
    for bigram, count in china_bigrams_02:
        print(f"  {bigram}: {count}")

    # PART 01
    italy_bigrams_03, china_bigrams_03 = part03(df)

    print("Top 20 Italy/Italian bigrams (no stopwords/punctuation):")
    for bigram, count in italy_bigrams_03:
        print(f"  {bigram}: {count}")

    print("\nTop 20 China/Chinese bigrams (no stopwords/punctuation):")
    for bigram, count in china_bigrams_03:
        print(f"  {bigram}: {count}")

if __name__ == "__main__":
    df = pd.read_excel('Data\All_Articles_Excel_Augustuntil9October2020.xlsx', index_col=False)
    main(df)