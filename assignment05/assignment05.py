import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import gensim
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

df_sample = pd.read_csv('filtered_discharge_notes.csv')

sentences = []
for note in df_sample['text'].dropna():
    # Split into sentences first
    for sent in sent_tokenize(note):
        # Tokenize each sentence
        tokens = word_tokenize(sent.lower())
        sentences.append(tokens)


poor_model = gensim.models.Word2Vec(
    sentences,
    vector_size=50,    
    window=2,           
    min_count=10,       
    epochs=1,           
    workers=4
)


better_model1 = gensim.models.Word2Vec(
    sentences,
    vector_size=150,   
    window=5,           
    min_count=3,        
    epochs=50,          
    workers=4
)


better_model2 = gensim.models.Word2Vec(
    sentences,
    vector_size=200,    
    window=7,           
    min_count=2,        
    epochs=100,         
    sg=1,               
    workers=4
)

# Test medical concept grouping
print("Poor model - retina:")
print(poor_model.wv.most_similar('retina'))

print("\nBetter model 1 - retina:")
print(better_model1.wv.most_similar('retina'))

print("\nBetter model 2 - retina:")
print(better_model2.wv.most_similar('retina'))

# Test medication relationships
print("\nMedication similarity - ibuprofen:")
print(better_model2.wv.most_similar('ibuprofen'))

