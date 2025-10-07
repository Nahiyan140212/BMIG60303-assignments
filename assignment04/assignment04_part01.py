import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDiA
from nltk.tokenize import casual_tokenize
from langdetect import detect, detect_langs, LangDetectException
from nltk.corpus import stopwords
import pyLDAvis.lda_model
import pyLDAvis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')


# Load the dataset
file_path = '../Data/All_Articles_Excel_Augustuntil9October2020.xlsx'
df = pd.read_excel(file_path)

def detect_language(text):
    try:
        if pd.isna(text) or str(text).strip() == '':
            return False
        return detect(str(text)) == 'en'
    except LangDetectException:
        return False
    except:
        return False
# Apply language detection
df_abstracts = pd.DataFrame({
    'Abstract': df['Abstract'],
    'English': df['Abstract'].apply(detect_language)
})



# Sample English
df_english = df_abstracts[df_abstracts['English'] == True].copy()

# Remove any NaN abstracts
df_english = df_english[df_english['Abstract'].notna()]


# Sample 5000 
df_sample = df_english.sample(n=5000, random_state=0).reset_index(drop=True)


# Vectorize the abstracts
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=1000)
doc_term_matrix = vectorizer.fit_transform(df_sample['Abstract'])

# Build and fit LDA model with 10 topics
lda_model = LatentDirichletAllocation(n_components=10, random_state=0, max_iter=10)
lda_output = lda_model.fit_transform(doc_term_matrix)

# Helper function
def get_topic_words(model, vectorizer, n_words=10):
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics[f'Topic {topic_idx}'] = top_words
    
    return topics

topics = get_topic_words(lda_model, vectorizer, n_words=10)

for topic_name, words in topics.items():
    print(f"\n{topic_name}:")
    print(", ".join(words))
