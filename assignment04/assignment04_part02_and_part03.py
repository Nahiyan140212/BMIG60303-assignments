from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
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

# Helper function to display topics
def get_topic_words(model, vectorizer, n_words=10):
    """Extract top words for each topic"""
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics[f'Topic {topic_idx}'] = top_words
    
    return topics


# Sample English
df_english = df_abstracts[df_abstracts['English'] == True].copy()

# Remove any NaN abstracts
df_english = df_english[df_english['Abstract'].notna()]


# Sample 5000 
df_sample = df_english.sample(n=5000, random_state=0).reset_index(drop=True)



# Get English stopwords
english_stopwords = stopwords.words('english')

# Store results for comparison
experiments = {}


# EXPERIMENT 1: Add stopwords (using NLTK stopwords)

vectorizer_exp1 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1000,
    stop_words=english_stopwords
)

doc_term_matrix_exp1 = vectorizer_exp1.fit_transform(df_sample['Abstract'])
lda_exp1 = LatentDirichletAllocation(n_components=10, random_state=0, max_iter=10)
lda_exp1.fit(doc_term_matrix_exp1)

topics_exp1 = get_topic_words(lda_exp1, vectorizer_exp1, n_words=10)
experiments['Exp1_Stopwords'] = topics_exp1

for topic_name, words in topics_exp1.items():
    print(f"\n{topic_name}: {', '.join(words)}")

# EXPERIMENT 2: Add stopwords + punctuation


# Combine stopwords with punctuation

stopwords_plus_punct = english_stopwords + list(string.punctuation)

vectorizer_exp2 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1000,
    stop_words=stopwords_plus_punct
)

doc_term_matrix_exp2 = vectorizer_exp2.fit_transform(df_sample['Abstract'])
lda_exp2 = LatentDirichletAllocation(n_components=10, random_state=0, max_iter=10)
lda_exp2.fit(doc_term_matrix_exp2)

topics_exp2 = get_topic_words(lda_exp2, vectorizer_exp2, n_words=10)
experiments['Exp2_Stopwords+Punct'] = topics_exp2

for topic_name, words in topics_exp2.items():
    print(f"\n{topic_name}: {', '.join(words)}")


# EXPERIMENT 3: TreebankWordTokenizer + Stopwords

tokenizer = TreebankWordTokenizer()

vectorizer_exp3 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1000,
    stop_words=english_stopwords,
    tokenizer=tokenizer.tokenize
)

doc_term_matrix_exp3 = vectorizer_exp3.fit_transform(df_sample['Abstract'])
lda_exp3 = LatentDirichletAllocation(n_components=10, random_state=0, max_iter=10)
lda_exp3.fit(doc_term_matrix_exp3)

topics_exp3 = get_topic_words(lda_exp3, vectorizer_exp3, n_words=10)
experiments['Exp3_Treebank+Stopwords'] = topics_exp3

for topic_name, words in topics_exp3.items():
    print(f"\n{topic_name}: {', '.join(words)}")

# EXPERIMENT 4: More features (2000 instead of 1000)

vectorizer_exp4 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=2000,  # Increased from 1000
    stop_words=english_stopwords
)

doc_term_matrix_exp4 = vectorizer_exp4.fit_transform(df_sample['Abstract'])
lda_exp4 = LatentDirichletAllocation(n_components=10, random_state=0, max_iter=10)
lda_exp4.fit(doc_term_matrix_exp4)

topics_exp4 = get_topic_words(lda_exp4, vectorizer_exp4, n_words=10)
experiments['Exp4_MoreFeatures'] = topics_exp4

for topic_name, words in topics_exp4.items():
    print(f"\n{topic_name}: {', '.join(words)}")


# EXPERIMENT 5: Fewer topics (5 topics instead of 10)


vectorizer_exp5 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1000,
    stop_words=english_stopwords
)

doc_term_matrix_exp5 = vectorizer_exp5.fit_transform(df_sample['Abstract'])
lda_exp5 = LatentDirichletAllocation(n_components=5, random_state=0, max_iter=10)
lda_exp5.fit(doc_term_matrix_exp5)

topics_exp5 = get_topic_words(lda_exp5, vectorizer_exp5, n_words=10)
experiments['Exp5_FewerTopics'] = topics_exp5

for topic_name, words in topics_exp5.items():
    print(f"\n{topic_name}: {', '.join(words)}")

# EXPERIMENT 6: More topics (15 topics)

vectorizer_exp6 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1000,
    stop_words=english_stopwords
)

doc_term_matrix_exp6 = vectorizer_exp6.fit_transform(df_sample['Abstract'])
lda_exp6 = LatentDirichletAllocation(n_components=15, random_state=0, max_iter=10)
lda_exp6.fit(doc_term_matrix_exp6)

topics_exp6 = get_topic_words(lda_exp6, vectorizer_exp6, n_words=10)
experiments['Exp6_MoreTopics'] = topics_exp6

for topic_name, words in topics_exp6.items():
    print(f"\n{topic_name}: {', '.join(words)}")


# EXPERIMENT 7: TfidfVectorizer instead of CountVectorizer

vectorizer_exp7 = TfidfVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1000,
    stop_words=english_stopwords
)

doc_term_matrix_exp7 = vectorizer_exp7.fit_transform(df_sample['Abstract'])
lda_exp7 = LatentDirichletAllocation(n_components=10, random_state=0, max_iter=10)
lda_exp7.fit(doc_term_matrix_exp7)

topics_exp7 = get_topic_words(lda_exp7, vectorizer_exp7, n_words=10)
experiments['Exp7_TfidfVectorizer'] = topics_exp7

for topic_name, words in topics_exp7.items():
    print(f"\n{topic_name}: {', '.join(words)}")


# EXPERIMENT 8: Custom stopwords 
print("\n" + "-"*70)
print("EXPERIMENT 8: Custom Stopwords (removing artifacts)")
print("-"*70)

# Add custom stopwords for COVID papers
custom_stopwords = list(english_stopwords) + [
    'covid', '19', 'covid-19', '2019', '2020', '2021',
    'de', 'la', 'quot', 'et', 'al',
    '95', 'ci', 'abstract', 'conclusion', 'background'
]

vectorizer_exp8 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1000,
    stop_words=custom_stopwords
)

doc_term_matrix_exp8 = vectorizer_exp8.fit_transform(df_sample['Abstract'])
lda_exp8 = LatentDirichletAllocation(n_components=10, random_state=0, max_iter=10)
lda_exp8.fit(doc_term_matrix_exp8)

topics_exp8 = get_topic_words(lda_exp8, vectorizer_exp8, n_words=10)
experiments['Exp8_CustomStopwords'] = topics_exp8

for topic_name, words in topics_exp8.items():
    print(f"\n{topic_name}: {', '.join(words)}")

# EXPERIMENT 9: Best config with more iterations

print("EXPERIMENT 9: Custom Stopwords + More Iterations (max_iter=20)")

vectorizer_exp9 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1500,
    stop_words=custom_stopwords
)

doc_term_matrix_exp9 = vectorizer_exp9.fit_transform(df_sample['Abstract'])
lda_exp9 = LatentDirichletAllocation(n_components=12, random_state=0, max_iter=20)
lda_exp9.fit(doc_term_matrix_exp9)

topics_exp9 = get_topic_words(lda_exp9, vectorizer_exp9, n_words=10)
experiments['Exp9_CustomStopwords_MoreIter'] = topics_exp9

for topic_name, words in topics_exp9.items():
    print(f"\n{topic_name}: {', '.join(words)}")

# EXPERIMENT 10: Keep COVID-19 terms but remove artifacts

print("\n" + "-"*70)
print("EXPERIMENT 10: Moderate Stopwords (keep COVID terms, remove artifacts)")
print("-"*70)

moderate_stopwords = list(english_stopwords) + [
    'de', 'la', 'quot', 'et', 'al',
    '95', 'ci', 'abstract', 'conclusion', 'background',
    'results', 'methods', 'study', 'also'
]

vectorizer_exp10 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1500,
    stop_words=moderate_stopwords
)

doc_term_matrix_exp10 = vectorizer_exp10.fit_transform(df_sample['Abstract'])
lda_exp10 = LatentDirichletAllocation(n_components=12, random_state=0, max_iter=15)
lda_exp10.fit(doc_term_matrix_exp10)

topics_exp10 = get_topic_words(lda_exp10, vectorizer_exp10, n_words=10)
experiments['Exp10_ModerateStopwords'] = topics_exp10

for topic_name, words in topics_exp10.items():
    print(f"\n{topic_name}: {', '.join(words)}")


# EXPERIMENT 11: All Languages (No English Filter)

# Sample 5000 from ALL abstracts (including non-English)

df_all_languages = df_abstracts[df_abstracts['Abstract'].notna()].copy()
df_sample_all = df_all_languages.sample(n=5000, random_state=0).reset_index(drop=True)

# Convert all abstracts to strings and handle any issues
df_sample_all['Abstract'] = df_sample_all['Abstract'].astype(str)

# Check language distribution
english_count = df_sample_all['English'].sum()
non_english_count = len(df_sample_all) - english_count
print(f"English abstracts: {english_count} ({english_count/50:.1f}%)")
print(f"Non-English abstracts: {non_english_count} ({non_english_count/50:.1f}%)")

# same configuration as Experiment 1
vectorizer_exp11 = CountVectorizer(
    max_df=0.95, 
    min_df=2, 
    max_features=1000,
    stop_words=english_stopwords  # Using English stopwords
)

doc_term_matrix_exp11 = vectorizer_exp11.fit_transform(df_sample_all['Abstract'])
print(f"\nDocument-term matrix shape: {doc_term_matrix_exp11.shape}")

lda_exp11 = LatentDirichletAllocation(n_components=10, random_state=0, max_iter=10)
lda_exp11.fit(doc_term_matrix_exp11)

topics_exp11 = get_topic_words(lda_exp11, vectorizer_exp11, n_words=10)
experiments['Exp11_AllLanguages'] = topics_exp11

print("\nTopics from ALL languages (English + Non-English):")
for topic_name, words in topics_exp11.items():
    print(f"\n{topic_name}: {', '.join(words)}")


# COMPARISON: English-only vs All Languages
print("\nEXPERIMENT 1 (English only) Sample Topics")
for i, (topic_name, words) in enumerate(list(experiments['Exp1_Stopwords'].items())[:5]):
    print(f"{topic_name}: {', '.join(words)}")

print("\n EXPERIMENT 11 (All languages) Sample Topics")
for i, (topic_name, words) in enumerate(list(topics_exp11.items())[:5]):
    print(f"{topic_name}: {', '.join(words)}")



#Part 3
import pyLDAvis
import pyLDAvis.lda_model

#the best model from Experiment 9
best_vectorizer = vectorizer_exp9
best_lda_model = lda_exp9
best_doc_term_matrix = doc_term_matrix_exp9


print(f"Model: {best_lda_model.n_components} topics")
print(f"Vocabulary size: {len(best_vectorizer.get_feature_names_out())}")

# visualization
vis_data = pyLDAvis.lda_model.prepare(
    best_lda_model, 
    best_doc_term_matrix, 
    best_vectorizer,
    mds='tsne',  
    sort_topics=False
)

# Save as HTML file
output_file = 'lda_visualization_part3.html'
pyLDAvis.save_html(vis_data, output_file)


# pyLDAvis.display(vis_data)