import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel('../Data/All_Articles_Excel_Augustuntil9October2020.xlsx')

def preprocess(df):
    df = df.dropna(subset=['Abstract'])
    df = df.reset_index(drop=True)
    return df

def part01(df):
    df_5000 = df.sample(n=5000, random_state=42)
    df_5000['Abstract'] = df_5000['Abstract'].astype(str)
    v1 = TfidfVectorizer(max_features=1200)
    mx1 = v1.fit_transform(df_5000['Abstract'])
    dis_vecs = pd.DataFrame(mx1.todense(), columns=v1.get_feature_names_out())
    return dis_vecs, df_5000

def part02(df_vectors):
    result = []
    for i in range(10):
        doc = df_vectors.iloc[i]
        top5 = doc.nlargest(5)
        result.append(list(top5.items()))
    return result

def part03(df_vectors, query_string):
    vocabulary = df_vectors.columns.tolist()
    query_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    query_vector = query_vectorizer.fit_transform([query_string])
    similarities = cosine_similarity(query_vector, df_vectors.values)
    similarity_scores = similarities[0]
    top_5_indices = similarity_scores.argsort()[-5:][::-1]
    return top_5_indices

def examine_search_results(df, indices, query):
    abstracts = []
    for i, idx in enumerate(indices):
        abstract = df.iloc[idx]['Abstract']
        abstracts.append(abstract)
    return abstracts

df = preprocess(df)
dis_vecs, df_5000 = part01(df)

first = dis_vecs.iloc[0]
top5 = first.nlargest(5)
top5_items = list(top5.items())

part02_results = part02(dis_vecs)

query1_results = part03(dis_vecs, 'covid vaccine causes fever')
query1_abstracts = examine_search_results(df_5000, query1_results, 'covid vaccine causes fever')

query2_results = part03(dis_vecs, 'covid is caused by corona virus')
query2_abstracts = examine_search_results(df_5000, query2_results, 'covid is caused by corona virus')

query3_results = part03(dis_vecs, 'What are the symptoms of coronavirus?')
query3_abstracts = examine_search_results(df_5000, query3_results, 'What are the symptoms of coronavirus?')