# Analysis
There are around 46,000 abstracts, almost one third of them are NaN. Then randomly selected 5000 abstract from the dataset and converted into TF-IDF vectors using scikit-learn limiting vocabulary to 1,200 features to manage memory, and returned a DataFrame where each row represents one abstract as a numerical vector showing the importance of each word. Then analyzed the TF-IDF vectors to extract the top 5 most important words from the first 10 documents. Using pandas' nlargest() method, created a tuple containing (word, score) pairs from each abstract. After that, built a search engine that takes a query string and finds the 5 most similar documents. It only returned the indices. To explore more, using the indices, the abstracts are retrieved. 

Finally, I tested the search function with queries like 'covid vaccine causes fever', 'coronavirus respiratory symptoms treatment' and 'What are the symptoms of coronavirus?'. It return most matched abstract and found that the top results are in different language other than English. I should have removed all other language than English to get the best match. Another important factor I noticed that the search working as key word matching engine rather than semantic similarity even though we used cosine similarity. The search system is also not able to return the response of a query in a structured format. For example, if we ask any question, it just returning the abstract where there might be the answer. Then we need to find the answer by reading the abstract. Some of the abstract it return that does not really made sense with the query we are asking. 

For example, I tried to find query 
```
covid vaccine causes fever

```
it returned:

```
Background The COVID-19 pandemic continues to adversely affect the U S , which leads globally in total cases and deaths As COVID-19 vaccines are under development, public health officials and policymakers need to create strategic vaccine-acceptance messaging to effectively control the pandemic and prevent thousands of additional deaths... Males (72%) compared to females, older adults (≥55 years;78%) compared to younger adults, Asians (81%) compared to other racial and ethnic groups, and college and/or graduate degree holders (75%) compared to people with less than a college degree were more likely to accept the vaccine...

```

This shows that the abstract mentioned something about covid vaccine, but I am not sure whether this answered my question or not.

This is definitely helpful for keyword matching or keyword search system. However, it is not as intelligent as other transoformer based model out there.

