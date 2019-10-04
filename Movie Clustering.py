#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk
#nltk.download('punkt')


# In[2]:


#Read data
movies_location = './dataset/movies.csv'

movies = pd.read_csv(movies_location)


# In[3]:


movies.head(10)


# In[4]:


#Joining IMDB Plot and Wiki Plot
movies['plot'] = movies['wiki_plot'].astype(str) +  "\n" + movies['imdb_plot'].astype(str)


# In[5]:


#defining tokenize and snowball stemming method
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re

#English language SnowballStemmer object
stemmer = SnowballStemmer("english")

def token_and_stem(para):
    tokens = [words for sent in nltk.sent_tokenize(para) for words in nltk.word_tokenize(sent)]
    
    #filtering to just words using list comprehensions
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]
    
    #stemming filtered tokens
    stemmed = [stemmer.stem(tok) for tok in filtered_tokens]
    
    return stemmed


# In[7]:


sent_tokenized = [sent for sent in nltk.sent_tokenize("""
                        It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. 
                        The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. 
                        Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. 
                        Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).
                        """)]

# Word Tokenize first sentence from sent_tokenized, save the result in a variable 'words_tokenized'
words_tokenized = [word for word in nltk.word_tokenize(sent_tokenized[0])]

filtered = [word for word in words_tokenized if re.search('[a-zA-Z]', word)]

# Let's observe words after tokenization
filtered


# In[8]:


stemmer = SnowballStemmer("english")

# let's observe words without stemming
print("Without stemming: ", filtered)


# In[10]:


stemmed_words = [stemmer.stem(word) for word in filtered]
# now let's check out after stemming
print("After stemming:   ", stemmed_words)


# In[11]:


#Creating TFIDFVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vector = TfidfVectorizer(stop_words='english',
                                tokenizer=token_and_stem)

plot_matrix = tfidf_vector.fit_transform([plot for plot in movies['plot']])


# In[12]:


#clustering with KMeans
from sklearn.cluster import KMeans

# Create a KMeans object with 5 clusters and save as km
k_means = KMeans(n_clusters=5)

# Fit the k-means object with tfidf_matrix
k_means.fit(plot_matrix)

clusters = k_means.labels_.tolist()

# Create a column cluster to denote the generated cluster for each movie
movies["cluster"] = clusters


# In[13]:


#calculating similarity distance
from sklearn.metrics.pairwise import cosine_similarity

# Calculate the similarity distance
sim_dis = 1 - cosine_similarity(plot_matrix)


# In[14]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from scipy.cluster.hierarchy import linkage, dendrogram

movies_sim_dis_matrix = linkage(sim_dis, method='complete')

movie_dendrogram = dendrogram(movies_sim_dis_matrix,
               labels=[x for x in movies["title"]],
               leaf_rotation=90,
               leaf_font_size=16,
)

fig = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
fig.set_size_inches(108, 21)

# Show the plotted dendrogram
plt.show()


# In[15]:


#makind a dictionary that held the most similar movies based on the ordering of the movies_sim_dis_matrix
similar_movies = {}

for movie in movies_sim_dis_matrix:
    movie_0 = int(movie[0])
    movie_1 = int(movie[1])
    similar_movies[movie_0] = movie_1
    
    


# In[16]:


#Generally we find that movies that are count as a similar value for an earlier value
#do not get their own individual entry to avoid repetition
#As a result we will take all the rated movies and their corresponding most similar movie into 2 separate lists that will be
#searched to create a method to show the most similar movie

rated_movies = []
similar_for_rated = []

for a in similar_movies:
    rated_movies.append(a)
    similar_for_rated.append(similar_movies[a])


# In[19]:


#predict method
def show_most_similar_movie():
    movie_title = input('Please Enter a movie title ').strip()    
    movies['title_lower'] = movies['title'].apply(lambda x: x.lower())
    
    
    
    if any(movies['title_lower'] == movie_title.lower()):
        movie_df = movies[movies['title_lower'] == movie_title]
    else:
        return "Movie does not exist. Please check your spelling and Capitalisations"
    
    
    rank = int(movie_df['rank'])
    
    if rank in rated_movies:
    
        sim_movie_df = movies[movies['rank'] == similar_movies[rank]]
    
        sim_movie = sim_movie_df.title.values
        
    elif rank in similar_for_rated:
        idx = similar_for_rated.index(rank)
        
        sim_movie_df = movies[movies['rank'] == rated_movies[idx]]
        
        sim_movie = sim_movie_df.title.values
        
        
        
    if sim_movie.size > 0:
        sel = sim_movie[0]
    else:
        sel = 'Sorry No Movie Available'
    
    return 'Most Similar movie to \'{}\' is: \'{}\''.format(movie_title, sel)
        


# In[20]:


show_most_similar_movie()

