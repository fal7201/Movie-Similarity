#!/usr/bin/env python
# coding: utf-8
import re
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
#nltk.download('punkt')
#Read data
MOVIES_LOCATION = './dataset/MOVIES.csv'
MOVIES = pd.read_csv(MOVIES_LOCATION)
#MOVIES.head(10)
#Joining IMDB Plot and Wiki Plot
MOVIES['plot'] = MOVIES['wiki_plot'].astype(str) + "\n" + MOVIES['imdb_plot'].astype(str)
#defining tokenize and snowball stemming method
#English language SnowballStemmer object
STEMMER = SnowballStemmer("english")
def token_and_stem(para):
    tokens = [words for sent in nltk.sent_tokenize(para) for words in nltk.word_tokenize(sent)]    
#filtering to just words using list comprehensions
    filtered_tokens = [token for token in tokens if re.search('[a-zA-Z]', token)]    
#stemming filtered tokens
    stemmed = [STEMMER.stem(tok) for tok in filtered_tokens]    
    return stemmed
SENT_TOKENIZED = [sent for sent in nltk.sent_tokenize("""
                        It is a long established fact that a reader will be 
                        distracted by the readable content of a page when 
                        looking at its layout. 
                        The point of using Lorem Ipsum is that it has a 
                        more-or-less normal distribution of letters, 
                        as opposed to using 'Content here, content here',
                        making it look like readable English. 
                        Many desktop publishing packages and web page editors 
                        now use Lorem Ipsum as their default model text, 
                        and a search for 'lorem ipsum' will uncover 
                        many web sites still in their infancy. Various versions 
                        have evolved over the years, sometimes by accident, 
                        sometimes on purpose (injected humour and the like).
                        """)]
#Word Tokenize first sentence from sent_tokenized, 
#save the result in a variable 'words_tokenized'
WORDS_TOKENIZED = [word for word in nltk.word_tokenize(SENT_TOKENIZED[0])]
FILTERED = [word for word in WORDS_TOKENIZED if re.search('[a-zA-Z]', word)]
# Let's observe words after tokenization
#print(FILTERED)
STEMMER = SnowballStemmer("english")
# let's observe words without stemming
print("Without stemming: ", FILTERED)
STEMMED_WORDS = [STEMMER.stem(word) for word in FILTERED]
# now let's check out after stemming
print("After stemming: ", STEMMED_WORDS)
#Creating TFIDFVectorizer
TFIDF_VECTOR = TfidfVectorizer(stop_words='english', tokenizer=token_and_stem)
PLOT_MATRIX = TFIDF_VECTOR.fit_transform([plot for plot in MOVIES['plot']])
#clustering with KMeans
# Create a KMeans object with 5 clusters and save as km
K_MEANS = KMeans(n_clusters=5)
# Fit the k-means object with tfidf_matrix
K_MEANS.fit(PLOT_MATRIX)
CLUSTERS = K_MEANS.labels_.tolist()
# Create a column cluster to denote the generated cluster for each movie
MOVIES["cluster"] = CLUSTERS
# Calculate the similarity distance
SIM_DIS = 1 - cosine_similarity(PLOT_MATRIX)
MOVIES_SIM_DIS_MATRIX = linkage(SIM_DIS, method='complete')
MOVIE_DENDOGRAM = dendrogram(MOVIES_SIM_DIS_MATRIX,
                             labels=[x for x in MOVIES["title"]],
                             leaf_rotation=90,
                             leaf_font_size=16,)
FIG = plt.gcf()
_ = [lbl.set_color('r') for lbl in plt.gca().get_xmajorticklabels()]
FIG.set_size_inches(108, 21)
# Show the plotted dendrogram
plt.show()
#making a dictionary that held the most similar MOVIES based on the 
#ordering of the MOVIES_sim_dis_matrix
SIMILAR_MOVIES = {}
for movie in MOVIES_SIM_DIS_MATRIX:
    movie_0 = int(movie[0])
    movie_1 = int(movie[1])
    SIMILAR_MOVIES[movie_0] = movie_1
#Generally we find that MOVIES that are count as a similar value for an 
#earlier value do not get their own individual entry to avoid repetition
#As a result we will take all the rated MOVIES and their corresponding most 
#similar movie into 2 separate lists that will be
#searched to create a method to show the most similar movie
RATED_MOVIES = []
SIMILAR_FOR_RATED = []
for a in SIMILAR_MOVIES:
    RATED_MOVIES.append(a)
    SIMILAR_FOR_RATED.append(SIMILAR_MOVIES[a])
#predict method
def show_most_similar_movie():
    movie_title = input('Please Enter a movie title ').strip() 
#making the movie_title input lower case and converting every title to lower case for comparisons
    MOVIES['title_lower'] = MOVIES['title'].apply(lambda x: x.lower())       
#checking that the entered movie exists in dataset
    if any(MOVIES['title_lower'] == movie_title.lower()):
        movie_df = MOVIES[MOVIES['title_lower'] == movie_title]
    else:
        return "Movie does not exist. Please check your spelling and Capitalisations"    
#converting the 'rank' of the movie according to the dataset; 
#it acts as an id of sorts, to an integer
    rank = int(movie_df['rank'])    
#checking if the rank appears in rated MOVIES, 
#if not converting then checking if it has been used as a 
#similar movie so that we can get the most similar movie for any movie
    if rank in RATED_MOVIES:    
        sim_movie_df = MOVIES[MOVIES['rank'] == SIMILAR_MOVIES[rank]]    
        sim_movie = sim_movie_df.title.values        
    elif rank in SIMILAR_FOR_RATED:
        idx = SIMILAR_FOR_RATED.index(rank)        
        sim_movie_df = MOVIES[MOVIES['rank'] == RATED_MOVIES[idx]]        
        sim_movie = sim_movie_df.title.values
    else:
        return 'Unknown Error, Movie does not exist'           
#this check here is used to enure that the similar movie exists
    if sim_movie.size > 0:
        sel = sim_movie[0]
    else:
        sel = 'Sorry No Movie Available'    
    return 'Most Similar movie to \'{}\' is: \'{}\''.format(movie_title, sel)
X = show_most_similar_movie()
print(X)
