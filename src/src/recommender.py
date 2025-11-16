import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class MovieRecommender:
def __init__(self, movies_csv_path):
self.movies = pd.read_csv(movies_csv_path)
self._prepare()


def _prepare(self):
# fillna and create a combined feature
for col in ['title','genres','overview']:
if col in self.movies.columns:
self.movies[col] = self.movies[col].fillna('')
self.movies['combined'] = (self.movies.get('title','') + ' ' +
self.movies.get('genres','') + ' ' +
self.movies.get('overview',''))


self.tfidf = TfidfVectorizer(stop_words='english')
self.tfidf_matrix = self.tfidf.fit_transform(self.movies['combined'])
self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
self.indices = pd.Series(self.movies.index, index=self.movies['title']).drop_duplicates()


def recommend(self, title, top_n=10):
if title not in self.indices:
raise ValueError(f"Title '{title}' not found in dataset")
idx = self.indices[title]
sim_scores = list(enumerate(self.cosine_sim[idx]))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
sim_scores = sim_scores[1: top_n+1]
movie_indices = [i[0] for i in sim_scores]
return self.movies.iloc[movie_indices][['movieId','title','genres']]


if __name__ == '__main__':
mr = MovieRecommender('data/sample_movies.csv')
print(mr.recommend('Toy Story (1995)', top_n=3))
