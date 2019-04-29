from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from nltk.corpus import stopwords

stopWords = stopwords.words('english')

def getcosinesim(*strs):
	vectors = [t for t in getvectors(*strs)]
	return cosine_similarity(vectors)[0,1]

def getvectors(*strs):
	text = [t for t in strs]
	vectorizer = CountVectorizer(text)
	vectorizer.fit(text)
	vecs = vectorizer.transform(text).toarray()
	res = []
	for vec in vecs:
		sqsum = 0
		for v in vec:
			sqsum = sqsum + v**2
		sqsum = math.sqrt(sqsum)
		new = [x/sqsum for x in vec]
		res.append(new)

	return res

str1 = 'AI is a friend of humans and it has been friendly'
str2 = 'AI and humans have always been friendly'

st1 = [w for w in word_tokenize(str1) if w not in stopWords]
st2 = [w for w in word_tokenize(str2) if w not in stopWords]

str1 = ' '.join(st1)
str2 = ' '.join(st2)

s1 = list()
s2 = list()
ps = SnowballStemmer('english')
for w in word_tokenize(str1):
	s1.append(ps.stem(w))

for w in word_tokenize(str2):
	s2.append(ps.stem(w))

#print(getvectors(s1, s2))

print(getcosinesim(' '.join(s1), ' '.join(s2)))