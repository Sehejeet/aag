from collections import Counter
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math
from nltk.corpus import stopwords
import PyPDF2

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

def readPdf(fileName):
	file = open(fileName, 'rb')
	pdfReader = PyPDF2.PdfFileReader(file)

	num = pdfReader.numPages
	text = ''
	for i in range(1, num):
		page = pdfReader.getPage(i)
		text += page.extractText()
	file.close()
	return text

def getSimilarityScore(f1, f2):	
	str1 = readPdf(f1)
	str2 = readPdf(f2)
	st1 = [w for w in word_tokenize(str1) if w.isalnum() and w not in stopWords]
	st2 = [w for w in word_tokenize(str2) if w.isalnum() and w not in stopWords]

	str1 = ' '.join(st1)
	str2 = ' '.join(st2)

	s1 = list()
	s2 = list()
	ps = SnowballStemmer('english')
	for w in word_tokenize(str1):
		s1.append(ps.stem(w))

	for w in word_tokenize(str2):
		s2.append(ps.stem(w))

	return (getcosinesim(' '.join(s1), ' '.join(s2)))
