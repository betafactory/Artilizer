from django.shortcuts import render, redirect

# Create your views here.

import nltk # Imports the library
import pandas as pd
import _pickle as cPickle
import string
from . import input
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from django.http import JsonResponse,HttpResponse
from django.views.decorators.csrf import csrf_exempt
import re
from nltk.cluster.util import cosine_distance
from google_images_download import google_images_download  

from collections import Counter
import numpy as np
import networkx as nx
stopwords.words('english')[0:10]
import moviepy.editor as mpe
import glob
import cv2
from wsgiref.util import FileWrapper
import mimetypes
from visualizer.models import Video
import os
from nltk.stem.porter import PorterStemmer
import os, shutil

folder = 'visualizer/images'

def index(request):

    return render(request,'index.html')


def downloadimages(query): 
	response = google_images_download.googleimagesdownload()  
	arguments = {"keywords": query, "format": "jpg", "limit":1,"print_urls":False, "size": "large", "aspect_ratio": "panoramic","no_directory":"1","output_directory":"visualizer/images/"} 
	try: 
		response.download(arguments) 
	except FileNotFoundError:
		arguments = {"keywords": query,"format": "jpg", "limit":1,  "print_urls":False,   "size": "large","no_directory":"1","output_directory":"visualizer/images/"} 
		try: 
			response.download(arguments) 
		except: 
			pass

def sentence_similarity(sent1, sent2, stopwords=None):
	if stopwords is None:
		stopwords = []
	sent1 = [w.lower() for w in sent1]
	sent2 = [w.lower() for w in sent2]
	all_words = list(set(sent1 + sent2))
	vector1 = [0] * len(all_words)
	vector2 = [0] * len(all_words)
	# build the vector for the first sentence
	for w in sent1:
		if w in stopwords:
			continue
		vector1[all_words.index(w)] += 1
	# build the vector for the second sentence
	for w in sent2:
		if w in stopwords:
			continue
		vector2[all_words.index(w)] += 1
	return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
	# Create an empty similarity matrix
	similarity_matrix = np.zeros((len(sentences), len(sentences)))
	for idx1 in range(len(sentences)):
		for idx2 in range(len(sentences)):
			if idx1 == idx2: #ignore if both are same sentences
				continue 
			similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
	return similarity_matrix

def text_process(mess):
	"""
	Takes in a string of text, then performs the following:
	1. Remove all punctuation
	2. Remove all stopwords
	3. Returns a list of the cleaned text
	"""
	# Check characters to see if they are in punctuation
	nopunc = [char for char in mess if char not in string.punctuation]

	# Join the characters again to form the string.
	nopunc = ''.join(nopunc)
	
	# Now just remove any stopwords
	return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

def predict():

	messages1 = pd.read_csv('visualizer/input/op.tsv', sep='\t',
							   names=["id", "message1"])

	with open('visualizer/models/emotion_detect_model.pkl', 'rb') as fin:
	  bow_transformer, emotion_detect_model = cPickle.load(fin)

	# fid = open('', 'rb')
	# emotion_detect_model = cPickle.load(fid)


	messages_bow_test = bow_transformer.transform(messages1['message1'].values.astype('U'))


	tfidf_transformer = TfidfTransformer()

	messages_tfidf_test = tfidf_transformer.fit_transform(messages_bow_test)

	all_predictions = emotion_detect_model.predict(messages_tfidf_test)

	numbers = all_predictions.tolist()
	b=sorted(numbers, key=Counter(numbers).get, reverse=True)
	return b[0]


def train(request):
	print("Working")
	messages = pd.read_csv('visualizer/input/data.bak3.tsv', sep='\t',
						   names=["label", "message"])
	bow_transformer = CountVectorizer(analyzer=text_process)
	X_train = bow_transformer.fit(messages['message'])
	messages_bow = X_train.transform(messages['message'])
	tfidf_transformer = TfidfTransformer()
	tf = tfidf_transformer.fit(messages_bow)
	messages_tfidf = tf.transform(messages_bow)
	emotion_detect_model = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None).fit(messages_tfidf, messages['label'])
	with open('visualizer/models/emotion_detect_model.pkl', 'wb') as fid:
		cPickle.dump((bow_transformer,emotion_detect_model), fid) 
	print("Done")

@csrf_exempt
def create(request):
	if request.method == 'POST':
		data = request.POST['parag']
		paragraph = data
		text = data.replace('\n', '')	
		data = text
		for k in text.split("\n"):
			text2 = re.sub(r"[^a-zA-Z0-9&]+", ' ', k)
		text = text2
		tokens = [t for t in text.split()]
		sr= stopwords.words('english')
		clean_tokens = tokens[:]
		for token in tokens:
			if token in stopwords.words('english'):
				
				clean_tokens.remove(token)
		freq = nltk.FreqDist(clean_tokens)

		s = [(k, freq[k]) for k in sorted(freq, key=freq.get, reverse=True)]
		title = s[0][0] 
		search_queries = [sorted(freq.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[0][0] +"  "+ sorted(freq.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)[1][0]]
		for query in search_queries: 
			downloadimages(query)

		stop_words = stopwords.words('english')
		summarize_text = []
		# Step 1 - Read text anc split it
		article = data.split(". ")
		sentences = []
		count_sentence = 0
		for sentence in article:
			count_sentence = count_sentence + 1
			sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
		sentences.pop() 
		top_n= int(count_sentence/3)
		# Step 2 - Generate Similary Martix across sentences
		sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)
		# Step 3 - Rank sentences in similarity martix
		sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
		scores = nx.pagerank(sentence_similarity_graph)
		# Step 4 - Sort the rank and pick top sentences
		ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)  
		for i in range(top_n):
		  summarize_text.append(" ".join(ranked_sentence[i][1]))
		# Step 5 - Offcourse, output the summarize texr
		m=1
	# Driver Code 
		with open("visualizer/input/op.tsv", "w") as text_file:
			text_file.write("content"+"\t"+"val"+'\n')
			for i in summarize_text:
				search_queries.append(i)
				text_file.write(i+"\t"+str(m)+'\n')
				m=m+1
		emotion = predict()
		for query in search_queries:
			review = re.sub('[^a-zA-Z]', ' ',query)
			review = review.lower()
			review = review.split()
			ps = PorterStemmer()
			review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
			review = ' '.join(review)
			downloadimages(review)  
		img_array = []
		for filename in glob.glob('visualizer/images/*.jpg'):
			img = cv2.imread(filename)
			height, width, layers = img.shape
			size = (width,height)
			img_array.append(img)
		 
		out = cv2.VideoWriter('visualizer/output/project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 0.2, size)
		for i in range(len(img_array)):
			img = img_array[i]
			img = cv2.resize(img, size)
			out.write(img)
		out.release()
		for the_file in os.listdir(folder):
		    file_path = os.path.join(folder, the_file)
		    try:
		        if os.path.isfile(file_path):
		            os.unlink(file_path)
		        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
		    except Exception as e:
		        print(e)
		my_clip = mpe.VideoFileClip('visualizer/output/project.avi')
		audio_background = mpe.AudioFileClip('visualizer/emotions/'+emotion+'.mp3')
		final_audio = mpe.CompositeAudioClip([audio_background])
		final_clip = my_clip.set_audio(final_audio)
		final_clip.write_videofile("visualizer/output/"+title+'.mp4')
		data = title
		file_path = 'visualizer/output/'+data+'.mp4'
		video = Video()
		video.data = paragraph
		video.name = data
		video.videofile = file_path
		video.save()
		return redirect(video.videofile.url)

	# if request.method == 'GET':
	# 	return render(
 #    	request,
 #        'index.html'
 #    )
