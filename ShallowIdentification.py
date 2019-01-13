import pandas as pd
import re
from nltk.corpus import stopwords
from bs4 import BeautifulSoup as bs
from sklearn.naive_bayes import MultinomialNB
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import islice
import time

class ShallowIdentifier:
    def __init__(self):
        self.fakeTF = {}
        self.realTF = {}


    def sentence_to_words(self, sentence):
        local_stopwords = stopwords.words("english")

        split_sentence = []

        soup = bs(sentence, "html5lib")
        cleaned_sentence = re.sub("[^a-zA-Z]", " ", soup.get_text())
        split_sentence = [w for w in cleaned_sentence for w in local_stopwords]
        return split_sentence


    def do_tf_stuff(self, ngram_lower, ngram_higher, n_training=1000):
        start = time.time()
        raw_data = pd.read_csv("news_ds.csv")
        raw_data = raw_data.iloc[0:n_training]
        thing = pd.DataFrame(raw_data["TEXT"].map(lambda x: str(x).lower()))
        raw_data = raw_data.assign(TEXT_LOWER=thing.values)

        stemmer = SnowballStemmer("english")
        raw_data["STEMMED"] = raw_data.TEXT_LOWER.map(
            lambda x: ' '.join([stemmer.stem(str(y)) for y in str(x).split(' ')]))
        raw_data.STEMMED.head()

        sentences = []

        cvec = CountVectorizer(ngram_range=(ngram_lower, ngram_higher))
        cvec.fit((raw_data["STEMMED"]))

        matrixFakeTF = cvec.transform(raw_data["STEMMED"])


        clf = MultinomialNB().fit(matrixFakeTF, list(islice(raw_data.LABEL, n_training)))

        raw_data2 = pd.read_csv("news_ds.csv")
        raw_data2 = raw_data2.iloc[1001:]
        thing = pd.DataFrame(raw_data2["TEXT"].map(lambda x: str(x).lower()))
        raw_data2 = raw_data2.assign(TEXT_LOWER=thing.values)
        raw_data2["STEMMED"] = raw_data2.TEXT_LOWER.map(
            lambda x: ' '.join([stemmer.stem(str(y)) for y in str(x).split(' ')]))
        raw_data2.STEMMED.head()

        docs_new = raw_data2["STEMMED"]
        X_new_counts = cvec.transform(docs_new)

        predicted = clf.predict(X_new_counts)
        total_relevant = 0
        num_retrieved = 0
        hits = 0

        incorrect = 0
        correct = 0
        for doc, actual, prediction in zip(raw_data2["TEXT"], raw_data2["LABEL"], predicted):
            if int(actual) == 0:
                total_relevant += 1
            if int(prediction) == 0:
                num_retrieved += 1
                if int(actual) == int(prediction):
                    hits += 1
            if int(actual) == int(prediction):
                correct += 1
            else:
                incorrect += 1
        end = time.time()
        print("***************")
        print("******TF*****y**")
        print("lower: ", ngram_lower, "higher: ", ngram_higher)
        print("Correct: ", hits, ", Incorrect: ", num_retrieved - hits)
        accuracy = correct / (incorrect + correct)
        precision = hits / num_retrieved
        recall = hits / total_relevant
        f1measure = 2 * (precision * recall) / (precision + recall)
        print("classification accuracy: ", accuracy)
        print("precision = ", precision)
        print("recall = ", recall)
        print("f1-measure = ", f1measure)
        print("Time taken: ", (end - start), "seconds")
        print("***************")

    def do_tfidf_stuff(self,ngram_lower,ngram_higher, n_training=1000):
        start = time.time()
        raw_data = pd.read_csv("news_ds.csv")
        raw_data = raw_data.iloc[0:n_training]
        thing = pd.DataFrame(raw_data["TEXT"].map(lambda x: str(x).lower()))
        raw_data = raw_data.assign(TEXT_LOWER=thing.values)

        stemmer = SnowballStemmer("english")
        raw_data["STEMMED"] = raw_data.TEXT_LOWER.map(lambda x: ' '.join([stemmer.stem(str(y)) for y in str(x).split(' ')]))
        raw_data.STEMMED.head()

        sentences = []

        cvec = CountVectorizer(ngram_range=(ngram_lower, ngram_higher))
        cvec.fit((raw_data["STEMMED"]))
        transformer = TfidfTransformer()

        matrixFakeTF = cvec.transform(raw_data["STEMMED"])
        matrixFakeTFIDF = transformer.fit_transform(matrixFakeTF)
        # From the vectorized count, we can use the TfidfTransformer to calculate the tf-idf weights



        clf = MultinomialNB().fit(matrixFakeTFIDF,list(islice(raw_data.LABEL,n_training)))

        raw_data2 = pd.read_csv("news_ds.csv")
        raw_data2 = raw_data2.iloc[1001:]
        thing = pd.DataFrame(raw_data2["TEXT"].map(lambda x: str(x).lower()))
        raw_data2 = raw_data2.assign(TEXT_LOWER=thing.values)
        raw_data2["STEMMED"] = raw_data2.TEXT_LOWER.map(lambda x: ' '.join([stemmer.stem(str(y)) for y in str(x).split(' ')]))
        raw_data2.STEMMED.head()

        docs_new = raw_data2["STEMMED"]
        X_new_counts = cvec.transform(docs_new)
        X_new_tfidf = transformer.transform(X_new_counts)

        predicted = clf.predict(X_new_tfidf)
        total_relevant = 0
        num_retrieved = 0
        hits = 0

        incorrect = 0
        correct = 0
        for doc, actual, prediction in zip(raw_data2["TEXT"], raw_data2["LABEL"], predicted):
            if int(actual) == 0:
                total_relevant += 1
            if int(prediction) == 0:
                num_retrieved += 1
                if int(actual) == int(prediction):
                    hits += 1
            if int(actual) == int(prediction):
                correct += 1
            else:
                incorrect += 1
        end = time.time()
        print("***************")
        print("*****TF-IDF****")
        print("lower: ", ngram_lower, "higher: ", ngram_higher)
        print("Correct: ", hits, ", Incorrect: ", num_retrieved - hits)
        accuracy = correct / (incorrect + correct)
        precision = hits / num_retrieved
        recall = hits / total_relevant
        f1measure = 2 * (precision * recall) / (precision + recall)
        print("classification accuracy: ", accuracy)
        print("precision = ", precision)
        print("recall = ", recall)
        print("f1-measure = ", f1measure)
        print("Time taken: ", (end - start), "seconds")
        print("***************")

if __name__ == "__main__":
    raw_data = pd.read_csv("news_ds.csv")
    identifier = ShallowIdentifier()
    identifier.do_tf_stuff(1,1)
    identifier.do_tfidf_stuff(1,1)
