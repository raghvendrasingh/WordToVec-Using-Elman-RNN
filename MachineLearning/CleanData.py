import re
import string
import sys
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import glob
import os


class CleanData(object):

    def remove_non_english_characters(self, data):
        return ''.join([i if ord(i) < 128 else ' ' for i in data])

    def make_lowercase(self, data):
        return data.lower()

    def tokenize_into_sentence(self, data):
        return sent_tokenize(data)

    def tokenize_into_words(self, data):
        return word_tokenize(data)

    def remove_punctuations(self, data):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        new_review = []
        for token in data:
            new_token = regex.sub(u'', token)
            if not new_token == u'':
                new_review.append(new_token)
        return new_review


    def remove_stop_words(self, data):
        new_term_vector = []
        for word in data:
            if word not in stopwords.words('english'):
                new_term_vector.append(word)
        return new_term_vector

    def lemmatize_data(self, data):
        final_doc = []
        wnl = WordNetLemmatizer()
        for word in data:
            final_doc.append(wnl.lemmatize(word))
        return final_doc

    def porter_stemmer(self, data):
        final_doc = []
        for word in data:
            final_doc.append(PorterStemmer.stem(word))
        return final_doc

    def snowball_stemmer(self, data):
        final_doc = []
        for word in data:
            final_doc.append(SnowballStemmer.stem(word))
        return final_doc

    def save_sentence_to_file(self, sentence, out_file):
        try:
            str = ",".join(sentence)
            out_file.write(str+"\n")
        except:
            print "Unexpected error while executing method save_sentence_to_file():", sys.exc_info()[0]
            raise


# data is saved one sentence per line. Each sentence in a line is a comma separated words
def clean_data_and_save_to_file(inp_data_file, out_file_name):
        clean_data_obj = CleanData()
        out_file = open(out_file_name, "a")
        try:
            with open(inp_data_file) as f:
                for line in f:
                    line_with_english_characters = clean_data_obj.remove_non_english_characters(line)
                    line_lowercase = clean_data_obj.make_lowercase(line_with_english_characters)
                    sentences = clean_data_obj.tokenize_into_sentence(line_lowercase)
                    for sentence in sentences:
                        words = clean_data_obj.tokenize_into_words(sentence)
                        words_without_punctuation = clean_data_obj.remove_punctuations(words)
                        words_without_stop_words = clean_data_obj.remove_stop_words(words_without_punctuation)
                        lemmatized_words = clean_data_obj.lemmatize_data(words_without_stop_words)
                        clean_data_obj.save_sentence_to_file(lemmatized_words, out_file)
            f.close()
            out_file.close()

        except:
            print "Unexpected error while executing method clean_data_and_save_to_file():", sys.exc_info()[0]
            out_file.close()
            raise



if __name__ == '__main__':
    filename = "cleaned_training_data.txt"
    if os.path.exists(filename):
        os.remove(filename)
    path = 'data/news.*'
    files = glob.glob(path)
    for train_file in files:
        clean_data_and_save_to_file(train_file, filename)
        print "Finished cleaning ", train_file
