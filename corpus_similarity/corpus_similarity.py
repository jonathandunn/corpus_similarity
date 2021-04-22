import os
import codecs
import gzip
import re
from cleantext import clean
import tinysegmenter
import jieba
from pythainlp import word_tokenize

#Class for loading and cleaning text data
class Load(object):

	def __init__(self, language, threshold = 10000000):

		self.language = language
		self.threshold = threshold

		if self.language == "jpn":
			self.segmenter = tinysegmenter.TinySegmenter()

		elif self.language == "zho":
			self.tk = jieba.Tokenizer()
			self.tk.initialize()
			self.tk.lock = True

	#--------------------------------------------------
	def load(self, data):

		#Initialize holder
		lines = []

		#Otherwise input is a list of strings
		if isinstance(data, list):
			lines = data

		#Load text file
		elif data.endswith(".txt"):
			with codecs.open(data, "r", encoding = "utf-8", errors = "replace") as fo:
				for line in fo:
					lines.append(line)
		
		#Load gzipped text file
		elif data.endswith(".gz"):
			with gzip.open(data, "r") as fo:
				for line in fo:
					line = line.decode("utf-8", errors = "replace")
					lines.append(line)

		#For each line, clean and prep
		new_lines = []
		counter = 0

		for text in lines:

			#Check if still need more data
			if counter < self.threshold and len(text) > 5:

				#Split into words in jpn, zho, tha
				if self.language == "jpn":
					text = " ".join(self.segmenter.tokenize(text))
					text = re.sub(r"\s+", " ", text)
													
				elif self.language == "zho":
					text = [x for x in self.tk.cut(text, cut_all = True, HMM = True) if x != ""]
					text = " ".join(text)
					text = re.sub(r"\s+", " ", text)
									
				elif self.language == "tha":
					text = word_tokenize(text, keep_whitespace = False)
					text = " ".join(text)
								
				#Now clean each line
				text = clean(text,
								fix_unicode = True,
								to_ascii = False,
								lower = True,
								no_line_breaks = True,
								no_urls = True,
								no_emails = True,
								no_phone_numbers = True,
								no_numbers = True,
								no_digits = True,
								no_currency_symbols = True,
								no_punct = True,
								replace_with_punct = "",
								replace_with_url = "",
								replace_with_email = "",
								replace_with_phone_number = "",
								replace_with_number = "<NUM>",
								replace_with_digit = "0",
								replace_with_currency_symbol = "",
								)
								
				length = len(text.split())
				counter += length
				new_lines.append(text)
							
		return new_lines
	#--------------------------------------------------