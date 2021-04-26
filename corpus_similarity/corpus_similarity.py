import os
import codecs
import gzip
import re
from cleantext import clean
import tinysegmenter
import jieba
from pythainlp import word_tokenize

import random
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import rankdata
from scipy.stats import chisquare
from scipy.stats import spearmanr
from scipy.spatial import distance



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
	
	

	
	
	
	
#===============================================================================

class training:
    
    def __init__(self, filename):
        self.filename=filename
       

    #---------------------------------------------------------------------
    # subcorpus function: split original corpus into smaller subcorpus for training
    # perc=1 : generate subcorpus from the first half of corpus (training)
    # perc=2 : generate subcorpus from the second half of corpus (validation)
    # perc=0 : generate subcorpus from the whole corpus range
    #
    # input file has been cleaned and saved in txt file
    #---------------------------------------------------------------------        
  
        
    def subcorpus(self, perc, corpusSize):

        with open(self.filename, encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
        
        line_num = i+1

        if perc==1:
            start_num=0
            bottom_num=math.ceil(line_num/2)
        elif perc==2:
            start_num=math.ceil(line_num/2)
            bottom_num=line_num-1            
        else:
            start_num=0
            bottom_num=line_num-1
            
        

        word_num = 0
        file = open(self.filename, encoding='utf-8')
        all_lines = file.readlines()
        combine_lines = ' '

        while True:
            random_num = random.randint(start_num, bottom_num)
            line_content = all_lines[random_num]
            word_num_line = len(line_content.split())
            word_num = word_num + word_num_line

            if word_num < corpusSize:
                combine_lines = combine_lines  +  line_content
            else:
                combine_lines = combine_lines  +  line_content
                combine_lines = combine_lines.replace('\n', ' ')  
                break
            
        print('word_num:', word_num)
        return combine_lines 
    
  

    #---------------------------------------------------------------------
    # feature_extraction function: extracting word/char frequency information
    # text_feature = word/char
    # n: n-gram
    #
    # corpuslist: training and validation subcorpora are put in this list
    # fre_array: return frequency features (word/char n-gram) in decreasing order
    #---------------------------------------------------------------------      
  

    
    def feature_extraction(self, corpuslist, text_feature, n):
        
        vectorizer = CountVectorizer(analyzer = text_feature, ngram_range=(n, n) )
        X = vectorizer.fit_transform(corpuslist)
        fre_array = X.toarray()
        fre_array_sum = np.sum(fre_array, axis=0)
        fre_array_sum_order = np.argsort(-fre_array_sum)

        # sort fre_array by decreasing fre
        
        for i in range(fre_array.shape[0]):
            fre_array[i]= fre_array[i][fre_array_sum_order]
            
        return fre_array
    
    
    #---------------------------------------------------------------------
    # same_regi_measure function: using word/char frequency information to measure corpus similarity
    # feature_N : feature number used in the measure
    # fre_array: an array containing frequency information of subcorpora from same register for training 
    # measure_type = chi_square/spearman/cosine_dis/euclidean_dis
    #
    # output: if fre_array.shape[0] is x, there are xC2 pairs of comparison for similarity measure,        
    #         stat_list contains similarity values with the specific measure for all the pairs 
    #---------------------------------------------------------------------



    def same_regi_measure(self, fre_array, feature_N, measure_type, S):
        fre_array_num = fre_array.shape[0]
    
        stat_list=[]
        
        if measure_type == "chi_square":
            
            # remove nan from array; avoid 0 as expected value
            # use top N features 
            
            fre_array_N = fre_array[:, :feature_N] +1

            fre_array_N = fre_array_N/S
            
            for i in range(fre_array_num-1):
                first_array=fre_array_N[i]
                for j in range(i+1, fre_array_num): 
                    second_array = fre_array_N[j]
                    
                    np_2array = np.array([first_array , second_array])
                    average_array = np.mean(np_2array, axis=0)

                    stat, pvalue = chisquare(first_array, f_exp=average_array)
                    stat_list.append(stat)
        
        elif measure_type == "spearman":
            
            fre_array_N = fre_array[:, :feature_N]
            
            
            for i in range(fre_array_num-1):
                first_array=fre_array_N[i]
                for j in range(i+1, fre_array_num):
                    second_array = fre_array_N[j]

                    first_array = rankdata(-first_array)
                    second_array = rankdata(-second_array)

                    coef, p = spearmanr(first_array, second_array)

                    if coef<0:
                        coef = -coef
                
                    stat_list.append(coef)
                    
                    
        elif measure_type == "cosine_dis": 
            
            
            # avoid 0 as denominator
            fre_array_N = fre_array[:, :feature_N] + 1
            fre_array_N = fre_array_N/S
            
            for i in range(fre_array_num-1):
                first_array = fre_array_N[i]
                for j in range(i+1, fre_array_num):
                    second_array = fre_array_N[j]

                    cos_simi = distance.cosine(first_array, second_array)
               
                    stat_list.append(cos_simi)


        elif measure_type == "euclidean_dis": 
                    
            # avoid 0 as denominator
            fre_array_N = fre_array[:, :feature_N] + 1
            fre_array_N = fre_array_N/S
            
            for i in range(fre_array_num-1):
                first_array = fre_array_N[i]
                for j in range(i+1, fre_array_num):
                    second_array = fre_array_N[j]

                    euc_simi = distance.euclidean(first_array, second_array)
               
                    stat_list.append(euc_simi)             
                    
    
        # nCr(n,r): n=fre_array_num  r=2 
        return stat_list 




    #---------------------------------------------------------------------
    # same_regi_measure function: using word/char frequency information to measure corpus similarity
    # feature_N : feature number used in the measure
    # fre_array: frequency information of word/char features from one register 
    # fre_array2: frequency information of word/char features from another register
    # measure_type = chi_square/spearman/cosine_dis/euclidean_dis
    #
    # output: if fre_array.shape[0]=x, fre_array2.shape[0]=y there are x*y pairs of comparison for similarity measure,        
    #         stat_list contains similarity values with the specific measure for all the pairs 
    #---------------------------------------------------------------------
    


    def diff_regi_measure(self, fre_array, fre_array2, feature_N, measure_type, S):
        
        fre_array_num = fre_array.shape[0]
        fre_array_num2 = fre_array2.shape[0]        
    
        stat_list=[]
        
        if measure_type == "chi_square":
            
            # avoid 0 as expected value
            fre_array_N = fre_array[:, :feature_N] + 1
            fre_array2_N = fre_array2[:, :feature_N] + 1

            fre_array_N = fre_array_N/S
            fre_array2_N = fre_array2_N/S
            
            for i in range(fre_array_num):
                first_array = fre_array_N[i] 
                
                for j in range(fre_array_num2): 
                    second_array = fre_array2_N[j]
                    
                    np_2array = np.array([first_array , second_array])
                    average_array = np.mean(np_2array, axis=0)

                    stat, pvalue = chisquare(first_array, f_exp=average_array)
                    stat_list.append(stat)
                    
                    
        elif measure_type == "spearman":         
            
            fre_array_N = fre_array[:, :feature_N]
            fre_array2_N = fre_array2[:, :feature_N]                  
                    
            for i in range(fre_array_num):
                first_array = fre_array_N[i] 
                
                for j in range(fre_array_num2): 
                    second_array = fre_array2_N[j]
                    
                    first_array = rankdata(-first_array)
                    second_array = rankdata(-second_array)  
                    
                    coef, p = spearmanr(first_array, second_array)

                    if coef<0:
                        coef = -coef

                    stat_list.append(coef)
            
     
        elif measure_type == "cosine_dis":         
            
            fre_array_N = fre_array[:, :feature_N] + 1 
            fre_array2_N = fre_array2[:, :feature_N] + 1

            fre_array_N = fre_array_N/S
            fre_array2_N = fre_array2_N/S
            
                    
            for i in range(fre_array_num):
                first_array = fre_array_N[i] 
                
                for j in range(fre_array_num2): 
                    second_array = fre_array2_N[j]
                    
                    cos_simi = distance.cosine(first_array, second_array)
                    stat_list.append(cos_simi)

               

        elif measure_type == "euclidean_dis":         
            
            fre_array_N = fre_array[:, :feature_N] + 1 
            fre_array2_N = fre_array2[:, :feature_N] + 1

            fre_array_N = fre_array_N/S
            fre_array2_N = fre_array2_N/S           
                    
            for i in range(fre_array_num):
                first_array = fre_array_N[i] 
                
                for j in range(fre_array_num2): 
                    second_array = fre_array2_N[j]
                    
                    euc_simi = distance.euclidean(first_array, second_array) 
                    stat_list.append(euc_simi)            
                
                 
        return stat_list            
            


    def measure_mean(self, stat_list):
        stat_array=np.array(stat_list)

        # remove NaN items
        stat_array = stat_array[~np.isnan(stat_array)]
      
        stat_mean=np.mean(stat_array)
        stat_std=np.std(stat_array)
        stat_len=len(stat_array)

        # print("measure mean is {m} std is {s} with {l} pairs".format(m = stat_mean, s = stat_std, l = stat_len ))
        return stat_mean, stat_std 


#==============================================================================================
		
	
