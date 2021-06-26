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
        
        self.filename = filename
       


    #---------------------------------------------------------------------
    # 'subcorpus' function: split an original corpus (which is in big size) into some smaller subcorpora for training/validation
    #                       Every line of strings put into one subcorpus is selected randomly from the original corpus, then the selected
    #                       line will be deleted from the original corpus, which make sure that no content is repeated
    #
    # corpusSize: the length counted by words
    #
    # Output: subcorpus_list is a list containing a number of (subcor_number) strings (i.e. subcorpus)     
    #---------------------------------------------------------------------        
         
    def subcorpus(self, corpusSize, subcor_number):

        with open(self.filename, encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
        
        line_num = i

        word_num = 0
        file = open(self.filename, encoding='utf-8')
        all_lines = file.readlines()

        subcorpus_list = []


        for k in range(subcor_number):

            combine_lines = ' '
            word_num = 0
            
            while True:
                if line_num < 1:
                    print('No sufficient words left in the corpus!')
                    break
   
                random_num = random.randint(0, line_num)
                line_content = all_lines[random_num]
                word_num_line = len(line_content.split())
                word_num = word_num + word_num_line

                if word_num < corpusSize:
                    combine_lines = combine_lines  +  line_content
                    del all_lines[random_num]
                    line_num = line_num - 1
                
                else:
                    combine_lines = combine_lines  +  line_content
                    combine_lines = combine_lines.replace('\n', ' ')
                    
                    subcorpus_list.append(combine_lines)

                    del all_lines[random_num]
                    line_num = line_num - 1   
                    break
            
            print('word_num:', word_num)

      
        return subcorpus_list







    #---------------------------------------------------------------------
    # Similar with the 'subcorpus' function:
    # For languages of 'jpn', 'tam', 'tha', 'zho', to remove spaces between words
    #---------------------------------------------------------------------
    
    def subcorpus_re_space(self, corpusSize, subcor_number):

        with open(self.filename, encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
        
        line_num = i

        word_num = 0
        file = open(self.filename, encoding='utf-8')
        all_lines = file.readlines()

        subcorpus_list = []


        for k in range(subcor_number):

            combine_lines = ' '
            word_num = 0
            
            while True:
                if line_num < 1:
                    print('No sufficient words left in the corpus!')
                    break
   
                random_num = random.randint(0, line_num)
                line_content = all_lines[random_num]
                word_num_line = len(line_content.split())
                word_num = word_num + word_num_line

                # remove sapce for tam, zho, jpn, tha
                line_content = line_content.replace(' ', '')
                

                if word_num < corpusSize:
                    combine_lines = combine_lines  +  line_content
                    del all_lines[random_num]
                    line_num = line_num - 1
                
                else:
                    combine_lines = combine_lines  +  line_content
                    combine_lines = combine_lines.replace('\n', '')
                    
                    subcorpus_list.append(combine_lines)

                    del all_lines[random_num]
                    line_num = line_num - 1   
                    break
            
            print('word_num:', word_num)

      
        return subcorpus_list







    #---------------------------------------------------------------------
    # Similar with the 'subcorpus' function:
    # For languages of 'jpn', 'tam', 'tha', 'zho' which do not have spaces between characters, then the corpusSize is counted by characters. 
    #---------------------------------------------------------------------



    def subcorpus_spaceless(self, corpusSize, subcor_number):

        with open(self.filename, encoding='utf-8') as f:
            for i, l in enumerate(f):
                pass
        
        line_num = i

        word_num = 0
        file = open(self.filename, encoding='utf-8')
        all_lines = file.readlines()

        subcorpus_list = []


        for k in range(subcor_number):

            combine_lines = ' '
            word_num = 0
            
            while True:
                if line_num < 1:
                    print('No sufficient words left in the corpus!')
                    break
   
                random_num = random.randint(0, line_num)
                line_content = all_lines[random_num]
                
                # word_num_line = len(line_content.split()) #==== words numbers =====
                
                word_num_line = len(line_content)    #==== characters numbers =====
                
                word_num = word_num + word_num_line

                if word_num < corpusSize:
                    combine_lines = combine_lines  +  line_content
                    del all_lines[random_num]
                    line_num = line_num - 1
                
                else:
                    combine_lines = combine_lines  +  line_content
                    combine_lines = combine_lines.replace('\n', ' ')
                    
                    subcorpus_list.append(combine_lines)

                    del all_lines[random_num]
                    line_num = line_num - 1   
                    break
            
            print('word_num:', word_num)

      
        return subcorpus_list






    #---------------------------------------------------------------------
    # 'feature_extraction' function: extracting word/char frequency information
    #
    # text_feature = word/char_wb
    # n: n-gram
    #
    # Output: fre_array is a numpy array containing features frequency information for each subcorpus in the subcorpus_list.
    #         wordlist is a list containing language features (word n-gram / char n-gram) in frequency decreasing order       
    #---------------------------------------------------------------------

    
    def feature_extraction(self, subcorpus_list, text_feature, n):
        
        vectorizer = CountVectorizer(analyzer = text_feature, ngram_range=(n, n) )
        X = vectorizer.fit_transform(subcorpus_list)
        
        fre_array = X.toarray()
        
        fre_array_sum = np.sum(fre_array, axis=0)
        fre_array_sum_order = np.argsort(-fre_array_sum)

        # sort fre_array by decreasing fre
        
        for i in range(fre_array.shape[0]):
            fre_array[i]= fre_array[i][fre_array_sum_order]
            

        get_voca = vectorizer.get_feature_names()
        
        myorder=fre_array_sum_order.tolist()
        
        wordlist = [get_voca[i] for i in myorder]

        # print('feature list: ', mylist)

        # save feature list in a file

            
        return fre_array, wordlist






    #---------------------------------------------------------------------
    # "get_simi_values" function: using word/char frequency information to measure the corpora similarity
    #                                                     
      
    # feature_N    : feature number used in the measure
    # fre_array1, fre_array2  : frequency information of two corpora are stored in two numpy arrays 
    #                          fre_array1 or fre_array2 can store frequency information for more than one subcorpus, which relate to more corpora compared,
    #                          the result 'stat_list' contains the similarity values of all combinations between the fre_array1 and fre_array2 
    #                         
    # measure_type : chi_square/spearman/cosine_dis/euclidean_dis
    #            S : corpus size
    
    # output:  stat_list is a list containing the corresponding similatiry values (with the chosen measure_type) 
    #         
    #---------------------------------------------------------------------



    def get_simi_values(self, fre_array1, fre_array2, feature_N, measure_type, corpusSize):
        
        fre_array_num1 = fre_array1.shape[0]
        fre_array_num2 = fre_array2.shape[0]        
    
        stat_list=[]
        
        if measure_type == "chi_square":
            
            # avoid 0 as expected value
            fre_array1_N = fre_array1[:, :feature_N] + 1
            fre_array2_N = fre_array2[:, :feature_N] + 1

            fre_array1_N = fre_array1_N/corpusSize
            fre_array2_N = fre_array2_N/corpusSize
            
            for i in range(fre_array_num1):
                first_array = fre_array1_N[i] 
                
                for j in range(fre_array_num2): 
                    second_array = fre_array2_N[j]
                    
                    np_2array = np.array([first_array , second_array])
                    average_array = np.mean(np_2array, axis=0)

                    stat, pvalue = chisquare(first_array, f_exp=average_array)
                    stat_list.append(stat)
                    
                    
        elif measure_type == "spearman":                          

            fre_array1_N = fre_array1[:, :feature_N] 
            fre_array2_N = fre_array2[:, :feature_N] 

                    
            for i in range(fre_array_num1):
                first_array = fre_array1_N[i] 
                
                for j in range(fre_array_num2): 
                    second_array = fre_array2_N[j]
                    
                    first_array = rankdata(-first_array)
                    second_array = rankdata(-second_array)  
                    
                    coef, p = spearmanr(first_array, second_array)

                    if coef<0:
                        coef = -coef

                    stat_list.append(coef)
            
     
        elif measure_type == "cosine_dis":         
            
            fre_array1_N = fre_array1[:, :feature_N] + 1
            fre_array2_N = fre_array2[:, :feature_N] + 1
            
            fre_array1_N = fre_array1_N/corpusSize
            fre_array2_N = fre_array2_N/corpusSize

            
            for i in range(fre_array_num1):
                first_array = fre_array1_N[i] 
                
                for j in range(fre_array_num2): 
                    second_array = fre_array2_N[j]
                    
                    cos_simi = distance.cosine(first_array, second_array)
                    stat_list.append(cos_simi)

               

        elif measure_type == "euclidean_dis":         
            
            fre_array1_N = fre_array1[:, :feature_N] + 1
            fre_array2_N = fre_array2[:, :feature_N] + 1
            
            fre_array1_N = fre_array1_N/corpusSize
            fre_array2_N = fre_array2_N/corpusSize         
                    
            for i in range(fre_array_num1):
                first_array = fre_array_N[i] 
                
                for j in range(fre_array_num2): 
                    second_array = fre_array2_N[j]
                    
                    euc_simi = distance.euclidean(first_array, second_array) 
                    stat_list.append(euc_simi)            
                
                
        # nCr(n,r): n=fre_array_num  r=2 
        return stat_list            
            






    #---------------------------------------------------------------------
    # get the mean and standard deviation of a list of similarity values (stat_list )
    #---------------------------------------------------------------------
    def measure_mean(self, stat_list):
        stat_array=np.array(stat_list)

        # remove NaN items
        stat_array = stat_array[~np.isnan(stat_array)]

        
        stat_mean=np.mean(stat_array)
        stat_std=np.std(stat_array)
        stat_len=len(stat_array)

        print("measure mean is {m} std is {s} with {l} pairs".format(m = stat_mean, s = stat_std, l = stat_len ))
        return stat_mean, stat_std 





    #---------------------------------------------------------------------
    # Based on the training data containing frequency information of corpora from same/different registers, this function calculates a 'middle point' as a
    # threshold value, which can be used to evaluate the accuracy of distinguishing corpora from same/different registers

    # sameRegi_mean_array: an array contains one or more means of similariy values (stat_list), depending on how many registers considered (e.g. we have three registers:
    #                      cc, tw, wk, then sameRegi_mean_array can include 3 mean values for cc_cc, tw_tw, wk_wk)
    #
    # diffRegi_mean_array: an array contains one or more means of similariy values (stat_list), depending on how many registers considered (e.g. we have three registers:
    #                      cc, tw, wk, then diffRegi_mean_array can include 3 mean values for cc_tw, tw_wk, wk_cc)

    #  measure_type considers 4 measures:  "chi_square", "spearman", "cosine_dis", "euclidean_dis"

    #---------------------------------------------------------------------


    def train_simi_middlevalue(self, sameRegi_mean_array, diffRegi_mean_array, measure_type):

        if measure_type == "spearman" :

            avg_same_regi = np.amin(sameRegi_mean_array)  
            avg_diff_regi = np.amax(diffRegi_mean_array)

        else:
            avg_same_regi = np.amam(sameRegi_mean_array)  
            avg_diff_regi = np.amin(diffRegi_mean_array)
               
        mid_v_same_vs_diff = (avg_same_regi + avg_diff_regi)/2 

        return mid_v_same_vs_diff  

    


    #---------------------------------------------------------------------   
    # Using the trained_wordlist (stored in files) to adjust the frequency array of extracted features (wordlist) from test/validation set,
    # make sure the extracted wordlist and its frequency array correspond to the same order of the trained wordlist
    #
    #  test_fre_array : is a numpy array, containing word/char frequency informain
    #  wordlist       : is a list, containing word/char features extracted from a corpus set  
    #
    #  test_fre_array, wordlist are generated from 'feature_extraction' function
    #---------------------------------------------------------------------    
                  
    def feature_array_transfer(self, test_fre_array, wordlist, traind_wordlist):


        len_wordlist = len(wordlist) # list length of original extracted features of test set

        len_traind_wordlist = len(traind_wordlist)   # it is 5k-length
  
        test_fre_array_1 = test_fre_array
    
        for i in range(len_traind_wordlist):
            for j in range(len_wordlist):
                  
                if traind_wordlist[i] == wordlist[j]:
                    test_fre_array_1[:, i] = test_fre_array_[:, j]
                    break
                if j == len_wordlist -1 :
                    test_fre_array_1[:, i] = 0      
                                  
        return test_fre_array_1

    
    



    #---------------------------------------------------------------------   
    #  Function 'get_ACC' generates the accuracy results by using the calculated similarity values (stat_list_same,  stat_list_diff) 
    #
    # stat_list_same: is a list containing similarity values of a number of pair of corpora from same registers
    # stat_list_diff: is a list containing similarity values of a number of pair of corpora from different registers
    #
    # stat_list_same/stat_list_diff can be obtained by using the 'get_simi_values' function

    # mid_v_same_vs_diff is a threshold value based on the training data, it can be calculated with 'train_simi_middlevalue' function  
    #
    #---------------------------------------------------------------------

    
    def get_ACC(self,  stat_list_same,  stat_list_diff, mid_v_same_vs_diff ):

        test_list = same_regi_list + diff_regi_list


        y_pred=[]    # 1: same registers    0: different registers
        


        if measure_type == "spearman":

            for test_v in test_list:
                
                if test_v > middle_v:   # for 'Spearman rho', a high value indicates the compared corpora are more similar
                    y_pred.append(1)
                else:
                    y_pred.append(0)

        else:
            
            for test_v in test_list:
                
                if test_v > middle_v:   # for other measures ("chi_square", "cosine_dis", "euclidean_dis"), a high value indicates the compared corpora are more different
                    y_pred.append(0)
                else:
                    y_pred.append(1)                        



        y_true = []   # ground truth: 1: same registers    0: different registers
    
        for i in range(len(stat_list_same)): 
            y_true.append(1)

        for i in range(len(stat_list_diff)): 
            y_true.append(0)


      
        ACC = precision_score(y_true, y_pred, average='micro')   # compute the accuracy 

        # print("acc is {A}".format(A = ACC))
 
        return ACC






#==============================================================================================
		
	
