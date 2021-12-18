import scipy.stats
from cleantext import clean
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata
from scipy.stats import chisquare
from scipy.stats import spearmanr
from scipy.spatial import distance
from scipy import stats
from pathlib import Path
from .sampler import Sampler
import numpy as np
import os
import codecs
import gzip
import json
import random

#Constants
SPACELESS_LANGS = ["jpn", "zho", "tha", "tam"]

#Best features per supported language
FEATURE_DICT = {'jpn': ('char_wb', 2),
                'zho': ('char_wb', 2),
                'bul': ('char_wb', 4),
                'cat': ('char_wb', 4),
                'ces': ('char_wb', 4),
                'dan': ('char_wb', 4),
                'deu': ('char_wb', 4),
                'ell': ('char_wb', 4),
                'eng': ('char_wb', 4),
                'fin': ('char_wb', 4),
                'glg': ('char_wb', 4),
                'heb': ('char_wb', 4),
                'hin': ('char_wb', 4),
                'hun': ('char_wb', 4),
                'kor': ('char_wb', 4),
                'lav': ('char_wb', 4),
                'nld': ('char_wb', 4),
                'nor': ('char_wb', 4),
                'pol': ('char_wb', 4),
                'ron': ('char_wb', 4),
                'rus': ('char_wb', 4),
                'slv': ('char_wb', 4),
                'swe': ('char_wb', 4),
                'tam': ('char_wb', 4),
                'tel': ('char_wb', 4),
                'tha': ('char_wb', 4),
                'tur': ('char_wb', 4),
                'ukr': ('char_wb', 4),
                'urd': ('char_wb', 4),
                'ara': ('word', 1),
                'est': ('word', 1),
                'fas': ('word', 1),
                'fra': ('word', 1),
                'ind': ('word', 1),
                'ita': ('word', 1),
                'por': ('word', 1),
                'spa': ('word', 1),
                'tgl': ('word', 1),
                'vie': ('word', 2),
                'ceb': ('char_wb', 4),
                'cha': ('char_wb', 4),
                'fij': ('char_wb', 4),
                'haw': ('char_wb', 4),
                'hmo': ('char_wb', 4),
                'ilo': ('char_wb', 4),
                'jav': ('char_wb', 4),
                'mlg': ('char_wb', 4),
                'mri': ('char_wb', 4),
                'msa': ('char_wb', 4),
                'smo': ('char_wb', 4),
                'sun': ('char_wb', 4),
                'tah': ('char_wb', 4),
                'tgl': ('char_wb', 4),
                'ton': ('char_wb', 4),
                'tvl': ('char_wb', 4),                
                'amh': ('char_wb', 4),
                'aze': ('char_wb', 4),
                'ben': ('char_wb', 4),
                'guj': ('char_wb', 4),
                'hat': ('char_wb', 4),
                'kan': ('char_wb', 4),
                'lit': ('char_wb', 4),
                'mal': ('char_wb', 4),
                'mar': ('char_wb', 4),
                'mkd': ('char_wb', 4),
                'mlg': ('char_wb', 4),
                'pan': ('char_wb', 4),
                'sin': ('char_wb', 4),
                'slk': ('char_wb', 4),
                'som': ('char_wb', 4),              
                'eus': ('word', 1),
                'gle': ('word', 1),
                'isl': ('word', 1),
                'sqi': ('word', 1),               
                'kat': ('word', 2),
                'mon': ('word', 2),
                'uzb': ('word', 2),
                }

#In-Domain Features (twitter, web, wikipedia)
IN_DOMAIN_PATH = os.path.join("in_domain_features")
if not os.path.isdir( IN_DOMAIN_PATH ) :
    IN_DOMAIN_PATH = Path(__file__).parent / os.path.join(".", IN_DOMAIN_PATH)

#Out-of-Domain Features (bibles, subtitles, news)
OUT_OF_DOMAIN_PATH = os.path.join("out_of_domain_features")
if not os.path.isdir( OUT_OF_DOMAIN_PATH ) :
    OUT_OF_DOMAIN_PATH = Path(__file__).parent / os.path.join(".", OUT_OF_DOMAIN_PATH)

#Number of features
N_FEATURES = "5k"

#Scaler values
SCALER_PATH = os.path.join("scaler_values")
if not os.path.isdir( SCALER_PATH ) :
    SCALER_PATH = Path(__file__).parent / os.path.join(".", SCALER_PATH)


#===============================================================================

class Similarity(object):

    languages = []
    scaler = None
    vectorizers = {}

    def __init__(self, language, threshold=1000000, feature_source="out"):
        self.feature_source = feature_source
        self.threshold = threshold
        self.language = language
        self.get_languages_features()
        self.scaler = self.get_scaler()

    def get_languages_features(self):
        self.Load = Load(self.language, self.threshold)

        if self.threshold < 10000:
            print("\nWARNING: Corpus sizes below 10k words do not have verified accuracy.\n")

        try:
            text_feature = FEATURE_DICT[self.language][0]
            n = FEATURE_DICT[self.language][1]

        except Exception as e:
            print("\nERROR: " + self.language + " is not currently supported.")
            print(e)
            exit()

        feature_file = self.language + "_" + N_FEATURES + "_" + self.feature_source.upper() + "_" + FEATURE_DICT[self.language][0][
                                                                                          :4] + str(
            FEATURE_DICT[self.language][1]) + ".json"

        if self.feature_source == "in":
            feature_file = os.path.join(IN_DOMAIN_PATH, feature_file)
        elif self.feature_source == "out":
            feature_file = os.path.join(OUT_OF_DOMAIN_PATH, feature_file)
        else:
            print("\nERROR: Feature source is not available.\n")
            exit()

        with codecs.open(feature_file, "r", encoding="utf-8") as fo:
            feature_set = json.load(fo)
            feature_set = list(feature_set[self.language].values())

        self.vectorizer = CountVectorizer(analyzer=text_feature,
                                          ngram_range=(n, n),
                                          vocabulary=feature_set)


    def get_scaler(self):
        """
        Attempts to instantiate a StandarScaler based on pre fit values from path.
        returns: scikit.StandarScaler or None.
        """
        scaler_path = os.path.join(SCALER_PATH, '.'.join((self.language, 'json')))
        scaler = None
        if os.path.exists(scaler_path):
            with open(scaler_path) as scaler_file_data:
                scaler_data = json.load(scaler_file_data)
                scaler = StandardScaler()
                scaler.mean_ = scaler_data['mean_']
                scaler.scale_ = scaler_data['scale_']
        else:
            print('WARNING: Scaler for this language [{}] is not in path, values will not be scaled.'.format(self.language))
        return scaler

    def scale(self, values):
        if self.scaler:
            return self.scaler.transform([[values, 1]])
        return values

    #--------------------------------------------------
    def get_features(self, lines):
        """
        Returns Features.
        :param lines:
        :param language:
        :return:
        """
    
        X = self.vectorizer.transform(lines)
        fre_array = X.toarray()
        fre_array_sum = np.sum(fre_array, axis=0)
        return fre_array_sum

    def get_samples(self, corpora_list, chunk_size, n):
        """
        :param corpora_list:
        :param chunk_size:
        :param n:
        :return:
        """
        sampler = Sampler()
        sampler.get_dataframes(corpora_list)
        acum = []
        for i in range(n):
            acum.append(sampler.get_sample(chunk_size, acumulative_dfs=acum))
        return [self.df_as_list(a) for a in acum]

    def df_as_list(self, df):
        return [str(x) for x in df.loc[:, 'Text'].values]

    def calculate_similarity(self, corpora_list1, corpora_list2, chunk_size=20000, n_pairs=100):
        """
        1- break corpora into 20k word subsets ( non overlapping ). (optional: add parameters with chunk size)
        2- get a N amount of pairs, run calculate for each of this pair-chunks comparisons.
        3- scale the array returned by all the calculate executions , returns 1 value for each similiarity value.
        4- over that array we get the bayesian mean values.
        Compares similarity between 2 same language corpora (eg: wiki/tw).
        :return:
        """
        sample1 = self.get_samples(corpora_list1, chunk_size, n_pairs)
        sample2 = self.get_samples(corpora_list2, chunk_size, n_pairs)
        result = [self.scale(self.calculate(sample1[i], sample2[i])) for i in range(len(sample1))]
        return scipy.stats.bayes_mvs(result)

    def calculate_similiarity_(self, corpora_list, chunk_size=20000, n_pairs=100):
        """
        same as corpus_similiartiy but with chunks of itself.
        :param corpus1:
        :return:
        """
        samples = self.get_samples(corpora_list, chunk_size, n_pairs*2)
        sample1 = samples[:len(samples) // 2]
        sample2 = samples[len(samples) // 2:]
        result = [self.scale(self.calculate(sample1[i], sample2[i])) for i in range(len(sample1))]
        return scipy.stats.bayes_mvs(result)

    def calculate_training_similarity(self, test_corpora, training_corpora, chunk_size=2000, n_pairs=100):
        """
        Calculates similarity with given standard training set.
        :param test_corpora:
        :param training_corpora:
        :param chunk_size:
        :param n_pairs:
        :return:
        """
        if len(training_corpora) != n_pairs:
            raise Exception('Training corpora must have same size as n_pairs')

        samples = self.get_samples([test_corpora], chunk_size, n_pairs*2)
        result = [self.scale(self.calculate(samples[i], training_corpora[i])) for i in range(len(samples))]
        return scipy.stats.bayes_mvs(result)
    #-------------------------------------------------
    def calculate(self, corpus1, corpus2):

        lines1 = self.Load.load(corpus1)
        lines2 = self.Load.load(corpus2)

        features1 = self.get_features(lines1)
        features2 = self.get_features(lines2)

        value = spearmanr(features1, features2)[0]

        return value

#===============================================================================
#Class for loading and cleaning text data

class Load(object):

    def __init__(self, language, threshold = 1000000):

        self.language = language
        self.threshold = threshold
        self.spaceless = False

        if self.language in SPACELESS_LANGS:
            self.spaceless = True            

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
                                
                #Spaceless
                if self.spaceless == True:
                    text = "".join(text.split())

                length = len(text.split())
                counter += length
                new_lines.append(text)
                            
        return new_lines
    #--------------------------------------------------
    
#===============================================================================


class training:
    

    def __init__(self, filename, language, threshold = 100000000):
        
        
        self.Load = Load(language = language, threshold = threshold)
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
        all_lines = self.Load.load(self.filename)

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
                
                if self.Load.spaceless == True:
                    word_num_line = len(line_content.split())
                    word_num = word_num + word_num_line

                elif self.Load.spaceless == False:
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



 
    #---------------------------------------------------------------------   
    #  Function 'get_z_scores' normalizes raw similarity values by z-scores, and calculate mean, variation,
    #  and standard deviation using Bayesian approach (with a 90% confidence level). 
    #
    # stat_list_same: is a list containing similarity values of a number of pair of corpora from same registers
    # stat_list_diff: is a list containing similarity values of a number of pair of corpora from different registers
    #
    #
    # stat_list_same/stat_list_diff can be obtained by using the 'get_simi_values' function
    #
    #
    # output: z_score: normalized similarity values 
    #         homo_mean, homo_var, homo_std: homogeneity estimate by mean, variation, standard deviation based
    #                                        on the z_scores of same register pairs (stat_list_same)      
    #---------------------------------------------------------------------

    
    def get_z_scores(self,  stat_list_same,  stat_list_diff ):

        test_list = same_regi_list + diff_regi_list

        z_score = stats.zscore(test_list)

        homo_len = len(same_regi_list)

        z_score_homo = z_score[0:homo_len]
        homo_mean, homo_var, homo_std = stats.bayes_mvs(z_score_homo)   # using Bayesian approach
        
        return z_score, homo_mean, homo_var, homo_std


      
#==============================================================================================
