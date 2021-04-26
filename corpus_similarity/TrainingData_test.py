"""
Created on Sat Apr 24 20:15:39 2021

@author: hli124
"""


from corpus_similarity import training
from sklearn.metrics import precision_score
import csv



#======================================================
# for each language, there are 3 register corpora
#
# name1, name2, name3 are corpora (txt) file names from 3 different registers (tw/cc/wiki)
# For each register, 30 subcorpora are collected (15 for training and 15 for validation) and saved in corpuslist1
#======================================================

def get_subcorpus(S, name1, name2, name3):

    train_data1 = training(name1)
    train_data2 = training(name2)
    train_data3 = training(name3)

    corpuslist1=[]


#=========== train_set 15 ========= 
    for i in range(15):
        str_i = train_data1.subcorpus(1, S)   # tw
        corpuslist1.append(str_i)

    for i in range(15):
        str_i = train_data2.subcorpus(1, S)   # cc
        corpuslist1.append(str_i)

    for i in range(15):
        str_i = train_data3.subcorpus(1, S)   # wiki
        corpuslist1.append(str_i)    


#=========== validation_set 15 ========= 
    for i in range(15):
        str_i = train_data1.subcorpus(2, S)   # tw
        corpuslist1.append(str_i)

    for i in range(15):
        str_i = train_data2.subcorpus(2, S)   # cc
        corpuslist1.append(str_i)


    for i in range(15):
        str_i = train_data3.subcorpus(2, S)   # wiki
        corpuslist1.append(str_i)


    return corpuslist1     





#================================================================
#   Training corpora in same/different registers using similarity measures
#   Using validation set and accuracy to validate the trained data
#   output accuracy values in ACC_LIST using different feature numbers (N)
#=================================================================

def train_validation_corpora_simi(train_fre_array, test_fre_array, S, measure_type, feature_type, n):
    
# ===============  same register ===============
    tw_fre_array = train_fre_array[0:15, :]
    cc_fre_array = train_fre_array[15:30, :]
    wk_fre_array = train_fre_array[30:45, :]
    

    S_str = 'S=' + str(S)
    
    ACC_LIST=[]
    ACC_LIST.append(S_str)

    for N in [50, 200, 500, 1000, 2000, 3000, 5000]:
        print('=====================================')
        print("S = {SIZE}, feature_N = {fN}, n-gram = {ng}, feature_type= {ft}".format(SIZE = S, fN = N, ng = n, ft = feature_type))
        print('=====================================')

# ===============  Training same register corpora 315 pairs ===============
    
        tw_tw_stat_list = train_data1.same_regi_measure(tw_fre_array, N, measure_type, S)
        cc_cc_stat_list = train_data1.same_regi_measure(cc_fre_array, N, measure_type, S)
        wk_wk_stat_list = train_data1.same_regi_measure(wk_fre_array, N, measure_type, S)

        tw_tw_train_mean1, tw_tw_train_std1 = train_data1.measure_mean(tw_tw_stat_list)
        cc_cc_train_mean1, cc_cc_train_std1 = train_data1.measure_mean(cc_cc_stat_list)
        wk_wk_train_mean1, wk_wk_train_std1 = train_data1.measure_mean(wk_wk_stat_list)

        avg_same_regi = (tw_tw_train_mean1 + cc_cc_train_mean1 + wk_wk_train_mean1)/3


# ===============  Training different register corpora 300 pairs ===============

        tw_fre_array2 = train_fre_array[0:10, :]
        cc_fre_array2 = train_fre_array[15:25, :]
        wk_fre_array2 = train_fre_array[30:40, :]


        tw_cc_stat_list = train_data1.diff_regi_measure(tw_fre_array2, cc_fre_array2, N, measure_type, S)
        wk_cc_stat_list = train_data1.diff_regi_measure(wk_fre_array2, cc_fre_array2, N, measure_type, S)
        tw_wk_stat_list = train_data1.diff_regi_measure(tw_fre_array2, wk_fre_array2, N, measure_type, S)


        tw_cc_train_mean1, tw_cc_train_std1 = train_data1.measure_mean(tw_cc_stat_list)
        wk_cc_train_mean1, wk_cc_train_std1 = train_data1.measure_mean(wk_cc_stat_list)
        tw_wk_train_mean1, tw_wk_train_std1 = train_data1.measure_mean(tw_wk_stat_list)


        avg_diff_regi = (tw_cc_train_mean1 + wk_cc_train_mean1 + tw_wk_train_mean1)/3

        mid_v_same_vs_diff = (avg_same_regi + avg_diff_regi)/2 



#=============================================
#============   Validation ===================
#=============================================


# =============== test/validating same register corpora 315 pairs ===============
        tw_test_array = test_fre_array[0:15, :]
        cc_test_array = test_fre_array[15:30, :]
        wk_test_array = test_fre_array[30:45, :]


        tw_tw_test_list = train_data1.same_regi_measure(tw_test_array, N, measure_type, S)
        cc_cc_test_list = train_data1.same_regi_measure(cc_test_array, N, measure_type, S)
        wk_wk_test_list = train_data1.same_regi_measure(wk_test_array, N, measure_type, S)

        same_regi_list = tw_tw_test_list + cc_cc_test_list + wk_wk_test_list


# =============== test/validating different register corpora 300 pairs ===============

        tw_test_array2 = test_fre_array[0:10, :]
        cc_test_array2 = test_fre_array[15:25, :]
        wk_test_array2 = test_fre_array[30:40, :]


        tw_cc_test_list = train_data1.diff_regi_measure(tw_test_array2, cc_test_array2, N, measure_type, S)
        wk_cc_test_list = train_data1.diff_regi_measure(wk_test_array2, cc_test_array2, N, measure_type, S)
        tw_wk_test_list = train_data1.diff_regi_measure(tw_test_array2, wk_test_array2, N, measure_type, S)

        diff_regi_list = tw_cc_test_list + wk_cc_test_list + tw_wk_test_list


        test_list = same_regi_list + diff_regi_list 


        y_pred=[]    # 1: same registers    0: different registers


        if measure_type == "spearman": 
            for test_v in test_list: 
                if test_v > mid_v_same_vs_diff : 
                    y_pred.append(1)
                else:
                    y_pred.append(0)
            
        else:
            for test_v in test_list: 
                if test_v > mid_v_same_vs_diff : 
                    y_pred.append(0)
                else:
                    y_pred.append(1)   
            
            
            
    

#---------------------------------------

        y_true = []   # ground truth: 1: same registers    0: different registers
    
        for i in range(len(same_regi_list)): 
            y_true.append(1)

        for i in range(len(diff_regi_list)): 
            y_true.append(0)


#----------------------------------------

        print("length of same-regi test pair is {n0}".format(n0 = len(same_regi_list)))      
        print("length of diff-regi test pair is {n1}".format(n1 = len(diff_regi_list)))


      
        ACC = precision_score(y_true, y_pred, average='micro')
        print("acc is {A}".format(A = ACC))
        
        

        ACC_LIST.append(ACC)
        print('=====================================')
        

    print('ACC_LIST: ', ACC_LIST)
    return ACC_LIST





# ===================
# for each language with 3 register corpora, we test parameters: S/N/n-gram/measures
# The training/validation procedures are repeated 840 times with accuracies stored in 120 rows (7 records in each row)
# The accuracy results are stored in a csv file for checking
# ===================

def save_acclist(final_list_120, csv_name):
    

    fields = [[' ', 'word-1', 'spearman ', '  ', '  ','  ', '  ', '  '],        
              [' ', 'word-1', 'chi_square ', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-1', 'cosine_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-1', 'euclidean_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-2', 'spearman ', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-2', 'chi_square ', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-2', 'cosine_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-2', 'euclidean_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-3', 'spearman ', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-3', 'chi_square ', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-3', 'cosine_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'word-3', 'euclidean_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-2', 'spearman ', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-2', 'chi_square ','  ', '  ','  ', '  ', '  '],
              [' ', 'char-2', 'cosine_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-2', 'euclidean_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-3', 'spearman ', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-3', 'chi_square ', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-3', 'cosine_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-3', 'euclidean_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-4', 'spearman ', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-4', 'chi_square ', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-4', 'cosine_dis', '  ', '  ','  ', '  ', '  '],
              [' ', 'char-4', 'euclidean_dis', '  ', '  ','  ', '  ', '  ']]


    new_final_list_120=[]

    for i in range(24):
        new_final_list_120.append(fields[i])
   
        for j in range(5):
            new_final_list_120.append(final_list_120[i + 24*j])


    fields11 = ['   ', ' N=50 ', ' N=200 ', ' N=500  ', ' N=1000 ',' N=2000 ', ' N=3000 ', ' N=5000 ']
              
    with open(csv_name, 'w',  newline='') as f:
        write = csv.writer(f)
        write.writerow(fields11)
        write.writerows(new_final_list_120)




#=====================================
#=====================================

# S: subcorpus size
# N: feature number
# measure_type = chi_square/spearman/cosine_dis/euclidean_dis
# n: n-gram
# feature_type = word/char





train_data1 = training('train')

# for each language, we will generate an accuracy list containing 120 rows (840 records)

final_list_120=[]

train_data1_fullname = 'data_tw_file'   
train_data2_fullname = 'data_cc_file' 
train_data3_fullname = 'data_wk_file'


for S in [10000, 30000, 50000, 100000, 500000]:

    corpuslist = get_subcorpus(S, train_data1_fullname, train_data2_fullname, train_data3_fullname)


    for n in [1, 2, 3]:
            
        feature_type = 'word'
            
        data_fre_array = train_data1.feature_extraction(corpuslist, feature_type, n)   # tw/cc/wk tw/cc/wk

        train_fre_array = data_fre_array[0:45, :]
        test_fre_array = data_fre_array[45:90, :]  # test data


        for measure_type in ['spearman', 'chi_square', 'cosine_dis', 'euclidean_dis']:

            acclist = train_test_corpora_simi(train_fre_array, test_fre_array, S, measure_type, feature_type, n)
            final_list_120.append(acclist)



    for n in [2, 3, 4]:
            
        feature_type = 'char'
            
        data_fre_array = train_data1.feature_extraction(corpuslist, feature_type, n)   # tw/cc/wk tw/cc/wk

        train_fre_array = data_fre_array[0:45, :]
        test_fre_array = data_fre_array[45:90, :]  # test data

        for measure_type in ['spearman', 'chi_square', 'cosine_dis', 'euclidean_dis']:

            acclist = train_test_corpora_simi(train_fre_array, test_fre_array, S, measure_type, feature_type, n)
            final_list_120.append(acclist)


csv_name = 'acc_test.csv'
save_acclist(final_list_120, csv_name)

