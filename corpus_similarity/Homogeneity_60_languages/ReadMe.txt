

"Bayes_mvs_60_languages.csv" contains register homogeneity information represented by z-score mean, variation, standard deviation for 60 languages.


For 3 registers (CC: Web, WK: Wikipedia, TW: Twitter), we extract 100 pairs of unique sub-corpora for each register pair (CC-CC, WK-WK, TW-TW, CC-WK, WK-TW, TW-CC), where each sub-corpus contains 20k words. Each pair is represented using the similarity, calculated using the frequency-based method. 

Similarity values are normalized by z-scores for each language.


To represent the register homogeneity,the mean z-score similarity is calculated, using a Bayesian approach to estimating the mean with a 90% confidence level. 

