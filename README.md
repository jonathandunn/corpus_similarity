# corpus_similarity

Measure the similarity between two corpora (text datasets). The measures work best when each corpus is at least 10k words. This package support 74 languages.

    from corpus_similarity import Similarity
    cs = Similarity(language = "eng")

    result = cs.calculate(corpus1, corpus2)

The package contains all preprocessing and training. Only the language needs to be specified. A list of supported languages is provided below.

# Input

The **Similarity.calculate** method requires two input corpora. These can be a list of strings or a filename (supports .txt and .gz files).

# Output

The output is a scalar measure of how similar the two corpora are. The values fall between 0 (very different) and 1 (very similar). The values are consistent within languages, but not across languages. For example, Swedish has higher relative similarity than Estonian.

# Installation

    pip install corpus_similarity

    pip install git+https://github.com/jonathandunn/corpus_similarity.git
    
# Languages

amh, Amharic

ara, Arabic

aze, Azerbaijani

ben, Bengali

bul, Bulgarian

cat, Catalan

ceb, Cebuano

ces, Czech

cha, Chamorro

dan, Danish

deu, German

ell, Greek

eng, English

est, Estonian

eus, Basque

fas, Farsi

fij, Fijian

fin, Finnish

fra, French

gle, Gaelic

glg, Galician

guj, Gujarati

hat, Hatian

haw, Hawaiian

heb, Hebrew

hin, Hindi

hmo, Hiri Motu

hun, Hungarian

ilo, Ilocano

ind, Indonesian

isl, Icelandic

ita, Italian

jav, Javanese

jpn, Japanese

kan, Kannada

kat, Georgian

kor, Korean

lav, Latvian

lit, Lithuanian

mal, Malayalam

mar, Marathi

mkd, Macedonian

mlg, Malagasy

mon, Mongolian

mri, te reo Māori

msa, Malay

nld, Dutch

nor, Norwegian

pan, Punjabi

pol, Polish

por, Portuguese

ron, Romanian

rus, Russian

sin, Sinhala

slk, Slovak

slv, Slovenian

smo, Samoan

som, Somali

spa, Spanish

sqi, Albanian

swe, Swedish

tah, Tahitian

tam, Tamil

tel, Telugu

tgl, Tagalog

tha, Thai

ton, Tongan

tur, Turkish

tvl, Tuvaluan

ukr, Ukrainian

urd, Urdu

uzb, Uzbek

vie, Vietnamese

zho, Chinese