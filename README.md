# corpus_similarity

Measure the similarity between two corpora (text datasets). The measures work best when each corpus is at least 10k words.

    from corpus_similarity import Similarity
    cs = Similarity(language = "eng")

    result = cs.calculate(corpus1, corpus2)

The package contains all preprocessing and training. Only the language needs to be specified. A list of supported languages is provided below.

# Input

The **Similarity.calculate** method requires two input corpora. These can be a list of strings or a filename (supports .txt and .gz files).

# Output

The output is a scalar measure of how similar the two corpora are. The values fall between 0 (very different) and 1 (very similar). The values are consistent within languages, but not across languages. For example, Swedish has higher relative similarity than Estonian.

# Installation

    pip install git+https://github.com/jonathandunn/corpus_similarity.git