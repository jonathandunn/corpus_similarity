


# read the trained language features (word/char) from .json files


f = open(json_filename)

data = json.load(f)

data is in the following format with language name as column name:

{'tam': {'0': 'கக', '1': 'தத', '2': 'கள', '3': 'டட', '4': 'ஙக',   '5': 'பப', '6': 'நத', '7': 'வர', '8': 'ரக', '9': 'கம',
         ...
        } }


Use the language name (e.g. 'tam') to read the data:

d1 = data['tam']

values = d1.values()

wordlist_list = list(values)  