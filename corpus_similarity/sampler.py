from collections import namedtuple
from polyglot.text import Text, Word
import pandas as pd
import math
import random

SamplerDataframe = namedtuple('SamplerDataframe', ['dataframe', 'mean_words'])

def count_words(row):
    return len(Text(row['Text']).words)

class Sampler(object):
    language = None
    full_dataframes = []
    full_df = None

    def get_dataframes(self, dataframes):
        for f in dataframes:
            df = pd.read_csv(f)
            df['count'] = df.apply(count_words, axis=1)
            import pdb;pdb.set_trace()
            self.full_dataframes.append(SamplerDataframe(df, self.get_mean_words(df)))


    def get_mean_words(self, df):
        return df['count'].mean()

    def remove_extra_duplicates(self, df, acumulative_dfs):
        for a_df in acumulative_dfs:
            merged_data = df.merge(a_df,
                                   how='inner',
                                   on='article_id')['article_id']
            df = df[~df['article_id'].isin(merged_data)]
        return df

    def get_sample(self, amount, current_amount=None, current_df=None, acumulative_dfs=None):
        random_df = self.full_dataframes[random.randrange(0, len(self.full_dataframes))]

        amount_to_search = amount if not current_amount else current_amount
        num_rows = math.ceil(amount_to_search/random_df.mean_words)

        new_df = random_df.dataframe.sample(n=num_rows)

        if current_df is not None:
            new_df = pd.concat([current_df, new_df]).drop_duplicates()

        new_df = self.remove_extra_duplicates(new_df, acumulative_dfs)
        sum_words = new_df['count'].sum()

        if sum_words <= (amount*-1.1):
            return self.get_sample(amount,
                                   current_amount=amount-sum_words,
                                   current_df=new_df,
                                   acumulative_dfs=acumulative_dfs)
        if sum_words >= (amount*1.1):
            return self.get_sample(amount,
                                   current_amount=current_amount,
                                   current_df=current_df,
                                   acumulative_dfs=acumulative_dfs)

        return new_df
