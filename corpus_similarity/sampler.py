from collections import namedtuple
import pandas as pd
import math
import random
import logging
import gzip

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

SamplerDataframe = namedtuple('SamplerDataframe', ['dataframe', 'mean_words'])

def count_words(row):
    if isinstance(row['Text'], float):
        return 0
    return len(row['Text'].split())

class Sampler(object):
    language = None
    full_dataframes = []
    full_df = None

    def get_dataframes(self, dataframes):
        logging.info('Fetching Data Frames')
        for f in dataframes:
            df = pd.read_csv(f)
            logging.info('Calculating number of words for {}'.format(f))
            df['count'] = df.apply(count_words, axis=1)
            mean_words = self.get_mean_words(df)
            logging.info('{} Mean Words: {}'.format(f, mean_words))
            self.full_dataframes.append(SamplerDataframe(df, mean_words))

    def build_dataframe_from_gzip(self, origin, destination):
        logging.info('opening file {}'.format(origin))
        data = gzip.open(origin, "rb").readlines()
        df = pd.DataFrame({'Text': data})
        logging.info('saving new df to  {}'.format(destination))
        df.to_csv(destination, index=False, compression='gzip')
        logging.info('new df saved')

    def get_mean_words(self, df):
        return df['count'].mean()

    def remove_extra_duplicates(self, df, acumulative_dfs):
        for a_df in acumulative_dfs:
            df = df[~df.index.isin(a_df.index)]
        return df

    def get_sample(self, amount, current_amount=None, current_df=None, acumulative_dfs=None):
        random_df = self.full_dataframes[random.randrange(0, len(self.full_dataframes))]

        amount_to_search = amount if not current_amount else current_amount
        logging.debug('Getting sample: {}'.format(amount_to_search))

        num_rows = math.ceil(amount_to_search/random_df.mean_words)

        new_df = random_df.dataframe.sample(n=num_rows)

        if current_df is not None:
            new_df = pd.concat([current_df, new_df]).drop_duplicates()

        new_df = self.remove_extra_duplicates(new_df, acumulative_dfs)
        sum_words = new_df['count'].sum()
        if sum_words <= amount-(amount*0.01):
            return self.get_sample(amount,
                                   current_amount=amount-sum_words,
                                   current_df=new_df,
                                   acumulative_dfs=acumulative_dfs)
        if sum_words >= (amount*1.01):
            return self.get_sample(amount,
                                   current_amount=current_amount,
                                   current_df=current_df,
                                   acumulative_dfs=acumulative_dfs)
        logging.debug('Sample fetched with {} words'.format(sum_words))

        return new_df