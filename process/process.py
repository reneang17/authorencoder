#!/usr/bin/env python3

import pandas as pd
from utils import *



data_dir = '../data/wrangled/'
data_file = 'wrangled_data.csv'
process_dir= '../data/processed/'


# Text is splited into paragraphs of at leas SPLIT_INTO words
SPLIT_INTO = 100
#In order to keep an author we ask to have at least:
WORDS_NEEDED_PER_AUTHOR = 2500


df = pd.read_csv(data_dir + data_file,index_col=0)



df['content'] = df.content.apply(lambda x: wspace_schars(x,
                chars_to_keep=".,'\n?!",
                no_white_space = False, no_newlines= False))


#Keeping authors which have at les
repeated_authors = df.groupby('author')['length_in_words'].sum()
repeated_authors_list= [i[0] for i in \
repeated_authors[repeated_authors > WORDS_NEEDED_PER_AUTHOR].items()]
df = df[df.author.isin(repeated_authors_list)]

df_final = df[:0].drop(columns= ['poetry_foundation_id','length_in_words'])


# Breaking poems into 1 or more paragraphs with least 100 words
for i in range(len(df)):
    (author, title, poem_pa)= poem_fragments( df.iloc[i], split_into = SPLIT_INTO)
    for j in poem_pa:
        df_final=df_final.append(pd.Series({'author': author,
        'title':title, 'content':j }),ignore_index=True)

authors_list= [i[0] for i in df_final.author.value_counts().items()]
author_dict = {j:i for i,j in enumerate(authors_list)}
df_final['author_label'] = df_final.author.apply(lambda x: author_dict[x])



# Take top hundred authors, ignore the first to no
MAX_N_AUTHORS= 10
MAX_LIST= authors_list[:MAX_N_AUTHORS]
df_final[df_final.author.isin(MAX_LIST)].drop(columns= ['author', 'title']).to_json(process_dir+'top_10_authors.json', orient='records', lines=True)

# Take top hundred authors, ignore the first to no
MAX_N_AUTHORS= 90
MAX_LIST= authors_list[:MAX_N_AUTHORS]
df_final[df_final.author.isin(MAX_LIST)].drop(columns= ['author', 'title']).to_json(process_dir+'top_90_authors.json', orient='records', lines=True)

# Take top hundred authors, ignore the first to no
MAX_N_AUTHORS= 100
MIN_N_AUTHORS = 90
MAX_LIST= authors_list[MIN_N_AUTHORS:MAX_N_AUTHORS]
df_final[df_final.author.isin(MAX_LIST)].drop(columns= ['author', 'title']).to_json(process_dir+'bottom_10_authors.json', orient='records', lines=True)



# Getting 10 longest poems
LONGEST_N= 10
longest_list= [i[0] for i in df.length_in_words.sort_values(ascending=False).items()][:LONGEST_N]


_df= df[df.index.isin(longest_list)]
df_longest10 = _df[:0].drop(columns= ['poetry_foundation_id','length_in_words'])

for i in range(len(_df)):
    (author,title, poem_pa  )= poem_fragments( _df.iloc[i], split_into = SPLIT_INTO)
    for j in poem_pa:
        df_longest10=df_longest10.append(pd.Series({'author': author, 'title':title, 'content':j }),ignore_index=True)

author_dict = {j:i for i,j in enumerate(df_longest10.author.unique())}
df_longest10['author_label'] = df_longest10.author.apply(lambda x: author_dict[x])
df_longest10.drop(columns= ['author', 'title']).to_json(process_dir+'longest10.json', orient='records', lines=True)
