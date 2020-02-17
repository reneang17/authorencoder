
import pandas as pd
from utils import poem_fragments, wspace_schars
import os
import argparse

#************************************************************
#                   Processing Input vars                   #
#************************************************************

parser = argparse.ArgumentParser()

def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)

parser.add_argument('--data_dir', type=dir_path, default = '../data/wrangled/',
                        help="dir of wrangled data to process (default: ../data/wrangled/)")

parser.add_argument('--data_file', type=str, default = 'wrangled_data.csv',
                        help="wrangled data-file to process (default: wrangled_data.csv)")

parser.add_argument('--process_dir', type=dir_path, default = '../data/processed/',
                        help="dir to dump processed data (default: ../data/processed/)")

parser.add_argument('--split_into', type=int, default=100,
                        help='max words per poem extract (default: 100)')

parser.add_argument('--min_words_per_author', type=int, default = 2500,
                        help="Author min corpus word length  (default: 2500)")

parser.add_argument('--chars_to_keep', type=str, default=".,'\n?!",
                        help="chars to keep (default: .,'\n?!)")

parser.add_argument('--no_white_space', type=bool, default=True,
                        help="Keep (False) or not (True) multiple white space (default: True)")

parser.add_argument('--no_newlines', type=bool, default=True,
                        help="Keep (False) or not (True) multiple new lines (default: True)")

parser.add_argument('--max_authors', type=int, default = 10,
                        help="Keep authors with larges corpus ending at (default: 10)")

parser.add_argument('--min_authors', type=int, default = 0,
                        help="Keep authors with larges corpus starting from (default: 0)")

parser.add_argument('--train_test_ratio', type=float, default = 0.9,
                        help="Train test ratio (default: 0.9)")

parser.add_argument('--seed', type=int, default = 1234,
                        help="Seed for replicability (default: 1234)")


args = parser.parse_args()
#************************************************************
#vars to process data

data_dir = args.data_dir
data_file = args.data_file
process_dir = args.process_dir
SPLIT_INTO = args.split_into
MIN_WORDS_PER_AUTHOR = args.min_words_per_author
CHARS_TO_KEEP = args.chars_to_keep
NO_WHITE_SPACE = args.no_white_space
NO_NEWLINES = args.no_newlines
MAX_AUTHORS = args.max_authors
MIN_AUTHORS = args.min_authors
TRAIN_TEST_RATIO = args.train_test_ratio
SEED = args.seed

if not os.path.exists(process_dir): # Make sure that the folder exists
    os.makedirs(process_dir)
    print('Created: ', process_dir)
else:
    print('Process dir already exist', process_dir)

if MIN_AUTHORS == 0:
    dump_train_data = 'top'+str(MAX_AUTHORS)+'_train.json'
    dump_test_data = 'top'+str(MAX_AUTHORS)+'_test.json'
elif MIN_AUTHORS >0:
    dump_train_data= 'top'+str(MAX_AUTHORS)+'_'+str(MIN_AUTHORS)+'_train.json'
    dump_test_data= 'top'+str(MAX_AUTHORS)+'_'+str(MIN_AUTHORS)+'_test.json'


data_path = os.path.join(data_dir,data_file)
dump_train_path = os.path.join(process_dir, dump_train_data)
dump_test_path = os.path.join(process_dir, dump_test_data)

#************************************************************


#read
df = pd.read_csv(data_path,index_col=0).drop(columns=['poetry_foundation_id'])

#parse poems content
df['content'] = df.content.apply(lambda x: wspace_schars(x,
                chars_to_keep= CHARS_TO_KEEP,
                no_white_space = NO_WHITE_SPACE,
                no_newlines= NO_NEWLINES))

#Keeping authors which have at les
def long_corpus_authors( df, min_words_per_author = MIN_WORDS_PER_AUTHOR):
    repeated_authors = df.groupby('author')['length_in_words'].sum()
    repeated_authors_list= [i[0] for i in \
    repeated_authors[repeated_authors > min_words_per_author].items()]
    df = df[df.author.isin(repeated_authors_list)].drop(columns= ['length_in_words'])
    return df
df = long_corpus_authors(df, min_words_per_author = MIN_WORDS_PER_AUTHOR)

#Split poems into meaninful fragments
def split_poems(df):
    df_aux = df[:0]
    for i in range(len(df)):
        (author, title, poem_pa)= poem_fragments( df.iloc[i], split_into = SPLIT_INTO)
        for j in poem_pa:
            df_aux=df_aux.append(pd.Series({'author': author,
            'title':title, 'content':j }),ignore_index=True)
    return df_aux
df_split=split_poems(df)

# author list, dictionary with labels
authors_list= [i[0] for i in df_split.author.value_counts().items()]
author_dict = {j:i for i,j in enumerate(authors_list)}
df_split['author_label'] = df_split.author.apply(lambda x: author_dict[x])

#List of authors to keep for training and testing
MAX_LIST= authors_list[MIN_AUTHORS: MAX_AUTHORS]
print('Creating corpus based on: {}'.format(MAX_LIST))
#Split training and test set
def split_train_test(df, authors=MAX_LIST,  frac = 0.90):
    df_train= df[:0]
    df_test= df[:0]
    for i in authors:
        df_aux = df[df.author==i]
        train = df_aux.sample(frac=frac, random_state=SEED)
        test = df_aux.drop(train.index)
        df_train = df_train.append(train)
        df_test = df_test.append(test)
    return df_train, df_test
df_train, df_test = split_train_test(df, authors=MAX_LIST,  frac = TRAIN_TEST_RATIO)
df_train=split_poems(df_train)
df_test=split_poems(df_test)

#Create authors labels
df_train['author_label'] = df_train.author.apply(lambda x: author_dict[x])
df_test['author_label'] = df_test.author.apply(lambda x: author_dict[x])

#Export files
df_train.drop(columns= ['author', 'title']).to_json(dump_train_path, orient='records', lines=True)
df_test.drop(columns= ['author', 'title']).to_json(dump_test_path, orient='records', lines=True)
