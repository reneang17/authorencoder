
import re

def wspace_schars(review, chars_to_keep=".,'\n" , no_white_space = True,
                  no_newlines= True):
    """
    TODO
    """

    to_keep= ""
    for i in chars_to_keep:
        to_keep+= i+'|'

    rep_special_chars= re.compile("[^\w\n|"+ (to_keep[:-1])+ "]|_")

    # Subs special chars by white space except chars_to_keep
    text=rep_special_chars.sub(' ', review)
    if no_white_space:
        text = re.sub('\n+', '\n',text) # Remove consecutive breaklines
    if no_newlines:
        text = re.sub(' +', ' ',text) # Remove consecutive white space
    return text

def poem_fragments(poem_series, split_into):
    """
    Gets wordlend of a poem,
    if larger than SPLIT_INTO partions into next paragraph
    return author, title and poem broken in this way
    """


    poem = poem_series
    poem_author = poem.author
    poem_title = poem.title
    poem_content = poem.content
    poem_pa= poem.content.split('.\n')
    i=0
    while ((i+1)!=(len(poem_pa))):
        if not (len(poem_pa[i].split())<split_into):
            if poem_pa[i][-1]!='.': poem_pa[i]=poem_pa[i]+'.'
            #print('FINAL')
            #print(poem_pa[i])
            i+=1
        else:
            #print('BEFORE')
            #print(poem_pa[i])
            poem_pa[i] =  poem_pa[i]+'.\n'+poem_pa[i+1]

            del poem_pa[i+1]
    return  (poem_author, poem_title  ,poem_pa)
