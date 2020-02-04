import re

rep_numbers=re.compile(r'\d+',re.IGNORECASE) # Numbers
rep_special_chars= re.compile("[^\w']|_") # Special character but not apostrophes

def apostrophes(text):
    return re.findall(r"\w+(?=n't)|n't|\w+(?=')|'\w+|\w+",
               text, re.IGNORECASE | re.DOTALL)

def text_to_words(text):     
    text=rep_special_chars.sub(' ', text) # Remove special characters but apostrophes    
    text = rep_numbers.sub('n', text) # substitute all numbers  
    words = text.lower()
    words = apostrophes(words)[:120]# Split string into words
    return words

def tokenize(word_dict, text):   
    words = text_to_words(text)
    words=[word_dict[w] if w in word_dict else word_dict['<unk>'] for w in words]
    return words