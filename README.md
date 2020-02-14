
![library.](./media/library_panoramic.jpg)
# AuthorEncoder

Find poetry authors who writes similar to your dearest poems.

Would it be great if,  after reading a touching poem, we could find authors who have similar ideas or style? To this end, I build a scalable recommendation system that takes  as input a poem and outputs authors that write similarly. Bookshops can scale this service to empower their customers, boosting sales and loyalty. The underlying AI results from applying computer vision techniques  to encode the legacy of a hundred of classic and modern poets. 

## How does it work?

Tipically, recomendation systems are based on what other people like. Instead, AuthorEncoder makes recomendations based on 
on the poems themselfs. How is this possible? you might be asking. The answer is bringing the triple tecnique of [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832) to natural language processing. Precisely, we wrangle and process poems, which are them used to build an authorencoder that multiple authors, which in turn can be used to give recomendation once the embedding itself. Run the main file to build this project yourself. 

## AuthorEncoder in action

![](./media/how_it_works.gif)

## About used data

The data to training this model can be found on [John Hallman repo](https://github.com/johnhallman/poem-data-processing) which to my understanding contains the larger [poetryfoundation.org](poetryfoundation.org). The required wrangling can be found on 
wrangling and EDA. 

## Requirements 

Appart from the requirements.txt, follow the instructions on [here](https://spacy.io/usage)
to install spaCy. 

## Current results and metrics

The present encoder is build with the 10 authors which produces the larges corpus of poems. 
