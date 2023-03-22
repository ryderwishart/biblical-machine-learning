# Biblical Machine Learning

This repo contains example notebooks covering various basic or more complex approaches to gaining insight through NLP techniques. Every notebook can be opened in Google Colab using the button at the top of the `.ipynb` file for easy reproduction.

## Structured Data Insights

These notebooks are focused on helping a user gain insight and identify patterns using a corpus of **structured** or **annotated** data.

### [macula-greek-pandas](https://github.com/ryderwishart/biblical-machine-learning/blob/main/macula_greek_pandas.ipynb)

Process [macula-greek](https://github.com/Clear-Bible/macula-greek) TSV data using the Pandas library. Exemplifies doing some basic derivation of new data columns on the basis of existing data, and leverages semantic domains to identify the most similar chapters in the New Testament.

### [topic-modelling](https://github.com/ryderwishart/biblical-machine-learning/blob/main/topic_modelling.ipynb)

Create a topic model from Macula Greek data using lemmas as the input data. The model is created using the ensemble technique (see this [thesis](https://www.sezanzeb.de/machine_learning/ensemble_LDA/)), where multiple models are generated initially, and then only the most stable topics are included in the final model. The topic model is presented as an interactive visualization using `pyLDAvis`.

### [domain-topic-modelling](https://github.com/ryderwishart/biblical-machine-learning/blob/main/domain_topic_modelling.ipynb)

Create a topic model from Macula Greek data using semantic domains, rather than lemmas, which is useful for people not familiar with original languages, and exemplifies the way semantic domains can capture some interesting generalizations.

Topic models can be based on individual books, or subcorpora (sets of books).

<img width="876" alt="Screen Shot 2023-03-16 at 10 46 52 AM" src="https://user-images.githubusercontent.com/19649268/225708295-df86e63c-d4f1-478b-91fa-0ac090f13526.png">

### [syntax-knowledge-graph](https://github.com/ryderwishart/biblical-machine-learning/blob/main/topic_modelling.ipynb)

Build a graph from macula-greek syntax and plot in notebook or download as `.graphml` for visualization in Gephi, or your preferred graphing software.

<img width="652" alt="Screen Shot 2023-03-22 at 10 23 25 AM" src="https://user-images.githubusercontent.com/19649268/226987274-329fee96-d466-4916-8acc-92ce40ecf87a.png">

You can also create a graph using the glosses as labels. Other kinds of relationships (such as coreference) could be used as well. This notebook is merely intended to exemplify one possibility.

<img width="1025" alt="Screen Shot 2023-03-22 at 1 09 13 PM" src="https://user-images.githubusercontent.com/19649268/227026321-2fa2f3a9-39e5-43ef-9651-2beb4f9b033d.png">

## Unstructured Data Insights

These notebooks are for:

1. Discovering insights in unstructured data using inference on unstructured data only, or 
2. Leveraging structured data to build a model that can be used to infer insight into other texts.

### [tfidf-model](https://github.com/ryderwishart/biblical-machine-learning/blob/main/tfidf_model.ipynb)

Create a term-frequency/inverse-document-frequency model of unlemmatized Greek texts. You can then input a text of any length (could be a book, paragraph, pericope, sentence, etc.) to this model and it will return the 10 most significant terms summarizing that text.

Example input:
`input_text = "Ἐν ἀρχῇ ἦν ὁ λόγος, καὶ ὁ λόγος ἦν πρὸς τὸν θεόν, καὶ θεὸς ἦν ὁ λόγος."`

Output:
```
0.58: θεόν
0.54: λόγος
0.47: ἀρχῇ
0.40: θεὸς
```

### [semantic-search-mvp](https://github.com/ryderwishart/biblical-machine-learning/blob/main/semantic_search_mvp.ipynb)

This notebook offers a minimal viable example of how one might go about vectorizing documents (in this case paragraphs/verse clusters in the KJV) and subsequently querying the model for similar documents. Since the underlying data is so small, the quality of the results is minimal, but the concept is at least illustrated.

Example input:
`input_sentence = "Love your neighbor"`

Output:
```
0.83 Remember I beseech thee that thou hast made me as the clay and wilt thou bring me into dust again  10:10 Hast thou not poured me out as milk and curdled me like cheese  10:11 Thou hast clothed me with skin and flesh and hast fenced me with bones and sinews
0.83 And the people answered and said God forbid that we should forsake the LORD to serve other gods 24:17 For the LORD our God he it is that brought us up and our fathers out of the land of Egypt from the house of bondage and which did those great signs in our sight and preserved us in all the way wherein we went and among all the people through whom we passed 24:18 And the LORD drave out from before us all the people even the Amorites which dwelt in the land therefore will we also serve the LORD for he is our God
0.83 This is the inheritance of the tribe of the children of Naphtali according to their families the cities and their villages
0.83 And Samuel called the people together unto the LORD to Mizpeh 10:18 And said unto the children of Israel Thus saith the LORD God of Israel I brought up Israel out of Egypt and delivered you out of the hand of the Egyptians and out of the hand of all kingdoms and of them that oppressed you 10:19 And ye have this day rejected your God who himself saved you out of all your adversities and your tribulations and ye have said unto him Nay but set a king over us Now therefore present yourselves before the LORD by your tribes and by your thousands
0.82 And now O Lord GOD thou art that God and thy words be true and thou hast promised this goodness unto thy servant 7:29 Therefore now let it please thee to bless the house of thy servant that it may continue for ever before thee for thou O Lord GOD hast spoken it and with thy blessing let the house of thy servant be blessed for ever
0.82 For thus saith the Lord GOD Because thou hast clapped thine hands and stamped with the feet and rejoiced in heart with all thy despite against the land of Israel 25:7 Behold therefore I will stretch out mine hand upon thee and will deliver thee for a spoil to the heathen and I will cut thee off from the people and I will cause thee to perish out of the countries I will destroy thee and thou shalt know that I am the LORD
0.82 O sing unto the LORD a new song for he hath done marvellous things his right hand and his holy arm hath gotten him the victory
0.82 Let my prayer come before thee incline thine ear unto my cry
0.82 How much more abominable and filthy is man which drinketh iniquity like water  15:17 I will shew thee hear me and that which I have seen I will declare 15:18 Which wise men have told from their fathers and have not hid it 15:19 Unto whom alone the earth was given and no stranger passed among them
0.82 Let them shout for joy and be glad that favour my righteous cause yea let them say continually Let the LORD be magnified which hath pleasure in the prosperity of his servant
```



