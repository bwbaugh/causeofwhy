System Components
=================

The QA system is made up of several major processes. Below is a general
overview of the flow through the system.

Overall Flowchart
-----------------

<pre>
                            +-------+
                            | Start |
                            +---+---+
+-------------------------------+
|                               |
| +---------------+        +----v-----+
| |   Document    |        |  Corpus  |
| | Preprocessing <---no---+ Indexed? |
| +-------+-------+        +----+-----+
|         |                     | yes
|   +-----v----+           +----v-----+
|   | Document |           |  Input   |
+---+ Indexing |           |  Query   |
    +-----+----+           +----+-----+
          |                     |
   +------+------+         +----v-----+
   |             |         | Question |
   |  IR System  |         | Analysis |
   |             |         +----+-----+
   +------+------+              |
          +---------------------+
                                |
                      +---------v----------+
                      | Candidate Document |
                      |     Selection      |
                      +---------+----------+
                                |
                      +---------v----------+
                      | Candidate Document |
                      |      Analysis      |
                      +---------+----------+
                                |
                          +-----v------+
                          |   Answer   |
                          | Extraction |
                          +-----+------+
                                |
                          +-----v------+
                          |  Response  |
                          | Generation |
                          +-----+------+
                                |
                             +--v--+
                             | End |
                             +-----+
</pre>

Individual Components
---------------------

Let's take a look at the steps involved with the individual components.

### Document Preprocessing

This process involves transforming the corpus from the format that it is
given in to a format that we are able to process more easily.

First we remove the MediaWiki markup using the
[*WikiExtractor.py* library](http://medialab.di.unipi.it/wiki/Wikipedia_extractor),
leaving just the plain text. Next we convert all characters from Unicode
to the closest ASCII equivalent using the
[*unidecode* library](http://pypi.python.org/pypi/Unidecode).

Future work on this module could include merging the steps of the
*Candidate Document Analysis* component into this one so that they also
don't have to be performed at run-time.
It should be noted, however, that certain steps such as POS tagging
require significant computational time, and can cause the time it takes
to initially generate the index by some order of magnitude.

### Document Indexing

After obtaining a plain-text version of an article, we index it so that
we may retrieve it later and run queries on the entire document
collection.

For each document we begin by segmenting each document into paragraphs,
and then by segmenting each sentence using the Punkt Sentence Segmenter
provided by the [*NLTK* library](http://nltk.org/). The Punkt sentence
segmenter comes trained on an English corpus, however we created a
custom trained segmenter by training on our entire *Simple English*
plain-text corpus.
We then run tokenize each sentence, and regularize the tokens through
case folding (convert to lowercase), stopword removal, and
lemmatization. We then count the tokens and create a mapping of the
`Page.ID` to the byte location in corpus file. A mapping from token to
the set of `Page.ID`s that contain that token is maintained during the
process. We also keep track of token frequency statistics for each
document and across the entire collection for use in the IR step, as
well as keep a dictionary mapping from a token string to an integer
token-id by using the [*Gensim* library](http://radimrehurek.com/gensim/).

### Information Retrieval System

The IR system is the collection of index files with functions for
getting a set of documents matching some criteria. The available
functions include:

* Retrieve a specific document from the corpus (in O(1) time).
* Get the set intersection of documents that contain all terms in a term
list.
* Get the union set of documents that contain any term in a term list.
* Get a ranked list of documents that are similar to a term list vector.

During our experiments with the full sized $English Wikipedia$, we found
that we had difficulty storing in memory the mapping from token to the
set of `Page.ID`s that contain that token. Without the mapping we would
have to compute the TF&ndash;IDF cosine similarity between every
document, since we could not throw out documents that don't contain any
of the terms in the query, which would cause execution time to be
extremely slow. Therefore we experimented with using
[*MongoDB*](http://www.mongodb.org/) as an on-disk database that we
could use to retrieve the set of `Page.ID`s that contained at least one
query term. Though this approach worked successfully, we decided not to
still not use the larger corpus because, in addition to a slowdown
during IR due to the more than 100-times greater number of articles, the
average length of each article was also greater causing the already slow
*Answer Extraction* phase to take even longer.

### Question Analysis

The input from the user needs to be analyzed so that an answer may be
returned. We process the user's query using the same steps that we used
on the plain-text corpus during the indexing phase, including stopword
removal and lemmatization.

An alternative approach would be to extract keywords using the following
guide:

1. Keyword extraction
  * Named entities
  * Complex nominals and adjective modifiers
  * Nouns
  * Verbs
1. Query generation

In an open-domain QA system it is necessary to determine what type of
question is being asked, and what the expected answer type might be.
Our system has been explicitly designed for *causal* questions, and
therefore question analysis has been put mostly on the back burner.
Question validation&mdash;the process of verifying that the question
is answerable by the system&mdash;might be added in the future to
confirm the question is truly asking about causation.

Some additional future work on this module might include:

* Identify relations that should hold in the answer
* Incorporate a user model
* Adjust UI if user is casual or expert
* Support an on-going dialog
* Question-type and answer-type detection
* Question validation

### Candidate Document Selection

We use the modified query to get a ranked list of documents based on a
similarity measure&mdash;cosine similarity of TF&ndash;IDF
values&mdash;compared to a query term list by using the indices and
functions of the IR component.

Because the *Candidate Document Analysis* takes a large amount of time
per document, we currently only examine the top-5 documents from the
ranked list to make execution time a little bit faster.

Instead of retrieving whole documents, potential future work could
include retrieving ranked lists of paragraphs that could be from any
document instead. In order to accomplish this, the original query term
would need to be expanded&mdash;most likely through computing the
semantic relatedness to every synset in WordNet&mdash;otherwise
paragraphs that do not contain the exact query term will not be
examined.

### Candidate Document Analysis (Information Extraction)

Since we retrieve whole documents, as most web search engines do, we
need to analyze each document to aid in answer extraction. Many of the
steps that could be done here, such as paragraph / sentence segmentation
or word tokenization, has been done during the *Document Preprocessing*
step. For each sentence we run part-of-speech (POS) tagging on the
tokens so that the list of WordNet synsets that we examine is shorter.

Future work in this module could include phrase chunking, named entity
(NE) tagging, and relation detection *(cause -> effect)* either through
a method, or through semantic role labeling (SRL) by using PropBank.

### Answer Extraction

We now use all of the available tagged information to create a ranked
list of sentences that the system believes to contain the answer.

As our system is meant to handle causal questions, once we have
retrieved the list of documents from the IR system we add an additional
special causation term to the query to be used during this phase. The
special causation term is the tuple formed by the token-string "cause",
and the WordNet synset "cause.v.01".

Our system currently returns answers in the form of single sentences. We
extract all sentences that contain a match&mdash;either exact or by
having a high enough semantic relatedness&mdash;for at least one query
term. We then order the sentences by computing a score using various
weighted scoring components.

* Define *page cosine similarity* as the cosine similarity of the page
the answer appears in, relative to the query.
* Define *word score* as the number of words from the question that are recognized in the sentence.
* Define *related score sum* as the sum of the maximum Leacock-Chodorow Similarity (LCH) scores for each query term when compared to each term
in the sentence.
* Define *related score average* as the related sum divided by the
number if query terms (normalization).
* In addition, we use the LCH score to determine if a sentence term
matches a query term by observing if the LCH score is > 2.16.
* We approximated an acceptable value for the LCH threshold empirically,
though a machine learning (ML) approach would be interesting to examine.
* Define *causal match* as a boolean value indicating if the special
causation term has a match in the answer or not.
* Define *position score* as the number of words that separate each
keyword from the next in the sentence.

We tried adding the *word score* and *related score* components equally
as our only features to score the answer, but later used machine
learning with the other features to compute better weights using
question-answer pairs&mdash;sometimes called learning to rerank.
Additional types of features that could be included, and would likely
greatly benefit the answer ranking.

Future work on this module includes:

* Compute and rank paragraph and answer windows, in order to return
answers of different length including smaller than a sentence and up to
an entire paragraph
* Use the relation detection from the information extraction module to
find *causes* that match what the user's question asked.
* Combine multiple answers together through summarization to output a
single answer (or a better single answer at rank-1).

### Response Generation

In addition to the ranked list of extracted answers, it would be
worthwhile to present the user with some additional information,
including:

* Original input query
* Modified query used for IR
* Link back to the source document

Some future work on this component could include:

* Rephrasing the answer to make more comprehensible
* More or less evidence (context)
* Gauging certainty about an answer so that the system could respond
that no answer was found in its knowledge base.

<!-- ### Future Work

In addition to the possibilities for improvement already mentioned,
there are others that could be included in one of the previous
components or could be made into its own module. -->
