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

1. Remove the MediaWiki markup, leaving just the plain text.
1. Convert all characters from Unicode to closest ASCII equivalent.

Currently these steps are done for every document when it is read from
disk. Future work on this module could include doing this work
once&mdash;saving the output to be used as a new corpus to be used for
all subsequent steps&mdash;thereby speeding up execution at run-time. If
this option were to be taken, it might also make sense to merge the
steps of the *Candidate Document Analysis* component into this one so
that they also don't have to be performed at run-time. It should be
noted, however, that certain steps such as POS tagging require
significant computational time, and can cause the time it takes to
initially generate the index by some order of magnitude.

### Document Indexing

1. Sentence segmentation
1. Word tokenization
1. Regularize tokens
    * Case folding (convert to lowercase)
    * Other
1. Count tokens
1. Create mapping of Page.ID to byte location in corpus file.
1. Create mapping of Token to set of Page.IDs that contain that token.
1. Keep track of token frequency statistics for each document and across
the entire collection.

### Information Retrieval System

The IR system is the collection of index files with functions for
getting a set of documents matching some criteria. The available
functions include:

* Retrieve a specific document from the corpus (in O(1) time).
* Get the set intersection of documents that contain all terms in a term
list.
* Get the union set of documents that contain any term in a term list.
* Get a ranked list of documents that are similar to a term list vector.

### Question Analysis

The input from the user needs to be analyzed so that an answer may be
returned. These steps include:

1. Keyword extraction
  * Named entities
  * Complex nominals and adjective modifiers
  * Nouns
  * Verbs
1. Query generation

In an open-domain QA system it is necessary to determine what type of
question is being asked, and what the expected answer type might be.
Our system has been explicitly designed for **causal** questions, and
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

Get a ranked list of documents based on a similarity measure
(TF&ndash;IDF) compared to a query term list by using the indices and
functions of the IR component.

If the next step of *Candidate Document Analysis* takes a large amount
of time per document, it may be worth evaluating how far down the
initial ranked list of documents to go before returning an answer.

Potential future work on this section could include switching from
retrieving whole documents to directly retrieving ranked lists of
paragraphs that could be from any document.

### Candidate Document Analysis (Information Extraction)

Since we retrieve whole documents, as most web search engines do, we
need to analyze each document to aid in answer extraction. The steps
involved include:

1. Paragraph segmentation
1. Sentence segmentation
1. Word tokenization
1. Part of speech (POS) tagging
1. Chunking
    * Phrase chunking
    * Named entity (NE) tagging
1. Relation detection *(cause -> effect)*

### Answer Extraction

We now use all of the available tagged information to create a ranked
list of paragraphs that the system believes to contain the answer.

1. Filter paragraphs either by a boolean method leaving only those that
contain the query keywords, or by some similarity measure to the query.
1. If a query term appears more than once in a single paragraph, in
order to consider each occurrence of a keyword separately we need to
compute paragraph windows.
1. Identify potential answers in each paragraph window by computing
answer windows.
1. Order the paragraph windows using the following criteria:
  * Define *same word score* as the number of words from the question
  that are recognized in the same sequence in the paragraph window.
  * Define *distance score* as the number of words that separate the
  most distant keywords in the window.
  * Define *missing keywords score* as the number of unmatched keywords,
  which is the same for all windows from the same paragraph.
  * Perform a radix sort of the answer windows by:
    1. largest *same word sequence score*
    1. largest *distance score*
    1. smallest *missing keyword score*

Future work on this module includes:

* Compute and rank answer windows, similarly to how it was done for
paragraph windows, in order to give more precise answers.
* Use the relation detection from the information extraction module to
find *causes* that match what the user's question asked.
* Combine multiple answers together through summarization to output a
single answer (or a better single answer at rank-1).

### Response Generation

In addition to the ranked list of extracted answers, it would be
worthwhile to present the user with some additional information,
including:

* Link back to the source document
* Highlight

Some future work on this component could include:

* Rephrasing the answer to make more comprehensible
* More or less evidence (context)
* Gauging certainty about an answer so that the system could respond
that no answer was found in its knowledge base.

<!-- ### Future Work

In addition to the possibilities for improvement already mentioned,
there are others that could be included in one of the previous
components or could be made into its own module. -->
