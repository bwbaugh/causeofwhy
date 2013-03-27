Cause of Why
============

The goal of this project is to implement a Question Answering (QA)
system that answers causal type questions. We use Wikipedia as a
knowledge base, extracting answers to user questions from the articles.

Currently we are focused on getting the system's engine working, so the
user interface is on the back burner right now. Please stay tuned for
lots of updates!

Causal Questions
----------------

Causal questions are generally why-questions. They ask for a reason or a
cause, such as "Why do birds sing?". This differs from other QA systems,
which usually try to answer factoid questions, such as "Where is the
Louvre located?".

Required Libraries
------------------

This project uses several libraries that either need to be installed or
need to be present in the project's `lib/` directory. The following is a
list of the required libraries, as well as at least one way (source) to
obtain the library.

### nltk

Natural Language Processing (NLP) functions such as sentence
segmentation, word tokenization, and more.

* <http://nltk.org/install.html>

#### nltk resources

In addition, you will need to download several nltk resources using
nltk.download() after you have the nltk library installed.

* 'taggers/maxent_treebank_pos_tagger/english.pickle'

### gensim

Some useful Information Retrieval (IR) algorithms including string to
vector functions and similarity queries such as TF-IDF. Also implements
topic modelling such as Latent Semantic Analysis.

* <http://radimrehurek.com/gensim/install.html>

### unidecode

Converts unicode strings to closest ASCII equivalent.

* <http://pypi.python.org/pypi/Unidecode>
* git clone http://www.tablix.org/~avian/git/unidecode.git

### Tornado

Provides a web server interface.

* <http://www.tornadoweb.org/>
* <https://github.com/facebook/tornado>

### WikiExtractor.py

Converts text from MediaWiki markup format to plain text.

* <http://medialab.di.unipi.it/wiki/Wikipedia_extractor>

Optional Libraries
------------------

### PyMongo

Tools for interacting with MongoDB databases. This is useful for working
with indices that can't be held entirely in memory, which is not a
problem for a smaller corpus like the *Simple English Wikipedia* but is
an issue for larger corpora like the full *English Wikipedia*.

* <http://api.mongodb.org/python/current/installation.html>
* <https://github.com/mongodb/mongo-python-driver/>

#### MongoDB

Since the PyMongo library is just an interface, we need an instance of
the actual database itself running.

* Pick the version for your platform.
* If using Windows 7 or higher get the Windows 2008+ build.
* Tested with version: 2.2.1

Start the database process before running the application.

* <http://www.mongodb.org/downloads>
