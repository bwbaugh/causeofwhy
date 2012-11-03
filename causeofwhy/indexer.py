# Copyright (C) 2012 Brian Wesley Baugh
"""Provides information retrieval (IR) functions and classes."""
# Python packages
from __future__ import division
from __future__ import with_statement
import os
import multiprocessing
import Queue
import logging
import collections
import operator
import codecs
import string
import math
try:
    import cPickle as pickle
except ImportError:
    import pickle


# External packages and modules
import gensim
try:
    import pymongo
except ImportError:
    pymongo = None
    print """\
          WARNING: pymongo package could not be found.
              If you are using a SMALL corpus---such as the *Simple English*
              version of Wikipedia---and the inverted token index can fit
              entirely in memory, then comment out this `raise` line.
              Otherwise, please install the PyMongo library!
          """
    raise
import nltk
from nltk.corpus import stopwords


# Project modules
# *** IMPORTED AT THE END OF THIS FILE ***


# MODULE CONFIGURATION

# How many page worker threads to use
NUMBER_OF_PROCESSES = max(1, multiprocessing.cpu_count() - 1)

# How many pages to send to each worker at a time.
CHUNK_SIZE = 32

# Arbitrary amount
MAX_QUEUE_ITEMS = NUMBER_OF_PROCESSES

# Logging display level
logging.basicConfig(level=logging.DEBUG)

# NLP (NLTK) settings
tokenizer = nltk.tokenize.TreebankWordTokenizer()
#stemmer = nltk.PorterStemmer()
stemmer = None
lemmatizer = nltk.WordNetLemmatizer()
PUNKT_FNAME = "wiki_punkt.pickle"
try:
    with open(PUNKT_FNAME, mode='rb') as f:
        sent_detector = pickle.load(f)
except (IOError, pickle.UnpicklingError):
    sent_detector = None
STOPWORDS = [lemmatizer.lemmatize(t) for t in stopwords.words('english')]


# CONSTANTS AND GLOBAL VARS
LINE_SEPARATOR = u'\u2028'
PARAGRAPH_SEPARATOR = u'\u2029'

# Bad page checks
page_length_limit = 1024


# EXCEPTIONS

class IndexLoadError(Exception):
    pass


# CLASSES

class Index(object):
    """The main information retrieval (IR) class.

    This class uses the previously created index files to create an
    Index object. This object represents the information retrieval (IR)
    portion of the system.

    Individual page objects can be retrieved from disk, or a set of
    pages that match a term list (either by union or intersection).

    Attributes:
        base_fname: The string filename (including path) of the corpus.
        mongo_conn: If present, is the Connection object for the MongoDB
            driver (interface).
        mongo_db: If present, is the MongoDB database, named after the
            corpus filename.
        dict: A Gensim Dictionary object for converting a string(token)
            to an int(token_id).
        pagi: The page index {page.ID -> page.start, doci.offset}. Used
            to store the location of an individual page in the corpus.
        doci: The document index {page.ID -> page.token_count} object.
        tokc: The token-count index {token -> count} where count is the
            number of times the term appears in the entire corpus.
        toki: The token-document index: {toki[token] -> set(documents)},
            where set(deocuments) is the set of document IDs where the
            token exists in the document.
    """

    def __init__(self, base_fname, doci_in_memory=False):
        """Loads all indices for the base_fname Wikipedia dump."""
        self.base_fname = base_fname
        check_plain_corpus(base_fname)
        if pymongo:
            self.load_mongo()
        self.load_dict()
        self.load_pagi()
        if doci_in_memory:
            self.load_doci()
        else:
            self.doci = DocI(self)
        self.load_tokc()
        self.load_toki()

    def load_mongo(self):
        """Connect to the MongoDB server and select the database."""
        self.mongo_conn = pymongo.Connection('localhost', 27017)
        self.mongo_db = self.mongo_conn[mongo_db_name(self.base_fname)]

    def load_dict(self):
        """Load the (gensim) Dictionary representing the vocabulary.

        The Dictionary object is mainly a {token -> token_id} mapping,
        but also contains {token_id -> document_frequency} information.
        """
        try:
            self.dict = (gensim.corpora.dictionary.Dictionary().
                         load_from_text(self.base_fname + '.dict'))
        except IOError:
            raise IndexLoadError

    def load_pagi(self):
        """Load the page index {page.ID -> page.start, doci.offset}"""
        self.pagi = dict()
        try:
            with codecs.open(self.base_fname + '.pagi', encoding='utf-8') as f:
                for line in f:
                    ID, start, offset = line.split('\t')
                    self.pagi[int(ID)] = (int(start), int(offset))
            if not self.pagi:
                raise IndexLoadError
        except IOError:
            raise IndexLoadError

    def load_doci(self):
        """Load the document index {page.ID -> page.token_count} in memory"""
        self.doci = collections.defaultdict(dict)
        try:
            with codecs.open(self.base_fname + '.doci', encoding='utf-8') as f:
                for line in f:
                    ID, token_counts = line.split('\t', 1)
                    for token_count in token_counts.split('\t'):
                        token, count = token_count.split(chr(26))
                        self.doci[int(ID)][int(token)] = int(count)
            if not self.doci:
                raise IndexLoadError
        except IOError:
            raise IndexLoadError

    def load_tokc(self):
        """Load the token-count index {tokc[token] -> count}"""
        try:
            with open(self.base_fname + '.tokc', mode='rb') as f:
                self.tokc = pickle.load(f)
            if not self.tokc:
                raise IndexLoadError
        except (IOError, pickle.UnpicklingError):
            raise IndexLoadError

    def load_toki(self):
        """Load the token-document index: {toki[token] -> set(documents)}"""
        self.toki = TokI(self)

    def get_page(self, ID):
        """Returns the corresponding Page object residing on disk."""

        def find_page(start):
            wiki_dump.seek(start)
            pages = plain_page_generator(wiki_dump)
            return next(pages)

        with open(self.base_fname + '.txt', mode='rb') as wiki_dump:
            try:
                iterator = iter(ID)
            except TypeError:
                start, offset = self.pagi[ID]
                return find_page(start)
            else:
                pages = []
                for page in ID:
                    start, offset = self.pagi[page]
                    pages.append(find_page(start))
                return pages

    def union(self, terms):
        """Returns set of Page.IDs that contain any term in the term list."""
        pages = set()
        try:
            terms = [self.dict.token2id[term] for term in terms]
        except KeyError:
            pass
        for term in terms:
            if term in self.toki:
                ID = self.toki[term]
                pages.update(ID)
        return pages

    def intersect(self, terms):
        """Returns set of Page.IDs that contain all terms in the term list."""
        try:
            terms = [self.dict.token2id[term] for term in terms]
        except KeyError:
            terms = list(terms)
        pages = set(self.toki[terms.pop()])
        for term in terms:
            if term in self.toki:
                ID = self.toki[term]
                pages.intersection_update(ID)
        return pages

    def ranked(self, terms):
        """Returns a ranked list of tuples of Page.IDs and similarity value."""
        try:
            terms = [self.dict.token2id[term] for term in terms]
        except KeyError:
            pass
        q_tfidf = self.query_tfidf(terms)
        pages = self.union(terms)
        ranked_pages = dict()
        for ID in pages:
            # Calculate document TF-IDF
            d_tfidf = dict()
            token_counts = self.doci[ID]
            max_count = max(token_counts.itervalues())
            for term in token_counts:
                # TF: Raw frequency divided by the maximum raw frequency
                # of any term in the document.
                tf = token_counts[term] / max_count
                # IDF: Total number of documents in the corpus divided by
                # the number of documents where the term appears.
                idf = math.log(len(self.doci) / self.dict.dfs[term])
                d_tfidf[term] = tf * idf
            # Calculate inner product
            inner_product = 0
            for term in terms:
                if term in token_counts:
                    inner_product += q_tfidf[term] * d_tfidf[term]
            # Calculate query length
            query_length = 0
            for term in q_tfidf:
                query_length += q_tfidf[term] ** 2
            query_length = math.sqrt(query_length)
            # Calculate document length
            doc_length = 0
            for term in d_tfidf:
                doc_length += d_tfidf[term] ** 2
            doc_length = math.sqrt(doc_length)
            # Calculate the cosine similarity
            cosine_sim = inner_product / (query_length * doc_length)
            ranked_pages[ID] = cosine_sim
        ranked_pages = sorted(ranked_pages.iteritems(),
                              key=operator.itemgetter(1), reverse=True)
        return ranked_pages

    def query_tfidf(self, terms):
        """Returns the {term: TF-IDF} dict for a query vector."""
        token_count = collections.defaultdict(int)
        try:
            terms = [self.dict.token2id[term] for term in terms]
        except KeyError:
            pass
        for term in terms:
            token_count[term] += 1
        max_count = max(token_count.itervalues())
        return {term: token_count[term] / max_count for term in token_count}


class DocI(object):
    """Wrapper class around the .doci index file; allows for doci[ID].

    This class allows the .doci {page.ID -> page.token_count} index to
    stay on disk without having to load the entire file into memory.
    There is a memory-speed tradeoff, but the speed lost with respect to
    the amount of memory saved for larger corpora will be noticeable.

    IMPORTANT: Save the resulting dictionary of a doci[Page.ID] call locally!
    Because the dictionary has to be created for every __getitem__()
    call, the operation becomes exponentially expensive!
    """
    def __init__(self, index):
        """Initialize the DocI object from disk."""
        self.index = index
        # Make sure the file can open and at least the first line is parsable.
        try:
            with codecs.open(self.index.base_fname + '.doci',
                             encoding='utf-8') as f:
                line = f.readline()
                ID, token_counts = line.split('\t', 1)
                for token_count in token_counts.split('\t'):
                    token, count = token_count.split(chr(26))
                    ID, token, count = int(ID), int(token), int(count)
        except IOError:
            raise IndexLoadError

    def __getitem__(self, ID):
        """Retrieve the dictionary result of: {page.ID -> page.token_count}"""
        counts = dict()
        with codecs.open(self.index.base_fname + '.doci',
                         encoding='utf-8') as f:
                start, offset = self.index.pagi[ID]
                f.seek(offset)
                line = f.readline()
        ID, token_counts = line.split('\t', 1)
        for token_count in token_counts.split('\t'):
            token, count = token_count.split(chr(26))
            counts[int(token)] = int(count)
        return counts

    def __len__(self):
        """Returns the equivalent length of self.index.pagi"""
        return len(self.index.pagi)


class TokI(object):
    """Wrapper class around the .toki index file; allows for toki[token].

    This class allows access to the toki {token -> set(page.IDs)} whether
    the underlying index is from a MongoDB or a defaultdict loaded entirely
    into memory.
    """

    def __init__(self, index):
        """Initialize the TokI object from a MongoDB or load from disk."""
        self.index = index
        if pymongo:
            if 'toki' in self.index.mongo_db.collection_names():
                self.mongo_toki = self.index.mongo_db['toki']
                if self.mongo_toki.count() == 0:
                    raise IndexLoadError
            else:
                raise IndexLoadError
        else:
            # Load into memory (not suitable for large corpora!)
            try:
                with open(self.index.base_fname + '.toki', mode='rb') as f:
                    self.toki = pickle.load(f)
                if not self.toki:
                    raise IndexLoadError
            except (IOError, pickle.UnpicklingError):
                raise IndexLoadError

    def __getitem__(self, token):
        """Retrieve a token's set of page IDs: {token -> set(page.IDs)}"""
        if pymongo:
            result = self.mongo_toki.find_one({'_id': token})
            try:
                return result['ID']
            except TypeError:
                print 'ERROR: bad token = {}'.format(token)
                raise
        else:
            return self.toki[token]

    def __contains__(self, key):
        """Checks if key exists in the index."""
        if pymongo:
            return self.mongo_toki.find_one({'_id': key}) is not None
        else:
            return key in self.toki


# FUNCTIONS

def check_plain_corpus(base_fname):
    """Attempts to make sure the plain-text corpus is available."""
    try:
        with open(base_fname + '.txt') as wiki_dump:
            pages = plain_page_generator(wiki_dump)
            if not next(pages):
                raise IndexLoadError
    except IOError:
        raise IndexLoadError


def mongo_db_name(base_fname):
    """Use the corpus filename to create the database name."""
    fname = base_fname.replace('\\', '/').rsplit('/', 1)[1]
    fname = fname.replace('.', '_')
    return fname


def regularize(tokens):
    """Returns a copy of a regularized version of the token list."""
    tokens = list(tokens)
    for i, token in enumerate(tokens):
        # Normalize text by case folding
        token = token.lower()
        # Lemmatize (birds -> bird)
        token = lemmatizer.lemmatize(token)
        # Stopword and punctuation removal
        if token in STOPWORDS or token in string.punctuation:
            token = None
        # Done; update value in list
        tokens[i] = token
    # Remove empty tokens
    tokens = [x for x in tokens if x is not None]
    return tokens


def first_pass_worker(taskq, doneq):
    """Processes pages to make a plain-text corpus from the original dump."""
    logger = logging.getLogger('worker')
    done_buff = []
    try:
        while True:
            chunk = taskq.get()
            if chunk is None:
                return
            for page in chunk:
                page.preprocess()
                if len(page.text) < page_length_limit:
                    continue
                # Need to get tokens so we can build our Dictionary
                page.regularize_text()
                done_buff.append(page)
            doneq.put(done_buff)
            done_buff = []
    finally:
        doneq.put(None)


def second_pass_worker(taskq, doneq):
    """Counts tokens from the plain-text corpus to create an index."""
    logger = logging.getLogger('worker')
    done_buff = []
    try:
        while True:
            chunk = taskq.get()
            if chunk is None:
                return
            for page in chunk:
                page.regularize_text()
                page.count_tokens()
                done_buff.append(page)
            doneq.put(done_buff)
            done_buff = []
    finally:
        doneq.put(None)


def first_pass_writer(doneq, wiki_location):
    """Extracts the Dictionary (vocabulary) and writes plain-text corpus."""
    pill_count = 0  # termination condition (poison pill)
    dictionary = gensim.corpora.dictionary.Dictionary()
    try:
        with codecs.open(wiki_location + '.txt',
                         mode='w',
                         encoding='utf-8') as txt:
            # Begin processing chunks as they come in.
            while True:
                chunk = doneq.get()
                if chunk is None:
                    pill_count += 1
                    if pill_count == NUMBER_OF_PROCESSES:
                        return
                    else:
                        continue
                for page in chunk:
                    # Send all tokens from document to Dictionary
                    all_tokens = []
                    para_sent = []
                    for para in page.paragraphs:
                        for sentence in para.sentence_tokens:
                            all_tokens.extend(sentence)
                        sent = LINE_SEPARATOR.join(para.sentences)
                        para_sent.append(sent)
                    para_sent = PARAGRAPH_SEPARATOR.join(para_sent)
                    dictionary.doc2bow(all_tokens, allow_update=True)
                    # page.text = unicode(page.text)
                    txt.write('\t'.join([str(page.ID), page.title, para_sent])
                              + '\n')
    finally:
        # Save token indices
        dictionary.filter_extremes(no_below=20, no_above=0.1)
        dictionary.save_as_text(wiki_location + '.dict')


def second_pass_writer(doneq, wiki_location):
    """Writes various index files for fast searching and retrieval of pages."""
    pill_count = 0  # termination condition (poison pill)
    if pymongo:
        mongo_conn = pymongo.Connection('localhost', 27017)
        mongo_db = mongo_conn[mongo_db_name(wiki_location)]
        mongo_toki = mongo_db['toki']
        # Delete any existing data
        mongo_toki.drop()
    else:
        token_docs = collections.defaultdict(set)
    token_counts = collections.defaultdict(int)
    dictionary = (gensim.corpora.dictionary.Dictionary().
                  load_from_text(wiki_location + '.dict'))
    try:
        with codecs.open(wiki_location + '.pagi',
                         mode='w',
                         encoding='utf-8') as pagi,\
             codecs.open(wiki_location + '.doci',
                         mode='w',
                         encoding='utf-8') as doci:
            # Begin processing chunks as they come in.
            while True:
                chunk = doneq.get()
                if chunk is None:
                    pill_count += 1
                    if pill_count == NUMBER_OF_PROCESSES:
                        return
                    else:
                        continue
                for page in chunk:
                    # Convert token from a string to an integer ID, and
                    # remove tokens that don't appear in our Dictionary.
                    page.token_count = [(dictionary.token2id[t], c) for t, c in
                                        page.token_count if t in
                                        dictionary.token2id]
                    pagi.write('\t'.join([str(page.ID).ljust(1),
                                          str(page.start).ljust(1),
                                          str(doci.tell()).ljust(1)]) + '\n')
                    doci.write('\t'.join([str(page.ID)] +
                                         [chr(26).join([str(k), str(v)]) for
                                          k, v in page.token_count]) +
                                         '\n')
                    if pymongo:
                        for token, count in page.token_count:
                            mongo_toki.update({'_id': token},
                                              {'$addToSet': {'ID': page.ID}},
                                              upsert=True)
                            token_counts[token] += int(count)
                    else:
                        for token, count in page.token_count:
                            token_docs[token].add(page.ID)
                            token_counts[token] += int(count)
                for f in (pagi, doci):
                    f.flush()
    finally:
        # Save token indices
        with open(wiki_location + '.tokc', mode='wb') as tokc:
            pickle.dump(token_counts, tokc, protocol=pickle.HIGHEST_PROTOCOL)
        if pymongo:
            mongo_conn.disconnect()
        else:
            with open(wiki_location + '.toki', mode='wb') as toki:
                pickle.dump(token_docs, toki, protocol=pickle.HIGHEST_PROTOCOL)


def create_punkt_sent_detector(fname, progress_count, max_pages=25000):
    """Makes a pass through the corpus to train a Punkt sentence segmenter."""
    logger = logging.getLogger('create_punkt_sent_detector')

    punkt = nltk.tokenize.punkt.PunktTrainer()

    logger.info("Training punkt sentence detector")

    wiki_size = os.path.getsize(fname)
    page_count = 0

    try:
        with open(fname, mode='rb') as wiki_dump:
            pages = page_generator(wiki_dump)
            for page in pages:
                page.preprocess()
                punkt.train(page.text, finalize=False, verbose=False)
                page_count += 1
                if page_count == max_pages:
                    break
                if page_count % progress_count == 0:
                    print(page_count, page.start,
                          (page.start / wiki_size * 100),
                          # taskq.qsize() if taskq is not None else 'n/a',
                          # doneq.qsize() if doneq is not None else 'n/a',
                          page.ID, page.title)
    except KeyboardInterrupt:
        print 'KeyboardInterrupt: Stopping the reading of the dump early!'

    logger.info('Now finalzing Punkt training.')

    punkt.finalize_training(verbose=True)
    learned = punkt.get_params()
    sbd = nltk.tokenize.punkt.PunktSentenceTokenizer(learned)
    with open(PUNKT_FNAME, mode='wb') as f:
        pickle.dump(sbd, f, protocol=pickle.HIGHEST_PROTOCOL)


def first_pass(fname, progress_count=None, max_pages=None):
    """Extract a Dictionary and create plain-text version of corpus."""
    logger = logging.getLogger('first_pass')

    wiki_size = os.path.getsize(fname)

    # Page task queues for parallel processing
    taskq = multiprocessing.Queue(MAX_QUEUE_ITEMS)
    doneq = multiprocessing.Queue(MAX_QUEUE_ITEMS)

    # Start worker processes
    logger.info('Starting workers')
    workers = []
    for i in range(NUMBER_OF_PROCESSES):
        p = multiprocessing.Process(target=first_pass_worker,
                                    args=(taskq, doneq))
        p.start()
        workers.append(p)

    # Start log writer process
    p = multiprocessing.Process(target=first_pass_writer, args=(doneq, fname))
    p.start()
    workers.append(p)

    # Process XML dump
    logger.info('Begining XML parse')

    wiki_size = os.path.getsize(fname)
    page_count = 0

    task_buff = []
    try:
        with open(fname, mode='rb') as wiki_dump:
            pages = page_generator(wiki_dump)
            for page in pages:
                task_buff.append(page)
                if len(task_buff) == CHUNK_SIZE:
                    taskq.put(task_buff)
                    task_buff = []
                page_count += 1
                if page_count == max_pages:
                    break
                if page_count % progress_count == 0:
                    print(page_count, page.start,
                          (page.start / wiki_size * 100),
                          taskq.qsize(), doneq.qsize(),
                          page.ID, page.title)
    except KeyboardInterrupt:
        print 'KeyboardInterrupt: Stopping the reading of the dump early!'
    finally:
        # Flush task buffer
        taskq.put(task_buff)
        task_buff = []
        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            taskq.put(None)

    logger.info('All done! Processed %s total pages.', page_count)

    # Wait for all child processes to stop (especially that writer!)
    for p in workers:
        p.join()


# Main function of module
def create_index(fname, progress_count=None, max_pages=None):
    """Processes a corpus to create a corresponding Index object."""
    logger = logging.getLogger('create_index')

    if sent_detector is None:
        create_punkt_sent_detector(fname=fname,
                                   progress_count=CHUNK_SIZE,
                                   max_pages=min(25000, max_pages))

    # Set params
    if progress_count is None:
        progress_count = CHUNK_SIZE * NUMBER_OF_PROCESSES

    # First pass, create Dictionary and plain-text version of corpus.
    try:
        dictionary = (gensim.corpora.dictionary.Dictionary().
                      load_from_text(fname + '.dict'))
        if not dictionary or check_plain_corpus(fname):
            raise IndexLoadError
    except (IOError, IndexLoadError):
        first_pass(fname, progress_count, max_pages)
    else:
        del dictionary

    # Page task queues for parallel processing
    taskq = multiprocessing.Queue(MAX_QUEUE_ITEMS)
    doneq = multiprocessing.Queue(MAX_QUEUE_ITEMS)

    # Start worker processes
    logger.info('Starting workers')
    workers = []
    for i in range(NUMBER_OF_PROCESSES):
        p = multiprocessing.Process(target=second_pass_worker,
                                    args=(taskq, doneq))
        p.start()
        workers.append(p)

    # Start log writer process
    p = multiprocessing.Process(target=second_pass_writer, args=(doneq, fname))
    p.start()
    workers.append(p)

    # We are now working with the plain-text corpus generated in the 1st pass.
    fname += '.txt'

    wiki_size = os.path.getsize(fname)

    # Process XML dump
    logger.info('Begining plain-text parse')

    wiki_size = os.path.getsize(fname)
    page_count = 0

    task_buff = []
    try:
        with open(fname, mode='rb') as wiki_dump:
            pages = plain_page_generator(wiki_dump)
            for page in pages:
                task_buff.append(page)
                if len(task_buff) == CHUNK_SIZE:
                    taskq.put(task_buff)
                    task_buff = []
                page_count += 1
                if page_count == max_pages:
                    break
                if page_count % progress_count == 0:
                    print(page_count, page.start,
                          (page.start / wiki_size * 100),
                          taskq.qsize(), doneq.qsize(),
                          page.ID, page.title)
    except KeyboardInterrupt:
        print 'KeyboardInterrupt: Stopping the reading of the dump early!'
    finally:
        # Flush task buffer
        taskq.put(task_buff)
        task_buff = []
        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            taskq.put(None)

    logger.info('All done! Processed %s total pages.', page_count)

    # Wait for all child processes to stop (especially that writer!)
    for p in workers:
        p.join()


# PROJECT MODULE IMPORTS

from wiki_dump_reader import page_generator, plain_page_generator
