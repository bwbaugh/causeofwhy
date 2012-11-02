# Copyright (C) 2012 Brian Wesley Baugh
"""Provides information retrieval (IR) functions and classes."""
# Python packages
from __future__ import division
from __future__ import with_statement
from pprint import pprint
import os
import sys
import multiprocessing
import Queue
import logging
import collections
import operator
import codecs
import re
import string
import math
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


# External packages and modules
import WikiExtractor
import gensim
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode


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
title_start_with_terms = ('User: Wikipedia: File: MediaWiki: Template: '
                          'Help: Category: Portal: Book: 28644448 Help:'
                          .upper().split(' '))
title_end_with_terms = '(disambiguation)'.upper().split(' ')
text_start_with_terms = '#REDIRECT {{softredirect'.upper().split(' ')
text_last_terms = '{{Disamb {{Dab stub}}'.upper().split(' ')


# EXCEPTIONS

class IndexLoadError(Exception):
    pass


# CLASSES

class Paragraph(object):
    """Container that holds sentences and their tokens."""

    def __init__(self, text):
        """Initialize the Paragraph object."""
        self.text = text
        self.sentences = None
        self.sentence_tokens = None

    def segment_sentences(self):
        """Segment the Paragraph text into a list of sentences."""
        # Sentence segmentation
        if LINE_SEPARATOR in self.text:
            self.sentences = [sent for sent in self.text.split(LINE_SEPARATOR)]
        else:
            self.sentences = sent_detector.tokenize(self.text,
                                                    realign_boundaries=True)

    def tokenize_sentences(self):
        """Tokenize each sentence in the list into a list of tokens."""
        if not self.sentences:
            self.segment_sentences()
        self.sentence_tokens = tokenizer.batch_tokenize(self.sentences)


class Page:
    """Holds all text and metadata (ID, title) of a page from the corpus."""

    def __init__(self, ID, title, text, start=None):
        """Initialize the Page object."""
        self.ID = ID
        self.title = title
        self.text = text
        self.start = start
        self.paragraphs = None
        self.token_count = None
        self.cosine_sim = None

    def remove_markup(self):
        """Remove wiki markup leaving just the plain-text."""
        self.text = WikiExtractor.clean(self.text)
        self.text = '\n'.join(WikiExtractor.compact(self.text))

    def unidecode(self):
        """Convert non-ascii to closest ASCII equivalent."""
        self.title = unidecode(self.title).strip()
        self.text = unidecode(self.text).strip()

    def preprocess(self):
        """Convenience method that removed markup does unidecode."""
        self.remove_markup()
        self.unidecode()

    def segment_paragraphs(self):
        """Segment the Page text into a list of paragraphs."""
        if PARAGRAPH_SEPARATOR in self.text:
            split = PARAGRAPH_SEPARATOR
        else:
            split = '\n'
        self.paragraphs = [Paragraph(text) for text in self.text.split(split)]

    def segment_sentences(self):
        """Segment each Paragraph into a list of sentences."""
        if not self.paragraphs:
            self.segment_paragraphs()
        for paragraph in self.paragraphs:
            paragraph.segment_sentences()

    def tokenize_sentences(self):
        """Tokenize the sentence list in the paragraphs into list of tokens."""
        if not self.paragraphs:
            self.segment_sentences()
        for paragraph in self.paragraphs:
            paragraph.tokenize_sentences()

    def regularize_text(self):
        """Regularizes all tokens for each sentence in each paragraph."""
        if not self.paragraphs:
            self.tokenize_sentences()
        for i, para in enumerate(self.paragraphs):
            for j, sent in enumerate(para.sentence_tokens):
                self.paragraphs[i].sentence_tokens[j] = regularize(sent)
            # Remove empty sentences
            self.paragraphs[i].sentence_tokens = [x for x in self.
                                                  paragraphs[i].sentence_tokens
                                                  if x]

    def count_tokens(self):
        """Count the frequency of text's tokens in a bag-of-words style."""
        self.token_count = collections.defaultdict(int)
        for paragraph in self.paragraphs:
            for sentence in paragraph.sentence_tokens:
                for token in sentence:
                    self.token_count[str(token)] += 1
        self.token_count = [(token, count) for (token, count) in\
                            sorted(self.token_count.iteritems(),
                                   key=operator.itemgetter(1),
                                   reverse=True)]

    def __str__(self):
        self.remove_markup()
        self.unidecode()
        f = StringIO()
        f.write('=' * 79 + '\n')
        f.write(str(self.ID) + ' ' + self.title + '\n')
        f.write('-' * 79 + '\n')
        f.write(self.text.encode('utf-8') + '\n')
        f.write('=' * 79 + '\n')
        output = f.getvalue()
        f.close()
        return output

    # def __eq__(self, other):
    #     return self.ID == other.ID

    # def __ne__(self, other):
    #     return not self.__eq__(other)

    # def __hash__(self):
    #     return hash((self.ID,))


class Index:
    """The main information retrieval (IR) class.

    This class uses the previously created index files to create an
    Index object. This object represents the information retrieval (IR)
    portion of the system.

    Individual page objects can be retrieved from disk, or a set of
    pages that match a term list (either by union or intersection).
    """
    def __init__(self, base_fname, doci_in_memory=False):
        """Loads all indices for the base_fname Wikipedia dump."""
        self.base_fname = base_fname
        check_plain_corpus(base_fname)
        self.load_dict()
        self.load_pagi()
        if doci_in_memory:
            self.load_doci()
        else:
            self.doci = DocI(self)
        self.load_tokc()
        self.load_toki()

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
        """Load the token counts {tokc[token] -> count}"""
        try:
            with open(self.base_fname + '.tokc', mode='rb') as f:
                self.tokc = pickle.load(f)
            if not self.tokc:
                raise IndexLoadError
        except (IOError, pickle.UnpicklingError):
            raise IndexLoadError

    def load_toki(self):
        """Load the token document index: {toki[token] -> set(documents)}"""
        try:
            with open(self.base_fname + '.toki', mode='rb') as f:
                self.toki = pickle.load(f)
            if not self.toki:
                raise IndexLoadError
        except (IOError, pickle.UnpicklingError):
            raise IndexLoadError

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
                tf = token_counts[term] / max_count
                idf = math.log(len(self.doci) / len(self.toki[term]))
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


class DocI:
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


def bad_page(title, text):
    for term in title_start_with_terms:
        if title[:len(term)].upper() == term:
            return True
    for term in title_end_with_terms:
        if title[-len(term):].upper() == term:
            return True
    if len(text) <= page_length_limit:
        return True
    for term in text_start_with_terms:
        if term == text[:len(term)].upper():
            return True
    for term in text_last_terms:
        if term in text[-8000:].upper():
            return True
    return False


def page_generator(file_obj, offset=None):
    """Parses a Wikipedia dump file and yields individual pages."""
    state = title = ID = text = start = None
    pos = next_pos = 0
    for line in file_obj:
        # Keep track of file pos for later start of page seeking
        pos = next_pos
        next_pos += len(line)
        line = line.decode('utf-8')
        if state is None:
            if '<page>' in line:
                state = 'page'
                start = pos
        elif state == 'page':
            title = re.search(r'<title>(.*?)</title>', line)
            if title:
                state = 'title'
                title = title.group(1)
        elif state == 'title':
            ID = re.search(r'<id>(\d+)</id>', line)
            if ID:
                state = 'id'
                ID = ID.group(1)
        elif state == 'id':
            if line.endswith('</text>\n'):
                text = re.search(r'<text[^>]*>(.*?)</text>', line).group(1)
                state = 'done'
            else:
                text = re.search(r'<text.*?>', line)
                if text:
                    text = [line[text.end():]]
                    state = 'text'
        elif state == 'text':
            if line.endswith('</text>\n'):
                text.append(line[:-8])
                text = ''.join(text)
                state = 'done'
            else:
                text.append(line)
        if state == 'done':
            state = None
            if bad_page(title, text):
                continue
            else:
                yield Page(int(ID), title, text, start)


def plain_page_generator(file_obj):
    """Yields individual pages from a generated plain-text corpus file."""
    title = ID = text = None
    pos = next_pos = 0
    for line in file_obj:
        # Keep track of file pos for later start of page seeking
        pos = next_pos
        next_pos += len(line)
        line = line.decode('utf-8')
        ID, title, text = line.split('\t')
        yield Page(int(ID), title, text, pos)


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
                    for token, count in page.token_count:
                        token_docs[token].add(int(page.ID))
                        token_counts[token] += int(count)
                for f in (pagi, doci):
                    f.flush()
    finally:
        # Save token indices
        with open(wiki_location + '.toki', mode='wb') as toki:
            pickle.dump(token_docs, toki, protocol=pickle.HIGHEST_PROTOCOL)
        with open(wiki_location + '.tokc', mode='wb') as tokc:
            pickle.dump(token_counts, tokc, protocol=pickle.HIGHEST_PROTOCOL)


def create_punkt_sent_detector(fname, progress_count, max_pages=25000):
    logger = logging.getLogger('create_punkt_sent_detector')

    punkt = nltk.tokenize.punkt.PunktTrainer()

    logger.info("Training punkt sentence detector")

    wiki_size = os.path.getsize(fname)
    page_count = 0

    try:
        with open(fname, mode='rb') as wiki_dump:
            pages = page_generator(wiki_dump)
            for page in pages:
                page.remove_markup()
                page.unidecode()
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
