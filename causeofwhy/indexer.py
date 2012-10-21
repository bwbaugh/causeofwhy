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
try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO


# External packages and modules
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode
import WikiExtractor


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

PUNCTUATION = """( ) ' '' `` : ; , ? % + ! - # = } { [ ]""".split(' ')
PARAGRAPH_SEPARATOR = u'\u2029'

# Bad page checks
page_length_limit = 1024
title_start_with_terms = ('User: Wikipedia: File: MediaWiki: Template: '
                       'Help: Category: Portal: Book: 28644448 Help:'
                       .upper().split(' '))
title_end_with_terms = '(disambiguation)'.upper().split(' ')
text_start_with_terms = '#REDIRECT {{softredirect'.upper().split(' ')
text_last_terms = '{{Disamb {{Dab stub}}'.upper().split(' ')


# CLASSES

class Page:
    def __init__(self, ID, title, text, start=None):
        self.ID = ID
        self.title = title
        self.text = text
        self.start = start
        self.sentences_of_tokens = None
        self.token_count = None

    def remove_markup(self):
        """Remove wiki markup leaving just the plain-text."""
        self.text = WikiExtractor.clean(self.text)
        self.text = '\n'.join(WikiExtractor.compact(self.text))

    def unidecode(self):
        """Convert non-ascii to closest ASCII equivalent."""
        self.title = unidecode(self.title).strip()
        self.text = unidecode(self.text).strip()

    def tokenize_sentences_of_tokens(self):
        self.sentences_of_tokens = []
        for paragraph in self.text.split(PARAGRAPH_SEPARATOR):
            # Sentence segmentation
            sentences = sent_detector.tokenize(paragraph,
                                               realign_boundaries=True)
            # Tokenization
            sentences = tokenizer.batch_tokenize(sentences)
            self.sentences_of_tokens.extend(sentences)

    def regularlize_text(self):
        for i, sentence in enumerate(self.sentences_of_tokens):
            for j, token in enumerate(sentence):
                # Normalize text by case folding
                self.sentences_of_tokens[i][j] = token.lower()
                # Remove punctuation
                if token in PUNCTUATION:
                    self.sentences_of_tokens[i][j] = None
                # Lemmatize
                self.sentences_of_tokens[i][j] = lemmatizer.lemmatize(token)
                # Stopword removal
                if token in STOPWORDS:
                    self.sentences_of_tokens[i][j] = None
            # Remove empty tokens
            self.sentences_of_tokens[i] = [x for x in
                                           self.sentences_of_tokens[i] if x is
                                           not None]
        # Remove empty sentences
        self.sentences_of_tokens = [x for x in self.sentences_of_tokens if x]

    def count_tokens(self):
        """Count the frequency of text's tokens in a bag-of-words style."""
        self.token_count = collections.defaultdict(int)
        for sentence in self.sentences_of_tokens:
            for token in sentence:
                self.token_count[token] += 1
        self.token_count = [(token, str(count)) for (token, count) in\
                            sorted(self.token_count.iteritems(),
                                   key=operator.itemgetter(1),
                                   reverse=True)]

    def __str__(self):
        f = StringIO()
        f.write('=' * 40 + '\n')
        f.write(self.ID, self.title + '\n')
        f.write('-' * 40 + '\n')
        f.write(self.text)
        f.write('=' * 40 + '\n')
        output = f.getvalue()
        f.close()
        return output


# FUNCTIONS

def index_exists(fname):
    """Specifies whether or not an index has been created for a wiki dump."""
    # TODO(bwbaugh): Implement this function.
    return False


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


def page_generator(file_obj):
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
                yield Page(ID, title, text, start)


def worker(taskq, doneq):
    logger = logging.getLogger('worker')
    done_buff = []
    try:
        while True:
            chunk = taskq.get()
            if chunk is None:
                return
            for page in chunk:
                page.remove_markup()
                if len(page.text) < page_length_limit:
                    continue
                page.unidecode()
                page.tokenize_sentences_of_tokens()
                page.regularlize_text()
                page.count_tokens()
                done_buff.append(page)
            doneq.put(done_buff)
            done_buff = []
    finally:
        doneq.put(None)


def writer(doneq, wiki_location):
    pill_count = 0  # termination condition (poison pill)
    token_docs = collections.defaultdict(set)
    token_counts = collections.defaultdict(int)
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
                    pagi.write('\t'.join([str(page.ID).ljust(1),
                                          str(page.start).ljust(1),
                                          str(doci.tell()).ljust(1)]) + '\n')
                    doci.write('\t'.join([page.ID] + [chr(26).join([k, v]) for
                                                      k, v in page.token_count]) +
                               '\n')
    #                txt.write('\t'.join([str(ID), title]) + '\n\n')
    #                for sentence in text:
    #                    txt.write(' '.join([token for token in sentence]) + '\n')
    #                txt.write('\n\n')
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


# Main function of module
def create_index(fname, progress_count=None, max_pages=None, skip_exists=True):
    if skip_exists and index_exists(fname):
        return

    if sent_detector is None:
        create_punkt_sent_detector(fname=fname,
                                   progress_count=CHUNK_SIZE,
                                   max_pages=min(25000, max_pages))

    # Set params
    if progress_count is None:
        progress_count = CHUNK_SIZE * NUMBER_OF_PROCESSES

    logger = logging.getLogger('create_index')

    wiki_size = os.path.getsize(fname)

    # Page task queues for parallel processing
    taskq = multiprocessing.Queue(MAX_QUEUE_ITEMS)
    doneq = multiprocessing.Queue(MAX_QUEUE_ITEMS)

    # Start worker processes
    logger.info('Starting workers')
    workers = []
    for i in range(NUMBER_OF_PROCESSES):
        p = multiprocessing.Process(target=worker, args=(taskq, doneq))
        p.start()
        workers.append(p)

    # Start log writer process
    p = multiprocessing.Process(target=writer, args=(doneq, fname))
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
