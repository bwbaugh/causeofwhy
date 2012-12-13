# Copyright (C) 2012 Brian Wesley Baugh
"""Provides functions and classes that deal with parsing a Wikipedia dump."""
# Python packages
import re
import collections
import operator
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

# External packages and modules
import WikiExtractor
from unidecode import unidecode

# Project modules
from indexer import (LINE_SEPARATOR, PARAGRAPH_SEPARATOR, sent_detector,
                     tokenizer, regularize, page_length_limit)


# MODULE CONFIGURATION

# Bad page checks
title_start_with_terms = ('User: Wikipedia: File: MediaWiki: Template: '
                          'Help: Category: Portal: Book: 28644448 Help:'
                          .upper().split(' '))
title_end_with_terms = '(disambiguation)'.upper().split(' ')
text_start_with_terms = '#REDIRECT {{softredirect'.upper().split(' ')
text_last_terms = '{{Disamb {{Dab stub}}'.upper().split(' ')


# CLASSES

class Paragraph(object):
    """Container that holds sentences and their tokens.

    Attributes:
        text: The original unaltered text string of the paragraph.
        sentences: A list of unaltered strings for each sentence in the
            paragraph.
        sentence_tokens: A list of sentences that contains a list of
            string tokens for each sentence.
    """

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


class Page(object):
    """Holds all text and metadata (ID, title) of a page from the corpus.

    Attributes:
        ID: An integer corresponding to the ID of the page in the corpus.
        title: A string of the document title.
        start: The integer offset this document begins at in the corpus.
            Used to seek in the corpus file when retrieving a Page ID.
        paragraphs: A list of Paragraph objects for this document.
        token_count: A defaultdict(int) providing {token -> count},
            where count is the number of times the token appears in the
            document (in all of the document's paragraphs).
        cosine_sim: If present, represents the similarity score for the
            query that was used to retrieve this document. This value
            is set by an Index object.
    """

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
        # First fix wiktioanry links that aren't being handled properly
        # by the WikiExtractor library.
        wikt = r"\[{2,}wikt:[^\|]+\|([^\]]+)\]{2,}"
        self.text = re.sub(wikt, r'\1', self.text)
        broken_wikt = r"{{broken wikt link\|([^\|}]+)(?:\|([^}]+))?}{2,}"
        self.text = re.sub(broken_wikt, r'\1', self.text)
        # Use the WikiExtractor library to finish processing
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
        """Creates a string including ID, title, and original text."""
        self.preprocess()
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


# FUNCTIONS

def bad_page(title, text):
    """Uses heuristics to see if a page shouldn't be processed."""
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
