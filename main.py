# Copyright (C) 2012 Brian Wesley Baugh
"""Main program to start the Cause of Why QA system."""
import sys

from causeofwhy import indexer


# Dump location
# wiki_location = ('R:/_Other/Wikipedia/enwiki-20120902-pages-articles-'
#                 'multistream.xml')
# wikilocation = 'C:/wiki/enwiki-20120902-pages-articles-multistream.xml'
wiki_location = 'R:/_Other/Wikipedia/simplewiki-20121002-pages-articles.xml'
# wiki_location = 'C:/wiki/simplewiki-20121002-pages-articles.xml'


def main():
    index = indexer.create_index(wiki_location)
    while True:
        print input('>>> ')


if __name__ == '__main__':
    sys.exit(main())
