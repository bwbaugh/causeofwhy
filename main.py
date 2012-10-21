# Copyright (C) 2012 Brian Wesley Baugh
"""Main program to start the Cause of Why QA system."""
import sys
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

from causeofwhy import indexer


# Dump location
# wiki_location = ('R:/_Other/Wikipedia/enwiki-20120902-pages-articles-'
#                 'multistream.xml')
# wikilocation = 'C:/wiki/enwiki-20120902-pages-articles-multistream.xml'
wiki_location = 'R:/_Other/Wikipedia/simplewiki-20121002-pages-articles.xml'
# wiki_location = 'C:/wiki/simplewiki-20121002-pages-articles.xml'


def page_list(page):
    try:
        iterator = iter(page)
    except TypeError:
        page.unidecode()
        return ' '.join([str(page.ID), page.title])
    else:
        f = StringIO()
        for page in iterator:
            page.unidecode()
            f.write(' '.join([str(page.ID), page.title]) + '\n')
        output = f.getvalue().rstrip()
        f.close()
        return output


def main():
    index = indexer.create_index(wiki_location)
    while True:
        print input('>>> ')


if __name__ == '__main__':
    sys.exit(main())
