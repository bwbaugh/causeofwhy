# Copyright (C) 2012 Brian Wesley Baugh
"""Main program to start the Cause of Why QA system."""
import sys
import errno
import ConfigParser

from causeofwhy import indexer, web


CONFIG_FNAME = 'causeofwhy.ini'


def create_default_config():
    """Used to create a default config file if one does not exist."""
    config = ConfigParser.SafeConfigParser()
    config.add_section('wiki')
    config.set('wiki', 'location', 'PATH/TO/WIKIPEDIA/DUMP.xml')
    config.add_section('web server')
    config.set('web server', 'port', '8080')
    with open(CONFIG_FNAME, mode='w') as f:
        config.write(f)


def read_config():
    """Reads a configuration file from disk."""
    config = ConfigParser.SafeConfigParser()
    try:
        with open(CONFIG_FNAME) as f:
            config.readfp(f)
    except IOError as e:
        if e.errno != errno.ENOENT:
            raise
        print 'Configuration file not found! Creating one...'
        create_default_config()
        print 'Please edit the config file named: ' + CONFIG_FNAME
        sys.exit(errno.ENOENT)
    return config


def load_index(wiki_location, doci_in_memory=False):
    """Loads an existing Index or creates one if it doesn't exist."""
    try:
        return indexer.Index(wiki_location, doci_in_memory)
    except indexer.IndexLoadError:
        indexer.create_index(wiki_location)
        return indexer.Index(wiki_location, doci_in_memory)


def main():
    """Loads the Index and starts a web UI according to a config file."""
    config = read_config()
    print 'Loading index'
    index = load_index(config.get('wiki', 'location'))
    print 'Starting web server'
    web.main(index, int(config.get('web server', 'port')))


if __name__ == '__main__':
    sys.exit(main())
