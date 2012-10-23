# Copyright (C) 2012 Brian Wesley Baugh
"""Main program to start the Cause of Why QA system."""
import sys
import errno
import ConfigParser

from causeofwhy import indexer, web


CONFIG_FNAME = 'causeofwhy.ini'


def create_default_config():
    config = ConfigParser.SafeConfigParser()
    config.add_section('wiki')
    config.set('wiki', 'location', 'PATH/TO/WIKIPEDIA/DUMP.xml')
    with open(CONFIG_FNAME, mode='w') as f:
        config.write(f)


def main():
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
        return errno.ENOENT
    print 'Loading index'
    index = indexer.create_index(config.get('wiki', 'location'))
    print 'Starting web server'
    web.main(index)


if __name__ == '__main__':
    sys.exit(main())
