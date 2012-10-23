# Copyright (C) 2012 Brian Wesley Baugh
"""Web interface allowing users to submit queries and get a response."""
import os
import multiprocessing

import tornado.ioloop
import tornado.web
import tornado.httpserver

from causeofwhy import indexer


# How many page worker threads to use
NUMBER_OF_PROCESSES = max(1, multiprocessing.cpu_count() - 1)


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class QueryHandler(tornado.web.RequestHandler):
    def initialize(self):
        self.index = self.application.settings.get('index')
        self.pool = self.application.settings.get('pool')

    @tornado.web.asynchronous
    def get(self):
        query = self.get_argument('query')
        num_top = int(self.get_argument('top', default=10))
        start = int(self.get_argument('start', default=0))
        ir_query = indexer.tokenizer.tokenize(query)
        ir_query = indexer.regularize(ir_query)
        answers = self.index.ranked(ir_query)
        num_results = len(answers)
        # Reduce number of pages we need to get from disk
        answers = answers[start:num_top]
        answers, similarity = zip(*answers)
        # Retrieve the Page objects from the list of Page.IDs
        answers = self.index.get_page(answers)
        args = (answers, similarity, query, ir_query, num_results, answers)
        self.pool.apply_async(handle_answers, (args,), callback=self.callback)

    def callback(self, result):
        query, ir_query, num_results, answers = result
        self.render("answer.html",
                    query=query,
                    ir_query=' '.join(ir_query),
                    num_results=num_results,
                    answers=answers)


def handle_answers(args):
        answers, similarity, query, ir_query, num_results, answers = args
        for answer, sim in zip(answers, similarity):
            answer.cosine_sim = sim
            answer.preprocess()
            answer.tokenize_sentences()
            sentences = []
            for sentence in answer.sentences:
                if (len(sentences) < 3 and
                   any(term in indexer.regularize(sentence) for term
                       in ir_query)):
                    sentences.append(sentence)
            answer.sentences = sentences
        return query, ir_query, num_results, answers


def main(index):
    pool = multiprocessing.Pool(NUMBER_OF_PROCESSES)
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/cause/", QueryHandler),
        ], template_path=os.path.join(os.path.dirname(__file__), "templates"),
        static_path=os.path.join(os.path.dirname(__file__), "static"),
        index=index,
        pool=pool)
    http_server = tornado.httpserver.HTTPServer(application, xheaders=True)
    http_server.listen(8080)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
