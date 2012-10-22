# Copyright (C) 2012 Brian Wesley Baugh
"""Web interface allowing users to submit queries and get a response."""
import os
import tornado.ioloop
import tornado.web

from causeofwhy import indexer


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("index.html")


class QueryHandler(tornado.web.RequestHandler):
    def initialize(self, index):
        self.index = index

    def get(self):
        query = self.get_argument('query')
        ir_query = indexer.tokenizer.tokenize(query)
        ir_query = indexer.regularize(ir_query)
        answers = self.index.intersect(ir_query)
        for answer in answers:
            answer.preprocess()
            answer.tokenize_sentences()
            sentences = []
            for sentence in answer.sentences:
                if (len(sentences) < 3 and
                   any(term in indexer.regularize(sentence) for term
                       in ir_query)):
                    sentences.append(sentence)
            answer.sentences = sentences
        self.render("answer.html",
                    query=query,
                    ir_query=' '.join(ir_query),
                    answers=answers)


def main(index):
    application = tornado.web.Application([
        (r"/", MainHandler),
        (r"/cause/", QueryHandler, dict(index=index)),
        ], template_path=os.path.join(os.path.dirname(__file__),
                                      "templates"))
    application.listen(8888)
    tornado.ioloop.IOLoop.instance().start()

if __name__ == "__main__":
    main()
