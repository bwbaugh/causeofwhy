# Copyright (C) 2012 Brian Wesley Baugh
"""Generates a Weka ARFF file from the labelled question-answer pairs."""

# Constants
CORPUS_FNAME = "log_training.txt"
CLASS_ATTRIBUTE = "__class__"
FEATURES = (
            'page_cosine_sim NUMERIC',
            'term_count NUMERIC',
            'related_sum NUMERIC',
            'related_average NUMERIC',
            'causal_match {True,False}',
            'position NUMERIC',
            'text_length NUMERIC',
           )
SPAN = len(FEATURES) + 1


def write_header(fileobj, class_list):
    comments = """\
% This file generated automatically from the following filename:
% {}
"""
    comments = comments.format(CORPUS_FNAME)
    fileobj.write(comments)

    relation = """\
@RELATION "{}"
"""
    relation = relation.format(CORPUS_FNAME)
    fileobj.write(relation)
    fileobj.write('\n')

    for attribute in FEATURES:
        fileobj.write('@ATTRIBUTE {}'.format(attribute) + '\n')
    fileobj.write('\n')

    class_attribute = """\
@ATTRIBUTE {} {{{}}}
"""
    class_attribute = class_attribute.format(CLASS_ATTRIBUTE,
                                             ','.join(class_list))
    fileobj.write(class_attribute)
    fileobj.write('\n')


def write_data(fileobj):
    fileobj.write('@DATA\n')
    with open(CORPUS_FNAME) as f:
        for line in f:
            ir_query, query, answer_positions, answers = line.split('\t', 3)
            answer_positions = answer_positions.split(',')
            if '0' in answer_positions:
                continue
            answers = answers.split('\t')
            answers = ['\t'.join(answers[i:i + SPAN]) for i in
                       xrange(0, len(answers), SPAN)]
            correct = []
            wrong = []
            for answer in answers:
                rank, answer = answer.split('\t', 1)
                answer = answer.split('\t')
                if len(answer) != len(FEATURES):
                    print 'ERROR ON LINE:'
                    print line
                    raise ValueError
                answer[-1] = answer[-1].strip()
                if rank in answer_positions:
                    answer.append('1')
                    correct.append(answer)
                else:
                    answer.append('-1')
                    wrong.append(answer)
            # Oversample the minority class (correct answers) in order
            # to have an equal number of correct and incorrect instances.
            for answer in correct * (len(wrong) / len(correct)) + wrong:
                fileobj.write(','.join(answer) + '\n')


def build_class_list():
    return set(['1', '-1'])


def main():
    with open(CORPUS_FNAME + '.arff', mode='w') as arff:
        print "Building class_list"
        class_list = build_class_list()
        print "Writing ARFF file"
        write_header(arff, class_list)
        write_data(arff)
        print "Done!"


if __name__ == '__main__':
    main()
