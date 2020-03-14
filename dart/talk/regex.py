import re


def is_word_in_list(term, sentence: list):
    if isinstance(term, list):
        for t in term:
            return _regex_search(term=t, sentence=sentence)
    return _regex_search(term=term, sentence=sentence)


def _regex_search(term, sentence):
    if re.search(term, ' '.join(sentence)) is not None:
        return True
    return False
