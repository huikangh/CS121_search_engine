
class Posting:

    def __init__(self, docid, tfidf):
        self.docid = docid
        self.tfidf = tfidf

    def __str__(self):
        return "(id:{}, freq:{})".format(self.docid, self.tfidf)

    def __repr__(self):
        return "Posting(id:{}, freq:{})".format(self.docid, self.tfidf)