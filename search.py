import time
from index import *
from tkinter import *
from nltk.stem import PorterStemmer

index_index = 0
id_dict = 0
file = 0
common_dict = {}


def build_common_dict(index_index):
    global common_dict
    ps = PorterStemmer()
    file1 = open("indexFinal2.txt", 'r')  # open the inverted index

    with open("stopWordsUpdated.txt") as file2:
        word = file2.readline()
        while word:
            posting_list = []                   # posting list for this term
            term = ps.stem(word.strip())        # stem the term
            position = index_index[term]        # get the starting position of term in the text file
            file1.seek(position)
            line = file1.readline()             # go to that term and retrieve the line
            line = line.split(':')[1]           # take only the postings of that line
            for post in line.split(';')[0:-1]:
                id = int(post.split('-')[0])
                tfidf = float(post.split('-')[1])
                my_post = Posting(id, tfidf)    # remake each posting into actual Posting object
                posting_list.append(my_post)    # append posting to the posting list for this term

            common_dict[term] = posting_list
            word = file2.readline()

    file1.close()


# find the intersecting postings from two lists of postings
def intersect(postings1, postings2):
    answer = []
    i = 0       # counter for postings1
    j = 0       # counter for postings2
    while i < len(postings1) and j < len(postings2):
        if postings1[i].docid == postings2[j].docid:
            # if a posting from the two lists matches, make a new posting with the combined tfidf
            docid = postings1[i].docid
            combined_tfidf = postings1[i].tfidf + postings2[j].tfidf
            new_posting = Posting(docid, combined_tfidf)
            # append that new posting to the "intersection" list
            answer.append(new_posting)
            i += 1
            j += 1
        elif postings1[i].docid < postings2[j].docid:
            i += 1
        else:
            j += 1
    return answer


# retrieve the list of postings for each query term from the inverted index
def retrieve(query, index_index, id_dict, file):
    global common_dict

    tp = textProcess()
    terms = tp.tokenize(query)
    bigrams = set(bigram(terms))           # set of bigram terms
    unigrams = set(terms)                  # set of unigram terms

    posting_lists2 = []                    # list of list of postings for bigram term
    posting_lists1 = []                    # list of list of postings for unigram term

    # gather the list of postings for all the BIGRAMS
    ps = PorterStemmer()
    for term in bigrams:
        posting_list = []                           # posting list for this term
        term = ps.stem(term)                        # stem the term
        if term in common_dict:
            posting_lists2.append(common_dict[term])     # load the in-memory posting list of term is common
        elif term in index_index:
            position = index_index[term]                # get the starting position of term in the text file
            file.seek(position)
            line = file.readline()                      # go to that term and retrieve the line
            line = line.split(':')[1]                   # take only the postings of that line
            for post in line.split(';')[0:-1]:
                id = int(post.split('-')[0])
                tfidf = float(post.split('-')[1])
                my_post = Posting(id, tfidf)            # remake each posting into actual Posting object
                posting_list.append(my_post)            # append posting to the posting list for this term
            posting_lists2.append(posting_list)          # add the corresponding list of postings to the list

    # gather the list of postings for all the UNIGRAMS
    ps = PorterStemmer()
    for term in unigrams:
        posting_list = []                           # posting list for this term
        term = ps.stem(term)                        # stem the term
        if term in common_dict:
            posting_lists1.append(common_dict[term])     # load the in-memory posting list of term is common
        elif term in index_index:
            position = index_index[term]                # get the starting position of term in the text file
            file.seek(position)
            line = file.readline()                      # go to that term and retrieve the line
            line = line.split(':')[1]                   # take only the postings of that line
            for post in line.split(';')[0:-1]:
                id = int(post.split('-')[0])
                tfidf = float(post.split('-')[1])
                my_post = Posting(id, tfidf)            # remake each posting into actual Posting object
                posting_list.append(my_post)            # append posting to the posting list for this term
            posting_lists1.append(posting_list)          # add the corresponding list of postings to the list

    # return immaturely if there are no posting list for the given query
    if len(posting_lists2) <= 0 and len(posting_lists1) <= 0:
        print("No data for the given query")
        return
    # if there's only one list of postings, that will be the only list
    elif len(posting_lists2) == 1:
        intersection = posting_lists2[0]
    elif len(posting_lists1) == 1:
        intersection = posting_lists1[0]
    # if there is no posting list for bigrams, set intersection to empty
    elif len(posting_lists2) <= 0:
        intersection = []
    # if there are multiple lists, find the intersection of the lists
    else:
        # first sort posting_lists by the length of each list
        posting_lists2 = sorted(posting_lists2, key=lambda x: len(x), reverse=False)
        # then start intersecting starting with the smallest list
        intersection = intersect(posting_lists2[0], posting_lists2[1])
        for i in range(2, len(posting_lists2)):
            intersection = intersect(intersection, posting_lists2[i])

    # if there are no intersection in the bigram posting list, use the unigram posting list
    if len(intersection) <= 0:
        # first sort posting_lists by the length of each list
        posting_lists1 = sorted(posting_lists1, key=lambda x: len(x), reverse=False)
        # then start intersecting starting with the smallest list
        intersection = intersect(posting_lists1[0], posting_lists1[1])
        for i in range(2, len(posting_lists1)):
            intersection = intersect(intersection, posting_lists1[i])


    # with the intersecting postings, retrieve their corresponding urls
    print("Top 5 Posting Urls:")
    # sort the intersection by tf-idf score
    intersection = sorted(intersection, key=lambda x: x.tfidf, reverse=True)
    for posting in intersection[0:5]:
        print(id_dict[str(posting.docid)], posting.tfidf)

    # print to GUI interface
    try:
        top1Result.configure(text= id_dict[str(intersection[0].docid)] )
        top1tf.configure(text= str(intersection[0].tfidf))
        top2Result.configure(text= id_dict[str(intersection[1].docid)] )
        top2tf.configure(text= str(intersection[1].tfidf))
        top3Result.configure(text= id_dict[str(intersection[2].docid)] )
        top3tf.configure(text= str(intersection[2].tfidf))
        top4Result.configure(text= id_dict[str(intersection[3].docid)] )
        top4tf.configure(text= str(intersection[3].tfidf))
        top5Result.configure(text= id_dict[str(intersection[4].docid)] )
        top5tf.configure(text= str(intersection[4].tfidf))
    except IndexError:
        pass


def clicked():
    print()
    print(entry.get())

    top1Result.configure(text="")
    top1tf.configure(text="")
    top2Result.configure(text="")
    top2tf.configure(text="")
    top3Result.configure(text="")
    top3tf.configure(text="")
    top4Result.configure(text="")
    top4tf.configure(text="")
    top5Result.configure(text="")
    top5tf.configure(text="")

    if entry.get() == "":
        print("ERROR: EMPTY")
    else:
        start = time.time()
        retrieve(entry.get(), index_index, id_dict, file)
        end = time.time()
        print("Time elapsed:", end - start)
        timeResult = str(end - start)
        timeLabelResult.configure(text=timeResult)




if __name__ == "__main__":

    print("Launching Search Engine")

    # load in our id-doc mapping and our index of index mapping
    file = open("DocID2.json", 'r')
    id_dict = json.load(file)
    file.close()

    file = open("indexOfIndexFinal2.json", 'r')
    index_index = json.load(file)
    file.close()

    # open the inverted index text file
    file = open("indexFinal2.txt", 'r')

    # load in the posting lists for all common words and build a dictionary in memory
    build_common_dict(index_index)

    # GUI interface
    root = Tk()
    root.geometry('900x600')

    label = Label(root, text="Type Query: ")
    label.grid(column=0, row=0)

    entry = Entry(root, width=50)
    entry.grid(column=1, row=0)

    enterButton = Button(root, text="Enter", command=clicked)
    enterButton.grid(column=2, row=0)

    searchResult = Label(root, text="Top 5 Posting Urls")
    searchResult.grid(column=0, row=1)

    top1Label = Label(root, text="1:")
    top1Label.grid(column=0, row=2)
    top1Result = Label(root, text="")
    top1Result.grid(column=1, row=2)
    top1tf = Label(root, text="")
    top1tf.grid(column=2, row=2)

    top2Label = Label(root, text="2:")
    top2Label.grid(column=0, row=3)
    top2Result = Label(root, text="")
    top2Result.grid(column=1, row=3)
    top2tf = Label(root, text="")
    top2tf.grid(column=2, row=3)

    top3Label = Label(root, text="3:")
    top3Label.grid(column=0, row=4)
    top3Result = Label(root, text="")
    top3Result.grid(column=1, row=4)
    top3tf = Label(root, text="")
    top3tf.grid(column=2, row=4)

    top4Label = Label(root, text="4:")
    top4Label.grid(column=0, row=5)
    top4Result = Label(root, text="")
    top4Result.grid(column=1, row=5)
    top4tf = Label(root, text="")
    top4tf.grid(column=2, row=5)

    top5Label = Label(root, text="5:")
    top5Label.grid(column=0, row=6)
    top5Result = Label(root, text="")
    top5Result.grid(column=1, row=6)
    top5tf = Label(root, text="")
    top5tf.grid(column=2, row=6)

    timeLabel = Label(root, text="Time Elapsed:")
    timeLabel.grid(column=0, row=7)
    timeLabelResult = Label(root, text="")
    timeLabelResult.grid(column=1, row=7)
    root.mainloop()

    file.close()
    print("Terminate Program")

    """
    # console prompt for a query
    print("Enter a query: ")
    query = input()
    while query:
        try:
            # search in the index and display the result
            print("\nSearch Results:")
            start = time.time()
            retrieve(query, index_index, id_dict, file)
            end = time.time()
            print("Time elapsed:", end-start)
        except KeyError:
            print("No data for the given query")
        #except Exception as e:
        #    print(type(e), str(e))

        # console prompt for a query
        print("\nEnter a query: ")
        query = input()

    file.close()
    print("Terminate Program")
    """