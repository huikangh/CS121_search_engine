# REFERENCES:
# https://www.datacamp.com/community/tutorials/pickle-python-tutorial
# https://stackoverflow.com/questions/24398302/bs4-featurenotfound-couldnt-find-a-tree-builder-with-the-features-you-requeste
# https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76

import json
import math
from PartA import *
from posting import *
from os import listdir
from os.path import isfile, isdir, join
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from urllib.parse import urldefrag


def bigram(text):
    model=[]
    # model will contain n-gram strings
    count=0
    for token in text[:len(text)-2+1]:
       model.append(text[count] + " " + text[count+1])
       count=count+1
    return model


# store all the json file paths in the directory into a list
# return a list of files in the given file path
def gather_file(file_dir):
    json_list = []
    for d in listdir(file_dir):
        dir = join(file_dir, d)  # join the file path
        if isdir(dir):
            for f in listdir(dir):
                file = join(dir, f)  # join the file path
                if isfile(file):
                    json_list.append(file)
    return json_list


# Save the given dictionary to a text file with the format:
# term:docid-tfidf;docid-tfidf;docid-tfidf...
# for each term in each line
# Also returns an index-the-index mapping
def index_to_disk(index, num):
    index_of_index = {}

    filename = 'index' + str(num) + ".txt"
    outfile = open(filename, 'w')

    for token in sorted(index.keys()):
        index_of_index[token] = outfile.tell()     # index the starting position of the token in the text file
        outfile.write(token + ":")
        for posting in index[token]:
            outfile.write(str(posting.docid) + "-" + str(posting.tfidf) + ";")
        outfile.write("\n")

    outfile.close()
    return index_of_index


# takes in a list of files
# return a list of "index of index" dictionaries, a id-doc dictionary, and a idf dictionary
# inverted index: {"token":[posting]}
# id_dict: {id, "url"}
def build_index(file_list):
    inverted_index = {}         # dictionary for keeping a partial inverted index
    index_index_list = []       # list of dictionaries, each dictionary has the "index of index" to each text file
    idf_dict = {}               # dictionary for idf
    id_dict = {}                # dictionary for mapping doc ID to urls
    id_ = 0

    print("Length of documents without filtering:", len(file_list))

    # parameter for controlling when to save dict to disk
    maxsize = 12000         # max number of files before offloading
    count = 0               # counter for number of files indexed
    numFiles = 10

    # for each json file in the json_list
    for i in range(len(file_list)):

        f = open(file_list[i])                  # open the json file
        data = json.load(f)                     # load the file into a dictionary of json fields
        f.close()

        # check if the document is a duplicate by looking at the url
        # if the current document has the same de-fragmented url as a document we seen before
        # skip this document
        defrag_url = urldefrag(data['url'])[0]
        if defrag_url in id_dict.values():
            continue

        # map the de-fragmented url to an integer as its doc ID
        id_ += 1
        id_dict[id_] = defrag_url
        print(defrag_url)

        # extract only text from the page content
        content = data['content']
        soup = BeautifulSoup(content, 'lxml')   # using the content string, construct a beautiful soup
        text = soup.get_text()                  # extract only text from the page content

        # now that we got the text from the page, tokenize the text
        tp = textProcess()
        token_list = tp.tokenize(text)          # list of tokens in this file

        # get the bi-gram of the token list
        bigram_list = bigram(token_list)
        token_list = bigram_list + token_list

        # we should process the tokens
        ps = PorterStemmer()
        token_list = [token for token in token_list if len(token) >= 2]     # remove len-1 tokens
        token_list = [ps.stem(token) for token in token_list]               # stem tokens

        # compute the frequency of tokens in this file
        token_map = tp.computeWordFrequencies(token_list)

        # keep track of the document frequency of each token for idf
        for token in token_map.keys():
            if token not in idf_dict:
                idf_dict[token] = 0
            idf_dict[token] += 1

        # for each unique token, create a posting and add it to the inverted index
        for token in token_map.keys():
            # construct a list for the token if token does not exist
            if token not in inverted_index:
                inverted_index[token] = []

            new_posting = Posting(id_, token_map[token])        # construct a posting object with id and basic tf
            inverted_index[token].append(new_posting)           # append the posting to the list

        count += 1

        # save dict to disk when maxsize is exceeded or all files are indexed
        if (count == maxsize) or (i == len(file_list)-1):
            # save dict to disk and empty dict
            index_of_index = index_to_disk(inverted_index, numFiles)    # returns the index of index in the text file
            index_index_list.append(index_of_index)         # append the "index of index" to list
            inverted_index = {}
            count = 0
            numFiles += 1
            print("Saving to file")

    # lastly, save the id->doc mapping to a local file
    file = open("DocID2.json", 'w')
    json.dump(id_dict, file)
    file.close()

    return index_index_list, idf_dict


# merge all the partial indices into one big text file of index
# return the final "index the index" mapping of the final inverted index
def merge_partial_index(index_index_list, idf_dict):

    # load all index_to_index dictionaries
    index_index0 = index_index_list[0]
    index_index1 = index_index_list[1]
    index_index2 = index_index_list[2]
    index_index3 = index_index_list[3]
    index_index4 = index_index_list[4]

    # open all partial indices
    file0 = open("index10.txt", 'r')
    file1 = open("index11.txt", 'r')
    file2 = open("index12.txt", 'r')
    file3 = open("index13.txt", 'r')
    file4 = open("index14.txt", 'r')

    # initialize the final large index and its "index of index"
    indexFinal = open("indexFinal2.txt", 'w')
    index_index_final = {}

    # idf_dict will have all our tokens, so use idf_dict to iterate through all the tokens
    all_tokens = sorted([token for token in idf_dict.keys()])     # sort the list of all tokens

    # for each token, retrieve its corresponding posting list from all partial index, and merge into one
    for token in all_tokens:
        # save the starting position of this term
        index_index_final[token] = indexFinal.tell()
        # the final posting list for this term
        final_posting = []

        # TEXT FILE 0
        if token in index_index0.keys():
            position = index_index0[token]      # get the position of the word in text file 0
            file0.seek(position)                # go to that position in text file 0
            line = file0.readline().strip()     # read that line
            line = line.split(':')[1]           # take only the postings of that line

            for post in line.split(';')[0:-1]:  # index 0 to -1 because we dont want the last one (just a newline)
                id = int(post.split('-')[0])
                tfidf = float(post.split('-')[1])
                tfidf = (1+math.log(tfidf)) * math.log(len(idf_dict)/idf_dict[token])   # re-calculate tfidf
                my_post = Posting(id, tfidf)
                final_posting.append(my_post)   # save the posting in the final posting list for this term

        # TEXT FILE 1
        if token in index_index1.keys():
            position = index_index1[token]      # get the position of the word in text file 0
            file1.seek(position)                # go to that position in text file 0
            line = file1.readline().strip()     # read that line
            line = line.split(':')[1]           # take only the postings of that line

            for post in line.split(';')[0:-1]:  # index 0 to -1 because we dont want the last one (just a newline)
                id = int(post.split('-')[0])
                tfidf = float(post.split('-')[1])
                tfidf = (1 + math.log(tfidf)) * math.log(len(idf_dict) / idf_dict[token])  # re-calculate tfidf
                my_post = Posting(id, tfidf)
                final_posting.append(my_post)  # save the posting in the final posting list for this term

        # TEXT FILE 2
        if token in index_index2.keys():
            position = index_index2[token]      # get the position of the word in text file 0
            file2.seek(position)                # go to that position in text file 0
            line = file2.readline().strip()     # read that line
            line = line.split(':')[1]           # take only the postings of that line

            for post in line.split(';')[0:-1]:  # index 0 to -1 because we dont want the last one (just a newline)
                id = int(post.split('-')[0])
                tfidf = float(post.split('-')[1])
                tfidf = (1 + math.log(tfidf)) * math.log(len(idf_dict) / idf_dict[token])  # re-calculate tfidf
                my_post = Posting(id, tfidf)
                final_posting.append(my_post)  # save the posting in the final posting list for this term

        # TEXT FILE 3
        if token in index_index3.keys():
            position = index_index3[token]      # get the position of the word in text file 0
            file3.seek(position)                # go to that position in text file 0
            line = file3.readline().strip()     # read that line
            line = line.split(':')[1]           # take only the postings of that line

            for post in line.split(';')[0:-1]:  # index 0 to -1 because we dont want the last one (just a newline)
                id = int(post.split('-')[0])
                tfidf = float(post.split('-')[1])
                tfidf = (1 + math.log(tfidf)) * math.log(len(idf_dict) / idf_dict[token])  # re-calculate tfidf
                my_post = Posting(id, tfidf)
                final_posting.append(my_post)  # save the posting in the final posting list for this term

        # TEXT FILE 4
        if token in index_index4.keys():
            position = index_index4[token]      # get the position of the word in text file 0
            file4.seek(position)                # go to that position in text file 0
            line = file4.readline().strip()     # read that line
            line = line.split(':')[1]           # take only the postings of that line

            for post in line.split(';')[0:-1]:  # index 0 to -1 because we dont want the last one (just a newline)
                id = int(post.split('-')[0])
                tfidf = float(post.split('-')[1])
                tfidf = (1 + math.log(tfidf)) * math.log(len(idf_dict) / idf_dict[token])  # re-calculate tfidf
                my_post = Posting(id, tfidf)
                final_posting.append(my_post)  # save the posting in the final posting list for this term

        # after the final posting list for this term is done
        # write the current term and its posting list to the final index file
        final_posting = sorted(final_posting, key=lambda x: x.docid)
        indexFinal.write(token + ":")
        for post in final_posting:
            indexFinal.write(str(post.docid) + "-" + str(post.tfidf) + ";")
        indexFinal.write("\n")

    # close all files
    file0.close()
    file1.close()
    file2.close()
    file3.close()
    file4.close()
    indexFinal.close()

    # as a final step, offload the final index of index mapping to a local file
    file = open("indexOfIndexFinal2.json", 'w')
    json.dump(index_index_final, file)
    file.close()




if __name__ == "__main__":

    path = "DEV"
    file_list = gather_file(path)
    index_index_list, idf_dict = build_index(file_list)
    print("Merging partial index")
    merge_partial_index(index_index_list, idf_dict)