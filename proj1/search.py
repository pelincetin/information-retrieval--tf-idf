from googleapiclient.discovery import build
import sys
import string
from collections import OrderedDict
from math import log10
import numpy as np

# we build our set of stopwords by reading them in from an external file (NLTK's list of English stopwords)
# this is where we got the file: https://gist.github.com/sebleier/554280
# ASSUMPTION: queries are made in english
def build_stop_words(stop_words):
    filepath = 'stopwords.txt'
    with open(filepath) as fp:
        line = fp.readline().strip()
        while line:
            stop_words.add(line)
            line = fp.readline().strip()

# input: string of raw words
# returns: string with punctuation filtered out
# disclaimer: this technique was taken from a stackoverflow post... https://stackoverflow.com/questions/265960/best-way-to-strip-punctuation-from-a-string
def filter_string(s):
    exclude = set(string.punctuation)
    s = ''.join(ch for ch in s if ch not in exclude)
    return s

def print_result_to_user(idx, item):
    print("Result " + str(idx) + ":")
    document = "["
    document += "\n  URL:" + item['link']
    document += "\n  Title:" + item['title']
    document += "\n  Summary:" + item['snippet']
    document += "\n]"
    document += "\n"
    print(document)

def compute_tf_idf_matrix(title_vectors, summary_vectors, all_words):
    tf_vectors = []
    df_vector = []
    for i in range(len(title_vectors)):
        doc_tf = []
        # this will be the same order everytime since this is an ordereddict
        # which is necessary... each corresponding index in each of the tf-idf vectors
        # must correspond to the same word,
        # and each vector in the tf-idf matrix must represent 1 document
        for word in all_words.keys():
            count = title_vectors[i].count(word) + summary_vectors[i].count(word)
            doc_tf.append(count)

        # builds up a matrix -- a list of lists, where each list is a vector of term frequencies
        # every vector represents a different doc, and every element within each vector represents a different word
        tf_vectors.append(doc_tf)

    # build a single list of document frequencies
    # each element in the list is a df for one of the words
    # obv, len(df_vector) == number of words in all_words
    for word in all_words.keys():
        count = 0
        for i in range(len(title_vectors)):
            if word in title_vectors[i] or word in summary_vectors[i]:
                count += 1
        df_vector.append(count)

    # number of documents is equal to the number of titles
    number_of_documents = float(len(title_vectors))

    # to quote the slides,
    # "a document with 10 occurences of a query term might be more relevant than a document with just one occurence,
    # but not 10 times more relevant!"
    # ... relevance does not grow linearly with term frequency, so we dampen it with logarithms
    tf_vectors_dampened = []
    for tf_vector in tf_vectors:
        tf_vector_dampened = []
        for tf in tf_vector:
            if tf == 0:
                tf_vector_dampened.append(float(0))
            else:
                tf_vector_dampened.append(1.0+log10(tf))
        tf_vectors_dampened.append(tf_vector_dampened)

    # we use log10(N/df) instead of N/df to dampen the effect of idf
    dampened_idf_vector = []
    for df in df_vector:
        dampened_idf = log10(float(number_of_documents) / df)
        dampened_idf_vector.append(dampened_idf)

    # unnormalized_tf_idf_matrix = np.asarray(dampened_idf_vector) * tf_vectors_dampened
    unnormalized_tf_idf_matrix = dampened_idf_vector * np.array(tf_vectors_dampened, dtype=float)

    normalized_tf_idf_matrix = (unnormalized_tf_idf_matrix.T / np.linalg.norm(unnormalized_tf_idf_matrix, axis=1)).T
    return normalized_tf_idf_matrix

def rocchios_algorithm(query, query_vector, relevance_vector, tf_idf_matrix, number_of_relevant, number_of_irrelevant, all_words):

    # make a vector of relevant words
    sum_r = np.zeros(len(tf_idf_matrix[0]), dtype=float)
    # make a vector of irrevelant words
    sum_i = np.zeros(len(tf_idf_matrix[0]), dtype=float)
    for i in range(len(relevance_vector)):
        if relevance_vector[i]:
            # print(tf_idf_matrix[i])
            sum_r += tf_idf_matrix[i]
            # print(sum_r)
        else:
            sum_i += tf_idf_matrix[i]
    if number_of_relevant != 0:
        sum_r /= number_of_relevant
    if number_of_irrelevant != 0:
        sum_i /= number_of_irrelevant

    # weights in the formula are determined empirically.
    # We picked these values because it was what Wikipedia said is most common: https://en.wikipedia.org/wiki/Rocchio_algorithm
    a = 1
    b = 0.8
    c = 0.1

    modified_query = (a * query_vector) + (b * sum_r) - (c * sum_i)
    # argsort returns a list of indices of the same shape as the list that would sort the array
    top_word_indices = np.argsort(modified_query)[::-1]

    items = list(all_words.items())
    count = 1

    # in the modified query vector that is calculated,
    # we find the two words with the highest tf-idf score (that aren't already in the original query)
    common_words = []
    for idx in top_word_indices:
        if items[idx][0] not in query:
            common_words.append(items[idx][0])

    return common_words[0], common_words[1]

def process_search_results(results, query, all_words, stop_words):

    idx = 1
    relevance_vector = []
    title_vectors = []
    summary_vectors = []

    for item in results['items']:

        # prints the search result to the user in the format specified in the reference implementation
        print_result_to_user(idx, item)

        is_relevant = input("Relevant (Y/N)?")

        # ASSUMPTION: everything not 'y' is considered 'n' since it's a boolean
        if is_relevant.lower().strip() == 'y':
            relevance_vector.append(True)
        else:
            relevance_vector.append(False)

        idx += 1

        # filter punctuation out of the title's words
        list_of_words_in_title = filter_string(item['title']).lower().split()
        title_vectors.append(list_of_words_in_title)

        # filter punctuation out of summary's words
        list_of_words_in_summary = filter_string(item['snippet']).lower().split()
        summary_vectors.append(list_of_words_in_summary)

    # now that we have populated our summary and title vectors, we're going to take everything inside of them
    # and build an ordered set of all words present in the titles and snippets returned by the gse api.
    # needs to be ordered so that each vector in our tf-idf matrix represents exactly 1 of the documents that's returned
    build_word_set(title_vectors, summary_vectors, all_words, stop_words)

    query_vector = []

    # build the query vector
    for word in all_words.keys():
        value = 0.0
        if word in query:
            value = 1.0
        query_vector.append(value)

    number_of_relevant_documents = 0

    # for every true value in the relevance vector, increment the number of relevant documents
    for b in relevance_vector:
        if b:
            number_of_relevant_documents += 1

    # by definition, the number of irrelevant documents is the number of all documents subtracted by the number
    # of relevant documents
    number_of_irrelevant_documents = len(results['items']) - number_of_relevant_documents

    # build a normalized tf-idf matrix
    tf_idf_matrix = compute_tf_idf_matrix(title_vectors, summary_vectors, all_words)

    # use rocchios algorithm to modify direction of original query vector in the hypersphere
    # Then, use that new vector to find two words to augment the query with
    # also checks to make sure those two new words aren't already present in the query
    word1, word2 = rocchios_algorithm(query, query_vector, relevance_vector, tf_idf_matrix, number_of_relevant_documents, number_of_irrelevant_documents, all_words)

    # augment the query
    query.extend([word1,word2])
    # TODO: comment this out before submitting
    print("The words " + word1 + " and " + word2 + " are the next words to augment the query with.")

    return relevance_vector

# build a set of all words that are present in the titles and summaries of search results
# we're using an OrderedDict to maintain a consist ordering for when we use this to build our query/document vectors,
# and just inserting a dummy None for the value
def build_word_set(title_vectors, summary_vectors, all_words, stop_words):
    for i in range(len(title_vectors)):
        for word in title_vectors[i]:
            if word not in stop_words:
                all_words[word] = None
        # obv there are an equal number of titles and summaries
        for word in summary_vectors[i]:
            if word not in stop_words:
                all_words[word] = None

def main():

    google_api_key = sys.argv[1]
    google_engine_id = sys.argv[2]
    desired_precision = float(sys.argv[3])
    query = [str.lower() for str in sys.argv[4].split()]
    real_precision = -1

    stop_words = set()
    build_stop_words(stop_words)

    while desired_precision > real_precision:

        service = build("customsearch", "v1",
                        developerKey=google_api_key)

        # send query to gcse api and get results
        results = service.cse().list(
            q=" ".join(query),
            cx=google_engine_id,
        ).execute()

        # we need an ordered set to model queries and documents -- we need to know all the words present in the query + document database
        # to build their corresponding vector representations
        all_words = OrderedDict()

        # process search results (augments the initial query) and returns a boolean list that represents relevant/irrelevant docs
        # that returned list is then used to calculate the precision
        relevance_vector = process_search_results(results, query, all_words, stop_words)
        number_of_search_results = len(results['items'])

        if number_of_search_results < 10:
            print("===============================")
            print("FEEDBACK SUMMARY")
            print("Query: " + " ".join(query))
            print("Less than 10 results...search already too niche to refine.")
            break

        number_of_relevant_documents = 0
        for b in relevance_vector:
            if b:
                number_of_relevant_documents += 1

        real_precision =  number_of_relevant_documents / number_of_search_results
        # when printing the feedback summaru
        # we aren't including the 2 words that would be added to the query next
        # since the next query hasn't been sent to the Google Search API yet
        x = " ".join(query[:-2])
        if real_precision == 0:
            print("===============================")
            print("FEEDBACK SUMMARY")
            print("Query: " + x)
            print("Below desired precision, but can no longer augment results (less than 10 results)")
            print("===============================")
            break

        if real_precision >= desired_precision:
            print("===============================")
            print("FEEDBACK SUMMARY")
            print("Query: " + x)
            print("Precision " + str(real_precision))
            print("Desired precision reached, done")
            print("We achieved precision of " + str(real_precision) + "! Your desired precision was " + str(
                desired_precision))
            print("===============================")
        else:
            print("===============================")
            print("FEEDBACK SUMMARY")
            print("Query: " + x)
            print("Precision " + str(real_precision))
            print("Still below the desired precision of " + str(desired_precision) + ". Beginning another round...")
            print("===============================")

if __name__ == '__main__':
    main()
