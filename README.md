##TEAMMATES: 
Pelin Cetin (pc2807) and Justin Zwick (jaz2130)

##FILES: 
proj1.tar.gz (which when uncompressed includes search.py and stopwords.txt), readme.txt, query-transcripts.txt

JSON API KEY: ""

SEARCH ENGINE ID: ""

## HOW TO RUN OUR CODE: 
	sudo apt install python3-pip
	pip3 install google-api-python-client
	pip3 install numpy
	python3 search.py <google api key> <google engine id> <precision> <query>

## CODE EXPLANATION: 
	INTERNAL DESIGN / QUERY MODIFICATION METHOD: 
		- First, we extract the basic arguments from the command line that we need to send a request to the gse api.
		- Then, we build our set of stopwords. We made the assumption that query's to our service would be made in English. We used the NLTK's list of English stop words, 
		which can be found here: https://gist.github.com/sebleier/554280
		- After sending the query to the gse api, we print the search results to the user and ask for relevance feedback, which is then used to construct a boolean list for later use.
		- Then, we build a vector of titles and summaries of each document, with the punctation stripped out
		- After that, we iterate over the title and summary vectors to build an OrderedDict of all the words used in every document
		- We then compute a tf-idf matrix using the title and summary vectors from before. We dampen the tf and idf using the logarithm technique discussed in the slides
		- We normalize the tf-idf matrix so that each of its component vectors has the same magnitude
		- Then, using the boolean relevance list we constructed earlier, and our tf-idf-matrix, we do Rocchio's algorithm
		- For our term weights, we use alpha = 1.0, beta = 0.8, and gamma = 0.1, since this is what Wikipedia said is most common.
		- Then, using the modified query vector calculated via Rocchio's algorithm, we take the two terms with the highest value (that aren't already in the query) and add them to the query
		- repeat the above until desired precision is achieved
	NOTES:
		- We don't do anything special for handling non-html files. We didn't download and analyze the full results -- only the summaries that Google returns -- so we didn't see a need for this.
	






