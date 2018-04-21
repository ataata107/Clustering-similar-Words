import logging
import os
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
import gensim
from gensim import corpora, models, similarities
from sklearn.cluster import KMeans


# Reads and returns the list of files from a directory
def read_directory(mypath):
    current_list_of_files = []

    while True:
        for (_, _, filenames) in os.walk(mypath):
            current_list_of_files = filenames
        logging.info("Reading the directory for the list of file names")
        return current_list_of_files


# Function you will be working with
def creating_subclusters(list_of_terms, name_of_file):
    # Your code that converts the cluster into subclusters and saves the output in the output folder with the same name as input file
    # Note the writing to file has to be handled by you.
    corpus = list_of_terms
    tok_corp= [nltk.word_tokenize(sent) for sent in corpus]
    model = gensim.models.Word2Vec(tok_corp, min_count=1,size=100)
    X=[]
    print(len(corpus))
    for i in corpus:
        X.append(model.wv[i])
    

    kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit_predict(X)
    #Z = X[y_kmeans==1,:]
    f = open("output/"+name_of_file,'w')
    for i in range(5):
        for j in range(len(corpus)):
            if(y_kmeans[j]==i):
                f.write(corpus[j]+" ")
        f.write("\n")        
    
    
    return X,y_kmeans
    #print( model.wv['finance'])
    #print(model.most_similar(['finance']))
    #pass


# Main function
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

    # Folder where the input files are present
    mypath = "input"
    list_of_input_files = read_directory(mypath)
    for each_file in list_of_input_files:
        with open(os.path.join(mypath, each_file), "r") as f:
            file_contents = f.read()
        list_of_term_in_cluster = file_contents.split()

        # Sending the terms to be converted to subclusters in your code
        X,y=creating_subclusters(list_of_term_in_cluster, each_file)
        

        # End of code