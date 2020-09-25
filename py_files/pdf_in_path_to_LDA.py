import os
#pip install pypdf2
import PyPDF2
import textract
#pip install gensim
from gensim.models import LdaMulticore, CoherenceModel
from gensim.parsing.preprocessing import remove_stopwords, stem_text
from gensim.corpora import Dictionary as d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
# pip install pyldavis
import pyLDAvis
import pyLDAvis.gensim  
# pip install wordcloud
from wordcloud import WordCloud, STOPWORDS 
import seaborn as sns
import pandas as pd
import numpy as np

directory = input('paste your parent directory here')


cwd = os.getcwd()

def get_files_dir(path):
    filenames = []
    for file in os.listdir(path):
        if file.endswith('.pdf'): #or whatever file you want
            #print(os.path.join(path, file))
            filenames.append(file)
    return filenames

def pdf_to_token_array(filename):

    pdfFileObj = open(filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
    count = 1
    num_pages = pdfReader.numPages
    text = ""

    #The while loop will read each page.
    while count < num_pages:
        pageObj = pdfReader.getPage(count)
        count +=1
        text += pageObj.extractText()
    #This if statement exists to check if the above library returned words. It's done because PyPDF2 cannot read scanned files.
    if text != "":
        text = text
    #If the above returns as False, we run the OCR library textract to #convert scanned/image based PDF files into text.
    else:
        pass
        #text = textract.process(fileurl, method='tesseract', language='eng')

    text = text.split()

    #trim short words
    text = [i for i in text if len(i)>5]
    text = [i.replace(',','') for i in text]
    # be sure to split sentence before feed into Dictionary
    dataset = [d.split() for d in text]

    return dataset, filename


def compute_coherence(model, texts, dictionary):
    coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model.get_coherence()
    return coherence_lda


# exception to not start this if certain requirements (dictionary of weird characters) are met
def build_lda(seed, texts, filename):
	#optional, in case data comes from a .csv:
    #texts = pd.read_csv(filename, names = colname ,converters = {'tweets': eval}, header = None, squeeze = True)
    #delete texts parameter accordingly
    dictionary = d(texts)
    print("-------------\n Output begins:")
    print(filename)
    print(dictionary)
    #remove words which appear <5 times in the dictionary + words which are more frequent than 10 per cent
    dictionary.filter_extremes(no_below=5, no_above=0.1, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in texts]
    #change num_topics according to own interest
    lda_model = LdaMulticore(bow_corpus, random_state=seed, num_topics=4, id2word=dictionary, passes=2, workers=3, per_word_topics=True)
    print("----\nLDA results:\nshown here:")
    for idx, topic in lda_model.print_topics(-1):
       print('Topic: {} \nWords: {}'.format(idx, topic))
    #Compute Perplexity and Coherence

    perplexity = lda_model.log_perplexity(bow_corpus)
    coherence = compute_coherence(lda_model, texts, dictionary)
    print('\nPerplexity: ',perplexity, "\nCoherence:",coherence)
    print("---")


    return bow_corpus, dictionary, lda_model, coherence

def create_word_cloud(model,fname):
    # more colors: 'mcolors.XKCD_COLORS'
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  
#put title of the file as title of the graphics
    cloud = WordCloud(background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    topics = model.show_topics(formatted=False)

    fig, axes = plt.subplots(2, 2, figsize=(10,10), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')


    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    #check if this is useful title. also maybe put as H1 the folder and then H2 the document.
    title = fname.split('/')
    plt.title(title[-1])#right now this is shown as title instead of "topic 3".

    fname = fname.split('.pdf')

    fname = fname[0]+'_wordcloud.svg'

    plt.savefig(fname, dpi=None, facecolor='w', edgecolor='w')
    plt.close(fig) #the plots still do not reset, so whenever one fails the next one picks up the words from the former.
    #plt.show()

def format_topics_sentences(ldamodel=None, corpus=None, texts=None):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)

def document_word_counts(dominant_topic):   #change to plotly for comparison
    print("Distribution of Document Word Counts by Dominant Topic")
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]

    fig, axes = plt.subplots(2,2,figsize=(6,6), dpi=160, sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):    
        df_dominant_topic_sub = dominant_topic.loc[dominant_topic.Dominant_Topic == i, :]
        doc_lens = [len(d) for d in df_dominant_topic_sub.Text]
        ax.hist(doc_lens, bins = 80, color=cols[i])
        ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
        sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
        ax.set(xlim=(0, 80), xlabel='Document Word Count')
        ax.set_ylabel('Number of Documents', color=cols[i])
        ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0,80,9))
    plt.show()


"""
function here which finds all files in the listet parent directory which end in .pdf and appends them to the list_of_files
then use this list.
"""

#insert all files to do LDA on:     #what way to shorten these long urls? #keep directory from the whole string.
list_of_files = get_files_dir(directory)

directory = directory+'/'

list_of_files = [directory+i for i in list_of_files]

for filepath in list_of_files:
##this could be it, master
    begin = pd.Timestamp.now()

    texts, filename = (pdf_to_token_array(filepath))

    try:
        #save returned results into variables to reuse?
        corpus, dictionary, lda_model, coherence = build_lda(42,texts,filename)
    except:
        print("failed lda for {}".format(filepath))

    
    
    """
    lda_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=corpus, texts=texts)

    # Format
    lda_dominant_topic = lda_topic_sents_keywords.reset_index()
    lda_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    lda_dominant_topic.head(10)"""

    #document_word_counts(lda_dominant_topic)

    #uncomment once you fixed and saved each thing, how to display multiple figures from plt btw?
    #document_word_counts(lda_dominant_topic)
    try:
        create_word_cloud(lda_model,filename)
    except:
        print("failed wordcloud for {}".format(filepath))

    end = pd.Timestamp.now()
    compilation_time = (end-begin).total_seconds()

    #printing filename -> https://stackoverflow.com/questions/4152963/get-name-of-current-script-in-python
    print(filename+'compilation time: {}s'.format(compilation_time))


