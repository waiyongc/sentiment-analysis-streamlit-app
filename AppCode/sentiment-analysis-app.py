import _imp
import streamlit as st
from xgboost import XGBClassifier
import pandas as pd
import joblib
import pickle 
import matplotlib.pyplot as plt
from plotly import express as px
from emot.emo_unicode import UNICODE_EMOJI
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from textblob import Word
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
import scattertext as sct
import spacy
import en_core_web_sm
nlp = spacy.load('en_core_web_sm')

st.set_page_config(page_title="Sentiment Analysis App",layout="wide")

@st.cache
def load_vectorizer():
    return pickle.load(open(r'latest_vectorizer.pk','rb'))

@st.cache
def load_model():
    return joblib.load(open(r'latest_XGB_Model.pkl','rb'))

TFIDF_vectorizer=load_vectorizer()
loaded_model=load_model()

@st.experimental_singleton 
def convert_emojis_to_words(text):
    for emot in UNICODE_EMOJI:
        text = text.replace(emot, "_".join(UNICODE_EMOJI[emot].replace(",","").replace(":","").split()))
    return text
 
def convert_polarity(Polarity):
    if Polarity == 0:
        return 'Negative'
    else:
        return 'Positive'

@st.experimental_singleton
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

def main():
    st.title("A Web-based Application for Sentiment Analysis on Product Review")
    input_type = ["Dataset","Text"]
    choice = st.sidebar.selectbox("Select input type: ",input_type)
    if choice == "Dataset":
        st.subheader("Upload your dataset")
        data_file=st.file_uploader("A dataset with review text in a csv file",type=["csv"])
        if data_file is not None:
            st.write(type(data_file))
            file_details={"filename":data_file.name,"filetype":data_file.type,
                          "filesize":data_file.size}
            st.write(file_details)
            df=pd.read_csv(data_file,header=0,names=['Review'])
            classified_dataset=dataModellingForFile(df)
            corpus = sct.CorpusFromPandas(classified_dataset, category_col='Sentiment_Label', text_col='Review', nlp=nlp).build()
            review_text = " ".join(review for review in classified_dataset.Review)
            stopwords = set(STOPWORDS)
            stopwords = stopwords.union(["ha", "thi", "now", "onli", "im", "becaus", "wa", "will", "even", "go", "realli", "didnt", "abl"])
            col1,col2=st.columns((4,6))
            with col1:
                st.markdown("### Sentiment Label Distribution")
                val_count=classified_dataset['Sentiment_Label'].value_counts()
                chart1=px.bar(x=val_count.index, y=val_count.values, color=val_count.index, text_auto=True, width=400)
                chart1.update_layout(xaxis_title="Sentiment", yaxis_title="Count", legend_title="Sentiment")
                st.write(chart1)
            with col2:
                st.markdown("### Most Frequent 3 Word Phrases in the dataset")
                common_words = get_top_n_trigram(classified_dataset['Review'], 20)
                df1 = pd.DataFrame(common_words, columns = ['Review_Text' , 'Count'])
                chart2=px.bar(df1,x='Count',y='Review_Text',orientation='h',text_auto=True, height=600)
                chart2.update_traces(textangle=0)
                chart2.update_layout(yaxis={'categoryorder':'total ascending'})
                st.write(chart2)
            st.markdown("### Top 100 Most Frequent Words in the dataset")
            wordcl = WordCloud(stopwords = stopwords, background_color='white', max_font_size = 50, max_words = 100).generate(review_text)
            chart3=plt.figure(figsize=(14, 10))
            plt.imshow(wordcl, interpolation='bilinear')
            plt.axis('off')
            plt.show()
            st.pyplot(chart3)
            st.markdown("### Detailed View")
            classified_dataset.index+=1
            st.dataframe(classified_dataset)
            csv=classified_dataset.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Press to Download the Labelled Review Dataset as CSV",
                data=csv,
                file_name="labelled_review_dataset.csv",
                mime="text/csv")
    else:
        st.subheader("Please type the review text in the text box below")
        text=st.text_input("Text box: ")
        st.write("Notes: ")
        st.write("-> Only accept product review in English. ")
        st.write("-> Please use full form. (i.e. do not)")
        st.write("-> Please press ENTER before you click 'Show Result'.")
        if st.button('Show Result'):
            label=dataModellingForText(text)
            st.write("The sentiment of the review is ",label)

@st.experimental_singleton
def dataModellingForFile(dataset):
    dataset['Cleaned_Review'] = dataset['Review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    dataset['Cleaned_Review'] = dataset['Cleaned_Review'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    dataset['Cleaned_Review'] = dataset['Cleaned_Review'].apply(convert_emojis_to_words)
    dataset['Cleaned_Review'] = dataset['Cleaned_Review'].str.replace('[^\w\s]', '')
    tf_X_demo = TFIDF_vectorizer.transform(dataset['Cleaned_Review'])
    prediction = loaded_model.predict(tf_X_demo)
    dataset['Polarity']=prediction
    dataset['Sentiment_Label']=dataset['Polarity'].apply(convert_polarity)
    dataset.drop('Polarity',inplace=True,axis=1)
    dataset.drop('Cleaned_Review',inplace=True,axis=1)
    return dataset

@st.experimental_singleton 
def dataModellingForText(text):
    text = text.lower()
    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text)                           
    text = convert_emojis_to_words(text)
    text = text.replace('[^\w\s]', '')
    text=[text]
    tf_text = TFIDF_vectorizer.transform(text)
    prediction=loaded_model.predict(tf_text)
    if prediction == 0:
        label='NEGATIVE'
    else:
        label='POSITIVE'
    return label


if __name__ == '__main__':
    main()
        







