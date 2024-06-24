import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import gensim
from gensim import corpora
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    if isinstance(text, str):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
        return tokens
    else:
        return []

def create_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    st.pyplot(plt)

def visualize_topics_plotly(lda_model):
    topics = lda_model.show_topics(formatted=False, num_words=10)
    topic_words = []
    for i, topic in enumerate(topics):
        topic_words_list = [word for word, _ in topic[1]]
        topic_words.append(", ".join(topic_words_list))
    
    fig = px.bar(x=list(range(1, len(topic_words)+1)), y=[1]*len(topic_words), 
                 hover_data={'x': list(range(1, len(topic_words)+1)), 'y': topic_words},
                 labels={'x': 'Topic', 'y': 'Top Words'}, 
                 title='Topic Distribution')
    
    return fig

st.title("NLP Preprocessing and Topic Modeling App")
st.write("""
Upload a CSV file, select a text column for NLP preprocessing,
and then wait for the model to perform topic modeling to visualize the topics using LDA.
""")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # text column
    text_column = st.selectbox("Select the text column for preprocessing", df.columns)

    df['processed_text'] = df[text_column].apply(preprocess_text)

    st.write("Preprocessed Text:")
    st.write(df[['processed_text']].head())

    df = df.dropna(subset=['processed_text'])
    df = df[df['processed_text'].apply(len) > 0]

    dictionary = corpora.Dictionary(df['processed_text'])
    corpus = [dictionary.doc2bow(text) for text in df['processed_text']]

    #LDA model
    num_topics = st.slider("Select number of topics", 2, 10, 5)
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    # topics
    st.write("LDA Topics:")
    topics = lda_model.print_topics(num_words=10)
    for topic in topics:
        st.write(topic)

    # Visualize topics with Plotly
    st.write("Topic Visualization:")
    fig = visualize_topics_plotly(lda_model)
    st.plotly_chart(fig, use_container_width=True)

    search_word = st.text_input("Enter a word to find related topics:")
    if search_word:
        search_topics = lda_model.get_term_topics(search_word)
        search_results = pd.DataFrame(search_topics, columns=['Topic', 'Probability']).sort_values(by='Probability', ascending=False)
        st.write(f"Topics related to '{search_word}':")
        st.write(search_results)

    for i, topic in enumerate(lda_model.show_topics(formatted=False, num_words=10)):
        st.write(f"Topic {i+1}")
        words = [word for word, _ in topic[1]]
        create_wordcloud(words, f"Topic {i+1}")
