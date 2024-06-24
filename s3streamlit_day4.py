# Check if Streamlit is installed correctly, print version
import streamlit as st
print(st.__version__)

# 1: Seeing your data in Streamlit with st.write
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
import openai
from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer
from skllm.models.gpt.classification.zero_shot import ZeroShotGPTClassifier
from openai import OpenAI

df = pd.read_csv("Data/rappler-2024-cleaned-st.csv")

api_key = open('Data/openaiapikey.txt').read()
SKLLMConfig.set_openai_key(api_key)

api_key = open('Data/openaiapikey.txt').read()
client = OpenAI(api_key=api_key)

def extract_keywords(text):
    system_prompt = 'You are a news analyst assistant tasked to extract keywords from news articles.'

    main_prompt = """
    ###TASK###
    - Extract the five most crucial keywords from the news article. 
    - Extracted keywords must be listed in a comma-separated list. 
    - Example: digital advancements, human rights, AI, gender, post-pandemic

    ###ARTICLE###
    """

    try:
        response = client.chat.completions.create(
            model='gpt-3.5-turbo', 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{main_prompt}\n{text}"}
            ]
        )
        top_keywords = response.choices[0].message.content
        return [kw.strip() for kw in top_keywords.split(',')]

    except:
        return []

# Add page options in a sidebar
my_page = st.sidebar.radio('Page Navigation',
                           ['About the data', 'Interactive highlights', 
                            'News summarization', 
                            'Sentiment-based recommendations',
                            'Keyword extraction'])


if my_page == 'About the data':
    st.title("Insight Out: A Rappler News Exploration App")
    st.markdown("This Streamlit app provides comprehensive analysis and exploration of the latest Rappler news data. Designed for **Eskwelabs Data Science Fellowship Cohort 13.**")

    st.header("Preview of the dataset")

    df = pd.read_csv("Data/rappler-2024-cleaned-st.csv")
    st.write(df.head())

    st.header("Quick stats from Rappler news articles")

    fig = plt.figure(figsize=(10, 6))

    df['date'] = pd.to_datetime(df['date'])
    ax = df.set_index('date')['id'].resample('D').count().plot(kind='line')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('date')
    plt.ylabel('count of articles')
    plt.title('Daily Volume of Rappler Articles, 2024Q1', fontsize=16)

    st.pyplot(fig)

elif my_page == 'Interactive highlights':
    st.title('Interacting with the Rappler dataset')
    df = pd.read_csv("Data/rappler-2024-cleaned-st.csv")

    # Add text input for filter to specific keywords
    keywords = st.text_input(
        label='Keywords for filtering the data. If multiple keywords, make a comma-separated list',
        value=''
    )

    keywords = [kw.strip() for kw in keywords.split(',')]

    st.write(keywords)

    # Add multi-select columns where keywords will be searched
    search_cols = st.multiselect(
        'Select columns where keywords will be searched',
        df.columns)

    if search_cols:
        marker_cols = list()
        for col in search_cols:
            df[col+'_marker'] = df[col].str.contains('|'.join(keywords), case=False)
            marker_cols.append(col+'_marker')

        search_markers = [x + '_marker' for x in search_cols]
        df['marker'] = df[search_markers].sum(axis=1)
        df['marker'] = df['marker'] > 0

        df = df[df['marker']]
        df.drop(columns = marker_cols, inplace=True)

        if st.toggle('Show preview of data', value=True):
            st.header("Preview of the dataset")
            st.write(df.head())

     # Add bar plot for top bigrams from filtered data
        st.header('Top bigrams from the filtered dataset')

        content = df['content.cleaned'].str.cat(sep=' ')
        tokens = word_tokenize(content)
        tokens = [word.lower() for word in tokens
                  if word not in stopwords.words('english')
                  and word.isalpha()]

        bigrams = list(nltk.bigrams(tokens))
        bigram_counts = nltk.FreqDist(bigrams)
        top_10_bigrams = bigram_counts.most_common(10)

        bigram_words = [f"{word1} {word2}" for (word1, word2), freq in top_10_bigrams]
        bigram_frequencies = [freq for (word1, word2), freq in top_10_bigrams]

        fig, ax = plt.subplots(figsize=(10, 6))

        plt.barh(bigram_words, bigram_frequencies)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.xlabel('frequency')
        plt.ylabel('bigrams')
        plt.title('Top 10 bigrams by frequency', fontsize=16)
        plt.gca().invert_yaxis()

        st.pyplot(fig)

elif my_page == 'News summarization':
    st.title('Summarizing Rappler articles')
    df = pd.read_csv("Data/rappler-2024-cleaned-st.csv").sort_values(
        'date', ascending=False
    )

    title = st.selectbox(
        'Select article title', df['title.cleaned'], index=None
    )

    if title:
        article = df[df['title.cleaned']==title].iloc[0]

        st.header(f"[{article['title.cleaned']}]({article['link']})")
        st.caption(f"__Published date:__ {article['date']}")

        col1, col2 = st.columns([3,1])

        focused_summary_toggle = col1.toggle(
            'Make focused summary', value=False
        )

        summary_button = col2.button('Summarize article')

        focus = None
        if focused_summary_toggle:
            focus = st.text_input('Input summary focus', value='')

            if focus == '':
                focus = None

        s = GPTSummarizer(
            model='gpt-3.5-turbo', max_words=50, focus=focus
        )

        if summary_button:
            st.subheader('Summary')
            article_summary = s.fit_transform([article['content.cleaned']])[0]
            st.write(article_summary)

        st.subheader('Article content')
        st.write(article['content.cleaned'])

elif my_page == 'Sentiment-based recommendations':
    st.title('Recommending articles based on predicted sentiments')
    df = pd.read_csv("Data/schools-sentiment-labeled.csv").sort_values(
        'date', ascending=False
    )
    
    title = st.selectbox(
        'Select article title', df['title.cleaned'], index=None
    )
    
    if title:
        article = df[df['title.cleaned']==title].iloc[0]
                           
        col1, col2 = st.columns([3,1])
        col1.header(f"[{article['title.cleaned']}]({article['link']})")
        col1.caption(f"__Published date:__ {article['date']}")
        
        clf = ZeroShotGPTClassifier(model="gpt-3.5-turbo")
        clf.fit(None, ["Positive", "Negative", "Neutral"])
        article_sentiment = clf.predict([article['content.cleaned']])[0]

        #code to add color
        if article_sentiment == 'Positive':
            col1.success(f'This article is **{article_sentiment.upper()}** based on the article content.')
        elif article_sentiment == 'Neutral':
            col1.info(f'This article is **{article_sentiment.upper()}** based on the article content.')
        elif article_sentiment == 'Negative':
            col1.error(f'This article is **{article_sentiment.upper()}** based on the article content.')
            
        col1.subheader('Full article content')
        col1.write(article['content.cleaned'])
              
        col2.caption('**SUGGESTED STORIES**')
        suggestions = df[df['gpt_sentiment']==article_sentiment].sample(3)
        
        for i, suggestion in suggestions.iterrows():
            col2.subheader(f"{suggestion['title.cleaned']}")
            col2.write(f"[Link to the article]({suggestion['link']})")
            
elif my_page == 'Keyword extraction':
    st.title('Tagging articles with their most relevant keywords')
    df = pd.read_csv("Data/rappler-2024-cleaned-st.csv").sort_values( #changed to -st file
        'date', ascending=False
    )
    
    title = st.selectbox(
        'Select article title', df['title.cleaned'], index=None
    )
    
    if title:
        article = df[df['title.cleaned']==title].iloc[0]
                           
        st.header(f"[{article['title.cleaned']}]({article['link']})")
        st.caption(f"__Published date:__ {article['date']}")

        st.caption('**TOP KEYWORDS**')
        top_keywords = extract_keywords(article['content.cleaned'])

        highlighted_keywords = ""
        for i, keyword in enumerate(top_keywords):
            highlighted_keywords += f"<span style='background-color:#ffcc99;padding: 5px; border-radius: 5px; margin-right: 5px;'>{keyword}</span>"

        st.markdown(highlighted_keywords, unsafe_allow_html=True) 
        
        st.subheader('Full article content')
        st.write(article['content.cleaned'])