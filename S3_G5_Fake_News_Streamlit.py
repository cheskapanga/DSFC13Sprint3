import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string
from datetime import datetime, date
import itertools
from ast import literal_eval
import seaborn as sns

from skllm.config import SKLLMConfig
from skllm.models.gpt.text2text.summarization import GPTSummarizer
from skllm.models.gpt.classification.zero_shot import MultiLabelZeroShotGPTClassifier
import openai
from openai import OpenAI

# Import libraries for nltk preprocessing
nltk.download('punkt') # Downloads the Punkt tokenizer models
nltk.download('stopwords') # Downloads the list of stopwords
nltk.download('wordnet') # Downloads the WordNet lemmatizer data

st.set_page_config(layout='wide')

api_key = st.secrets["api_key"]##  open('openaiapikey.txt').read() ##
client = OpenAI(api_key=api_key)
SKLLMConfig.set_openai_key(api_key)

df = pd.read_csv("data/Philippine Fake News Corpus - cleaned.csv")
df['Date'] = pd.to_datetime(df['Date'])
df['Start of Month'] = (df['Date'].dt.to_period('M').dt.to_timestamp()).dt.date
df['Start of Week'] = (df['Date'] - pd.to_timedelta(df['Date'].dt.dayofweek, unit='D')).dt.date
df['Date'] = df['Date'].dt.date
df['Authors'] = df['Authors'].apply(literal_eval)

source_author_df = df[['Source', 'Authors']]
source_author_df  = source_author_df.explode('Authors')
source_author_df = source_author_df.drop_duplicates()
sources = df['Source'].unique()

source_author_df = source_author_df.sort_values(by = ['Authors'], ascending=True, na_position='last')
source_author_df['Authors'] = source_author_df['Authors'].fillna(source_author_df['Source'] +" (No Author)")
authors = source_author_df['Authors'].unique()

df['Authors_'] = df.apply(lambda x: [x['Source'] +" (No Author)"] if x['Authors'] == [] else x['Authors'] , axis=1)

date_min = df['Date'].min()
date_max = df['Date'].max()

#helper functions


def article_count_num(df,label_suffix):
    html = f'''
        <div style="background-color: transparent; padding: 20px 5px 5px 5px; text-align: center; justify: center; height:600">
            <div style="font-size: 3rem; font-weight: bold; margin-bottom: 0px; margin-top: 20px;">{df['Headline'].count()}</div>
            <div style="font-size: 1.5rem; font-weight: bold; margin-bottom: 0px;">{label_suffix}News Articles</div>
            <div style="font-size: 1.1rem; ">from {df['Date'].min().strftime("%B %d, %Y")}</div>
            <div style="font-size: 1.1rem; ">to {df['Date'].max().strftime("%B %d, %Y")}</div>
    
        </div>
        '''
    return html
    

def has_intersection(names, target_names):
    return bool(set(names) & set(target_names))
    
def article_count_period(df, period, period_column, label_suffix,line_color ):
    df_period_count = df.groupby([period_column])['Headline'].count().to_frame('article_count').reset_index()
    chart = alt.Chart(df_period_count).mark_line(
            color=line_color,
            opacity=1
            ).encode(
            x=alt.X(period_column),
            y=alt.Y('article_count'),
            tooltip=[period_column, 'article_count']
            ).properties(
                title=f'{period} Number of {label_suffix}News Articles',
  
                height=300
            ).configure_axis(
                # labelFontSize=0,  # Set label font size to 0 to hide axis labels
                title=None  # Set axis title to None to hide axis title
            ).configure_title(
                fontSize=20,  # Adjust title font size
                # anchor='start',  # Title alignment
                # color='blue'  # Title color
            )
    return chart

def source_count(df,bar_color,label_suffix):
    st.header(f':writing_hand: Top Sources for {label_suffix}News Dataset')
    df_source_count = df.groupby(['Source']).size().to_frame('article_count').reset_index()
    df_source_count.sort_values(by = 'article_count', ascending = False)
    
    chart = alt.Chart(df_source_count).mark_bar(color = bar_color).encode(
        x= 'article_count:Q',
        y=alt.Y('Source:N', sort='-x'),
        tooltip=['Source:N', 'article_count:Q'] 
    ).properties(
        # title = f"{label_suffix}News Sources",
        width=600,
        height=alt.Step(40)
    ).configure_axis(
        # labelFontSize=0,  # Set label font size to 0 to hide axis labels
        title=None  # Set axis title to None to hide axis title
    ).configure_title(
        fontSize=20,  # Adjust title font size
        # anchor='start',  # Title alignment
        # color='blue'  # Title color
    )

    return chart

def author_count(df,bar_color,label_suffix,top_n):
    st.header(f':writing_hand: Top Authors for {label_suffix}News Dataset')
    df_author_count = df.explode('Authors')
    df_author_count['Author'] = df_author_count['Authors'].fillna(df_author_count['Source'] +" (No Author)")
    df_author_count = df_author_count.drop('Authors', axis = 1)
    df_author_count = df_author_count.groupby(['Author']).size().to_frame('article_count')
    df_author_count.sort_values(by = 'article_count', ascending = False)
    df_author_count = df_author_count.reset_index()
    df_author_count = df_author_count.tail(top_n)
    
    chart = alt.Chart(df_author_count).mark_bar(color = bar_color).encode(
        x= 'article_count:Q',
        y=alt.Y('Author:N', sort='-x'),
        tooltip=['Author:N', 'article_count:Q'] 
    ).properties(
        # title = f"Top {label_suffix}News Authors",
        width=600,
        height=alt.Step(40)
    ).configure_axis(
        # labelFontSize=0,  # Set label font size to 0 to hide axis labels
        title=None  # Set axis title to None to hide axis title
    ).configure_title(
        fontSize=20,  # Adjust title font size
        # anchor='start',  # Title alignment
        # color='blue'  # Title color
    )

    return chart

def tokenize_contents(df):
    content = df['Content'].str.cat(sep=' ')
    tokens = word_tokenize(content)
    tokens = [word.lower() for word in tokens
              if word not in stopwords.words('english')
              and word.isalpha()]
    return content,tokens
    
def top_bigrams(tokens,label_suffix, top_n,bar_color):
    st.header(f":bar_chart: Top bigrams from the {label_suffix}News Dataset")

    bigrams = list(nltk.bigrams(tokens))
    bigram_counts = nltk.FreqDist(bigrams)
    top_10_bigrams = bigram_counts.most_common(top_n)

    bigram_words = [f"{word1} {word2}" for (word1, word2), freq in top_10_bigrams]
    bigram_frequencies = [freq for (word1, word2), freq in top_10_bigrams]

    df_bigrams = pd.DataFrame({'Bigrams': bigram_words, 'Frequency': bigram_frequencies})
    chart = alt.Chart(df_bigrams).mark_bar(color = bar_color).encode(
        x= 'Frequency:Q',
        y=alt.Y('Bigrams:N', sort='-x'),
        tooltip=['Bigrams:N', 'Frequency:Q'] 
    ).properties(
        # title = f"Top {label_suffix}News Authors",
        width=600,
        height=alt.Step(40)
    ).configure_axis(
        # labelFontSize=0,  # Set label font size to 0 to hide axis labels
        title=None  # Set axis title to None to hide axis title
    ).configure_title(
        fontSize=20,  # Adjust title font size
        # anchor='start',  # Title alignment
        # color='blue'  # Title color
    )

    return chart

def word_cloud(content,label_suffix, color = None):
    st.header(f':capital_abcd: Wordcloud from the {label_suffix}News Dataset')

    stop_words=stopwords.words('english')
    wordcloud = WordCloud(width = 1000, height = 600,
                          background_color ='white',
                          stopwords = stop_words,
                          min_font_size = 10).generate(content)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    plt.tight_layout(pad=0)        
    return fig
        



# Function to label an article
def label_article(Headline):
    clf = MultiLabelZeroShotGPTClassifier(openai_model="gpt-3.5-turbo", max_labels=3)    
    # Define the candidate labels
    candidate_labels = [
        'Politics',
        'Economy',
        'Business',
        'Technology',
        'Health',
        'Environment',
        'Education',
        'Entertainment',
        'Sports',
        'International Relations',
        'Lifestyle',
    ] 
    clf.fit(None, [candidate_labels])
    return clf.predict([Headline])[0]


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

    except e:        
        return []

# # Main Function, generate a response based on a given prompt
# @st.cache_resource(show_spinner = "Loading...please wait")
# def generate_response(article, prompt):
#     response = client.chat.completions.create(
#         model='gpt-3.5-turbo', 
#         messages=[
#             {'role': 'system', 
#              'content': f"Perform the specified tasks based on this article: {article}"},
#             {'role': 'user', 
#              'content': prompt}
#         ]
#     )
#     return response.choices[0].message.content


def generate_response(article, prompt):
    response = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages=[
            {'role': 'system', 
             'content': 
             f"Perform the specified tasks based on this article:\n\n{article}"},
            {'role': 'user', 'content': prompt}
        ]
    )
    return response.choices[0].message.content

def identify_entities(article):
    prompt = f"""
    Return a list of persons and organization entities mentioned in the article
    """
        #     - Entities must be listed in a comma-separated list
        # - Return the list only. 
        # - Example: Elon Musk,Rodrigo Duterte,PhilHealth

    entities = generate_response(article, prompt)
    return entities

def identify_entity_sentiment(article, entities):
    prompt = f"""
    Identify the article's sentiment on each mentioned entity whether POSTITIVE, NEUTRAL or NEGATIVE:\n\n{entities}.        
    """
        # - List them in a semi-colon-separated format along with their sentiment, POSTITIVE, NEUTRAL or NEGATIVE. 
        # - Return List Only
        # - Example: Elon Musk,POSITIVE; Rodrigo Duterte,NEGATIVE; PhilHealth,'NEUTRAL'
    
    
    sentiment = generate_response(article, prompt)
    return sentiment

def support_claims(article, entity_sentiment):
    prompt = f""""
        f"For each entity and sentiment in {entity_sentiment}, identify claims that support the sentiment on entity"   
    """
    # st.write(1)
    descriptions = generate_response(article, prompt)
    return descriptions


    
def identify_entities_with_sentiment(article):
    prompt = f"Identify only the person and organization entities mentioned in the article along with the article's sentiment on each mentioned entity (whether Positive, Neutral or Negative)."
    entity = generate_response(article, prompt)
    return entity



@st.cache_resource(show_spinner = "Loading...please wait")
def identify_claims(article):
    prompt = f"""
    Identify five claims made on this: {article}, and the entities who made such claim. 
    Each claim should be formatted as a full sentence.
    Return the claims as a semicolon-separated list, do not add numbers.
    """
    claims = generate_response(article, prompt)
    return claims

@st.cache_resource(show_spinner = "Loading...please wait")
def verify_claims(article, claim):
    system_prompt = 'You are a news analyst tasked to fact-check the claims made in news articles.'
    main_prompt = f"""Identify whether or not the {claim} made in this {article} is more likely True or 
    more likely not credible. 
    First, state whether or not the {claim} is more likely True or more likely False. 
    If {claim} is more likely False, give an explanation for the average reader 
    why this is so. Explain this as if you were educating the average reader."""
    response = client.chat.completions.create(
        model='gpt-3.5-turbo', 
        messages = [
            {'role': 'system', 
             'content': system_prompt},
            {'role': 'user', 
             'content': main_prompt}
        ]
    )
    return response.choices[0].message.content



############SET COMMON FILTERS#########
# filter_default_df = pd.read_csv("data/filter_default.csv")
# st.write(filter_default_df)
# st.write(filter_default_df)

def initialize_session_state():
    st.session_state.label = 'Not Credible'
    st.session_state.date_filter = True
    st.session_state.start_date = date_min
    st.session_state.end_date = date_max
    st.session_state.source_filter = True
    st.session_state.source = []
    st.session_state.author_filter = True
    st.session_state.author = []
    st.session_state.keyword_filter = True
    st.session_state.keywords = ''
    st.session_state.search_cols = []
    st.session_state.df_filtered = pd.DataFrame(columns = ['Headline'])
    st.session_state.show_common_filter_interactive_highlights = True
    st.session_state.show_common_filter_summarization = True
    st.session_state.filter_article_summarization = False
    st.session_state.show_common_filter_ml_classifier = True
    st.session_state.filter_article_ml_classifier = False
    st.session_state.show_common_filter_interactive_quiz = True
    st.session_state.filter_article_interactive_quiz = False
    st.session_state.article = None
    st.session_state.headline = None
    st.session_state.headline_index = None

def get_session_state():
    session_state = st.session_state
    if 'start_date' not in session_state:
        initialize_session_state()
    return session_state
    
def common_filter():
    session_state = get_session_state()
    frow1_col1, frow1_col2, frow1_col3 , frow1_col4 , frow1_col5, frow1_col6= st.columns([2.5,1,1,2,0.5,2])
    frow2_col1, frow2_col2= st.columns([1,8])
    frow3_col1, frow3_col2= st.columns([1,8])
    frow4_col1, frow4_col2= st.columns([1,8])
    frow5_col1, frow5_col2, frow5_col3= st.columns([7,1,1])

    labels = ['Credible', 'Not Credible', 'All']
    label_temp = frow1_col1.selectbox('Label:', labels, index=labels.index(session_state.label))
    frow1_col3.write("<div style='height: 30px;font-size: 14px'>Filter Date:</div>", unsafe_allow_html=True)
    date_filter_temp = frow1_col3.toggle('', value=session_state.date_filter , key = 'date_filter_key')
    if date_filter_temp == True:
        start_date_temp = frow1_col4.date_input('',
                              value = session_state.start_date,
                              min_value=date_min,
                              max_value=date_max,
                              # key = 'start_date'
                              )
        frow1_col5.write("<div style='padding-top:35px; text-align:center;font-size: 14px' > to </div>", unsafe_allow_html=True)
        end_date_temp = frow1_col6.date_input('',
                              value = session_state.end_date,
                              min_value=session_state.start_date,
                              max_value=date_max,
                              # key = 'end_date'
                            )

        
    frow2_col1.write("<div style='height: 30px;font-size: 14px'>Filter Source:</div>", unsafe_allow_html=True)
    source_filter_temp = frow2_col1.toggle('', value=session_state.source_filter, key = 'source_filter_key')
    
    if source_filter_temp == True:
        for s in  session_state.source:
            if s not in sources:
                session_state.source.remove(s)
        source_temp = frow2_col2.multiselect('', sources, session_state.source)

    frow3_col1.write("<div style='height: 30px;font-size: 14px'>Filter Author:</div>", unsafe_allow_html=True)    
    author_filter_temp = frow3_col1.toggle('', value=session_state.author_filter, key = 'author_filter_key')
    if author_filter_temp == True:
        author_temp = frow3_col2.multiselect('', authors, session_state.author)

    frow4_col1.write("<div style='font-size: 14px'>Filter Keywords:</div>", unsafe_allow_html=True)    
    keyword_filter_temp = frow4_col1.toggle('', value=session_state.keyword_filter, key = 'keyword_filter_key')
    
    if keyword_filter_temp == True:        
        keywords_temp = frow4_col2.text_input(
            label='Keywords for filtering the data. If multiple keywords, make a comma-separated list',
            value=session_state.keywords
        )
        
        search_cols_temp = frow4_col2.multiselect('Select columns where keywords will be searched',
                            ['Headline', 'Content', 'URL'], session_state.search_cols)
    
    # if frow5_col2.button("Clear", key='clear-button', type="secondary", use_container_width=True):
    #     session_state.source.clear()
    #     session_state.author.clear()
    #     session_state.search_cols.clear()
    #     session_state.keywords = ''
    #     st.experimental_rerun()    
    
    if frow5_col3.button("Apply", key='apply-button', type="primary", use_container_width=True):
        session_state.label = label_temp
        session_state.date_filter = date_filter_temp
        if date_filter_temp:
            session_state.start_date = start_date_temp
            session_state.end_date = end_date_temp
        
        session_state.source_filter = source_filter_temp
        if source_filter_temp:
            session_state.source = source_temp
        
        session_state.author_filter = author_filter_temp
        if author_filter_temp:
            session_state.author = author_temp
        
        session_state.keyword_filter = keyword_filter_temp
        if keyword_filter_temp:
            session_state.keywords = keywords_temp 
            session_state.search_cols = search_cols_temp 
        
        if session_state.label == 'All':
            df_filtered = df
        else:
            df_filtered = df[df['Label'] == session_state.label]

        if session_state.date_filter == True and len(df_filtered)>0:
            df_filtered = df_filtered[(df_filtered['Date'] >= session_state.start_date) & (df_filtered['Date'] <= session_state.end_date) ]            

        if session_state.source_filter == True and len(df_filtered)>0:
            df_filtered = df_filtered[df_filtered['Source'].isin(session_state.source)]
        
        if session_state.author_filter== True and len(df_filtered)>0:
            df_filtered = df_filtered[df_filtered['Authors_'].apply(lambda x: has_intersection(x, session_state.author))]

        if df_filtered.shape[0] > 0 and len(df_filtered)>0:
            keywords_list = [kw.strip() for kw in session_state.keywords.split(',')] 
            
            if session_state.search_cols:
                marker_cols = list()
                for col in session_state.search_cols:
                    df_filtered[col+'_marker'] = df_filtered[col].str.contains('|'.join(keywords_list), case=False)               
                    marker_cols.append(col+'_marker')
        
                search_markers = [x + '_marker' for x in session_state.search_cols]
                df_filtered['marker'] = df_filtered[search_markers].sum(axis=1)
                df_filtered['marker'] = df_filtered['marker'] > 0
        
                df_filtered = df_filtered[df_filtered['marker']]
                
        if len(df_filtered)>0:
            df_filtered.reset_index(inplace = True, drop = True)
        session_state.df_filtered = df_filtered  
        
    return session_state.df_filtered

    
# Page 1 content
def about_the_data():
    st.title(":mag_right: A Philippine Fake News Exploration App")
    st.markdown("This Streamlit app is a response to the campaign against [misinformation and disinformation](https://www.un.org/en/countering-disinformation) in the Philippines.")
    st.markdown("It offers an in-depth analysis of the Philippine Fake News Corpus, which comprises web-scraped news articles from January 1, 2016, to November 16, 2018. For this prototype app, only the articles published from 2018 will be used. For more information about the dataset, you may access [this link](https://github.com/aaroncarlfernandez/Philippine-Fake-News-Corpus/tree/master).")

    st.header("Preview of the dataset")
    st.write(df.head())
    st.header("Quick stats from News articles")

    col1, col2 =st.columns([2,5])
    col1.write(article_count_num(df[['Headline', 'Content', 'Date', 'Source', 'Authors', 'URL', 'Label']],''), unsafe_allow_html=True)    
    col2.altair_chart(article_count_period(df, 'Monthly' ,'Start of Month', '', '#1E90FF'), use_container_width=True)
    
# Page 2 content


    
def interactive_highlights():
    session_state = get_session_state()
    st.title(':newspaper: Interacting with the 2018 Philippine Fake News Corpus')
    show_common_filter = st.toggle('Show Filter', value=True , key = 'show_common_filter_interactive_key')
    if show_common_filter == True:
        df_filtered = common_filter()
    else:
        df_filtered = session_state.df_filtered
        
    if st.session_state.label == 'Credible':
        label_suffix = 'Credible '
        color =  'green'#'#33FF57'
    elif st.session_state.label == 'Not Credible':  
        label_suffix = 'Fake '
        color = 'red'#'#FF5733'
    else:
        label_suffix = ''
        color =  '#1E90FF'
    
    if df_filtered.shape[0]>0:            
        viz_col1,viz_col2 = st.columns([2,5])
        period = viz_col2.selectbox('Date Column:', ['Daily', 'Weekly', 'Monthly'], index=1)
        if period == 'Daily':
            date_column = 'Date'
        elif period == 'Weekly':
            date_column = 'Start of Week'
        elif period == 'Monthly':
            date_column = 'Start of Month'

        viz_col1.write(article_count_num(df_filtered,label_suffix), unsafe_allow_html=True)    
        viz_col2.altair_chart(article_count_period(df_filtered, period , date_column, label_suffix, color ), use_container_width=True)
        st.altair_chart(source_count(df_filtered, color , label_suffix), use_container_width=True)
        st.altair_chart(author_count(df_filtered,'grey',label_suffix,top_n = 10), use_container_width=True)
        
        content, tokens = tokenize_contents(df_filtered)
        st.altair_chart(top_bigrams(tokens,label_suffix,10,color), use_container_width=True)

        word_cloud_fig = word_cloud(content,label_suffix) 
        st.pyplot(word_cloud_fig)
        
        show_filtered_data = st.toggle('Show Filtered Data', value=False , key = 'show_filtered_data_key')               
        if show_filtered_data:
            st.write(df_filtered[['Headline', 'Content', 'Date', 'Source', 'Authors', 'URL', 'Label']])
        
    else:
        st.write("<div style='height: 80px;font-size: 30px;text-align: center; padding:20px'>No Article Found</div>", unsafe_allow_html=True)   

def news_summarization():
    st.title('Summarizing 2018 Articles')
    session_state = get_session_state()
    
    sf_col1, sf_col2, sf_col3 = st.columns([2,2,5])
    session_state = get_session_state()
    filter_headlines = sf_col1.toggle('Filter Articles', value=session_state.filter_article_summarization, key = 'filter_article_summarization_key')
    df_filtered = df 

    if filter_headlines == True:
        show_common_filter= sf_col2.toggle('Show Filter', value=False, key = 'show_common_filter_summarization_key')    
        
        if show_common_filter == True:
             common_filter()        
        df_filtered = session_state.df_filtered

    headline_temp = None
    if len(df_filtered) >0:
        headlines = df_filtered['Headline'].to_list()
        art_col1, art_col2 = st.columns([7,2])
        
        headline_temp = art_col1.selectbox('Select article title', headlines , index=session_state.headline_index)
        
        art_col2.write("<div style='height:30px'> </div>", unsafe_allow_html=True)           
        apply_article = art_col2.button('Apply',use_container_width = True, key = 'article_apply_key')
       
        if apply_article:
            session_state.filter_article_summarization = filter_headlines
            session_state.headline = headline_temp            
            if headline_temp is None:
                session_state.headline_index = None
                session_state.article = None
            else:
                session_state.headline_index = headlines.index(headline_temp)
                article_temp = df_filtered[df_filtered['Headline']==headline_temp].iloc[0]
                session_state.article = article_temp       

            
        if apply_article and session_state.article is not None:
            article = session_state.article 
            st.header(f"[{article['Headline']}]({article['URL']})")
            st.caption(f"__Published date:__ {article['Date']}")
                        # Predict labels           
            predicted_labels = label_article(article['Content'])
            # Display predicted labels as blue tabs
            st.caption('**PREDICTED LABELS**')
            labeled_categories = ""
            for label in predicted_labels:
                if label!= "":
                    labeled_categories += f"<span style='background-color:#0041C2;padding: 5px; border-radius: 5px; margin-right: 5px;'>{label}</span>"
            st.markdown(labeled_categories, unsafe_allow_html=True)
    
            
            top_keywords = extract_keywords(article['Content'])
            if len(top_keywords) >0:
                st.caption('**TOP KEYWORDS**')  
            highlighted_keywords = ""
            for i, keyword in enumerate(top_keywords):
                highlighted_keywords += f"<span style='background-color:#0041C2;padding: 5px; border-radius: 5px; margin-right: 5px;'>{keyword}</span>"
    
            st.markdown(highlighted_keywords, unsafe_allow_html=True)    
            st.caption('**ENTITIES WITH SENTIMENTS**')
            # try:
            #     st.subheader("Article's sentiment on entities mentioned")                     
            #     identify_entities2 = identify_entities_with_sentiment(article)
            #     st.write(identify_entities2)
            # except:
            #      pass
            try:
                entities = identify_entities(article['Content'])
                # st.write(entities)
                entity_sentiment = identify_entity_sentiment(article['Content'], entities)
                # st.write(entity_sentiment)
                entity_sentiment_claims = support_claims(article['Content'], entity_sentiment)
                st.write(entity_sentiment_claims)
            except:
                pass
            # entities_ = [ent.strip() for ent in entities.split(',')]
             
            # entities_positive_list = []
            # entities_negative_list = []
            # entities_neutral_list = []
            # st.write(entity_sentiment)
            # for ent_sen in entity_sentiment.split(';') :
            #     items = ent_sen.split(',')
            #     ent = ','.join(items[:-1])
            #     sen = items[-1]
            #     if sen == 'POSITIVE':
            #         entity_positive_list += [sen]
            #     elif sen == 'NEGATIVE':
            #         entity_negative_count  += [sen]
            #     else:
            #         entity_neutral_count  += [sen]

            #     st.write(ent)
            #     st.write(sen)
            #     entity_sentiment_claims = support_claims(article['Content'], ent, sen)
            #     st.write(f"<div style='height: 80px;font-size: 30px;text-align: left; padding:20px'>{ent} ({sen})</div>", unsafe_allow_html=True) 
            #     st.write(entity_sentiment_claims)


     
            # summary_col1, summary_col2 = st.columns([6,3])
    
            
            # focused_summary_toggle = summary_col1.toggle('Make focused summary', value=False, key = 'foccused_summary_key')            
            # summary_button = summary_col2.button('Summarize article',use_container_width = True, key = 'summary_button_key')
            
            # focus = None
            # if focused_summary_toggle:
            #     focus = summary_col1.text_input('Input summary focus', value='', key = 'focus_key')                
            #     if focus == '':
            #         focus = None            
            # if summary_button:
            #     s = GPTSummarizer(
            #         model='gpt-3.5-turbo', max_words=50, focus=focus
            #     )

            #     summary_col1.write('Summary')
            #     article_summary = s.fit_transform([article['Content']])[0]
            #     st.write(article_summary)
            
            # st.write('Article content')
            st.caption('**Content**')  
            st.write(article['Content'])


    else:
        st.write("<div style='height: 80px;font-size: 30px;text-align: center; padding:20px'>No Article Found</div>", unsafe_allow_html=True)   

    


def ml_news_classifier():
    st.title("Using Traditional Machine Learning and Large Language Models to Assess the Credibility of a News Article")
    st.markdown("### Pitting the Different Machine Learning and OpenAI Large Language Models Against Each Other")
    
    st.write("While more sophisticated Large Language Models (LLM) exist which can perform complex Natural Language Processing (NLP) tasks, let's see if we can still use traditional machine learning (ML) models to determine a news articles credibility based on its content alone.")
    
    st.markdown(" ##### Methodology")
    st.write("We trained different types machine learning models traditionally used for classification tasks with the articles as training data. Each article content was cleaned, pre-processed, and tokenized. The features for the training and validation data were extracted using `TfidfVectorizer`. Each model was evaluated based on the Accuracy and the F1 score. A `DummyClassifier` model was used as sanity check.")
    st.write("Aside from this, we also tested LLM Classifiers `ZeroShotGPTClassifier` and `FewShotGPTClassifier` on a subset of the dataset for comparison.")
    
    st.markdown(" ##### Results and Discussion")
    st.markdown(" ###### Traditional ML Models")
    
    
    df = pd.read_csv('data/traditional_ML_NLP.csv')
    df.columns = ["Model", 'Train Accuracy', "Validation Accuracy", "Train F1", "Validation F1", "Run Time (s)"]
    df.set_index(df["Model"], inplace = True)
    df.drop("Model", axis = 1, inplace = True)
    
    # We plot the results
    data_acc = df['Validation Accuracy'].sort_values()
    
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
    
    ax.barh(y = data_acc.index, 
               width = data_acc.values,
               color = ['mediumaquamarine' if x == data_acc.max() else 'lightgray' for x in data_acc.values])
    
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_title("")
    ax.set_yticks(list(data_acc.index))
    ax.set_yticklabels(list(data_acc.index))
    ax.set_title("ML Models by Accuracy Score")
    ax.spines[["top","right", "bottom", "left"]].set_visible(False)
    
    for i in range(len(data_acc)):
        ax.text(
            y  = ax.get_yticks()[i]-0.12,
            x = np.round(data_acc.values[i],2),
            s = f'{np.round(data_acc.values[i]*100,2)}%',
            fontsize = 10)
    
        
    dummy_acc = df.loc['Dummy Classifier', 'Validation Accuracy'] 
    
    # we add the dummy classifier accuracy score as sanity check
    ax.axvline(x = dummy_acc, color = 'lightcoral', alpha = 0.8 , linestyle = '--')
    ax.text(x = dummy_acc-0.25,
               y = -0.8,
               s = f"Dummy Classifier Accuracy: {np.round(dummy_acc * 100, 2)}%",
               color = 'lightcoral')
        
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Accuracy scores of each model in identifying credible from not credible news.")
    
    data_f1 = df['Validation F1'].sort_values()
    
    fig, ax = plt.subplots(1, 1, figsize = (10, 5))
        
    ax.barh(y = data_f1.index, 
               width = data_f1.values,
               color = ['mediumaquamarine' if x == data_f1.max() else 'lightgray' for x in data_f1.values])
    
    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_title("")
    ax.set_yticks(list(data_f1.index))
    ax.set_yticklabels(list(data_f1.index))
    ax.set_title(f"ML Models by F1 Score")
    ax.spines[["top","right", "bottom", "left"]].set_visible(False) 
    
    for j in range(len(data_f1)):
        ax.text(
            y  = ax.get_yticks()[j]-0.12,
            x = np.round(data_f1.values[j],2),
            s = f'{np.round(data_f1.values[j]*100,2)}%',
            fontsize = 10)
    
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("F1 scores of each model in identifying credible from not credible news.")
    
    st.write("In terms of accuracy, we note that most traditional models are able to perform at least better than random chance, but most however are not able to beat the `DummyClassifier` model, which serves as our sanity check model.")
             
    st.write("Recall that we mainly used the Term Frequency-Inverse Document Frequency (TF-IDF) matrix as the features for which we train our traditional ML models with. This method has several limitations:")
    
    st.markdown("**1. TF-IDF vectors are typically very high-dimensional.** This can make it difficult for some ML models to learn effectively, particularly those that don't handle high-dimensional data quite well.")
    st.markdown("**2. TF-IDF only takes into consideration the frequencies of each word in each document and occurences within the corpus**. This may not encapsulate or capture all the relationships between words, the order of words, nuances, and specific contexts.")
    st.markdown("**3. TF-IDF's vocabulary is restricted only within the corpus determined during the training period**. This means that the ML model will not be able to handle an encounter with a new word outside of its vocabularity. This would require retraining the models with a new training data every time, thus making it impractical.")
    
    st.caption("Summary of the scores and run times:")
    st.dataframe(df[['Validation Accuracy', 'Validation F1', "Run Time (s)"]].sort_values(by = 'Validation Accuracy', 
                                                                                         ascending = False))  
    st.markdown(" ###### LLM Classifiers")
    
    st.write("Aside from this, we also tested LLM Classifiers `ZeroShotGPTClassifier` and `FewShotGPTClassifier` on a subset of 200 articles with a distribution of 'Credible' and 'Not Credible' news, comparable to the original dataset. For the Few Shot GPT Classifier, a subset of 20 articles (10 each labeled as 'Credible' and 'Not Credible'), were used to train the model, and the remainder was used for its evaluation." )
    
    st.write("A summary of the results are as follows:")
    
    llm_results = pd.read_csv('data/llm_results.csv') 
    llm_results.columns = ['Model', 'Accuracy', 'F1 Score', 'Run Time (s)']
    data_acc_llm = llm_results['Accuracy']
    data_f1_llm = llm_results['F1 Score']
    
    metrics = ['Accuracy', 'F1 Score']
    
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 3), sharey = True)
    
    for i, data in enumerate([data_acc_llm, data_f1_llm]):
    
        ax[i].bar(x = data.index, height = data.values,
                color = ['steelblue' if x == data.max() else 'lightgray' for x in data.values])
        ax[i].spines[['left', 'right', 'top', 'bottom']].set_visible(False)
        ax[i].set_xticks(list(data.index))
        ax[i].set_xticklabels(['Zero Shot', 'Few Shot'], fontsize = 8)
        ax[i].set_ylabel('')
        ax[i].set_yticks([])
        ax[i].set_xlabel(f"{metrics[i]}")
        
        for j in range(len(data)):
            ax[i].text(
                x  = ax[i].get_xticks()[j]-0.12,
                y = np.round(data.values[j],2)+0.01,
                s = f'{np.round(data.values[j]*100,2)}%')
            
    fig.suptitle("LLM Performance")
    fig.tight_layout()
    st.pyplot(fig)
    
    st.write("The findings show that large language models, which are trained on vast amounts of data and are more complex in design, also did not seem to fair any better in terms of their accuracy. Of note, LLM's are superior to the traditional ML models in terms of the F1 score. On the other hand, the run time for these models are considerably much longer (see below).")
    st.write()
    
    st.dataframe(llm_results.set_index('Model'))
    
    st.markdown("We note the limitations of LLM's ability to assess a news articles' credibility:")
    
    st.markdown("1. A **broader context or external references** may be needed to ascertain the accuracy or credibility of the claims made in an article. Information might be incomplete, misleading, or taken out of context.")
    
    st.markdown("2. **Identifying bias in an article is a complex task** which required broad domain knowledge, and information on the author's, sources', and publication's potential biases; LLM's might be able to pick up on thes subtle nuances.")
    
    st.markdown("3. **Verifying the factual accuracy of statements in an article typically requires cross-referencing with other reliable sources**. Analyzing content alone limits this fact-checking ability.")
    
    st.markdown("4. **Some misinformation are crafted to appear highly credible**, using legitimate sources out of context or blending true information with falsehoods in a convincing manner. These subtle differences can be missed easily.")
    
    st.markdown(" ##### Summary and Conclusion")
    st.markdown("To summarize, _all of the models have varying degrees of difficulty in differentiating credible from not credible news based on the content of the material alone_. The GPT Classifiers had the best F1 score amongst all the model, while also having the longest run time.")
    
    st.markdown("*__These findings highlight the complexity involved in the very heavy task of differentiating real, credible news from 'fake' news.__* Further measures must be put in place to ensure proper assessment of the credibility of news articles.")
    
    ## I'll try to add here yung choose a model and see their predictions part if may time tonight, i-paste lang below this divider
    st.divider()


def interactive_quiz():
    df = pd.read_csv('data/Philippine Fake News Corpus - cleaned.csv')
    df_ui = pd.read_csv('data/verified_samples.csv')
    
    st.title("Test Yourself! :brain:")
    st.header("Can You Identify Which Claims are More Likely to be False?")
    
    st.write("Below are headlines of a few examples of verified and fact-checked articles. Choose a headline below to view its contents.")
    st.write("A few statements derived from this article will be given to you. Take your time and read through the article. For each claim or statement, decide whether or not this is more likely true :white_check_mark:, or more likely false :x:. Click on the button to see if your answer is correct.")
    st.write("At the end of each section, you can also try to see what chatGPT had to say :nerd_face:.")
    
    @st.cache_resource(show_spinner = "loading response...please wait :hourglass:")
    def verify_claim(article, claim, label):
        system_prompt = 'You are a news analyst tasked to fact-check the claims made in news articles.'
        main_prompt = f"""
        The {article} is said to be {label}.
        Do not mention the word 'Series'.
        If {claim} is said to be {label}, give an explanation for the average reader why this is so. 
        Explain this as if you were trying to educate the average reader in a respectful tone."""
        response = client.chat.completions.create(
            model='gpt-3.5-turbo', 
            messages = [{'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': main_prompt}])
        return response.choices[0].message.content
    
    title = st.selectbox('', df_ui['Headline'].unique(), index=None)
    try:
        if title:
            sample_df = df_ui[df_ui['Headline'] == title]
            article = df[df['Headline'] == title]
            article_content = df[df['Headline']==title].iloc[0]['Content']
            st.markdown("*Please read the entire article down below to help you decide.*")
            with st.expander("Here is the full article: click to expand"):
                st.header(title)
                st.write(article_content)
        
        statements = [text for text in sample_df['Statements']]
        verified_claims = [text for text in sample_df['Cross reference']]
        
        for i, text in enumerate(statements):
            st.markdown(f"##### Statement #{i+1}: {text}")
            choice = st.radio(label = '',
                              options = ['This is more likely True.', 'This is more likely False.'],
                              key = f"radio_{i}")
            if st.button("Check Answer", key = f"button_{i}", type = "primary"):
                if (choice == 'This is more likely True.') & (sample_df.iloc[i]['Verification']):
                    st.write("**You are correct!**")
                elif (choice == 'This is more likely False.') & ~(sample_df.iloc[i]['Verification']):
                    st.write("**You are correct!**")
                else:
                    st.write("**Sorry! Unfortunately, your choice was wrong. Please try again.**")
                
                if sample_df.iloc[i]['Verification']:
                    st.write("**This statement is True.**")
                else:
                    st.write("**This statement is False.**")
                st.write(verified_claims[i])
                response = verify_claim(article = article_content, claim = text, label = sample_df.iloc[i]['Label'])
                st.write("*Want to see what else chatGPT have to say?*")
                with st.expander("Click to see what else chatGPT had to say."):
                    st.write(response)
            st.divider()
        
        with st.expander("Read the :warning:Disclaimer:warning:"):
            st.markdown("#### DISCLAIMER:")
            st.write("The information and responses provided by this application are generated using OpenAI's ChatGPT. While we strive to provide accurate and helpful information, it is important to note the following: ")
            st.markdown("1. **Accuracy:** The responses generated by ChatGPT are based on patterns and information from a wide range of sources, but they may not always be accurate, complete, or up-to-date. Users should independently verify any information provided by the application with credible news sources.")
            st.markdown("2. **Context:** ChatGPT may not always fully understand the context of the news article, which can lead to responses that may be off-topic or out of context.")
            st.markdown("3. **Bias:** The responses may reflect biases present in the training data. OpenAI has made efforts to reduce such biases, but they may still occasionally be present in the output.")
            st.markdown("4. **Limitations:** ChatGPT cannot provide legal, medical, financial, or other professional advice. This application's purpose is to _guide the reader's discernment when encountering a news article containing facts or statements which may or may not be factual or credible._")
            st.markdown("5. **Responsibility:** The use of this application is at your own risk. We are not responsible for any consequences arising from the use of the information provided by ChatGPT.")
            st.markdown("By using this application, you acknowledge and accept these limitations and agree to use the information provided responsibly.")
    except:
        pass
 

def main():

    my_page = st.sidebar.radio('Page Navigation',
                           ['About the data', 'Interactive highlights', 
                            'News summarization', 
                            'ML News Classifier',
                            'Interactive quiz: Test your knowledge'])



    if my_page == "About the data":
        about_the_data()
    elif my_page == "Interactive highlights":
        interactive_highlights()
    elif my_page == "News summarization":
        news_summarization()
    elif my_page == "ML News Classifier":
        ml_news_classifier()
    elif my_page == "Interactive quiz: Test your knowledge":
        interactive_quiz()
        
if __name__ == "__main__":
    main()
