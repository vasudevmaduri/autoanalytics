import streamlit as st
from PIL import Image
import time

import numpy as np
import pandas as pd
import sys
import os
from  PIL import Image

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
# from config import DBConfig
import datetime
import time

from model.classify import prediction
#plotly related
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

import os
import glob
import tensorflow as tf
import  _thread
import time
# We'll render HTML


st.set_page_config(page_title = "Auto Analytics", page_icon = "🚘", layout = 'wide')

max_width_str = f"max-width: 2000px;"
st.markdown(
    f"""
    <style>
    .reportview-container .main .block-container{{{max_width_str}}}
    </style>""", unsafe_allow_html=True
)
pd.options.plotting.backend = "plotly"

image = Image.open("Dashboard//logo.png")
st.sidebar.image(image)

###Read Data
@st.cache()
def get_sapres():
    df = pd.read_csv("Dashboard//data//spare_parts_final.csv")
    return df

##Read Sales data
@st.cache()
def get_sales():
    df = pd.read_csv("Dashboard//data//car_sales.csv")
    return df

##Read Forecast Data 
@st.cache()
def get_forecast():
    df= pd.read_csv("Dashboard//data//stocks_data.csv")
    df["Date"] = pd.to_datetime(df['Date'])
    return df

@st.cache()
def get_gtrends():
    df= pd.read_csv("Dashboard//data//gtrends_car_data.csv")
    df["Date"] = pd.to_datetime(df['Date'])
    return df

# st.sidebar.image(image, user_column_width = False, width = 150 )

def auto_analysis():

    st.markdown("<h2 style='text-align: center; color: black;'><b>Automobile Analysis 📉📈<b></h1>", unsafe_allow_html=True)
    
    df_sales = get_sales()
    col1, col2  = st.columns(2)
    with col1:
        df_multi = pd.read_csv("Dashboard//data//multiTimeline.csv")
        df_multi["Week"] = pd.to_datetime(df_multi['Week'])
        df_multi.sort_values(by=['Week'], inplace=True, ascending=False)
        # st.dataframe(df_multi)
        fig = px.line(df_multi, x="Week", y="Car",
                title='Automobile Trends')
        fig.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    },
                    )
        
        st.plotly_chart(fig)
    
    with col2:
        df_top = pd.read_csv("Dashboard//data//bybrandspares.csv")
        df_top = df_top.head(10)
        df_top = df_top.sort_values(['count'], ascending=True)
        fig = px.bar(df_top, x='count', y='brand', text='count')
        fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
        fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                },
                )
        st.plotly_chart(fig)
    with col1:
        fig = px.line(df_sales, x="Date", y=df_sales.columns,
              title='Sales by Brand')
        fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                },
                )

    
        st.plotly_chart(fig)

    df_stocks = get_forecast()
    with col2:
        # st.write(df_stocks)
        temp = df_stocks.loc[df_stocks['Brand'] == "Hyundai"]
        temp.sort_values(by=['Date'], inplace=True, ascending=False)
        fig = px.line(temp, x="Date", y="Close",
              title='Stock Forecast by Brand - {}'.format("Hyundai"))
        fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                },
                )
        st.plotly_chart(fig)
    
    df_trends = get_gtrends()
    
    # st.write(df_trends)
    temp1 = df_trends.loc[df_trends['keyword'] == "Hyundai"]
    temp1.sort_values(by=['Date'], inplace=True, ascending=False)
    fig = px.line(temp1, x="Date", y="data",
              title='Trends by Brand - {}'.format("Hyundai"))
    fig.update_layout({
                'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                },
                )
    st.plotly_chart(fig)

############# Spare Parts Analysis #############

def spare_parts():
    st.markdown("<h2 style='text-align: center; color: black;'><b>Spare Parts Analysis 📉📈<b></h1>", unsafe_allow_html=True)
    df = get_sapres()
    col1, col2, col3, col4 = st.columns([2,3,2,1])
    try:
        product_1 =  sorted(df["product_1"].unique())
        product_1.insert(0, "All Products")
        with col1:
            lvl_1 = st.selectbox("Select product Type", options = product_1, index = product_1.index("All Products"))
        
        analysis_df = df if lvl_1 == "All Products" else df[df["product_1"]==lvl_1]
        
        product_2 = ((analysis_df["product"].unique()).tolist())
        product_2.insert(0, "All Products")
        with col2:
            lvl_2 = st.selectbox("Select product Type", options = product_2, index = product_2.index("Front Bumper Bracket "))

        analysis_df = analysis_df if lvl_2 == "All Products" else df[df["product"]==lvl_2]
        brand = ((analysis_df["brand"].unique()).tolist())
        brand.insert(0, "All Brands")
        with col3:
            lvl_3 = st.selectbox("Select Vehicle Brand", options = brand, index = brand.index("All Brands"))

        analysis_df = analysis_df if lvl_3 == "All Brands" else df[df["brand"]==lvl_3]
        model = ((analysis_df["model"].unique()).tolist())
        model.insert(0, "All Models")
        with col4:
            lvl_4 = st.selectbox("Select Vehicle Model", options = model, index = model.index("All Models"))
        
        analysis_df = analysis_df if lvl_4 == "All Models" else df[df["model"]==lvl_4]

        # st.dataframe(analysis_df)

        c1,c2 = st.columns(2)
        t = analysis_df
        t = t.groupby("model").sum(["new_price_","old_price_"])
        with c1:
            fig = go.Figure()
            fig.add_trace(go.Bar(y=t.index,
                            x=t["new_price_"],
                            name='New Price',
                            marker_color='rgb(26, 118, 255)', orientation='h'
                            ))

            fig.add_trace(go.Bar(y=t.index,
                            x=t["old_price_"],
                            name='Old Price',
                            marker_color='rgb(55, 83, 109)', orientation='h'
                            ))

            fig.update_layout(
                title='Price Analysis for Front Bumper Bracket for brand HONDA',
                xaxis_tickfont_size=14,
                yaxis=dict(
                    title='INR Rupees',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                barmode='group',
                bargap=0.15, # gap between bars of adjacent location coordinates.
                bargroupgap=0.1 # gap between bars of the same location coordinate.
            )
            fig.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    },
                    )
            st.plotly_chart(fig)
        
        with c2:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(y=t.index,
                            x=t["pct_change"],
                            name='New Price',
                            marker_color='rgb(26, 118, 255)', orientation='h'
                            ))


            fig1.update_layout(
                title='Price Analysis for Front Bumper Bracket for brand HONDA',
                xaxis_tickfont_size=14,
                yaxis=dict(
                    title='INR Rupees',
                    titlefont_size=16,
                    tickfont_size=14,
                ),
                barmode='group',
                bargap=0.15, # gap between bars of adjacent location coordinates.
                bargroupgap=0.1 # gap between bars of the same location coordinate.
            )
            fig1.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    },
                    )
            st.plotly_chart(fig1)
        
        with c1:
            #Avg Price change
            avg_change = analysis_df["pct_change"].mean()
            color = "green" if avg_change > 50 else "red"
            gauge_fig_1 = go.Figure()
            gauge_fig_1.add_trace(
                    go.Indicator(
                    mode = "gauge+number",
                    value = avg_change,
                    domain = {'x': [0, 1], 'y': [0.5, 1]},
                    title = {'text': 'Percentage Change'},
            #         delta = {'reference': delta},
                    gauge = {'bar': {'color': color},'axis': {'range': [0, 100]}})
                                )
            gauge_fig_1.update_layout({
                    'plot_bgcolor': 'rgba(0, 0, 0, 0)',
                    'paper_bgcolor': 'rgba(0, 0, 0, 0)',
                    },
                    )
            gauge_fig_1.update_layout(height=600, width=650)
            st.plotly_chart(gauge_fig_1)
        
        # with c2:
        #     temp_df = analysis_df[analysis_df["product"] == lvl_2]
        #     st.dataframe(temp_df)
    
    except:
        st.write("No Data found for the selected combination")
        img = Image.open("Dashboard//no.png")
        st.image(img)
        # st.button()




############# Twitter Analyser###############
def twitter_analyser():

    with st.spinner("**Sit Back**, Autoanalytics is reasoning 🧠 "):
        time.sleep(1)
    st.markdown("<h2 style='text-align: center; color: black;'>Twitter Analyser📨</h2>", unsafe_allow_html=True)
    @st.cache(allow_output_mutation=True, show_spinner=False)
    def get_con():
        # USER = "postgres"
        # PWORD = "zOpKJDxG13fFFkFx"#
        # HOST = "35.197.148.152"#"34.136.184.102"
        # return create_engine('postgresql://{}:{}@{}/postgres'.format(USER, PWORD, HOST),
        #                     convert_unicode=True)
        db_user = os.environ["DB_USER"]
        db_pass = os.environ["DB_PASS"]
        db_name = os.environ["DB_NAME"]
        db_socket_dir = os.environ.get("DB_SOCKET_DIR", "/cloudsql")
        cloud_sql_connection_name = os.environ["CLOUD_SQL_CONNECTION_NAME"]

        pool = create_engine(

            # Equivalent URL:
             'postgresql+pg8000://{}:{}@/{}?unix_sock={}/{}/.s.PGSQL.5432'.format(db_user,db_pass,db_name,db_socket_dir,cloud_sql_connection_name),convert_unicode=True
          
        )
        return pool


    @st.cache(allow_output_mutation=True, show_spinner=False, ttl=5*60)
    def get_data():
        timestamp = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        df = pd.read_sql_table('tweets', get_con())
        df = df.rename(columns={'body': 'Tweet', 'tweet_date': 'Timestamp',
                                'followers': 'Followers', 'sentiment': 'Sentiment',
                                'keyword': 'Subject'})
        return df, timestamp


    @st.cache(show_spinner=False)
    def filter_by_date(df, start_date, end_date):
        df_filtered = df.loc[(df.Timestamp.dt.date >= start_date) & (df.Timestamp.dt.date <= end_date)]
        return df_filtered


    @st.cache(show_spinner=False)
    def filter_by_subject(df, subjects):
        return df[df.Subject.isin(subjects)]


    @st.cache(show_spinner=False)
    def count_plot_data(df, freq):
        plot_df = df.set_index('Timestamp').groupby('Subject').resample(freq).id.count().unstack(level=0, fill_value=0)
        plot_df.index.rename('Date', inplace=True)
        plot_df = plot_df.rename_axis(None, axis='columns')
        return plot_df


    @st.cache(show_spinner=False)
    def sentiment_plot_data(df, freq):
        plot_df = df.set_index('Timestamp').groupby('Subject').resample(freq).Sentiment.mean().unstack(level=0, fill_value=0)
        plot_df.index.rename('Date', inplace=True)
        plot_df = plot_df.rename_axis(None, axis='columns')
        return plot_df


    

    data, timestamp = get_data()

    # st.header('Twitter Analyser')
    # st.markdown("<h1 style='text-align: center; color: black;'>Twitter Analyser📨{}</h1>", unsafe_allow_html=True)
    st.write('Total tweet count: **{}**'.format(data.shape[0]))
    st.write('Data last loaded {} (In GMT +0 Timezone)'.format(timestamp))

    # col1, col2 = st.beta_columns(2)

    date_options = data.Timestamp.dt.date.unique()
    start_date_option = st.sidebar.selectbox('Select Start Date', date_options, index=0)
    end_date_option = st.sidebar.selectbox('Select End Date', date_options, index=len(date_options)-1)

    keywords = data.Subject.unique()
    keyword_options = st.sidebar.multiselect(label='Subjects to Include:', options=keywords.tolist(), default=keywords.tolist())

    data_subjects = data[data.Subject.isin(keyword_options)]
    data_daily = filter_by_date(data_subjects, start_date_option, end_date_option)

    top_daily_tweets = data_daily.sort_values(['Followers'], ascending=False).head(10)

    plot_freq_options = {
        'Hourly': 'H',
        'Four Hourly': '4H',
        'Daily': 'D'
    }
    plot_freq_box = st.sidebar.selectbox(label='Plot Frequency:', options=list(plot_freq_options.keys()), index=0)
    plot_freq = plot_freq_options[plot_freq_box]

    st.subheader('Tweet Volumes')
    plotdata = count_plot_data(data_daily, plot_freq)
    st.line_chart(plotdata)

    st.subheader('Sentiment')
    plotdata2 = sentiment_plot_data(data_daily, plot_freq)
    st.line_chart(plotdata2)

    
    st.subheader('Influential Tweets')
    st.dataframe(top_daily_tweets[['Tweet', 'Timestamp', 'Followers', 'Subject']].reset_index(drop=True), 1000, 400)

    st.subheader('Recent Tweets')
    st.table(data_daily[['Tweet', 'Timestamp', 'Followers', 'Subject']].sort_values(['Timestamp'], ascending=False).
                reset_index(drop=True).head(10))

def detector():
    st.markdown("<h2 style='text-align: center; color: black;'><b>Car Damage Detector🚘 - Spare Parts <b></h1>", unsafe_allow_html=True)
    st.markdown("***", unsafe_allow_html = True)
    col1, col2, col3 = st.columns(3)
    col1.markdown("<h3 style='text-align: center; color: black;'>{}</h1>".format("Type"), unsafe_allow_html=True)
    col2.markdown("<h3 style='text-align: center; color: black;'>Severity Level</h1>", unsafe_allow_html=True)
    col3.markdown("<h3 style='text-align: center; color: black;'>Score</h1>", unsafe_allow_html=True)
    c1 = col1.empty()
    c2=col2.empty()
    c3=col3.empty()
    img = col2.empty()
    image_file  = st.file_uploader("Upload an image", type=["jpg","jpeg"])
    if image_file is not None:
        with open(image_file.name,"wb") as f:
            f.write(image_file.getbuffer())
        header = st.empty()
        st.success("Saved File")
        image_path = max(glob.glob(r'*.jpg'), key=os.path.getctime)
        with tf.Graph().as_default():
            human_string, score= prediction(image_path)
        print('model one value' + str(human_string))
        print('model one value' + str(score))
        if (human_string == 'car'):
            color = "green" if score < 50 else "red"
            c1.markdown("<h3 style='text-align: center; color: green;'>{}</h1>".format(human_string), unsafe_allow_html=True)
            c2.markdown("<h3 style='text-align: center; color: black;'>NA</h1>", unsafe_allow_html=True)
            c3.markdown("<h3 style='text-align: center; color: {};'>{}</h1>".format(color, score), unsafe_allow_html=True)
            label_text = 'This is not a damaged car with confidence ' + str(score) + '%. Please upload a damaged car image'
            print(image_path)
            # return render_template('front.html', text = label_text, filename= image_path)
        elif (human_string == 'low'):
            color = "green" if score < 50 else "red"
            if color == "red":
                lvl = "High"
            else:
                lvl  = "Low"
            c1.markdown("<h3 style='text-align: center; color: green;'>{}</h1>".format("Car"), unsafe_allow_html=True)
            c2.markdown("<h3 style='text-align: center; color: {};'>{}</h1>".format(color, lvl), unsafe_allow_html=True)
            c3.markdown("<h3 style='text-align: center; color: {};'>{}</h1>".format(color, score), unsafe_allow_html=True)
            label_text = 'This is a low damaged car with '+ str(score) + '% confidence.'
            print(image_path)
            # return render_template('front.html', text = label_text, filename= image_path)
        elif (human_string == 'high'):
            color = "green" if score < 50 else "red"
            c1.markdown("<h3 style='text-align: center; color: green;'>{}</h1>".format("Car"), unsafe_allow_html=True)
            c2.markdown("<h3 style='text-align: center; color: green;'>High</h1>", unsafe_allow_html=True)
            c3.markdown("<h3 style='text-align: center; color: {};'>{}</h1>".format(color, score), unsafe_allow_html=True)
            label_text = 'This is a low damaged car with '+ str(score) + '% confidence.'
            label_text = 'This is a high damaged car with '+ str(score) + '% confidence.'
            print(image_path)
            # return rende/r_template('front.html', text = label_text, filename= image_path)
        elif (human_string == 'not'):
            color = "green" if score < 50 else "red"
            c1.markdown("<h3 style='text-align: center; color: red;'>{}</h1>".format(" Not a Car"), unsafe_allow_html=True)
            c2.markdown("<h3 style='text-align: center; color: green;'>NA</h1>", unsafe_allow_html=True)
            c3.markdown("<h3 style='text-align: center; color: {};'>{}</h1>".format(color, score), unsafe_allow_html=True)
            label_text = 'This is a low damaged car with '+ str(score) + '% confidence.'
            label_text = 'This is not the image of a car with confidence ' + str(score) + '%. Please upload the car image.'
            print(image_path)
        img.image(image_file, width = 300)
        os.remove(image_path)

user_preference_options = [  "Automobile Analysis", "Spare part Analysis🔍", "Reporting Live Data 🔴📡 ", "Car Damage Detector🚘"]
user_preference = st.sidebar.radio(label="Want to know?", options=user_preference_options, index=0)

if user_preference == user_preference_options[3]:
    detector()
elif user_preference == user_preference_options[2]:
    twitter_analyser()
elif user_preference == user_preference_options[0]:
    auto_analysis()
elif user_preference == user_preference_options[1]:
    spare_parts()


st.sidebar.title("About")
st.sidebar.info(
        """
        This project is created & maintained by **Auto Analytics Team**. You can learn more about us at
        [our documentation](https://github.com/vasudevmaduri/autoanalytics/tree/dev).

        **Our Team**:
        - [Ashish Shingla](https://www.linkedin.com/in/ashish--singla)
        - [Charika Bhatia](https://www.linkedin.com/in/charika-bhatia-21569a96)
        - [Mahima Sharma](https://www.linkedin.com/in/mahima-sharma/)
        - [Sudha Saini](http://linkedin.com/in/sudha-saini)
        - [Vasudev Maduri](https://www.linkedin.com/in/vasudevmaduri/)
"""
    )