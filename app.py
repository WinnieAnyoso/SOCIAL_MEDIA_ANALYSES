
#

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
from plotly import tools
import plotly.offline as py
import plotly.express as px

st.set_option('deprecation.showPyplotGlobalUse', False)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

##############################################################################################################################


@st.cache
def load_data():
    """loading data"""
    df=pd.read_csv("/Users/winnieanisa/Downloads/Moringa  Technical Projects/facebook_clean.csv",index_col="text")

    num_df=df.select_dtypes(['float','int'])
    num_cols=num_df.columns

    
    text_df=df.select_dtypes(['object'])
    text_cols=text_df.columns

    
    # year_col=df['year']
    # unique_years=year_col.unique()

    return df,num_cols,text_cols

df,num_cols,text_cols=load_data()

#Dashboard Title

st.title("**♟**Moringa School Social Media Analysis Facebook**♟**") 

#Dataset on Dashboard

if st.checkbox('Show Dataset'):
   st.write(df)

#sidebar title

st.sidebar.title("Dashboard Settings")
feature_selection=st.sidebar.multiselect(label="Features to plot",options=num_cols)
print('feature_selection')

######################################################################################################################################################

#Average Likes Per Year 
cohort_df = df.groupby(by=['year']).agg({"likes":'mean'})
cohort_df = cohort_df.reset_index()

#Plotly visualization on the average likes per year
fig = px.bar(data_frame=cohort_df, x='year', y='likes',
                title = 'Average likes per year',
                text = cohort_df['likes'],
                hover_data= ['year', 'likes'],
                color_discrete_sequence=["royalblue","green"], )
fig.update_traces(
                  texttemplate='%{text:.3s}',
                  textposition='inside',
                  )
st.plotly_chart(fig)

##############################################################################################################################

#Sum of comments Per Year 
cohort_df = df.groupby(by=['year']).agg({"comments":'sum'})
cohort_df = cohort_df.reset_index()

#Plotly visualization on the average likes per year
fig = px.bar(data_frame=cohort_df, x='year', y='comments',
                title = 'Sum of comments per year',
                text = cohort_df['comments'],
                hover_data= ['year', 'comments'],
                color_discrete_sequence=["royalblue","green"], )
fig.update_traces(
                  texttemplate='%{text:.3s}',
                  textposition='inside',
                  )
st.plotly_chart(fig)


#################################################################################################################################

#Sum of Likes Per Year 
cohort_df = df.groupby(by=['year']).agg({"likes":'sum'})
cohort_df = cohort_df.reset_index()

#Plotly visualization on the average likes per year
fig = px.bar(data_frame=cohort_df, x='year', y='likes',
                title = 'Sum of likes per year',
                text = cohort_df['likes'],
                hover_data= ['year', 'likes'],
                color_discrete_sequence=["royalblue","green"], )
fig.update_traces(
                  texttemplate='%{text:.3s}',
                  textposition='inside',
                  )
st.plotly_chart(fig)

################################################################################################################################

grouped_fb = df.groupby('year')['likes'].sum().reset_index(name="likes of posts per year")
grouped_fb

x_axis = st.sidebar.selectbox('Which value do you want to explore?', num_cols)

fig = px.scatter(df,
                x=x_axis,
                y='likes',
                title=f'Likes vs. {x_axis}')

st.plotly_chart(fig)

######################################################################################################################################################
# WORDCLOUD

st.set_option('deprecation.showPyplotGlobalUse', False)

def cloud(image, text, max_word, max_font, random):

    wc = WordCloud(background_color="white", colormap="hot", max_words=max_word, mask=image,
                stopwords=stopwords, max_font_size=max_font, random_state=random)
    # generate word cloud
    wc.generate(text)

    # create coloring from image
    image_colors = ImageColorGenerator(image)

    # show
    plt.figure(figsize=(100,100))
    fig, axes = plt.subplots(1,2, gridspec_kw={'width_ratios': [3, 2]})
    axes[0].imshow(wc, interpolation="bilinear")
    # recolor wordcloud and show
    # we could also give color_func=image_colors directly in the constructor
   # axes[1].imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
    axes[1].imshow(image, cmap=plt.cm.gray, interpolation="bilinear")
    for ax in axes:
        ax.set_axis_off()
    
    st.pyplot()
    
   


def main():
    st.write("# Text Summarization with a WordCloud")
    max_word = st.sidebar.slider("Max words", 200, 3000, 200)
    max_font = st.sidebar.slider("Max Font Size", 50, 350, 60)
    random = st.sidebar.slider("Random State", 30, 100, 42 )
    image = st.file_uploader("Choose a file(preferably a silhouette)")
    text = st.text_area("Add text ..")
    if image and text is not None:
        if st.button("Plot"):
            st.write("### Original image")
            image = np.array(Image.open(image))
            # st.image(image, width=100, use_column_width=True)
       
            st.write("### Word cloud")
            st.write(cloud(image, text, max_word, max_font, random), use_column_width=True)


if __name__=="__main__":
    main()

st.sidebar.subheader("About")


if st.sidebar.button("Project Editor"):
    st.sidebar.text("Rose Delilah Gesicho")
    st.sidebar.text("rose.delilah@gmail.com")
  
