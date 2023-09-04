import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans 
#Progreebar
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu

css = """
    <style>
        .stSlider label {
            font-size: 12px; 
        }
    </style>
"""

st.markdown(css, unsafe_allow_html=True)
# Load the preprocessed Spotify dataset (you can use your own dataset)
data = pd.read_csv("spotify_dataset.csv")

# Preprocess data and create feature matrix X and target vector y
# (Assuming you already have the data preprocessed)

# defining predictors
X = data[["energy","danceability","liveness","acousticness","instrumentalness"]]
#x = np.array(x).reshape(-1, 1)

# target variable
y = data['popularity']
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the RandomForestRegressor model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Function to predict song popularity based on input features
def predict_popularity(acousticness, danceability, energy, instrumentalness, liveness, speechiness, tempo, valence):
    features = np.array([[acousticness, danceability, energy, instrumentalness, liveness]])
    prediction = model.predict(features)
    return prediction[0]


# Function to add a banner image
def add_banner():
    st.image("Spotify-Banner-1.png", use_column_width=True)


#----------------------------------------------Recommendation system--------------------------------------------------

# function to normalize columns
def normalize_column(col):
    """
    col - column in the dataframe which needs to be normalized
    """
    max_d = data[col].max()
    min_d = data[col].min()
    data[col] = (data[col] - min_d)/(max_d - min_d)

#spotify_dataset.drop(['duration_min'],inplace=True,axis=1)
#spotify_dataset.drop(['year'],inplace=True,axis=1)

#Normalize allnumerical columns so that min value is 0 and max value is 1
num_types = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
num = data.select_dtypes(include=num_types)
        
for col in num.columns:
    normalize_column(col)

# k-means clustering 
km = KMeans(n_clusters=25)
pred = km.fit_predict(num)
data['pred'] = pred
normalize_column('pred')

class Song_Recommender():
    """
    Neighbourhood Based Collaborative Filtering Recommendation System using similarity Metrics
    Manhattan Distance is calculated for all songs and Recommend Songs that are similar to it based on any given song
    """
    def __init__(self, data):
        self.data_ = data
    
    # function which returns recommendations; you can also choose the number of songs to be recommended
    def get_recommendations(self, song_name, n_top):
        distances = []
        # choosing the given song_name and dropping it from the data
        song = self.data_[(self.data_['track'].str.lower() == song_name.lower())].head(1).values[0]
        rem_data = self.data_[self.data_['track'].str.lower() != song_name.lower()]
        
        for r_song in tqdm(rem_data.values):
            dist = 0
            for col_index, col_value in enumerate(r_song):
                # indices of non-numerical columns (id, Release date, name, artists)
                if col_index not in [3, 7, 13] and isinstance(col_value, (int, float)):
                    dist += abs(float(song[col_index]) - float(col_value))
            distances.append(dist)
        
        rem_data['distance'] = distances
        # sorting our data in ascending order by the 'distance' feature
        rem_data = rem_data.sort_values('distance')
        columns = ['artists', 'track']
        return rem_data[columns][:n_top]
    
recommender = Song_Recommender(data)

#--------------------------------------------------------------------------

# Streamlit app
def spr_sidebar():
    menu=option_menu(
        menu_title=None,
        options=['Home','Visualizations','About','Log'],
        icons=['house','book','info-square','terminal'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal'
    )
    if menu=='Home':
        st.session_state.app_mode = 'Home'
    elif menu=='Visualizations':
        st.session_state.app_mode = 'Visualizations'
    elif menu=='About':
        st.session_state.app_mode = 'About'
    elif menu=='Log':
        st.session_state.app_mode = 'Log'
        
css = r'''
    <style>
        [data-testid="stForm"] {border: 0px}
    </style>
'''

st.markdown(css, unsafe_allow_html=True)
    
def home_page():
    st.title("Spotify Song Popularity Prediction and Recommendation")
    st.subheader("Enter songs and get recommendations")
    st.sidebar.title("Try predicting popularity based on audio features !")
    
    # Define the sidebar form components
    with st.sidebar.form(key='prediction_form'):
        st.slider("Acousticness", 0.0, 1.0, 0.5, key='acousticness')
        st.slider("Danceability", 0.0, 1.0, 0.5, key='danceability')
        st.slider("Energy", 0.0, 1.0, 0.5, key='energy')
        st.slider("Instrumentalness", 0.0, 1.0, 0.5, key='instrumentalness')
        st.slider("Liveness", 0.0, 1.0, 0.5, key='liveness')
        st.slider("Speechiness", 0.0, 1.0, 0.5, key='speechiness')
        st.slider("Tempo", 50, 200, 120, key='tempo')
        st.slider("Valence", 0.0, 1.0, 0.5, key='valence')
        submitted = st.form_submit_button("Predict Popularity")
        
    # Check if the form is submitted
    if submitted:
        with st.spinner("Predicting..."):
            prediction = predict_popularity(
                st.session_state.acousticness,
                st.session_state.danceability,
                st.session_state.energy,
                st.session_state.instrumentalness,
                st.session_state.liveness,
                st.session_state.speechiness,
                st.session_state.tempo,
                st.session_state.valence
            )
            st.sidebar.success(f"Predicted Song Popularity: {prediction:.2f}")

    with st.form(key='recommendation_form'):
        song_name = st.text_input("Enter song name:")
        song_num = st.slider("Number of Songs", 1, 20, 10)
        submitted = st.form_submit_button("Recommend")
        
    if submitted:
        with st.spinner("Loading..."):
            recommendations = recommender.get_recommendations(song_name,song_num)
            st.success("Recommended Songs:")
            st.write(recommendations)
        
 
def Log_page():
    log=st.checkbox('Display Output', True, key='display_output')
    if log == True:
     if 'err' in st.session_state:
        st.write(st.session_state.err)
    with open('spotify_dataset.csv') as f:
        st.download_button('Download Dataset', f,file_name='spotify_dataset.csv')
            
def About_page():
    st.header('Development')
    """
    Check out the [repository](https://github.com/abdelrhmanelruby/Spotify-Recommendation-System) for the source code and approaches, and don't hesitate to contact me if you have any questions. I'm excited to read your review.
    [Github](https://github.com/abdelrhmanelruby)  [Linkedin](https://www.linkedin.com/in/abdelrhmanelruby/) Email : maniyalilbaijuakshaya@gmail.com
    """
    st.subheader('Spotify Million Playlist Dataset')
    """
    For this project, I'm using the Million Playlist Dataset, which, as its name implies, consists of one million playlists.
    contains a number of songs, and some metadata is included as well, such as the name of the playlist, duration, number of songs, number of artists, etc.
    """
    st.subheader('Audio Features Explanation')

def Visualizations_page():
    st.header('Visualization Dashboard')
    with st.spinner("Loading..."):
        st.image("Dashboard_spotify.png", use_column_width=True)
    
def main():
    
    add_banner()  

    spr_sidebar()        
    if st.session_state.app_mode == 'Home':
        home_page()
    if st.session_state.app_mode == 'Visualizations':
        Visualizations_page()
    if st.session_state.app_mode == 'About' :
        About_page()
    if st.session_state.app_mode == 'Log':
        Log_page()

if __name__ == "__main__":
    main()







