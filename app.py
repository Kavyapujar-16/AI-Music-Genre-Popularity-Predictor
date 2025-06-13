# 🎵 AI Music Classifier & Popularity Predictor
# ✅ Full code with final improvements (bold clear result)

# 📦 Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from sklearn.metrics.pairwise import cosine_similarity

# ✅ MUST BE FIRST Streamlit command
st.set_page_config(page_title="Music Genre & Popularity Predictor", layout="wide")

# 🔧 Set Background Image Function (dynamic based on page)
def set_background(image_path=None):
    if image_path is not None:
        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{encoded}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
                color: white !important;
            }}
            .stMarkdown, .stHeader, .stDataFrame, .stTable {{
                color: white !important;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        # Plain black background for Welcome page
        st.markdown(
            """
            <style>
            .stApp {
                background-color: black;
                color: white !important;
            }
            .stMarkdown, .stHeader, .stDataFrame, .stTable {
                color: white !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

# ✅ Setup Session State
if "page" not in st.session_state:
    st.session_state.page = "Welcome"

# 🚀 Welcome Page
if st.session_state.page == "Welcome":
    set_background()  # Plain black

    st.markdown("<h1 style='text-align: center;'>🎵 Welcome To Spotify Music Genre Prediction App!</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>'Check the Genre of your Track!'</h3>", unsafe_allow_html=True)

    # Button CSS → nice look
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #1DB954;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
        display: block;
        margin: 0 auto;
    }
    div.stButton > button:hover {
        background-color: #1ED760;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    # "Let's Go" button
    if st.button("🎬 Let's Go!"):
        st.session_state.page = "Main"

    # Spotify logo → centered → bigger
    with open("spotify_logo.jpg", "rb") as img_file:
        logo_encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <img src="data:image/png;base64,{logo_encoded}" style="display:block;margin-left:auto;margin-right:auto;width:600px;">
        """,
        unsafe_allow_html=True
    )

# 🚀 Main App
else:
    set_background("background.png")  # Now use background.png

    # 🎯 Load Models and Transformers
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    genre_model = joblib.load('genre_classifier.pkl')
    pop_model = joblib.load('popularity_regressor.pkl')

    # 🎨 Page Heading
    st.title("🎶 AI Music Classifier & Popularity Predictor")
    st.markdown("""
    Welcome to the **AI-powered Music Intelligence App**! 🎧 

    > This app allows you to:
    > - Predict a track's **Genre** 🎼 and **Popularity Score** 📊
    > - Upload a **CSV of tracks** for batch predictions
    > - Visualize the predicted genres and popularity distribution
    > - Recommend similar tracks based on audio features

    ---
    """)

    # 📋 Feature Input UI
    st.sidebar.header("🎛️ Input Audio Features")
    input_mode = st.sidebar.radio("Choose Input Mode", ["Single Track", "Batch CSV Upload"])

    feature_list = [
        'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness',
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms'
    ]

    # 🎯 Predict Single Track with Confidence %
    def predict_single(features):
        features_scaled = scaler.transform([features])
        genre_index = genre_model.predict(features_scaled)[0]
        proba = genre_model.predict_proba(features_scaled)[0]
        confidence = np.max(proba)
        genre_pred = le.inverse_transform([genre_index])[0]
        pop_pred = int(pop_model.predict(features_scaled)[0])
        return genre_pred, confidence, pop_pred

    # 🎵 Single Track Prediction
    if input_mode == "Single Track":
        user_inputs = []
        for feat in feature_list:
            val = st.sidebar.slider(f"{feat.replace('_', ' ').title()}", 0.0, 1.0, 0.5) if feat not in ['key', 'loudness', 'mode', 'tempo', 'duration_ms'] else \
                st.sidebar.number_input(f"{feat.replace('_', ' ').title()}", value=0.0)
            user_inputs.append(val)

        if st.sidebar.button("🔍 Predict Track Info"):
            genre, confidence, popularity = predict_single(user_inputs)

            # ✅ Use bold clear result blocks
            st.markdown(
                f"""
                <div style="background-color:#1db954; padding:12px; border-radius:10px; margin-bottom:10px;">
                <h4 style="color:white; text-align:left; font-size:20px;">🎧 Predicted Genre: <b>{genre}</b> ({confidence*100:.2f}% confidence)</h4>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div style="background-color:#5353ec; padding:12px; border-radius:10px; margin-bottom:10px;">
                <h4 style="color:white; text-align:left; font-size:20px;">🔥 Predicted Popularity Score: <b>{popularity}/100</b></h4>
                </div>
                """,
                unsafe_allow_html=True
            )

    # 📂 Batch Prediction
    else:
        uploaded_file = st.sidebar.file_uploader("Upload CSV with Audio Features", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.subheader("📄 Uploaded Data Preview")
            st.dataframe(df.head())

            if all(col in df.columns for col in feature_list):
                scaled = scaler.transform(df[feature_list])
                df['Predicted Genre'] = le.inverse_transform(genre_model.predict(scaled))
                df['Predicted Popularity'] = pop_model.predict(scaled).astype(int)

                st.subheader("✅ Prediction Results")
                st.dataframe(df[['Predicted Genre', 'Predicted Popularity']].head())

                # 🎨 Genre Distribution
                st.subheader("🎼 Genre Distribution (Bar Chart)")
                genre_counts = df['Predicted Genre'].value_counts()
                fig1, ax1 = plt.subplots(figsize=(4, 2), facecolor='none')
                ax1.set_facecolor('none')
                sns.barplot(x=genre_counts.values, y=genre_counts.index, palette='viridis', ax=ax1)
                ax1.set_xlabel("Number of Tracks", fontsize=7)
                ax1.set_ylabel("Genre", fontsize=7)
                ax1.xaxis.label.set_color('white')
                ax1.yaxis.label.set_color('white')
                ax1.tick_params(axis='x', colors='white', labelsize=6)
                ax1.tick_params(axis='y', colors='white', labelsize=6)
                st.pyplot(fig1, use_container_width=True)

                # 📊 Popularity Score Distribution
                st.subheader("📈 Popularity Score Distribution")
                fig2, ax2 = plt.subplots(figsize=(4, 2), facecolor='none')
                ax2.set_facecolor('none')
                sns.histplot(df['Predicted Popularity'], bins=20, kde=True, ax=ax2, color='skyblue')
                ax2.set_xlabel("Popularity", fontsize=7)
                ax2.set_ylabel("Count", fontsize=7)
                ax2.xaxis.label.set_color('white')
                ax2.yaxis.label.set_color('white')
                ax2.tick_params(axis='x', colors='white', labelsize=6)
                ax2.tick_params(axis='y', colors='white', labelsize=6)
                st.pyplot(fig2, use_container_width=True)

                # 📁 Download Predictions
                st.download_button(
                    "⬇️ Download Predictions as CSV",
                    data=df.to_csv(index=False),
                    file_name="music_predictions.csv",
                    mime="text/csv"
                )
            else:
                st.error("Uploaded CSV is missing required audio feature columns.")

    # 🚀 Content-Based Recommendation System
    st.markdown("---")
    st.header("🎧 Content-Based Recommendation System")

    df_full = pd.read_csv("spotify_data.csv")
    df_full.drop(columns=['Unnamed: 0'], inplace=True)
    top_20_genres = df_full['track_genre'].value_counts().head(20).index.tolist()
    df_filtered = df_full[df_full['track_genre'].isin(top_20_genres)].copy()
    df_filtered_sample = df_filtered.sample(n=1000, random_state=42).reset_index(drop=True)
    X_scaled_sample = scaler.transform(df_filtered_sample[feature_list])
    similarity_matrix = cosine_similarity(X_scaled_sample)

    track_selected = st.selectbox("🎵 Select a Track:", df_filtered_sample['track_name'].unique())
    track_index = df_filtered_sample[df_filtered_sample['track_name'] == track_selected].index[0]

    def recommend_similar_tracks_streamlit(track_index, df_original, similarity_matrix, top_n=5):
        similarity_scores = similarity_matrix[track_index]
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        recommendations = df_original.iloc[similar_indices][['track_name', 'artists', 'track_genre']]
        return recommendations

    if st.button("🔍 Recommend Similar Tracks"):
        st.write(f"🎧 Recommendations for: **{track_selected}**")
        recs = recommend_similar_tracks_streamlit(track_index, df_filtered_sample, similarity_matrix)
        st.dataframe(recs)
