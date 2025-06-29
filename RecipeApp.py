!pip install -q streamlit

import streamlit as st
import pandas as pd
import ast
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import hstack


!git clone https://github.com/B-Wayne00/CapFiles.git
%cd CapFiles/RecipeApp.py

# ---------- Load and Preprocess Data ----------
@st.cache_data
def load_data():
    df = pd.read_csv('RecipesFin.csv')

    if 'cuisine_type' not in df.columns:
        st.error("The column 'cuisine_type' is missing in RecipesFin.csv")
        st.stop()

    df['ingredient_list'] = df['ingredients'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else [])

    modifiers = {
        'ground', 'fresh', 'chopped', 'minced', 'diced', 'low-fat', 'boneless',
        'skinless', 'sliced', 'crushed', 'cooked', 'raw', 'shredded', 'frozen',
        'grated', 'large', 'small', 'extra', 'light', 'lean', 'reduced-fat', 'whole'
    }

    def simplify_ingredient(ingredient):
        words = ingredient.lower().split()
        filtered = [word for word in words if word not in modifiers]
        return ' '.join(filtered) if filtered else ingredient.lower()

    df['simplified_ingredients'] = df['ingredient_list'].apply(
        lambda lst: [simplify_ingredient(ing) for ing in lst]
    )
    df['ingredient_string'] = df['simplified_ingredients'].apply(lambda lst: ' '.join(sorted(lst)))

    df = df.drop_duplicates(subset='ingredient_string').reset_index(drop=True)

    vectorizer_dedupe = TfidfVectorizer()
    X_dedupe = vectorizer_dedupe.fit_transform(df['ingredient_string'])
    cos_sim = cosine_similarity(X_dedupe)
    threshold = 0.9
    to_remove = set()
    for i in range(cos_sim.shape[0]):
        for j in range(i + 1, cos_sim.shape[1]):
            if cos_sim[i, j] > threshold:
                to_remove.add(j)

    df = df.drop(df.index[list(to_remove)]).reset_index(drop=True)

    all_ingredients = [ing for sublist in df['simplified_ingredients'] for ing in sublist]
    ingredient_counts = Counter(all_ingredients)
    frequent_ingredients = {ing for ing, count in ingredient_counts.items() if count >= 5}

    df['filtered_ingredients'] = df['simplified_ingredients'].apply(
        lambda lst: [ing for ing in lst if ing in frequent_ingredients]
    )
    df['ingredient_string'] = df['filtered_ingredients'].apply(lambda lst: ' | '.join(lst))

    vectorizer = TfidfVectorizer(
        tokenizer=lambda x: x.split('|'),
        lowercase=False,
        ngram_range=(1, 2),
        token_pattern=None
    )
    X_ingredients = vectorizer.fit_transform(df['ingredient_string'])

    encoder = OneHotEncoder(handle_unknown='ignore')
    X_cuisine = encoder.fit_transform(df[['cuisine_type']])

    X = hstack([X_ingredients, X_cuisine])

    kmeans = KMeans(n_clusters=8, init='k-means++', random_state=42)
    df['cluster'] = kmeans.fit_predict(X)

    knn_model = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn_model.fit(X)

    return df, knn_model, X


# ---------- Recommendation Function ----------
def recommend_similar_recipes(recipe_name, df, knn_model, feature_matrix, n_recommendations=5):
    try:
        recipe_idx = df.index[df['name'] == recipe_name].tolist()[0]
    except IndexError:
        return []

    recipe_vector = feature_matrix[recipe_idx]
    distances, indices = knn_model.kneighbors(recipe_vector, n_neighbors=n_recommendations + 1)
    recommended_indices = indices.flatten()[1:]  # skip the recipe itself

    recommendations = []
    for idx, dist in zip(recommended_indices, distances.flatten()[1:]):
        rec_name = df.iloc[idx]['name']
        rec_cuisine = df.iloc[idx]['cuisine_type']
        recommendations.append({
            'Recipe Name': rec_name,
            'Cuisine': rec_cuisine,
            'Similarity Score': f"{(1 - dist):.3f}"
        })

    return recommendations


# ---------- Streamlit UI ----------
st.set_page_config(page_title="Recipe Recommender", layout="centered")
st.title("🍽️ Recipe Recommender")

df, knn_model, X = load_data()

st.markdown("You can either:")
st.markdown("1. **Select a cuisine and recipe**, _or_  \n2. **Enter a User ID** (for future personalization)")

st.divider()

# Option 1: Choose a cuisine and recipe
st.subheader("🔍 Option 1: Choose a Cuisine and a Recipe")

cuisines = sorted(df['cuisine_type'].unique())
selected_cuisine = st.selectbox("Select a cuisine:", cuisines)

filtered_df = df[df['cuisine_type'] == selected_cuisine]
recipe_names = filtered_df['name'].sort_values().unique()
selected_recipe = st.selectbox("Select a recipe:", recipe_names)

# Option 2: Enter user ID
st.subheader("👤 Option 2: Enter Your User ID")
user_id = st.text_input("User ID (optional):")

# Trigger recommendations
if st.button("Get Recommendations"):
    if user_id:
        st.info(f"🔐 Personalized recommendations for User ID **{user_id}** coming soon!")
        # Placeholder for future user-personalized logic
    else:
        recommendations = recommend_similar_recipes(selected_recipe, df, knn_model, X)
        if recommendations:
            st.success(f"Top recommendations similar to **{selected_recipe}**:")
            st.table(recommendations)
        else:
            st.warning("No recommendations found for the selected recipe.")
