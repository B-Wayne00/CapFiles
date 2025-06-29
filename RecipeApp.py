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
import matplotlib.pyplot as plt

# ---------- Load and Preprocess Data ----------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/B-Wayne00/CapFiles/main/RecipesFin.csv"
    df = pd.read_csv(url)

    if 'cuisine_type' not in df.columns:
        st.error("Missing 'cuisine_type' column.")
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


# ---------- Recommendation Logic ----------
def recommend_similar_recipes(recipe_name, df, knn_model, feature_matrix, n_recommendations=5):
    try:
        recipe_idx = df.index[df['name'] == recipe_name].tolist()[0]
    except IndexError:
        return []

    recipe_vector = feature_matrix[recipe_idx]
    distances, indices = knn_model.kneighbors(recipe_vector, n_neighbors=n_recommendations + 1)
    recommended_indices = indices.flatten()[1:]

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

# ---------- Visualization Functions ----------
def plot_elbow(X):
    inertia = []
    K_range = range(2, 16)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X)
        inertia.append(km.inertia_)

    fig, ax = plt.subplots()
    ax.plot(K_range, inertia, marker='o')
    ax.set_title("Elbow Method for Optimal k")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    st.pyplot(fig)

def plot_svd_clusters(X, df):
    svd = TruncatedSVD(n_components=2, random_state=42)
    X_reduced = svd.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], c=df['cluster'], cmap='tab10', alpha=0.7)
    ax.set_title("2D Visualization of Recipe Clusters")
    ax.set_xlabel("SVD Component 1")
    ax.set_ylabel("SVD Component 2")
    st.pyplot(fig)

def plot_common_ingredients(df):
    all_ings = [ing for sublist in df['filtered_ingredients'] for ing in sublist]
    top_ingredients = Counter(all_ings).most_common(10)
    ingredients, counts = zip(*top_ingredients)

    fig, ax = plt.subplots()
    ax.barh(ingredients[::-1], counts[::-1], color='salmon')
    ax.set_title("Top 10 Most Common Ingredients")
    ax.set_xlabel("Frequency")
    st.pyplot(fig)

def plot_cuisine_distribution(df):
    cuisine_counts = df['cuisine_type'].value_counts()

    fig, ax = plt.subplots()
    cuisine_counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Recipe Count per Cuisine Type")
    ax.set_ylabel("Number of Recipes")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# ---------- Streamlit Layout ----------
st.set_page_config(layout='wide')
st.title("🥘 Recipe Recommendation App")

df, knn_model, X = load_data()

# Sidebar filters
st.sidebar.header("🔎 Filters & Options")
selected_cuisine = st.sidebar.selectbox("Filter by Cuisine", ['All'] + sorted(df['cuisine_type'].unique()))
user_id = st.sidebar.text_input("Optional User ID for Personalization")

# Filter recipes
filtered_df = df if selected_cuisine == 'All' else df[df['cuisine_type'] == selected_cuisine]
recipe_options = sorted(filtered_df['name'].unique())
selected_recipe = st.selectbox("Select a Recipe", recipe_options)

if st.button("🔍 Get Recommendations"):
    if selected_recipe:
        st.subheader(f"Recommendations similar to **{selected_recipe}**")
        if user_id:
            st.caption(f"(Personalization not yet enabled. Using generic model.)")
        recs = recommend_similar_recipes(selected_recipe, df, knn_model, X)
        if recs:
            st.dataframe(pd.DataFrame(recs))
        else:
            st.warning("No similar recipes found.")

# Visualizations section
st.markdown("---")
st.header("📊 Visual Insights")

tab1, tab2, tab3, tab4 = st.tabs(["Cluster Elbow Plot", "SVD Clusters", "Top Ingredients", "Cuisine Distribution"])

with tab1:
    plot_elbow(X)

with tab2:
    plot_svd_clusters(X, df)

with tab3:
    plot_common_ingredients(df)

with tab4:
    plot_cuisine_distribution(df)
