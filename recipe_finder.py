# File: recipe_finder_real_data_translated.py
import pandas as pd
from llama_index.legacy import VectorStoreIndex, Document  # Updated import from legacy
from langchain_community.vectorstores import Chroma  # Updated Chroma import
from langchain_huggingface import HuggingFaceEmbeddings  # Updated HuggingFaceEmbeddings import

import streamlit as st
##
import nltk
import os
import zipfile

"""
# Define a directory for downloading NLTK data (create it if it doesn't exist)
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download the stopwords corpus to the custom directory
nltk.download('stopwords', download_dir=nltk_data_dir)
"""


# Define the path to the zip file and extraction directory
zip_file_path = os.path.join(os.getcwd(), 'nltk_data.zip')
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')

# Extract the zip file if the directory doesn't exist
if not os.path.exists(nltk_data_dir):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(nltk_data_dir)

# Point NLTK to the extracted directory
nltk.data.path.append(nltk_data_dir)

#


# Step 1: Load and Preprocess the Recipe Data from CSV
def load_and_preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    
    df_selected = df[['RecipeName', 'Ingredients', 'PrepTimeInMins', 
                      'CookTimeInMins', 'TotalTimeInMins', 'Servings', 'Cuisine', 
                      'Course', 'Diet', 'Instructions', 'URL']]
    df_cleaned = df_selected.dropna(subset=['RecipeName', 'Ingredients', 'Instructions'])
    return df_cleaned

# Step 2: Convert DataFrame into Document Format for Indexing
def create_recipe_documents_from_df(df):
    documents = []
    for index, row in df.iterrows():
        content = f"Title: {row['RecipeName']}\nIngredients: {row['Ingredients']}\nInstructions: {row['Instructions']}"
        documents.append(Document(text=content))
    return documents

# Step 3: Create Embeddings using HuggingFace Model for Chroma
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")  # Pass model name directly

# Step 4: Store Recipe Embeddings in ChromaDB
def store_embeddings_in_chromadb(documents):
    texts = [doc.get_content() for doc in documents]
    vectorstore = Chroma.from_texts(texts, embedding=embeddings)
    return vectorstore

# Step 5: Query Recipes by Ingredients
def find_recipe_by_ingredients(ingredients, vectorstore):
    query = ", ".join(ingredients)
    results = vectorstore.similarity_search(query)
    
    # Extract unique results based on content to avoid duplicates
    unique_recipes = []
    seen_contents = set()
    
    for result in results:
        content = result.page_content  # Extract content of the recipe
        if content not in seen_contents:
            seen_contents.add(content)
            unique_recipes.append(content)
    
    return unique_recipes

# Streamlit App
def main():
    st.title("Recipe Finder with Ingredient Search")

    # Load the CSV file internally (provide the correct path)
    csv_file_path = 'IndianFood-cleaned.csv'  # Path to your CSV file
    recipe_data = load_and_preprocess_data(csv_file_path)

    # Convert DataFrame into documents for indexing
    documents = create_recipe_documents_from_df(recipe_data)

    # Store recipe embeddings in ChromaDB
    vectorstore = store_embeddings_in_chromadb(documents)

    # Input ingredients
    user_ingredients = st.text_input("Enter ingredients separated by commas", value="chicken, garlic, lemon")

    # When the user presses the button, search for matching recipes
    if st.button("Find Recipes"):
        if user_ingredients:
            # Split user input into a list of ingredients
            ingredients_list = [ingredient.strip() for ingredient in user_ingredients.split(",")]

            # Find matching recipes
            matching_recipes = find_recipe_by_ingredients(ingredients_list, vectorstore)

            # Display the matching recipes
            st.write("### Matching Recipes")
            if matching_recipes:
                for i, recipe in enumerate(matching_recipes):
                    # Display the clean recipe information
                    st.write(f"**Recipe {i+1}:**")
                    st.write(recipe)
            else:
                st.write("No matching recipes found.")

if __name__ == "__main__":
    main()