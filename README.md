# Recipe Finder with Ingredient Search

This project allows users to search for recipes based on the ingredients they have. It uses a machine learning model from HuggingFace to create embeddings for recipes and stores them in ChromaDB for efficient ingredient-based searching. The user interface is built with Streamlit, allowing users to input ingredients and receive a list of matching recipes.

## Features

- **Recipe Data Preprocessing:** Clean and format recipe data from a CSV file.
- **Document Creation:** Convert the recipe data into documents suitable for embedding.
- **Embeddings with HuggingFace:** Use the `sentence-transformers/all-MiniLM-L6-v2` model to generate recipe embeddings.
- **Recipe Search:** Search for recipes by ingredients using similarity search on the stored embeddings in ChromaDB.
- **Streamlit UI:** A simple web-based UI for users to input ingredients and receive recipe suggestions.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>

2. Install dependencies: Make sure to have Python installed and run the following command to install the necessary packages:
pip install -r requirements.txt

3. Prepare the data:

Ensure you have a CSV file containing the recipe data. In the script, the file is named IndianFood-cleaned.csv. Place your CSV file in the project directory.

4. Run the Streamlit app
streamlit run recipe_finder.py

5. Using the app:

Input ingredients (e.g., "tomato, garlic, lemon") in the text box.
Press "Find Recipes" to see a list of matching recipes.

## Data Format

Ensure your CSV file includes the following columns:

- **RecipeName**: The name of the recipe.
- **Ingredients**: Ingredients used in the recipe.
- **PrepTimeInMins**: Preparation time in minutes.
- **CookTimeInMins**: Cooking time in minutes.
- **TotalTimeInMins**: Total time (preparation + cooking) in minutes.
- **Servings**: Number of servings.
- **Cuisine**: Cuisine type (e.g., Indian, Italian, etc.).
- **Course**: Type of course (e.g., Appetizer, Main Course, Dessert).
- **Diet**: Type of diet (e.g., Vegetarian, Vegan, Non-Vegetarian).
- **Instructions**: Step-by-step cooking instructions.
- **URL**: A URL to the recipe for further details (optional).

## Future Enhancements

- Support for more complex queries (e.g., exclude certain ingredients).
- Integration with external recipe APIs.
- Support for multiple embedding models.

