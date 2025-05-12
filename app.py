import pandas as pd
from flask import Flask, request, render_template
from sklearn.tree import DecisionTreeClassifier

# Initialize Flask app
app = Flask(__name__)

# Load and clean the data
df = pd.read_csv('vgsales.csv')  # Replace with your file
df.dropna(inplace=True)  # Drop rows with missing values

# Create target column: 1 if sales > 1 million, else 0
df['Is_Hit'] = df['Global_Sales'].apply(lambda x: 1 if x > 1.0 else 0)

# Select features for the model (exclude 'Name', 'Global_Sales', and target)
features = ['Platform', 'Year', 'Genre', 'Publisher',
            'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

# Prepare the feature matrix (X) and target vector (y)
X = df[features]
y = df['Is_Hit']

# One-hot encode categorical columns (Platform, Genre, Publisher)
X_encoded = pd.get_dummies(X, columns=['Platform', 'Genre', 'Publisher'])

# Train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_encoded, y)

# Function to recommend a similar hit game
def recommend_game(df, genre, platform, publisher):
    filtered = df[
        (df['Genre'] == genre) &
        (df['Platform'] == platform) &
        (df['Publisher'] == publisher) &
        (df['Is_Hit'] == 1)
    ]
    if not filtered.empty:
        return filtered.iloc[0]['Name']
    else:
        fallback = df[df['Is_Hit'] == 1]
        if not fallback.empty:
            return fallback.sample(1).iloc[0]['Name']
        return "No hit game found"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input data from the form
        game_name = request.form['game_name']
        platform = request.form['platform']
        year = int(request.form['year'])
        genre = request.form['genre']
        publisher = request.form['publisher']
        na_sales = float(request.form['na_sales'])
        eu_sales = float(request.form['eu_sales'])
        jp_sales = float(request.form['jp_sales'])
        other_sales = float(request.form['other_sales'])

        # Prepare the data for prediction
        input_data = {
            'Platform': platform,
            'Year': year,
            'Genre': genre,
            'Publisher': publisher,
            'NA_Sales': na_sales,
            'EU_Sales': eu_sales,
            'JP_Sales': jp_sales,
            'Other_Sales': other_sales
        }

        # One-hot encode and prepare the input features
        input_df = pd.DataFrame([input_data])
        input_encoded = pd.get_dummies(input_df, columns=['Platform', 'Genre', 'Publisher'])

        # Match columns with training data
        missing_cols = set(X_encoded.columns) - set(input_encoded.columns)
        for col in missing_cols:
            input_encoded[col] = 0
        input_encoded = input_encoded[X_encoded.columns]

        # Predict using the model
        prediction = model.predict(input_encoded)
        probability = model.predict_proba(input_encoded)[:, 1][0]

        # Calculate implied global sales
        predicted_sales = na_sales + eu_sales + jp_sales + other_sales

        # Get prediction result
        prediction_result = "HIT 🎯" if prediction[0] == 1 else "Not a Hit 😞"

        # Recommend similar hit game based on input
        recommended_game = recommend_game(df, genre, platform, publisher)

        # Render the result
        return render_template('index.html', 
                               prediction=prediction_result, 
                               probability=f"{probability*100:.2f}%", 
                               global_sales=f"{predicted_sales:.2f} million",
                               recommended_game=recommended_game)

    except KeyError as e:
        return f"Missing key: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
