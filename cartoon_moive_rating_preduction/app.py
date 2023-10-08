from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import joblib

app = Flask(__name__)

# Load the pre-trained random forest model
data = pd.read_csv('anime.csv', escapechar='\\')
data['episodes'] = data['episodes'].replace('Unknown', 0)
X = data[['genre', 'type', 'episodes', 'members']]
y = data['rating']
y.fillna(y.mean(), inplace=True)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(X[['genre', 'type']])
X_encoded = encoder.transform(X[['genre', 'type']])
encoded_feature_names = encoder.get_feature_names_out(['genre', 'type'])
X_encoded_df = pd.DataFrame(X_encoded, columns=encoded_feature_names)
X_final = pd.concat([X_encoded_df, X[['episodes', 'members']]], axis=1)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_final, y)

# Function to predict ratings
def predict_rating(genre, type_, episodes, members):
    input_data = pd.DataFrame({
        'genre': [genre],
        'type': [type_],
        'episodes': [int(episodes)],
        'members': [int(members)]
    })

    input_encoded = encoder.transform(input_data[['genre', 'type']])
    input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)
    input_final = pd.concat([input_encoded_df, input_data[['episodes', 'members']]], axis=1)

    predicted_rating = rf_model.predict(input_final)
    return predicted_rating[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            genre = request.form['genre']
            type_ = request.form['type']
            episodes = request.form['episodes']
            members = request.form['members']

            predicted_rating = predict_rating(genre, type_, episodes, members)

            return render_template('result.html', predicted_rating=predicted_rating)

        except Exception as e:
            error_message = "An error occurred. Please try again later."
            return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
