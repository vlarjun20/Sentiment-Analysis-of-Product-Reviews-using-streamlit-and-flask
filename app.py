from flask import Flask, render_template, request, redirect
import requests

app = Flask(__name__)

# Route to display the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle form submission
@app.route('/submit', methods=['POST'])
def submit():
    reviews = [request.form.get(f'review{i}') for i in range(1, 6)]  # Get 5 reviews
    # Send reviews as query parameters to the Streamlit page
    reviews_query = "&".join([f'reviews={review}' for review in reviews])
    return redirect(f"http://localhost:8501/?{reviews_query}")  # Redirect to Streamlit

if __name__ == '__main__':
    app.run(port=5000)
