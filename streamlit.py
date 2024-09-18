import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest").to('cuda')

# Store reviews (this simulates the Flask -> Streamlit data transfer)
if 'reviews' not in st.session_state:
    st.session_state.reviews = []

# Define function to get sentiment scores
def get_sentiment_scores(reviews):
    encoded_inputs = tokenizer(reviews, padding=True, truncation=True, return_tensors="pt").to('cuda')
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    scores = torch.softmax(outputs.logits, dim=1).cpu().numpy()
    
    # Print scores and shape for debugging
    st.write("Scores shape:", scores.shape)
    st.write("Scores:", scores)
    
    return scores

# Get query parameters
query_params = st.query_params

if 'reviews' in query_params:
    reviews = query_params['reviews']
    st.session_state.reviews = reviews

# Display the Streamlit app UI
st.title("Product Review Sentiment Analysis")

if st.session_state.reviews:
    reviews = st.session_state.reviews
    st.write(f"Reviews submitted from Flask")

    # Perform sentiment analysis
    scores = get_sentiment_scores(reviews)

    # Check the shape of scores and adjust the labels accordingly
    if scores.shape[1] == 2:
        labels = ["Negative", "Positive"]  # Model provides only two sentiment categories
    elif scores.shape[1] == 3:
        labels = ["Negative", "Neutral", "Positive"]  # Model provides three sentiment categories

    sentiment_df = pd.DataFrame(scores * 100, columns=labels)

    # Calculate the percentages for the sentiments
    avg_sentiment = sentiment_df.mean()

    # Create the data for the Plotly gauge chart
    sentiment_data = {
        'Sentiment': labels,
        'Percentage': avg_sentiment.tolist()
    }
    df = pd.DataFrame(sentiment_data)

    # Identify the sentiment with the highest percentage
    max_sentiment = df.loc[df['Percentage'].idxmax()]

    # Define the color zones for the gauge
    color_zones = {
        "Negative": [0, 33],
        "Neutral": [33, 66],
        "Positive": [66, 100]
    }
    
    # Determine the end value and color for the gauge
    end_value = avg_sentiment[labels.index(max_sentiment['Sentiment'])]
    color = "red" if max_sentiment['Sentiment'] == "Negative" else "green"

    # Define the animation frames for the gauge
    frames = []
    for i in np.linspace(0, end_value, 50):
        frames.append(go.Frame(
            data=[go.Indicator(
                mode="gauge+number",
                value=i,
                title={'text': f"Sentiment: {max_sentiment['Sentiment']}"},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "black"},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 33], 'color': "red"},
                        {'range': [33, 66], 'color': "yellow"},
                        {'range': [66, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 5},
                        'thickness': 0.8,
                        'value': i
                    },
                }
            )],
            name=f"Frame {int(i)}"
        ))

    # Create initial figure for the Plotly gauge
    fig = go.Figure(
        data=[go.Indicator(
            mode="gauge+number",
            value=end_value,
            title={'text': f"Sentiment: {max_sentiment['Sentiment']}"},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "black"},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 33], 'color': "red"},
                    {'range': [33, 66], 'color': "yellow"},
                    {'range': [66, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 5},
                    'thickness': 0.8,
                    'value': end_value
                },
            }
        )],
        layout=go.Layout(
            title={'text': "Animated Sentiment Gauge"},
            updatemenus=[{
                'buttons': [
                    {
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate', 'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'direction': 'left',
                'pad': {'r': 10, 't': 87},
                'showactive': True,
                'type': 'buttons',
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                'active': 0,
                'currentvalue': {'prefix': 'Needle Position: '},
                'pad': {'b': 10},
                'steps': [{'label': f"{int(i)}", 'method': 'animate', 'args': [[f"Frame {int(i)}"], {'mode': 'immediate', 'frame': {'duration': 50, 'redraw': True}}]} for i in np.linspace(0, end_value, 50)]
            }]
        ),
        frames=frames
    )

    # Display the animated gauge chart in Streamlit
    st.plotly_chart(fig)

    # Show the sentiment percentages
    st.write("### Sentiment Percentages")
    st.write(df)

else:
    st.write("No reviews submitted yet.")
