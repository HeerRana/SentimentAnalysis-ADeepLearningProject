# ── Load and Predict — No retraining needed ──────────────────
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
import os

print("Loading model...")

# Load the base model
model = keras.models.load_model("sentiment_model.keras")

# Rebuild the vectorization layer and load its config
vectorize_layer = layers.TextVectorization(
    max_tokens=20000,
    output_mode="int",
    output_sequence_length=250,
)

# Load saved vocabulary
with open("vectorizer_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
vectorize_layer.set_vocabulary(vocab)

# Build export model with vectorizer included
export_model = keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation("linear"),
])

print("Model loaded successfully!")
print("="*60)
print("   SENTIMENT CLASSIFIER — IMDb Movie Review")
print("="*60)

def predict_sentiment(review_text):
    prediction = export_model.predict([review_text], verbose=0)[0][0]
    confidence  = prediction if prediction > 0.5 else 1 - prediction
    sentiment   = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
    print(f"\nReview    : {review_text[:80]}...")
    print(f"Sentiment : {sentiment}")
    print(f"Confidence: {confidence * 100:.1f}%")
    print(f"Raw Score : {prediction:.4f}")
    print("-" * 60)

while True:
    user_input = input("\nEnter a movie review (or type 'quit' to exit):\n> ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Exiting. Goodbye!")
        break
    if len(user_input.strip()) == 0:
        print("Please enter some text.")
        continue
    predict_sentiment(user_input)