# app.py
import sys
import os
# Add the parent directory to the path so we can import model.py and db.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import gradio as gr
import numpy as np
import pickle
from model import Perceptron  # Import the custom model definition
from db import create_table, insert_submission  # Import our DB functions

# Ensure the database table exists (this creates the table if it doesn't exist)
create_table()

# Load the trained model from the models folder
model_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "models", "perp.pkl")
with open(r'models/perp.pkl', "rb") as f:
    perceptron_model = pickle.load(f)

def predict_employability(name, degree, course, general_appearance, manner_speaking,
                          mental_alertness, self_confidence, present_ideas, comm_skills):
    """
    Takes student details and ratings, predicts employability using the perceptron model,
    stores the submission in the database, and returns a message.
    """
    # Construct the input feature vector (order must match training)
    X = np.array([[general_appearance, manner_speaking, mental_alertness,
                    self_confidence, present_ideas, comm_skills]]) / 5.0
    
    # Get the raw prediction (a probability)
    raw_prediction = perceptron_model.predict(X)
    # Use .item() to extract scalar value before rounding
    predicted_class = 1 if raw_prediction.item() > 0.5 else 0
    
    # Determine the result message based on prediction
    if predicted_class == 1:
        result = f"😊 Yeah, {name}! You are Good to be Employed !!!!"
    else:
        result = f"☹️ Sorry, {name}. You need to Work On Yourself !!!"
    
    # Store the submission in the database
    insert_submission(name, degree, course, general_appearance, manner_speaking,
                      mental_alertness, self_confidence, present_ideas, comm_skills, result)
    
    return result

# Create the Gradio Interface with inputs matching the training features
demo = gr.Interface(
    fn=predict_employability,
    inputs=[
        gr.Textbox(label="Name"),
        gr.Textbox(label="Degree"),
        gr.Textbox(label="Course"),
        gr.Slider(0, 5, step=1, label="General Appearance"),
        gr.Slider(0, 5, step=1, label="Manner of Speaking"),
        gr.Slider(0, 5, step=1, label="Mental Alertness"),
        gr.Slider(0, 5, step=1, label="Self-Confidence"),
        gr.Slider(0, 5, step=1, label="Ability to Present Ideas"),
        gr.Slider(0, 5, step=1, label="Communication Skills")
    ],
    outputs=gr.Textbox(label="Employability Result"),
    title="Student Employability Checker",
    description="""Enter your details and rate yourself on the given criteria.
    Click "Submit" to see if you're employable or need more practice!"""
)

if __name__ == "__main__":
    # Set share=True to create a public link if needed
    demo.launch(share=True)
