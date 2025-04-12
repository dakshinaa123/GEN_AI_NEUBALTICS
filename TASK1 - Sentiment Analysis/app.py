import gradio as gr
from transformers import pipeline

# Load the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text):
    result = sentiment_pipeline(text)[0]
    return f"{result['label']} ({result['score']:.4f})"

# Custom CSS for styling
custom_css = """
body {
    background-color: #111;
    color: white;
    text-align: center;
}
h1 {
    color: white;
}
"""

# Create the Gradio Interface
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("<h1>Sentiment Analysis API</h1>")
    gr.Markdown("Enter a sentence, and the model will predict if it’s **POSITIVE or NEGATIVE**.")

    with gr.Column():
        input_text = gr.Textbox(label="Enter Text")
        output_text = gr.Textbox(label="Sentiment Output", interactive=False)

    with gr.Row():
        submit_btn = gr.Button("Submit")
        clear_btn = gr.Button("Clear")

    submit_btn.click(analyze_sentiment, inputs=input_text, outputs=output_text)
    clear_btn.click(lambda: ("", ""), inputs=[], outputs=[input_text, output_text])

# Launch the app
if __name__ == "__main__":
    demo.launch()
