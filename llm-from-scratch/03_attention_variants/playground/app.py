"""Interactive playground: prompt -> generation with adjustable temperature /
top-k / top-p, plus a live bar chart of the next-token probability distribution.

Usage:
    python app.py
"""
import gradio as gr
import torch


def predict_next_token_distribution(prompt, temperature, top_k, top_p):
    # TODO: load a trained checkpoint + tokenizer, run one forward pass,
    # apply temperature/top-k/top-p, return a dict {token_str: probability}
    # for the top ~20 candidates so gr.BarPlot can render it.
    raise NotImplementedError


def build_app():
    with gr.Blocks() as demo:
        prompt = gr.Textbox(label="Prompt")
        temperature = gr.Slider(0.1, 2.0, value=1.0, label="Temperature")
        top_k = gr.Slider(0, 100, value=0, step=1, label="Top-k (0 = disabled)")
        top_p = gr.Slider(0.0, 1.0, value=1.0, label="Top-p")
        output = gr.BarPlot(label="Next-token probability")
        run_btn = gr.Button("Predict next token")
        run_btn.click(
            predict_next_token_distribution,
            inputs=[prompt, temperature, top_k, top_p],
            outputs=output,
        )
    return demo


if __name__ == "__main__":
    build_app().launch()
