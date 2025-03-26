import gradio as gr
import json
from entity_risk_analysis import analyze_text

def analyze_and_format(input_text):
    result = analyze_text(input_text)
    return json.dumps(result, indent=2)

with gr.Blocks() as demo:
    gr.Markdown("""
    # 🛡️ AI-Powered Entity Risk Analyzer
    Analyze entities in transaction descriptions for risk scores and justifications.
    """)
    input_text = gr.Textbox(lines=4, label="Enter Transaction Description")
    analyze_btn = gr.Button("Analyze")
    output = gr.Code(label="Risk Analysis JSON Output")
    analyze_btn.click(fn=analyze_and_format, inputs=input_text, outputs=output)

if __name__ == "__main__":
    demo.launch()
