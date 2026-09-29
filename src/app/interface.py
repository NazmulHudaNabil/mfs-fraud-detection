import gradio as gr
import requests
import os

PORT = os.getenv("PORT", "8000")
API_URL = f"http://127.0.0.1:{PORT}/predict"

def predict_fraud(step, type_val, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest):
    payload = {
        "step": int(step),
        "type": type_val,
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest)
    }
    
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            result = response.json()
            pred = result["prediction"]
            prob = result["fraud_probability"]
            
            return f"Result: {pred}\nFraud Probability: {prob:.2%}"
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"Failed to connect: {str(e)}"

# Clean, simple professional theme
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

with gr.Blocks(theme=custom_theme, title="MFS Fraud Detection") as demo:
    gr.Markdown("# MFS Fraud Detection Simulator")
    gr.Markdown("Enter transaction details to predict if it is fraudulent.")
    
    with gr.Row():
        with gr.Column():
            step = gr.Number(label="Step (Time)", value=1)
            type_val = gr.Dropdown(choices=["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], label="Transaction Type", value="TRANSFER")
            amount = gr.Number(label="Amount", value=1000.0)
            oldbalanceOrg = gr.Number(label="Old Balance Origin", value=1000.0)
            newbalanceOrig = gr.Number(label="New Balance Origin", value=0.0)
            oldbalanceDest = gr.Number(label="Old Balance Dest", value=0.0)
            newbalanceDest = gr.Number(label="New Balance Dest", value=1000.0)
            
            predict_btn = gr.Button("Predict Fraud", variant="primary")
            
        with gr.Column():
            output = gr.Textbox(label="Prediction Result", lines=6)
            
    predict_btn.click(
        fn=predict_fraud,
        inputs=[step, type_val, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
