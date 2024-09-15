from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

# Load the new model and tokenizer
def load_model():
    try:
        # Use the specified model and tokenizer from Hugging Face with TensorFlow weights
        print("Loading model with TensorFlow weights: Reflection-Llama-3.1-70B")
        # Load model directly
 
        tokenizer = AutoTokenizer.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
        model = AutoModelForCausalLM.from_pretrained("mattshumer/Reflection-Llama-3.1-70B")
        
        print("Model loaded successfully!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Function to generate text
def generate_text(prompt, max_length=50):
    tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        return "Model loading failed. Check the logs for more details."
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=1000)
    outputs = model.generate(inputs.input_ids, max_length=max_length, num_return_sequences=1)
    
    # Decode the generated summary or response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
