#App using gpt2 getting custom data and generating text using nlp
#steps - loading the data,tokenizing the data, pre-training using gpt 2, fine tuning the model, evaluation

from flask import Flask, request, jsonify, render_template
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
import logging
from sklearn.model_selection import train_test_split
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the custom text dataset
def load_custom_dataset(file_path):
    try:
        dataset = load_dataset('text', data_files={'train': file_path})
        logger.info("Dataset loaded successfully.")
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise

# Tokenize the dataset
def tokenize_dataset(dataset, tokenizer):
    try:
        def tokenize_function(examples):
            tokens = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        logger.info("Dataset tokenized successfully.")
        return tokenized_datasets
    except Exception as e:
        logger.error(f"Error tokenizing dataset: {e}")
        raise

# Load pre-trained GPT-2 model and tokenizer
def load_model_and_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.resize_token_embeddings(len(tokenizer))
        logger.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}")
        raise

# Fine-tune the model
def fine_tune_model(model, train_dataset, eval_dataset):
    try:
        training_args = TrainingArguments(
            output_dir='./results',
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            num_train_epochs=3,
            weight_decay=0.01,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        
        trainer.train()
        logger.info("Model fine-tuned successfully.")
        return trainer
    except Exception as e:
        logger.error(f"Error during fine-tuning: {e}")
        raise

# Evaluate the model
def evaluate_model(trainer):
    try:
        results = trainer.evaluate()
        logger.info(f"Model evaluation results: {results}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

# Generate text
def generate_text(prompt, model, tokenizer):
    try:
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        generated = generator(prompt, max_length=100, num_return_sequences=1)
        logger.info(f"Generated text: {generated}")
        return generated[0]['generated_text']
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise

# Detect greetings
def detect_greeting(user_input):
    greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
    for greeting in greetings:
        if greeting in user_input.lower():
            return True
    return False

app = Flask(__name__)

# Endpoint to handle chat
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['input']

    # Check for greetings
    if detect_greeting(user_input):
        return jsonify({'response': 'Hello! How can I assist you today?'})

    # Generate text response for non-greetings
    response = generate_text(user_input, model, tokenizer)
    return jsonify({'response': response})

# Main function to run all steps
def main():
    file_path = 'dataset.txt'  # Path to your dataset file
    model_name = 'distilgpt2'  # Pre-trained model name

    dataset = load_custom_dataset(file_path)
    model, tokenizer = load_model_and_tokenizer(model_name)
    tokenized_datasets = tokenize_dataset(dataset, tokenizer)
    
    # Split the dataset into train and eval sets
    train_size = 0.8
    train_dataset, eval_dataset = tokenized_datasets['train'].train_test_split(test_size=1-train_size).values()
    
    trainer = fine_tune_model(model, train_dataset, eval_dataset)
    evaluate_model(trainer)
    
    # Save the trained model and tokenizer for later use
    model.save_pretrained('./model')
    tokenizer.save_pretrained('./tokenizer')

if __name__ == "__main__":
    # Load the trained model and tokenizer for the Flask app
    if os.path.exists('./model') and os.path.exists('./tokenizer'):
        model, tokenizer = load_model_and_tokenizer('./model')
    else:
        main()
    app.run(debug=True)



#----------------------------------------------------------code using gpt 4-------------------------------------------------
# from flask import Flask, request, jsonify, render_template
# from datasets import load_dataset
# from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
# import logging
# from sklearn.model_selection import train_test_split
# import os

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load the custom text dataset
# def load_custom_dataset(file_path):
#     try:
#         dataset = load_dataset('text', data_files={'train': file_path})
#         logger.info("Dataset loaded successfully.")
#         return dataset
#     except Exception as e:
#         logger.error(f"Error loading dataset: {e}")
#         raise

# # Tokenize the dataset
# def tokenize_dataset(dataset, tokenizer):
#     try:
#         def tokenize_function(examples):
#             tokens = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)
#             tokens["labels"] = tokens["input_ids"].copy()
#             return tokens
        
#         tokenized_datasets = dataset.map(tokenize_function, batched=True)
#         logger.info("Dataset tokenized successfully.")
#         return tokenized_datasets
#     except Exception as e:
#         logger.error(f"Error tokenizing dataset: {e}")
#         raise

# # Load GPT-4 model and tokenizer
# def load_model_and_tokenizer(model_name):
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_name)
#         if tokenizer.pad_token is None:
#             tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
        
#         model = AutoModelForCausalLM.from_pretrained(model_name)
#         model.resize_token_embeddings(len(tokenizer))
#         logger.info("GPT-4 model and tokenizer loaded successfully.")
#         return model, tokenizer
#     except Exception as e:
#         logger.error(f"Error loading GPT-4 model or tokenizer: {e}")
#         raise

# # Fine-tune the model
# def fine_tune_model(model, train_dataset, eval_dataset):
#     try:
#         training_args = TrainingArguments(
#             output_dir='./results',
#             eval_strategy="epoch",
#             learning_rate=2e-5,
#             per_device_train_batch_size=4,
#             per_device_eval_batch_size=4,
#             num_train_epochs=3,
#             weight_decay=0.01,
#         )
        
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=eval_dataset,
#         )
        
#         trainer.train()
#         logger.info("GPT-4 model fine-tuned successfully.")
#         return trainer
#     except Exception as e:
#         logger.error(f"Error during fine-tuning: {e}")
#         raise

# # Evaluate the model
# def evaluate_model(trainer):
#     try:
#         results = trainer.evaluate()
#         logger.info(f"Model evaluation results: {results}")
#     except Exception as e:
#         logger.error(f"Error during evaluation: {e}")
#         raise

# # Generate text using GPT-4
# def generate_text(prompt, model, tokenizer):
#     try:
#         generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
#         generated = generator(prompt, max_length=100, num_return_sequences=1)
#         logger.info(f"Generated text: {generated}")
#         return generated[0]['generated_text']
#     except Exception as e:
#         logger.error(f"Error during text generation: {e}")
#         raise

# # Detect greetings and generate appropriate responses
# def detect_greeting(user_input):
#     greetings = ['hello', 'hi', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
#     for greeting in greetings:
#         if greeting in user_input.lower():
#             return "Hello! How can I assist you today?"
#     return None

# app = Flask(__name__)

# # Endpoint to handle chat
# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.form['input']

#     # Check for greetings
#     greeting_response = detect_greeting(user_input)
#     if greeting_response:
#         return jsonify({'response': greeting_response})

#     # Generate text response for non-greetings
#     response = generate_text(user_input, model, tokenizer)
#     return jsonify({'response': response})

# # Main function to run all steps
# def main():
#     file_path = 'dataset.txt'  # Path to your dataset file
#     model_name = 'gpt-4'  # Using GPT-4 model name

#     dataset = load_custom_dataset(file_path)
#     model, tokenizer = load_model_and_tokenizer(model_name)
#     tokenized_datasets = tokenize_dataset(dataset, tokenizer)
    
#     # Split the dataset into train and eval sets
#     train_size = 0.8
#     train_dataset, eval_dataset = tokenized_datasets['train'].train_test_split(test_size=1-train_size).values()
    
#     trainer = fine_tune_model(model, train_dataset, eval_dataset)
#     evaluate_model(trainer)
    
#     # Save the trained model and tokenizer for later use
#     model.save_pretrained('./model')
#     tokenizer.save_pretrained('./tokenizer')

# if __name__ == "__main__":
#     # Load the trained model and tokenizer for the Flask app
#     if os.path.exists('./model') and os.path.exists('./tokenizer'):
#         model, tokenizer = load_model_and_tokenizer('./model')
#     else:
#         main()
#     app.run(debug=True)