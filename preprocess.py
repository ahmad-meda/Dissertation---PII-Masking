import os
import re
import srsly
import argparse

from tqdm import tqdm
from datasets import load_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "true"

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./data")
parser.add_argument("--dataset_name", type=str, default="ai4privacy/pii-masking-300k")

args = parser.parse_args()

print(f"Loading dataset: {args.dataset_name}")

dataset = load_dataset(args.dataset_name)
train, validation = dataset['train'], dataset['validation']


def tokenize_text(text):
    """Tokenizes the input text into a list of tokens."""
    tokens = re.findall(r'\w+(?:[-_]\w+)*|\S', text)
    return tokens

def char_to_token_spans(text, char_spans):
    """Convert character spans to token spans."""
    tokens = tokenize_text(text)
    
    # Create mapping from character positions to token positions
    char_to_token = {}
    char_pos = 0
    
    for token_idx, token in enumerate(tokens):
        token_start = text.find(token, char_pos)
        token_end = token_start + len(token)
        
        # Map each character position in this token to its token index
        for pos in range(token_start, token_end):
            char_to_token[pos] = token_idx
            
        char_pos = token_end
    
    # Convert character spans to token spans
    token_spans = []
    for start, end, label in char_spans:
        if start in char_to_token and (end - 1) in char_to_token:
            token_start = char_to_token[start]
            token_end = char_to_token[end - 1]
            token_spans.append((token_start, token_end, label))
    
    return token_spans

def process_example(example):
    tokenized_text = tokenize_text(example['source_text'])
    
    char_spans = srsly.json_loads(example['span_labels'])
    token_spans = char_to_token_spans(example['source_text'], char_spans)
    
    return { 'tokenized_text': tokenized_text, 'ner': token_spans }


train_data = [process_example(example) for example in tqdm(train, desc="Processing train:")]
validation_data = [process_example(example) for example in tqdm(validation, desc="Processing validation:")]

# Construct output paths
os.makedirs(args.output_dir, exist_ok=True)

train_output_path = os.path.join(args.output_dir, 'train.json')
validation_output_path = os.path.join(args.output_dir, 'validation.json')

print(f"Saving training data to {train_output_path}")
srsly.write_json(train_output_path, train_data)

print(f"Saving validation data to {validation_output_path}")
srsly.write_json(validation_output_path, validation_data)

print("Processed dataset successfully!")