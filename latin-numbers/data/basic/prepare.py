"""
Generates training data for Latin to English number translation.
Example output:
    unus one
    duo two
    tres three
    quinque five
"""
import os
import pickle
import numpy as np
import random

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Latin to English number mappings (1-20 plus key larger numbers)
NUMBERS = {
    'unus': 'one',
    'duo': 'two',
    'tres': 'three',
    'quattuor': 'four',
    'quinque': 'five',
    'sex': 'six',
    'septem': 'seven',
    'octo': 'eight',
    'novem': 'nine',
    'decem': 'ten',
    'undecim': 'eleven',
    'duodecim': 'twelve',
    'tredecim': 'thirteen',
    'quattuordecim': 'fourteen',
    'quindecim': 'fifteen',
    'sedecim': 'sixteen',
    'septendecim': 'seventeen',
    'duodeviginti': 'eighteen',
    'undeviginti': 'nineteen',
    'viginti': 'twenty',
    'triginta': 'thirty',
    'quadraginta': 'forty',
    'quinquaginta': 'fifty',
    'sexaginta': 'sixty',
    'septuaginta': 'seventy',
    'octoginta': 'eighty',
    'nonaginta': 'ninety',
    'centum': 'hundred',
}

# Generate training data
def generate_translation_data(num_sequences=10000):
    """Generate Latin to English number translations"""
    data = []
    latin_words = list(NUMBERS.keys())
    
    for _ in range(num_sequences):
        # Pick a random Latin number
        latin = random.choice(latin_words)
        english = NUMBERS[latin]
        
        line = f"{latin} {english}\n"
        data.append(line)
    
    return ''.join(data)

# Generate data
print("Generating Latin to English translation data...")
train_data = generate_translation_data(num_sequences=8000)
val_data = generate_translation_data(num_sequences=1000)

# Get all unique characters
all_text = train_data + val_data
chars = sorted(list(set(all_text)))
vocab_size = len(chars)
print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {chars}")

# Create character to index mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# Encode function
def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# Encode the data
train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Export to binary files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# Save metadata
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

print("Data preparation complete!")
print(f"Files created: train.bin, val.bin, meta.pkl")

# Show sample of training data
print("\nSample training data:")
print(train_data[:300])
