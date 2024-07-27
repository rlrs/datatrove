import os
import gzip
import glob
import json
from typing import List, Dict, Any
from collections import defaultdict
from nltk import word_tokenize, sent_tokenize
import re

# Function to load JSONL files
def load_jsonl_gz(file_path: str) -> List[Dict[str, Any]]:
    with gzip.open(file_path, 'r') as f:
        return [json.loads(line) for line in f]

# Function to get all JSONL files in a directory and its subdirectories
def get_jsonl_files(directory: str) -> List[str]:
    return glob.glob(os.path.join(directory, '**', '*.jsonl.gz'), recursive=True)

# Function to group data based on directory structure
def group_data(base_path: str) -> Dict[str, List[Dict[str, Any]]]:
    grouped_data = defaultdict(list)
    for file_path in get_jsonl_files(base_path):
        relative_path = os.path.relpath(file_path, base_path)
        group = os.path.dirname(relative_path)
        grouped_data[group].extend(load_jsonl_gz(file_path))
    return dict(grouped_data)

def analyze_text_cleanliness(text: str) -> Dict[str, Any]:
    """Analyze the cleanliness of a single text."""
    return {
        'word_count': len(word_tokenize(text)),
        'sentence_count': len(sent_tokenize(text)),
        'avg_word_length': sum(len(word) for word in word_tokenize(text)) / len(word_tokenize(text)),
        'special_char_ratio': len(re.findall(r'[^a-zA-Z0-9\s]', text)) / len(text),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text),
        'newline_count': text.count('\n'),
    }

def analyze_corpus_cleanliness(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze the cleanliness of the entire corpus."""
    cleanliness_metrics = [analyze_text_cleanliness(doc['text']) for doc in data]
    return {
        'avg_word_count': sum(m['word_count'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_sentence_count': sum(m['sentence_count'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_word_length': sum(m['avg_word_length'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_special_char_ratio': sum(m['special_char_ratio'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_uppercase_ratio': sum(m['uppercase_ratio'] for m in cleanliness_metrics) / len(cleanliness_metrics),
        'avg_newline_count': sum(m['newline_count'] for m in cleanliness_metrics) / len(cleanliness_metrics),
    }
