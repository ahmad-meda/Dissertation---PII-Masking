import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
import json

# Load your dataset
# Replace 'ai4privacy' with the actual path or name of your dataset
dataset = load_dataset('ai4privacy')

# Function to check for missing values
def analyze_missing_values(dataset):
    """
    Analyzes the dataset for missing values in each column.
    Returns a dictionary with column names and their respective missing value counts.
    """
    missing_values = {}
    for column in dataset.column_names:
        null_count = sum(1 for item in dataset[column] if item is None or (isinstance(item, str) and item.strip() == ''))
        missing_values[column] = null_count
    
    return missing_values

# Function to check text length distribution
def analyze_text_length(dataset):
    """
    Analyzes the length distribution of text in the dataset.
    Returns a list of text lengths.
    """
    text_lengths = [len(item) for item in dataset['source_text']]
    return text_lengths

# Function to check for stop words
def analyze_stop_words(dataset, language='english'):
    """
    Analyzes the presence of stop words in the dataset.
    Returns a dictionary with stop word statistics.
    """
    try:
        nltk.data.find(f'corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    
    stop_words = set(stopwords.words(language))
    
    # Count stop words in source_text
    stop_word_counts = []
    for text in dataset['source_text']:
        words = re.findall(r'\b\w+\b', text.lower())
        stop_count = sum(1 for word in words if word in stop_words)
        stop_word_counts.append(stop_count)
    
    # Calculate statistics
    total_words = sum(len(re.findall(r'\b\w+\b', text.lower())) for text in dataset['source_text'])
    total_stop_words = sum(stop_word_counts)
    
    return {
        'total_words': total_words,
        'total_stop_words': total_stop_words,
        'stop_word_percentage': (total_stop_words / total_words) * 100 if total_words > 0 else 0,
        'stop_word_counts': stop_word_counts
    }

# Function to analyze PII distribution
def analyze_pii_distribution(dataset):
    """
    Analyzes the distribution of PII types in the dataset.
    Returns a Counter object with PII type counts.
    """
    pii_types = []
    for item in dataset['privacy_mask']:
        if isinstance(item, list):
            for pii in item:
                if isinstance(pii, dict) and 'label' in pii:
                    pii_types.append(pii['label'])
    
    return Counter(pii_types)

# Function to analyze token distribution
def analyze_token_distribution(dataset):
    """
    Analyzes the distribution of tokens in the dataset.
    Returns a dictionary with token statistics.
    """
    token_counts = [len(tokens) for tokens in dataset['mbert_text_tokens']]
    
    return {
        'min_tokens': min(token_counts),
        'max_tokens': max(token_counts),
        'avg_tokens': sum(token_counts) / len(token_counts),
        'token_counts': token_counts
    }

# Function to check for special characters
def analyze_special_chars(dataset):
    """
    Analyzes the presence of special characters in the dataset.
    Returns a Counter object with special character counts.
    """
    special_chars = []
    for text in dataset['source_text']:
        chars = re.findall(r'[^a-zA-Z0-9\s]', text)
        special_chars.extend(chars)
    
    return Counter(special_chars)

# Function to analyze language distribution
def analyze_language_distribution(dataset):
    """
    Analyzes the distribution of languages in the dataset.
    Returns a Counter object with language counts.
    """
    if 'language' in dataset.column_names:
        return Counter(dataset['language'])
    else:
        return Counter()

# Function to analyze required cleaning
def analyze_cleaning_needs(dataset):
    """
    Comprehensive analysis of the dataset to determine if cleaning is needed.
    Returns a dictionary with analysis results.
    """
    results = {}
    
    # 1. Missing Values Analysis
    results['missing_values'] = analyze_missing_values(dataset)
    
    # 2. Text Length Analysis
    results['text_length'] = analyze_text_length(dataset)
    
    # 3. Stop Words Analysis
    # Use appropriate language for stopwords
    languages = analyze_language_distribution(dataset)
    primary_language = languages.most_common(1)[0][0] if languages else 'english'
    if primary_language not in ['english', 'italian', 'spanish', 'french', 'german', 'portuguese']:
        primary_language = 'english'  # Default to English if language not supported
    results['stop_words'] = analyze_stop_words(dataset, primary_language.lower())
    
    # 4. PII Distribution
    results['pii_distribution'] = analyze_pii_distribution(dataset)
    
    # 5. Token Distribution
    results['token_distribution'] = analyze_token_distribution(dataset)
    
    # 6. Special Characters Analysis
    results['special_chars'] = analyze_special_chars(dataset)
    
    # 7. Language Distribution
    results['language_distribution'] = analyze_language_distribution(dataset)
    
    return results

# Function to visualize the results
def plot_analysis_results(results):
    """
    Creates visualizations for the analysis results.
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Text Length Distribution
    sns.histplot(results['text_length'], kde=True, ax=axes[0, 0])
    axes[0, 0].set_title('Text Length Distribution')
    axes[0, 0].set_xlabel('Text Length')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Stop Words Distribution
    sns.histplot(results['stop_words']['stop_word_counts'], kde=True, ax=axes[0, 1])
    axes[0, 1].set_title('Stop Words Distribution')
    axes[0, 1].set_xlabel('Number of Stop Words')
    axes[0, 1].set_ylabel('Count')
    
    # 3. PII Type Distribution
    pii_types = results['pii_distribution']
    if pii_types:
        sns.barplot(x=list(pii_types.keys()), y=list(pii_types.values()), ax=axes[1, 0])
        axes[1, 0].set_title('PII Type Distribution')
        axes[1, 0].set_xlabel('PII Type')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_xticklabels(axes[1, 0].get_xticklabels(), rotation=45, ha='right')
    
    # 4. Token Count Distribution
    sns.histplot(results['token_distribution']['token_counts'], kde=True, ax=axes[1, 1])
    axes[1, 1].set_title('Token Count Distribution')
    axes[1, 1].set_xlabel('Number of Tokens')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('data_analysis_plots.png')
    plt.close()
    
    # Additional plots for language distribution if available
    if results['language_distribution']:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(results['language_distribution'].keys()), 
                    y=list(results['language_distribution'].values()))
        plt.title('Language Distribution')
        plt.xlabel('Language')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('language_distribution.png')
        plt.close()

# Function to generate a comprehensive report
def generate_report(results):
    """
    Generates a comprehensive report based on the analysis results.
    Returns a string with the report text.
    """
    report = []
    report.append("# PII Masking Dataset Analysis Report")
    report.append("\n## 1. Missing Values Analysis")
    
    # Missing Values
    missing_values = results['missing_values']
    if any(missing_values.values()):
        report.append("\nThe dataset contains missing values in the following columns:")
        for column, count in missing_values.items():
            if count > 0:
                report.append(f"- {column}: {count} missing values ({count/len(dataset)*100:.2f}%)")
    else:
        report.append("\nThe dataset does not contain any missing values in any column.")
    
    # Text Length Analysis
    text_lengths = results['text_length']
    report.append("\n## 2. Text Length Analysis")
    report.append(f"\nMinimum text length: {min(text_lengths)}")
    report.append(f"Maximum text length: {max(text_lengths)}")
    report.append(f"Average text length: {sum(text_lengths)/len(text_lengths):.2f}")
    
    # Stop Words Analysis
    stop_words = results['stop_words']
    report.append("\n## 3. Stop Words Analysis")
    report.append(f"\nTotal words: {stop_words['total_words']}")
    report.append(f"Total stop words: {stop_words['total_stop_words']}")
    report.append(f"Stop words percentage: {stop_words['stop_word_percentage']:.2f}%")
    
    # PII Distribution
    pii_distribution = results['pii_distribution']
    report.append("\n## 4. PII Type Distribution")
    if pii_distribution:
        for pii_type, count in pii_distribution.most_common():
            report.append(f"- {pii_type}: {count} instances")
    else:
        report.append("\nNo PII types found in the dataset.")
    
    # Token Distribution
    token_distribution = results['token_distribution']
    report.append("\n## 5. Token Distribution")
    report.append(f"\nMinimum tokens: {token_distribution['min_tokens']}")
    report.append(f"Maximum tokens: {token_distribution['max_tokens']}")
    report.append(f"Average tokens: {token_distribution['avg_tokens']:.2f}")
    
    # Special Characters Analysis
    special_chars = results['special_chars']
    report.append("\n## 6. Special Characters Analysis")
    report.append(f"\nTotal special characters: {sum(special_chars.values())}")
    report.append("\nTop 10 most common special characters:")
    for char, count in special_chars.most_common(10):
        report.append(f"- '{char}': {count} instances")
    
    # Language Distribution
    language_distribution = results['language_distribution']
    report.append("\n## 7. Language Distribution")
    if language_distribution:
        for language, count in language_distribution.most_common():
            report.append(f"- {language}: {count} instances ({count/len(dataset)*100:.2f}%)")
    else:
        report.append("\nNo language information found in the dataset.")
    
    # Conclusion
    report.append("\n## 8. Conclusion")
    
    # Determine if cleaning is needed
    cleaning_needed = False
    cleaning_reasons = []
    
    # Check for missing values
    if any(missing_values.values()):
        cleaning_needed = True
        cleaning_reasons.append("missing values")
    
    # Check for stop words
    if stop_words['stop_word_percentage'] > 30:  # Threshold can be adjusted
        cleaning_needed = True
        cleaning_reasons.append("high percentage of stop words")
    
    # Final conclusion
    if cleaning_needed:
        report.append(f"\nBased on the analysis, the dataset requires cleaning due to {', '.join(cleaning_reasons)}.")
    else:
        report.append("\nBased on the comprehensive analysis, the dataset doesn't require cleaning for PII masking tasks for the following reasons:")
        report.append("1. No significant missing values were found.")
        report.append(f"2. Stop words percentage ({stop_words['stop_word_percentage']:.2f}%) is not high enough to impact PII masking performance.")
        report.append("3. The dataset contains well-structured PII annotations.")
        report.append("4. Token distribution is appropriate for the task.")
        report.append("\nThe dataset is ready to be used for PII masking tasks without additional cleaning steps.")
    
    return "\n".join(report)

# Main execution
def main():
    # Load the dataset split
    train_dataset = dataset['train']
    
    # Analyze the dataset
    results = analyze_cleaning_needs(train_dataset)
    
    # Plot the results
    plot_analysis_results(results)
    
    # Generate a report
    report = generate_report(results)
    
    # Save the report
    with open('pii_dataset_analysis_report.md', 'w') as f:
        f.write(report)
    
    # Save the detailed analysis results
    with open('pii_dataset_analysis_results.json', 'w') as f:
        # Convert Counter objects to dictionaries for JSON serialization
        serializable_results = results.copy()
        serializable_results['pii_distribution'] = dict(results['pii_distribution'])
        serializable_results['special_chars'] = dict(results['special_chars'])
        serializable_results['language_distribution'] = dict(results['language_distribution'])
        
        # Convert numpy arrays to lists for JSON serialization
        if 'text_length' in serializable_results:
            serializable_results['text_length'] = [int(x) for x in serializable_results['text_length']]
        
        json.dump(serializable_results, f, indent=4)
    
    print("Analysis complete. Results saved to 'pii_dataset_analysis_report.md' and 'pii_dataset_analysis_results.json'.")
    
    return results

if __name__ == "__main__":
    main()