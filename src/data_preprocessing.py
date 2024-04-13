from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    return tokenizer(data, padding=True, truncation=True, return_tensors='pt')

positive_data = tokenize_data('positive.txt')
negative_data = tokenize_data('negative.txt')