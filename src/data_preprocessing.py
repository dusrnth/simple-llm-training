import torch
from transformers import AutoTokenizer

# 주어진 파일 경로에서 데이터를 읽어와 리스트로 반환
def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read().splitlines()
    return data

# 데이터를 토큰화하고 패딩 및 잘림을 적용하여 텐서로 변환
def tokenize_data(data, tokenizer):
    return tokenizer(data, padding=True, truncation=True, return_tensors='pt')

# 긍정 리뷰 파일과 부정 리뷰 파일을 로드하고 토큰화하며, 입력 ID, 어텐션 마스크, 레이블을 생성
def preprocess_data(positive_file, negative_file, tokenizer):
    positive_data = load_data(positive_file)
    negative_data = load_data(negative_file)

    positive_tokenized = tokenize_data(positive_data, tokenizer)
    negative_tokenized = tokenize_data(negative_data, tokenizer)

    labels = torch.tensor([1] * len(positive_tokenized['input_ids']) + [0] * len(negative_tokenized['input_ids']))

    input_ids = torch.cat((positive_tokenized['input_ids'], negative_tokenized['input_ids']), dim=0)
    attention_masks = torch.cat((positive_tokenized['attention_mask'], negative_tokenized['attention_mask']), dim=0)

    return input_ids, attention_masks, labels