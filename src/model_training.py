# TrainingArguments: 훈련 관련 인자를 설정
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

# 모델, 입력 ID, 어텐션 마스크, 레이블을 받아 모델을 훈련
def train_model(model, input_ids, attention_masks, labels):
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=torch.utils.data.TensorDataset(input_ids, attention_masks, labels),
    )

    trainer.train()

    return trainer