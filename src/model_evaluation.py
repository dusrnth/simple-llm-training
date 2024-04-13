# 훈련된 모델을 평가하고 평가 결과를 출력 및 파일로 저장
def evaluate_model(trainer):
    eval_results = trainer.evaluate()
    print(eval_results)

    with open('results/eval_results.txt', 'w') as file:
        file.write(str(eval_results))