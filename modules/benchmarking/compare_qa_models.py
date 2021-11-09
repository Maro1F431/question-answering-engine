from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, Trainer, AutoModelForQuestionAnswering
from modules.question_answering.model_evaluation import get_processed_predictions, compute_metrics

def eval_model(model_checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    datasets = load_dataset("squad_v2")
    validation_dataset = datasets['validation']
    trainer = Trainer(model)
    processed_predictions = get_processed_predictions(validation_dataset, trainer, tokenizer)
    metrics = load_metric("squad_v2")
    computed_metrics = compute_metrics(processed_predictions, validation_dataset, metrics)
    print(compute_metrics)


