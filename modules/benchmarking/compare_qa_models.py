import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, Trainer, AutoModelForQuestionAnswering
from question_answering.model_evaluation import get_processed_predictions, compute_metrics , show_badly_classified

def eval_model(model_checkpoint, local=False):
    '''
     Computes the given metrics with the predictions of the given dataset.

            Parameters:
                    model_checkpoint (Huggingface model checkpoint): Model to evaluate.

            Prints metrics and returns:
                   badly_classified_examples (List): A list of the badly classified examples with the context, possible answers and the prediction.
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, local_files_only=local)
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint, local_files_only=local)
    datasets = load_dataset("squad_v2")
    trainer = Trainer(model)
    processed_predictions = get_processed_predictions(datasets, trainer, tokenizer)
    metrics = load_metric("squad_v2")
    computed_metrics = compute_metrics(processed_predictions, datasets['validation'], metrics)
    print(computed_metrics)
    badly_classified_examples = show_badly_classified(processed_predictions, datasets['validation'])
    return badly_classified_examples


