import functools
import transformers
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
import datasets

def _prepare_train_feature(examples : datasets.arrow_dataset.Dataset, 
                           tokenizer : object, 
                           max_length : int, 
                           doc_stride : int) -> datasets.arrow_dataset.Dataset:
    '''
    Tokenize and prepare our training dataset for model training.

            Parameters:
                    examples (Dataset, huggingface): Batch of examples to process.
                    tokenizer (Tokenizer, huggingface): Tokenizer used for our model.
                    max_length (int): max length possible for a feature.
                    doc_stride (int): Possible overlap between the features.

            Returns:
                   tokenized_examples (Dataset, huggingface): Tokenized and prepared examples.
    ''' 
    #Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    pad_on_right = tokenizer.padding_side == "right"

    # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    tokenized_examples = tokenizer(
        examples["question" if pad_on_right else "context"],
        examples["context" if pad_on_right else "question"],
        truncation="only_second" if pad_on_right else "only_first",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Since one example might give us several features if it has a long context, we need a map from a feature to
    # its corresponding example. This key gives us just that.
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    # The offset mappings will give us a map from token to character position in the original context. This will
    # help us compute the start_positions and end_positions.
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Let's label those examples!
    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # We will label impossible answers with the index of the CLS token.
        input_ids = tokenized_examples["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)

        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        # If no answers are given, set the cls_index as answer.
        if len(answers["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(cls_index)
            tokenized_examples["end_positions"].append(cls_index)
        else:
            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

    return tokenized_examples

def preprocessing(model_checkpoint : object, 
                  datasets : datasets.arrow_dataset.Dataset, 
                  max_length : int, 
                  doc_stride : int) -> (object, datasets.arrow_dataset.Dataset):
    '''
    Tokenize and prepare our dataset for model training.

            Parameters:
                    model_checkpoint (obj, hugginface): Checkpoint of the model to be trained.
                    datasets (Dataset, huggingface): Huggingface dataset containing all the splits (train, validation and test).
                    max_length (int): max length possible for a feature.
                    doc_stride (int): Possible overlap between the features.

            Returns:
                    tokenizer (Tokenizer, huggingface): The used tokenizer.
                    tokenized_datasets (Dataset, huggingface): Tokenized and prepared dataset.
    ''' 
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    prepare_train_feature = functools.partial(_prepare_train_feature, tokenizer=tokenizer, max_length=max_length, doc_stride=doc_stride)
    tokenized_datasets = datasets.map(prepare_train_feature, batched=True, remove_columns=datasets["train"].column_names)
    return tokenizer, tokenized_datasets

def fine_tuning(model_checkpoint : object, 
                tokenizer : object, 
                tokenized_datasets : datasets.arrow_dataset.Dataset, 
                batch_size : int, 
                resume : bool = False) -> object:
    """
    Fine tunes the model with the tokenized dataset. The model is saved locally as "test-squad-trained".

            Parameters:
                    model_checkpoint (obj, hugginface): Checkpoint of the model to be trained.
                    tokenizer (Tokenizer, huggingface): Tokenizer used for our model.
                    tokenized_datasets (Dataset, huggingface): Tokenized and prepared dataset.
                    batch_size (int): Batch size used for the training.
                    resume (bool): Boolean stating whether or not the training should start from scratch.

            Returns:
                    trainer (Trainer, huggingface): The trained model.
    """
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned-squad",
        evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        resume_from_checkpoint=resume
    )
    data_collator = default_data_collator

    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model("test-squad-trained")
    return trainer
