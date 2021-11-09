from tqdm.auto import tqdm
import functools
import collections
import numpy as np

def _prepare_validation_features(examples, tokenizer, max_length, doc_stride):
    '''
    Tokenize and prepare our validation dataset for easier postprocessing of predictions.

            Parameters:
                    examples (Dataset, huggingface): Batch of examples to process.
                    tokenizer (Tokenizer, huggingface): Tokenizer used for our model.
                    max_length (int): max length possible for a feature.
                    doc_stride (int): Possible overlap between the features.

            Returns:
                   tokenized_examples (Dataset, huggingface): Tokenized and prepared examples.
    '''
    # Some of the questions have lots of whitespace on the left, which is not useful and will make the
    # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
    # left whitespace
    examples["question"] = [q.lstrip() for q in examples["question"]]

    pad_on_right = tokenizer.padding_side == "right"

    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
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

    # We keep the example_id that gave us this feature and we will store the offset mappings.
    tokenized_examples["example_id"] = []

    for i in range(len(tokenized_examples["input_ids"])):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_examples.sequence_ids(i)
        context_index = 1 if pad_on_right else 0

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = sample_mapping[i]
        tokenized_examples["example_id"].append(examples["id"][sample_index])

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples["offset_mapping"][i] = [
            (o if sequence_ids[k] == context_index else None)
            for k, o in enumerate(tokenized_examples["offset_mapping"][i])
        ]

    return tokenized_examples

def _postprocess_qa_predictions(examples, features, raw_predictions, tokenizer, n_best_size = 20, max_answer_length = 30):
    '''
    Postprocess raw predictions of our qa model to give the answers in a string format.

            Parameters:
                    examples (Dataset, huggingface): Batch of examples used to creates the features.
                    features (Dataset, huggingface): features created from the examples. An examples can
                    have multiplie features linked to it. The link between features and examples is made
                    easily thanks to the preprocessing made by the _prepare_validation_features function.
                    raw_predictions (list[list[list[float]]]): The raw predictions of our qa models for each feature.
                    The list is composed of two list of list. One for the 'start' logits and the other for the 'end' logits.
                    In each of this list, we have the logits for each features.

            Returns:
                   predictions (Dict): Predictions in a string format for each example. Keys are examples ids. Values are the predictions.
    '''
    all_start_logits, all_end_logits = raw_predictions
    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    predictions = collections.OrderedDict()

    # Logging.
    print(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None # Only used if squad_v2 is True.
        valid_answers = []
        
        context = example["context"]
        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]

            # Update minimum null prediction.
            cls_index = features[feature_index]["input_ids"].index(tokenizer.cls_token_id)
            feature_null_score = start_logits[cls_index] + end_logits[cls_index]
            if min_null_score is None or min_null_score < feature_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue

                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        {
                            "score": startP_logits[start_index] + end_logits[end_index],
                            "text": context[start_char: end_char]
                        }
                    )
        
        if len(valid_answers) > 0:
            best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            best_answer = {"text": "", "score": 0.0}

        # Let's pick our final answer: the best one or the null answer
        answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
        predictions[example["id"]] = answer

    return predictions

def get_processed_predictions(datasets, trainer, tokenizer, n_best_size = 20, max_answer_length = 30):
    '''
    Process the validation dataset for predictions, predict and then postprocess the raw predictions.

            Parameters:
                    datasets (Dataset, huggingface): Dataset containg all splits (train, validation, test).
                    trainer (Trainer, huggingface): Trained qa model.
                    tokenizer (Tokenizer, huggingface): Tokenizer used for the training of the qa model.
                    n_best_size (int): Parameter used for predictions postprocessing. Number of possible answers
                    to consider from the predicted logits.
                    max_answer_length (int):  Max length authorized for the considered answers.  

            Returns:
                   processed_predictions (Dict): Predictions in a string format for each example. Keys are examples ids. Values are the predictions.
    '''
    max_length = 384
    doc_stride = 128

    functools.partial(_prepare_validation_features, tokenizer=tokenizer, max_length=max_length, doc_stride=doc_stride)
    validation_features = datasets["validation"].map(
    functools.partial(_prepare_validation_features, tokenizer=tokenizer, max_length=max_length, doc_stride=doc_stride),
    batched=True,
    remove_columns=datasets["validation"].column_names
    )

    raw_predictions = trainer.predict(validation_features)
    validation_features.set_format(type=validation_features.format["type"], columns=list(validation_features.features.keys()))
    processed_predictions = _postprocess_qa_predictions(datasets['validation'], validation_features, raw_predictions.predictions, tokenizer,n_best_size, max_answer_length)
    return processed_predictions

def compute_metrics(processed_predictions, validation_dataset, metrics):
    '''
     Computes the given metrics with the predictions of the given dataset.

            Parameters:
                    processed_predictions (Dict): Predictions in a string format for each example. Keys are examples ids. Values are the predictions.
                    validation_dataset (Dataset, huggingface): The validation dataset.
                    metrics (Metric, huggingface): Huggingface metrics to compute.

            Returns:
                   computed_metrics (Dict): The computed metrics.
    '''
    formatted_predictions = [{"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in processed_predictions.items()]
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in validation_dataset]
    computed_metrics = metrics.compute(predictions=formatted_predictions, references=references)
    return computed_metrics