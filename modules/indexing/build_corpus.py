from datasets import load_dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import nltk
nltk.download('punkt')
from nltk import word_tokenize
import json
import os
from collections import Counter
from collections import defaultdict

def build_corpus(nb_dbpedia_sample):
    '''
    Build the corpus that serves as a knowledge base for our QA system.

            Parameters:
                    nb_dbpedia_sample (int): Number of samples to take from the dbpedia dataset.

            Returns:
                    corpus (list[dict]): The built corpus.
                    squad_valid (Dataset (from huggingface)): Validation split of the squad_v2 dataset. Used to compute MRR.
                    map_question_context (list[int]): List where each index corresponds to the question 
                    at the same index in the validation squad set. The value at the index gives the
                    index of the context corresponding to the question in our corpus.
    '''

    squad_valid = load_dataset("squad_v2", split="validation")
    squad_set_context = list(set(squad_valid['context']))
    map_question_context = []
    for context in squad_valid['context']:
      map_question_context.append(squad_set_context.index(context))
    corpus_squad = [{'dataset':'squad_v2', 'text': text} for text in squad_set_context]

    if os.path.exists('sample_corpus_db.json'):
        f = open("sample_corpus_db.json",)
        corpus = json.load(f)
        f.close()
    else:
        dataset = "dbpedia-entity"
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = util.download_and_unzip(url, "datasets")
        corpus_db, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")

        sample_corpus_db = []
        for k,e in corpus_db.items():
            if len(sample_corpus_db) > nb_dbpedia_sample:
                break
            if len(word_tokenize(e['text'])) > 50:
                sample_corpus_db.append({'dataset':'dbpedia', 'text':e['text']})
        json_dumped = json.dumps(sample_corpus_db)
        f = open("sample_corpus_db.json","w")
        f.write(json_dumped)
        f.close()
        
        
        corpus = corpus_squad + sample_corpus_db

    # squad_valid and map_question_context are used to evaluate the indexing with mrr.
    return corpus, squad_valid, map_question_context