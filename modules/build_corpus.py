import transformers
from datasets import load_dataset
from beir import util
from beir.datasets.data_loader import GenericDataLoader
import nltk
nltk.download('punkt')
from nltk import word_tokenize

def build_corpus():

    squad_val = load_dataset("squad_v2", split="validation")
    corpus_squad = [{'dataset':'squad_v2', 'text': text} for text in list(set(squad_val["context"]))]

    dataset = "dbpedia-entity"
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = util.download_and_unzip(url, "datasets")
    corpus_db, _, _ = GenericDataLoader(data_folder=data_path).load(split="test")

    sample_corpus_db = []
    for k,e in corpus_db.items():
        if len(sample_corpus_db) > 9000:
            break
        if len(word_tokenize(e['text'])) > 50:
            sample_corpus_db.append({'dataset':'dbpedia', 'text':e['text'], 'key':k})
    
    corpus = corpus_squad + sample_corpus_db
    # squad_val, db_queries and db_qrels are used to evaluate the indexing with mrr.
    return corpus, squad_val
    
if __name__ == "__main__":
    import json
    corpus = build_corpus()
    json = json.dumps(corpus)
    f = open("corpus.json","w")
    f.write(json)
    f.close()




