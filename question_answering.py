import argparse
from modules.indexing.build_corpus import build_corpus
from modules.indexing.indexing import corpus_embedding, get_annoy_index, batch_indexing, annoy_indexing
from modules.question_answering.best_answer import pick_best_answer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--annoy",action="store_true", help="uses annoy indexing")
    parser.add_argument("--nb-dbpedia",type=int, help="number of dbpedia")
    parser.add_argument("--index_model", type=str, help="name of the huggingface indexing model to use")
    args = parser.parse_args()
    annoy = args.annoy
    nb_dbpedia = 9000
    if args.nb_dbpedia:
        nb_dbpedia = args.nb_dbpedia
    indexing_model_name = 'sentence-transformers/msmarco-distilbert-dot-v5'
    if args.index_model:
        indexing_model_name = args.index_model
    corpus, squad_valid, _ = build_corpus(nb_dbpedia)
    indexing_model = SentenceTransformer(indexing_model_name)
    embedded_corpus = corpus_embedding(corpus, indexing_model)

    while True:

        query = input("Please enter a question: ")
        if annoy:
            annoy_index = get_annoy_index(256, embedded_corpus, 768, indexing_model_name)
            ranked_corpus_ids = annoy_indexing(annoy_index, query, indexing_model, 10)
        else:
            ranked_corpus_ids = batch_indexing(indexing_model, embedded_corpus, [query])[0]

        qa_model_name = 'mvonwyl/distilbert-base-uncased-finetuned-squad2'
        tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        qa_nlp = pipeline('question-answering', model=qa_model, tokenizer=tokenizer)

        answer = pick_best_answer(query, ranked_corpus_ids[:10], embedded_corpus, qa_nlp)
        print('\n Guessed answer is: {} \n'.format(answer))





