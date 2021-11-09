import argparse
from modules.indexing.build_corpus import build_corpus
from modules.indexing.indexing import corpus_embedding, get_annoy_index, indexing, annoy_indexing
from modules.question_answering.best_answer import pick_best_answer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    while True:
        query = input("Please enter a question: ")
        corpus, squad_valid = build_corpus()
        indexing_model_name = 'sentence-transformers/msmarco-distilbert-dot-v5'
        indexing_model = SentenceTransformer(indexing_model_name)
        embedded_corpus = corpus_embedding(corpus, indexing_model)
        annoy_index = get_annoy_index(256, embedded_corpus, 768, indexing_model_name)
        ranked_corpus_ids = annoy_indexing(annoy_indexing, query, indexing_model, 10)

        qa_model_name = 'mvonwyl/distilbert-base-uncased-finetuned-squad2'
        tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
        qa_nlp = pipeline('question-answering', model=qa_model, tokenizer=tokenizer)

        answer = pick_best_answer(query, ranked_corpus_ids, embedded_corpus, qa_nlp)
        print('\n Guessed answer is: {} \n'.format(answer))





