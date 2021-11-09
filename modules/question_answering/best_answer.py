def pick_best_answer(question, ranked_corpus_ids, embedded_corpus, qa_nlp):
    max_score = 0
    answer = ''
    for id in ranked_corpus_ids:
        context = embedded_corpus[id]['text']
        QA_input = {
            'question': question,
            'context': context
        }
        res = qa_nlp(QA_input)
        if res['score'] > max_score:
            max_score = res['score']
            answer = res['answer']
    return answer