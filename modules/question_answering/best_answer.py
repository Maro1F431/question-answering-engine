def pick_best_answer(question, ranked_corpus_ids, embedded_corpus, qa_nlp):
    '''
    Picks the best answer to the given question. Uses each document in the 
    given indexed corpus as context, one by one.

            Parameters:
                    question (str): The question we want an answer for.
                    ranked_corpus_ids (list[int]): A an indexing of our corpus, where documents are 
                    referenced by their index in the corpus. Documents are ranked in descending order.
                    embedded_corpus (list[dict]): Our corpus with the embedding of each entry.
                    qa_nlp (obj, huggingface pipeline): huggingface pipeline built from our chosen qa model.
            Returns:
                    answer (str): Estimated best answer, retrieved from the given context.
    '''
    max_score = 0
    answer = ''
    print('len corpus ids :')
    print(len(ranked_corpus_ids))
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