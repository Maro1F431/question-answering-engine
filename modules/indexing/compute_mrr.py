from indexing import indexing

def compute_mrr_squad(embedded_corpus, torch_comparison_metric, indexing_model, squad_validation_dataset, nb_queries):
    queries = squad_validation_dataset['question'][:nb_queries]
    list_ranked_documents_indexes = [indexing(indexing_model, embedded_corpus, query, torch_comparison_metric) for query in queries]
    sum_reci_rank = 0
    for i,query in enumerate(queries):
        query_index = squad_validation_dataset['question'].index(query)
        relevant_doc = squad_validation_dataset['context'][query_index]
        ranked_documents_indexes = list_ranked_documents_indexes[i]
        j = 0
        while(embedded_corpus[ranked_documents_indexes[j]]['dataset'] != 'squad_v2' or embedded_corpus[ranked_documents_indexes[j]]['text'] != relevant_doc):
            j += 1
        sum_reci_rank += 1/(j+1)
    mrr = sum_reci_rank/len(queries)
    return mrr