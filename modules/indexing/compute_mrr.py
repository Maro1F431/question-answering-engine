from indexing import batch_indexing

def compute_mrr_squad(embedded_corpus, map_question_context, torch_comparison_metric, indexing_model, squad_validation_dataset):
    queries = squad_validation_dataset['question']
    list_ranked_documents_indexes = batch_indexing(indexing_model, embedded_corpus, queries, torch_comparison_metric)
    sum_reci_rank = 0
    for i in range(len(queries)):
        ranked_documents_indexes = list_ranked_documents_indexes[i]
        rank = ranked_documents_indexes.index(map_question_context[i]) + 1
        sum_reci_rank += 1/rank
    mrr = sum_reci_rank/len(queries)
    return mrr