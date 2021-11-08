def compute_mrr_squad(queries, list_retrived_documents, squad_validation_dataset):
    sum_reci_rank = 0
    for i,query in enumerate(queries):
        query_index = squad_validation_dataset['question'].index(query)
        relevant_doc = squad_validation_dataset['context'][query_index]
        retrived_documents = list_retrived_documents[i]
        j = 1
        while(retrived_documents[j]['dataset'] != 'squad_v2' or retrived_documents[j]['text'] != relevant_doc):
            j += 1
        sum_reci_rank += 1/j
    mrr = sum_reci_rank/len(queries)
    return mrr

