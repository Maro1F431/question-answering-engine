def compute_mrr_dbpedia(query_keys, list_retrived_documents, qrels):
    for i,query_key in enumerate(query_keys):
        sum_reci_rank = 0
        retrived_documents = list_retrived_documents[i]
        j=0
        # dbpedia as 3 level of relevancy, 0 for none, 1 for medium and 2 for high.
        # We made the choice to only consider high relevancy for the MRR.
        while(qrles[query_key][retrived_documents[j]['key']] < 2):
            if retrived_documents[j]['dataset'] != 'dbpedia':
                next
            j += 1
        sum_reci_rank += 1/j
    mrr = sum_reci_rank/len(query_keys)
    return mrr

def compute_mrr_squad(queries, list_retrived_documents, squad_validation_dataset):
    for i,query in enumerate(queries):
        sum_reci_rank = 0
        query_index = squad_validation_dataset['question'].index(query)
        relevant_doc = squad_validation_dataset['context'][query_index]
        retrived_documents = list_retrived_documents[i]
        j = 0
        while(retrived_documents[j]['text'] != relevant_doc):
            if retrived_documents[j]['dataset'] != 'squad_v2':
                next
            j += 1
        sum_reci_rank += 1/j
    mrr = sum_reci_rank/len(queries)
    return mrr

