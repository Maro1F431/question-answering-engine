import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import datasets

from indexing.indexing import batch_indexing, get_annoy_index, batch_annoy_indexing

def compute_mrr_squad(embedded_corpus: list, 
                      map_question_context : list, 
                      indexing_model: object, 
                      squad_validation_dataset : datasets.arrow_dataset.Dataset,
                      comparison_metric : str ='dot') -> float:
    '''
    computes the MRR of our model with all the queries in the squad_v2 validation set.

            Parameters:
                    embedded_corpus (list[dict]): Our corpus with the embedding of each entry.
                    map_question_context (list[int]): Built by the build_corpus function. 
                    Needed to retrive easily the relevant context of a given query.
                    indexing_model (obj): Model used to embed the corpus and the queries.
                    squad_validation_dataset (Dataset (huggingface)): squad_v2 validation set.


            Returns:
                    mrr (float): mrr score on squad_v2.
    '''
    queries = squad_validation_dataset['question']
    list_ranked_documents_indexes = batch_indexing(indexing_model, embedded_corpus, queries, comparison_metric)
    sum_reci_rank = 0
    for i in range(len(queries)):
        ranked_documents_indexes = list_ranked_documents_indexes[i]
        rank = ranked_documents_indexes.index(map_question_context[i]) + 1
        sum_reci_rank += 1/rank
    mrr = sum_reci_rank/len(queries)
    return mrr

def compute_mrr_squad_annoy(embedded_corpus: list, 
                            map_question_context : list, 
                            indexing_model: object, 
                            indexing_model_name: str, 
                            top_k: int, 
                            squad_validation_dataset : datasets.arrow_dataset.Dataset, 
                            comparison_metric : str ='dot') -> float:
    '''
    computes the MRR of our model using annoy indexing with all the queries in the squad_v2 validation set.

            Parameters:
                    embedded_corpus (list[dict]): Our corpus with the embedding of each entry.
                    map_question_context (list[int]): Built by the build_corpus function. 
                    Needed to retrive easily the relevant context of a given query.
                    indexing_model (obj): Model used to embed the corpus and the queries.
                    indexing_model_name (str): Name of the indexing model.
                    top_k (int): Top k results to retrieve fron annoy indexing.
                    squad_validation_dataset (Dataset (huggingface)): squad_v2 validation set.


            Returns:
                    mrr (float): mrr score on squad_v2.
    '''
    queries = squad_validation_dataset['question']
    annoy_index = get_annoy_index(256, embedded_corpus, 768, indexing_model_name, comparison_metric)
    list_ranked_documents_indexes = batch_annoy_indexing(annoy_index, queries, indexing_model, top_k)
    sum_reci_rank = 0
    for i in range(len(queries)):
        ranked_documents_indexes = list_ranked_documents_indexes[i]
        try:
            rank = ranked_documents_indexes.index(map_question_context[i]) + 1
        except ValueError:
            rank = 0
        if rank != 0:
            sum_reci_rank += 1/rank
    mrr = sum_reci_rank/len(queries)
    return mrr
