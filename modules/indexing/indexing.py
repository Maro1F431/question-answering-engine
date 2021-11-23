from beir import util
import torch
import tqdm as tq
import numpy as np
import os
from annoy import AnnoyIndex
import json

def corpus_embedding(corpus, indexing_model):
    '''
    Embed the corpus with the indexing model.

            Parameters:
                    corpus (list[dict]): Our corpus, built by build_corpus().
                    indexing_model (obj): Model used to embed the corpus and the queries.

            Returns:
                    embedded_corpus (list[dict]): our corpus with the added embeddings.
    '''
    # Enables GPU if avalaible for embedding. It is strongly adviced to run this on a GPU
    # since the computation on a CPU can take up to an hour on a 10k dataset.
    if os.path.exists('{}-embedded_corpus.json'.format(len(corpus))):
        f = open('{}-embedded_corpus.json'.format(len(corpus)),)
        embedded_corpus = json.load(f)
        f.close()
    else:
        device = 'cuda:0' if torch.cuda.is_available() else None
        text_only_corpus = [elm['text'] for elm in corpus]
        text_embedded_corpus = indexing_model.encode(text_only_corpus,device=device, show_progress_bar=True)
        embedded_corpus = corpus
        for i,elm in enumerate(embedded_corpus):
            elm['text_embedding'] = text_embedded_corpus[i].tolist()
        json_dumped = json.dumps(embedded_corpus)
        f = open('{}-embedded_corpus.json'.format(len(corpus)),"w")
        f.write(json_dumped)
        f.close()
    return embedded_corpus


def batch_indexing(indexing_model, embedded_corpus, queries, comparison_metric='dot'):
    '''
    Indexes our corpus for multiple given queries.

            Parameters:
                    indexing_model (obj): Model used to embed the corpus and the queries.
                    embedded_corpus (list[dict]): Our corpus with the embedding of each entry.
                    queries (list[string]): queries used to index the corpus.
                    comparison_metric (str, either 'dot' or 'angular): metric used to compare
                    the embedded queries and documents.

            Returns:
                   list_ranked_corpus_ids (list[list[int]]): List containing a list of int for each
                   input queries. The list of int is an indexing of our corpus, where documents are 
                   referenced by their index in the corpus. Documents are ranked in descending order.
    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Embedding queries:')
    embedded_queries = indexing_model.encode(queries,device=device, show_progress_bar=True)
    corpus_embeddings = [doc['text_embedding'] for doc in embedded_corpus]
    tensor_embedded_query = torch.FloatTensor(embedded_queries).to(device)
    tensor_corpus_embeddings = torch.FloatTensor(corpus_embeddings).to(device)
    if (comparison_metric == 'angular'):
        similarity_tensor = util.cos_sim(tensor_embedded_query, tensor_corpus_embeddings)
    else:
        similarity_tensor = util.dot_score(tensor_embedded_query, tensor_corpus_embeddings)

    _, tensor_ranked_corpus_ids = torch.sort(similarity_tensor, descending=True, dim=1) # torch.sort return sorted, indexes
    list_ranked_corpus_ids = tensor_ranked_corpus_ids.tolist()
    return list_ranked_corpus_ids

def get_annoy_index(n_trees, embedded_corpus, embedding_size, indexing_model_name, comparison_metric='dot'):
    '''
    Builds the annoy index of our corpus. It is highly recommended to use a GPU for this.

            Parameters:
                    n_trees (int): Number of trees used by the annoy algorithm.
                    embedded_corpus (list[dict]): Our corpus with the embedding of each entry.
                    embedding_size (int): Size of the embeddings made by the indexing model.
                    model_name (str): Name of the indexing model
                    comparison_metric (str, either 'dot' or 'angular): metric used to compare
                    the embedded queries and documents.

            Returns:
                   annoy_index (obj): The annoy index built on our embedded corpus.
    '''
    annoy_index_path = '{}-embdsize{}-annoy_index-{}-trees.ann'.format(indexing_model_name.replace('/', '_'), embedding_size, n_trees)
    if not os.path.exists(annoy_index_path):
        # Create Annoy Index
        corpus_embeddings = [doc['text_embedding'] for doc in embedded_corpus]
        print("Create Annoy index with {} trees.".format(n_trees))
        annoy_index = AnnoyIndex(embedding_size, comparison_metric)

        for i in range(len(corpus_embeddings)):
            annoy_index.add_item(i, corpus_embeddings[i])

        annoy_index.build(n_trees)
        annoy_index.save(annoy_index_path)
    else:
        #Load Annoy Index from disc
        annoy_index = AnnoyIndex(embedding_size, comparison_metric)
        annoy_index.load(annoy_index_path)
    return annoy_index

def annoy_indexing(annoy_index, query, indexing_model, top_k_hits):
    '''
    Uses the annoy index to index the top k documents on a given query.

            Parameters:
                    annoy_index (obj): annoy index built by get_annoy_index.
                    query (str): The query used to index our corpus.
                    indexing_model (obj): Model used to embed the corpus and the queries.
                    top_k_hits (int): Number of documents to retrieve.

            Returns:
                   ranked_corpus_ids (list[int]): A an indexing of our corpus, where documents are 
                   referenced by their index in the corpus. Documents are ranked in descending order.
    '''
    query_embedding = indexing_model.encode(query)
    ranked_corpus_ids, _ = annoy_index.get_nns_by_vector(query_embedding, top_k_hits, include_distances=True)
    return ranked_corpus_ids

def batch_annoy_indexing(annoy_index, queries, indexing_model, top_k_hits):
    '''
    Uses the annoy index to index the top k documents on a given query.

            Parameters:
                    annoy_index (obj): annoy index built by get_annoy_index.
                    queries (list[string]): queries used to index the corpus.
                    indexing_model (obj): Model used to embed the corpus and the queries.
                    top_k_hits (int): Number of documents to retrieve.

            Returns:
                   ranked_corpus_ids (list[int]): A an indexing of our corpus, where documents are 
                   referenced by their index in the corpus. Documents are ranked in descending order.
    '''
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Embedding queries:')
    embedded_queries = indexing_model.encode(queries,device=device, show_progress_bar=True)
    list_ranked_corpus_ids = []
    for query in embedded_queries:
            ranked_corpus_ids, _ = annoy_index.get_nns_by_vector(query, top_k_hits, include_distances=True)
            list_ranked_corpus_ids.append(ranked_corpus_ids)
    return list_ranked_corpus_ids
    

    

