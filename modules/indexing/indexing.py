import torch
import tqdm as tq
import numpy as np
import os

def corpus_embedding(corpus, model):
    # Enables GPU if avalaible for embedding. It is strongly adviced to run this on a GPU
    # since the computation on a CPU can take up to an hour on a 10k dataset.
    device = 'cuda:0' if torch.cuda.is_available() else None
    text_only_corpus = [elm['text'] for elm in corpus]
    text_embedded_corpus = model.encode(text_only_corpus,device=device, show_progress_bar=True)
    embedded_corpus = corpus
    for i,elm in enumerate(embedded_corpus):
        elm['text_embedding'] = text_embedded_corpus[i]
    return embedded_corpus

def indexing(model, embedded_corpus, query, comparison_metric):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    embedded_query = model.encode(query)
    corpus_embeddings = [doc['text_embedding'] for doc in embedded_corpus]
    similarity_list = list(map(lambda x : comparison_metric(embedded_query, x), corpus_embeddings))
    ranked_corpus_ids = np.argsort(similarity_list)[::-1]
    return ranked_corpus_ids

def batch_indexing(model, embedded_corpus, queries, torch_comparison_metric):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Embedding queries:')
    embedded_queries = model.encode(queries,device=device, show_progress_bar=True)
    corpus_embeddings = [doc['text_embedding'] for doc in embedded_corpus]
    tensor_embedded_query = torch.FloatTensor(embedded_queries).to(device)
    tensor_corpus_embeddings = torch.FloatTensor(corpus_embeddings).to(device)
    similarity_tensor = torch_comparison_metric(tensor_embedded_query, tensor_corpus_embeddings.t())
    _, tensor_ranked_corpus_ids = torch.sort(similarity_tensor, descending=True, dim=1)
    return tensor_ranked_corpus_ids.tolist()

def get_annoy_index(n_trees, embedded_corpus, embedding_size, model_name, comparison_metric='dot'):
    annoy_index_path = '{}-embdsize{}-annoy_index-{}-trees.ann'.format(model_name.replace('/', '_'), embedding_size, n_trees)
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

def annoy_indexing(annoy_index, query, indexing_model, top_k_hits):
    query_embedding = indexing_model.encode(query)
    ranked_corpus_ids, scores = annoy_index.get_nns_by_vector(query_embedding, top_k_hits, include_distances=True)
    return ranked_corpus_ids
    

    

