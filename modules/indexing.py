import torch
import tqdm as tq

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

def indexing(model, embedded_corpus, query):
    embedded_query = model.encode(query)
    ranked_documents = sorted(tq.tqdm(embedded_corpus), key=lambda x : util.pytorch_cos_sim(embedded_query, x['text_embedding']))
    return ranked_documents

