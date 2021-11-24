import matplotlib.pyplot as plt
import matplotlib
import time
from indexing.build_corpus import build_corpus
from indexing.compute_mrr import compute_mrr_squad, compute_mrr_squad_annoy

from indexing.indexing import annoy_indexing, batch_indexing, batch_indexing_cpu, corpus_embedding, get_annoy_index
from sentence_transformers import SentenceTransformer, util

def plot_mrr_dot(list_of_models : list) -> None:
  '''
    Plot the MRR of a list of models that use dot products.

            Parameters:
                    list_of_models : list[str]: Our list of models.

            Returns:
                    None: We are ploting the time taken to embed models and the MRR score.
    '''
  time_taken = []
  res_mrrs = []

  fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15))
  corpus, squad_valid, map_question_context = build_corpus(9000)
  for index_model_name in list_of_models:
    model = SentenceTransformer(index_model_name)
    start = time.clock()
    embeded_corpus = corpus_embedding(corpus, model, index_model_name)
    time_taken.append(time.clock() - start)
    res_mrr = compute_mrr_squad(embeded_corpus, map_question_context, model, squad_valid, 'dot')
    res_mrrs.append(res_mrr)
  ax1.bar(list_of_models, time_taken, width = 0.2)
  ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
  ax2.bar(list_of_models, res_mrrs, width = 0.2)


def plot_mrr_cos(list_of_models : list) -> None:
  '''
    Plot the MRR of a list of models that use cosine similarity.

            Parameters:
                    list_of_models : list[str]: Our list of models.

            Returns:
                    None: We are ploting the time taken to embed models and the MRR score.
  '''
  time_taken = []
  res_mrrs = []

  fig, (ax1, ax2) = plt.subplots(2, figsize=(15, 15))
  corpus, squad_valid, map_question_context = build_corpus(9000)
  for index_model_name in list_of_models:
    model = SentenceTransformer(index_model_name)
    start = time.clock()
    embeded_corpus = corpus_embedding(corpus, model, index_model_name)
    time_taken.append(time.clock() - start)
    res_mrr = compute_mrr_squad(embeded_corpus, map_question_context, model, squad_valid, 'angular')
    res_mrrs.append(res_mrr)
  ax1.bar(list_of_models, time_taken, width = 0.2)
  ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
  ax2.bar(list_of_models, res_mrrs, width = 0.2)

  

def compute_annoy(best_model : str, top_k : int) -> None:
    '''
    Compute the time taken to index using the annoy method and the MRR score,
    We use here the best model that we found using the two functions above

            Parameters:
                    best_model (str): The name of our best model.
                    top_k (int):  Number of documents to retrieve.

            Returns:
                    time_taken (int): The time taken to index using this method
                    res_mrr (int): The MRR score of the method using the best model
                        
    '''
    time_taken = 0
    res_mrr = 0

    corpus, squad_valid, map_question_context = build_corpus(9000)
      
    model = SentenceTransformer(best_model)

    query = 'Where is Normandy ?'

    embeded_corpus = corpus_embedding(corpus, model, best_model)
    start = time.clock()
    annoy_index = get_annoy_index(256, embeded_corpus, 768, best_model)
    ranked_corpus_ids = annoy_indexing(annoy_index, query, model, top_k)

    time_taken = time.clock() - start

    res_mrr = compute_mrr_squad_annoy(embeded_corpus, map_question_context, model, best_model, top_k, squad_valid, 'dot')

    return time_taken, res_mrr

def compute_normal(best_model):
    '''
    Compute the time taken to index using the normal method and the MRR score,
    We use here the best model that we found using the two functions above
            Parameters:
                    best_model (str): The name of our best_model.

            Returns:
                    time_taken (int): The time taken to index using this method
                    res_mrr (int): The MRR score of the method using the best model
    '''
    time_taken = 0
    res_mrr = 0

    corpus, squad_valid, map_question_context = build_corpus(9000)
      
    model = SentenceTransformer(best_model)

    query = 'Where is Normandy ?'

    embeded_corpus = corpus_embedding(corpus, model, best_model)
    start = time.clock()
    ranked_corpus_ids = batch_indexing(model, embeded_corpus, [query])[0]
    time_taken = time.clock() - start

    res_mrr = compute_mrr_squad(embeded_corpus, map_question_context, model, squad_valid, 'dot')

    return time_taken, res_mrr


def compare(best_model : str, top_k : int) -> None:
  '''
  Compare the normal and annoy methods using a specific model
  We use here the best model that we found using the two functions above
            Parameters:
                    best_model (str): The name of our best_model.
                    top_k (int):  Number of documents to retrieve.

            Returns:
                    None: We bar plot the differences
  '''
  time_taken_annoy, res_mrr_annoy = compute_annoy(best_model, top_k)
  time_taken_normal, res_mrr_normal = compute_normal(best_model)

  fig, (ax1, ax2) = plt.subplots(2,  figsize=(15, 15))

  ax1.bar(['Annoy', 'Normal'], [time_taken_annoy, time_taken_normal], width = 0.2)
  ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
  ax2.bar(['Annoy', 'Normal'], [res_mrr_annoy, res_mrr_normal], width = 0.2)

def compute_normal_cpu(best_model : str):
  '''
  Compute time taken and MRR score using normal methods using a CPU instead of a GPU
            Parameters:
                    best_model (str): The name of our best_model.

            Returns:
                    time_taken (int): The time taken to index using this method
                    res_mrr (int): The MRR score of the method using the best model
  '''
  time_taken = 0
  res_mrr = 0

  corpus, squad_valid, map_question_context = build_corpus(9000)
      
  model = SentenceTransformer(best_model)

  query = 'Where is Normandy ?'

  embeded_corpus = corpus_embedding(corpus, model, best_model)
  start = time.clock()
  ranked_corpus_ids = batch_indexing_cpu(model, embeded_corpus, [query])[0]
  time_taken = time.clock() - start

  res_mrr = compute_mrr_squad(embeded_corpus, map_question_context, model, squad_valid, 'dot')

  return time_taken, res_mrr

def compare_cpu(best_model : str, top_k : int):
  '''
  Compare the normal using a CPU and annoy methods using a specific model
  We use here the best model that we found using the two functions above
            Parameters:
                    best_model (str): The name of our best_model.
                    top_k (int):  Number of documents to retrieve.

            Returns:
                    None: We bar plot the differences
  '''
  time_taken_annoy, res_mrr_annoy = compute_annoy(best_model, top_k)
  time_taken_normal, res_mrr_normal = compute_normal_cpu(best_model)

  fig, (ax1, ax2) = plt.subplots(2,  figsize=(15, 15))

  ax1.bar(['Annoy', 'Normal'], [time_taken_annoy, time_taken_normal], width = 0.2)
  ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
  ax2.bar(['Annoy', 'Normal'], [res_mrr_annoy, res_mrr_normal], width = 0.2)




  
