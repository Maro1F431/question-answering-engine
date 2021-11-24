import matplotlib.pyplot as plt
import time
from indexing.build_corpus import build_corpus
from indexing.compute_mrr import compute_mrr_squad, compute_mrr_squad_annoy

from indexing.indexing import corpus_embedding
from sentence_transformers import SentenceTransformer, util


def plot_mrr_dot(list_of_models : list) -> None:
  time_taken = []
  res_mrrs = []

  fig, (ax1, ax2) = plt.subplots(2)
  corpus, squad_valid, map_question_context = build_corpus(9000)
  for index_model_name in list_of_models:
    model = SentenceTransformer(index_model_name)
    embeded_corpus = corpus_embedding(corpus, model)
    start = time.clock()
    res_mrr = compute_mrr_squad(embeded_corpus, map_question_context, model, squad_valid, 'dot')
    time_taken.append(time.clock() - start)
    res_mrrs.append(res_mrr)
  ax1.hist(time_taken)
  ax2.hist(res_mrrs)




def plot_mrr_cos(list_of_models : list):
  time_taken = []
  res_mrrs = []

  fig, (ax1, ax2) = plt.subplots(2)
  corpus, squad_valid, map_question_context = build_corpus(9000)
  for index_model_name in list_of_models:
    model = SentenceTransformer(index_model_name)
    embeded_corpus = corpus_embedding(corpus, model)
    start = time.clock()
    res_mrr = compute_mrr_squad(embeded_corpus, map_question_context, model, squad_valid, 'cos')
    time_taken.append(time.clock() - start)
    res_mrrs.append(res_mrr)
  ax1.hist(time_taken)
  ax2.hist(res_mrrs)

  def plot_mrr_annoy_dot(list_of_models : list, top_k : int) -> None:
    time_taken = []
    res_mrrs = []

    fig, (ax1, ax2) = plt.subplots(2)
    corpus, squad_valid, map_question_context = build_corpus(9000)
    for index_model_name in list_of_models:
      model = SentenceTransformer(index_model_name)
      embeded_corpus = corpus_embedding(corpus, model)
      start = time.clock()
      res_mrr = compute_mrr_squad_annoy(embeded_corpus, map_question_context, model, index_model_name, top_k, squad_valid, 'dot')
      time_taken.append(time.clock() - start)
      res_mrrs.append(res_mrr)
    ax1.hist(time_taken)
    ax2.hist(res_mrrs)

  def plot_mrr_annoy_cos(list_of_models : list, top_k : int):
    time_taken = []
    res_mrrs = []

    fig, (ax1, ax2) = plt.subplots(2)
    corpus, squad_valid, map_question_context = build_corpus(9000)
    for index_model_name in list_of_models:
      model = SentenceTransformer(index_model_name)
      embeded_corpus = corpus_embedding(corpus, model)
      start = time.clock()
      res_mrr = compute_mrr_squad_annoy(embeded_corpus, map_question_context, model, index_model_name, top_k, squad_valid, 'cos')
      time_taken.append(time.clock() - start)
      res_mrrs.append(res_mrr)
    ax1.hist(time_taken)
    ax2.hist(res_mrrs)




  
