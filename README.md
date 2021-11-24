# IR/QA system

Install the requirements (adviced to use a virutal env) and use the question_answering.py script to test the system:
```
pip install -r requirements.txt
python question_answering.py [args]
```
Args for question_answering.py:

--annoy: uses annoy indexing
--nb-dbpedia: number of dbpedia samples to add to the corpus
--topk: number of contexts to consider
--indexmode: name of the huggingface indexing model to use
--qamodel : name of the huggingface qa model to use


Architecture description:

./question_answering.py : Main script to run the demo of the system.

./modules/ : Directory containing the reusable code.

    ./modules/indexing/ : Directory containing all the code about the indexing part of the system.
        ./modules/indexing/build_corpus.py : Contains all functions used to build the corpus.
        ./modules/indexing/compute_mrr.py : Contains all functions used to compute the MRR.
        ./modules/indexing/indexing.py : Contains all functions used to index the corpus.

    ./modules/question_answering/ : Directory containing all the code related to the question answering part of the system.
        ./modules/question_answering/best_answer.py : Contains the function used to pick the best answers for a given question and contexts.
        ./modules/question_answering/model_evaluation.py : Contains all functions related to the evaluation of QA models.
        ./modules/question_answering/model_training.py : Contains all functions related to the fine-tuning of QA models.

    ./modules/benchmarking/ : Directory containing all the code used for benchmarking.
        ./modules/benchmarking/compare_qa_models.py: Contains the function used to evaluate a QA model for given metrics.
        ./modules/benchmarking/score_full_system.py: Contains the function used to score the system as a whole.
        ./modules/benchmarking/plot_mrr.py