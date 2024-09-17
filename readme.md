## Unlocking LLM Insights: A Dataset for Automatic Model Card Generation  

Language models (LMs) are no longer restricted to the ML community, and instruction-following LMs have led to a rise in autonomous AI agents. As the accessibility of LMs grows, it is imperative that an understanding of their capabilities, intended usage, and development cycle also improves. Model cards are a widespread practice for documenting detailed information about an ML model. To automate model card generation, we introduce a dataset of 500 question-answer pairs for 25 LMs that cover crucial aspects of the model, such as its training configurations, datasets, biases, architecture details, and training resources. We employ annotators to extract the answers from the original paper. Further, we explore the capabilities of LMs in generating model cards by answering questions. We experiment with three configurations: zero-shot generation, retrieval-augmented generation, and fine-tuning on our dataset. The fine-tuned Llama 3 model shows an improvement of 7 points over the retrieval-augmented generation setup. This indicates that our dataset can be used to train models to automatically generate model cards from paper text and reduce the human effort in the model card curation process.  


### Directory Organization  
The `qa` directory contains the code to reproduce zeroshot, retrieval-augmented generation (rag), and supervised fine-tuning (sft) results.   
The `evals` directory contains the code to compute metrics.  

The `data` directory contains the modelcard dataset. The train and test splits are in directory `data/modelcards`. 

### Setting up the repository
`conda create -n modelcards python=3.10.14`  
`conda activate modelcards`  
`pip install -r requirements.txt`  

Gated models such as Llama require setting the appropriate huggingface token:  
huggingface-cli login HFTOKEN  

