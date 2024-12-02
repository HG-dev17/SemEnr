# readme
`My-try folder is my contribute`
Prepare Environment contributed by licyk
Environment add ollama.
this is my try and help:[https://hg-dev17.github.io/2024/12/02/ollama-install/](https://hg-dev17.github.io/2024/12/02/ollama-install/)
# SemEnr
Code Semantic Enrichment for Deep Code Search

## Dependency
> Tested in Ubuntu 16.04
* Python 2.7-3.6
* Keras 2.1.3 or newer
* Tensorflow-gpu 1.7.0
* lucene 7.7.1


## Prepare Environment
```
conda create -f environment.yml
conda activate semenr
pip install -r requirements.txt
```


## Usage

   ### DataSets
  The datasets used in our paper will be found at: https://drive.google.com/drive/folders/1j-0xukLQWGrJ8-Lxw7vFAbubFTyXJT2C?usp=sharing
  
   ### Data Process
   If you want to reprocess the data, you can process it into a usable form for the model by following steps:
   
   1.Build corpus for each features (i.e., description, tokens):
   
   `python createCorpus.py` `python createVocab.py` `python vocab2pkl.py`
   
   2.Processing training data and testing data according to the corpus:
   
   `python txt2pkl.py`
   
   ### Code Enrichment Module
   Build retrieval base: `python Index.py`
   
   Perform search: `python Search.py`
   
   Remove stop words: `python deleteStopWords.py`
   
   ### Code Search Module
   
   #### Configuration
   Put the data set into the `data/github` directory under `keras`
   
   Edit hyper-parameters and settings in `config.py`
   
   #### Train and Evaluate
   
   ```bash
   python main.py --mode train
   python main.py --mode eval
