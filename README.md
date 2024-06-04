# Negative-Free Self-Supervised Gaussian Embedding of Graphs

Official Implementation of Negative-Free Self-Supervised Gaussian Embedding of Graphs.



## Dependencies

- dgl
- torch
- scikit-learn



## Reproduction

Copy hyper-parameters from [params.txt](./params.txt) to [main.py](./main.py) and run it.



## Datasets

For the Cora, CiteSeer, PubMed, WikiCS, Computer, and CoauthorCS datasets, we use the processed version provided by [Deep Graph Library](https://docs.dgl.ai/api/python/dgl.data.html). For the ArXiv dataset, we use the processed version provided by [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv).

| Dataset    | Type          | #Nodes  | #Edges    | #Features | #Classes |
| ---------- | ------------- | ------- | --------- | --------- | -------- |
| Cora       | citation      | 2,708   | 10,556    | 1,433     | 7        |
| CiteSeer   | citation      | 3,327   | 9,228     | 3,703     | 6        |
| PubMed     | citation      | 19,717  | 88,651    | 500       | 3        |
| WikiCS     | reference     | 11,701  | 431,726   | 300       | 10       |
| Computer   | co-purchase   | 13,752  | 491,722   | 767       | 10       |
| CoauthorCS | co-authorship | 18,333  | 163,788   | 6,805     | 15       |
| ArXiv      | citation      | 169,343 | 2,315,598 | 128       | 40       |