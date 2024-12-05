# Explainable Course Recommendation System (XCRS)

This repository hosts code and data associated with the XCRS project.

[comment]: # (This repository hosts code and data associated with the paper:)

[comment]: # (```bash @inproceedings{,  author    = {},  year      = ,  title     = {},  booktitle = {}}```)


## Reproduce the results

 Firstly, to use the LLM APIs, get API keys from each LLM provider (cohere, google, mistral, openai, voyage) and put under this directory: 
 * [embedding-generation/api_keys](embedding-generation/api_keys) 
 (e.g. embedding-generation/api_keys/***_api_key.txt)

Then, follow the instructions in this file to generate necessary data and embeddings:
 * [README for embedding-generation](embedding-generation/README.md)


To start frontend component:

* [README for frontend](frontend/README.md)

*The frontend application will be ready on [``http://localhost:3000``](http://localhost:3000)*

To start backend component:

* [README for backend](backend/README.md)


### Note

Courses are taken from: [Udemy](https://udemy.com)

Roadmaps are taken from: [roadmap.sh Github](https://github.com/kamranahmedse/developer-roadmap) - [roadmap.sh](https://roadmap.sh)

