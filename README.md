# Explainable Course Recommendation System (XCRS)

This repository hosts code and data associated with the XCRS project.

[comment]: # (This repository hosts code and data associated with the paper:)

[comment]: # (```bash @inproceedings{,  author    = {},  year      = ,  title     = {},  booktitle = {}}```)

## Repository Structure

Our package is divided in folders and it is organized as follows:
- [**backend**](backend): This folder contains the implementation of backend application, backbone of our system. 
- [**embedding-generation**](backend): This folder contains the embedding generation component of the XCRS. 
- [**frontend**](frontend): This folder contains the frontend component of the XCRS.
- [**evaluation-materials**](evaluation-materials): This folder contains the user study materials (protocol, questions, and responses).

## Reproduce the results

 Firstly, to use the LLM APIs, get API keys from each LLM provider (cohere, google, mistral, openai, voyage) and put under this directory: 
 * [embedding-generation/api_keys](embedding-generation/api_keys) 
 (e.g. embedding-generation/api_keys/***_api_key.txt)

Then, follow the instructions in this file to generate necessary data and embeddings:
 * [instructions for embedding-generation](embedding-generation/README.md)


To start frontend component:

* [instructions for frontend](frontend/README.md)

*The frontend application will be ready on [``http://localhost:3000``](http://localhost:3000)*

To start backend component:

* [instructions for backend](backend/README.md)


### Note

Courses are taken from: [Udemy](https://udemy.com)

Roadmaps are taken from: [roadmap.sh Github](https://github.com/kamranahmedse/developer-roadmap) - [roadmap.sh](https://roadmap.sh)


# Contact
If you have any questions or are interested in contributing to this project, please don't hesitate to contact me:

* Muhammed Yasin Horasanli (muhammed.horasanli@std.bogazici.edu.tr)
