import pandas as pd
import numpy as np
import os

import google.generativeai as palm
import voyageai
from openai import OpenAI
from mistralai.client import MistralClient

import tiktoken


PALM_MODEL = "embedding-gecko-001"
VOYAGE_MODEL = "voyage-large-2"
OPENAI_MODEL = "text-embedding-3-small"
MISTRAL_MODEL = "mistral-embed"

class EmbeddingGenerator:

    def __init__(self, udemy_courses_df: pd.DataFrame, roadmap_nodes_df: pd.DataFrame, df_path: str, model_name: str):
        self.udemy_courses_df = udemy_courses_df
        self.roadmap_nodes_df = roadmap_nodes_df
        self.model_name = model_name

        if model_name == PALM_MODEL:
            palm_api_key = open("../api_keys/palm_api_key.txt").read().strip()
            palm.configure(api_key=palm_api_key)
            self.emb_type = "palm_emb"
        elif model_name == VOYAGE_MODEL:
            api_key = open("../api_keys/voyage_api_key.txt").read().strip()
            self.model = voyageai.Client(api_key=api_key)
            self.emb_type = "voyage_emb"
        elif model_name == OPENAI_MODEL:
            api_key = open("../api_keys/openai_api_key.txt").read().strip()
            os.environ["OPENAI_API_KEY"] = api_key
            self.model = OpenAI()
            self.emb_type = "openai_emb"
        elif model_name == MISTRAL_MODEL:
            api_key = open("../api_keys/mistral_api_key.txt").read().strip()
            self.model = MistralClient(api_key=api_key)
            self.emb_type = "mistral_emb"
        else:
            raise Exception("System does not support this model: " + model_name)

        self.df_path = df_path + self.emb_type + "/"

    def palm_embed_fn(self, text):
        return palm.generate_embeddings(model="models/" + self.model_name, text=text)["embedding"]

    def voyage_embed_fn(self, text):
        return self.model.embed([text], model=self.model_name).embeddings[0]
    
    def openai_embed_fn(self, text):
        text = text.replace("\n", " ")
        return self.model.embeddings.create(input = [text], model=self.model_name).data[0].embedding
    
    def mistral_embed_fn(self, text):
        return self.model.embeddings(model=self.model_name, input=text).data[0].embedding
        

    def generate_embeddings_for_courses(self):
        if self.emb_type == "palm_emb":
            self.udemy_courses_df[self.emb_type] = self.udemy_courses_df["concat_text"].apply(self.palm_embed_fn)
        elif self.emb_type == "voyage_emb":
            self.udemy_courses_df[self.emb_type] = self.udemy_courses_df["concat_text"].apply(self.voyage_embed_fn)
        elif self.emb_type == "openai_emb":
            self.udemy_courses_df[self.emb_type] = self.udemy_courses_df["concat_text"].apply(self.openai_embed_fn)
        elif self.emb_type == "mistral_emb":
            self.udemy_courses_df[self.emb_type] = self.udemy_courses_df["concat_text"].apply(self.mistral_embed_fn)

        # Column null check
        # TODO: logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
        # logging.debug('...')
        print("Number of null embedding value for Courses Dataframe: " + str(self.udemy_courses_df[self.emb_type].isnull().sum()))

        udemy_courses_emb_df = self.udemy_courses_df[[self.emb_type]].rename(columns={self.emb_type: "emb"})
        udemy_courses_emb_df.to_csv(self.df_path + "udemy_courses_{}.csv".format(self.model_name))

        return udemy_courses_emb_df

    def mean_of_embeddings(self, indices, dataframe):
        embeddings = dataframe.loc[indices]
        embeddings = embeddings[embeddings["type"] == "concept"]
        embedding_array = np.array(embeddings[self.emb_type])
        mean_embedding = np.mean(embedding_array, axis=0)
        return mean_embedding.tolist()

    def embed_concepts(self, row):
        if row["type"] == "concept":
            if self.emb_type == "palm_emb":
                return self.palm_embed_fn(row["content"])
            elif self.emb_type == "voyage_emb":
                return self.voyage_embed_fn(row["content"])
            elif self.emb_type == "openai_emb":
                return self.openai_embed_fn(row["content"])
            elif self.emb_type == "mistral_emb":
                return self.mistral_embed_fn(row["content"])
        else:
            return None

    def embed_topics(self, row):
        if row["type"] == "concept":
            return row[self.emb_type]
        elif row["type"] == "topic":
            subtopic_list = list(filter(lambda x: str(x).startswith(str(row.name)), self.roadmap_nodes_df.index))
            subtopic_list.remove(row.name)
            return self.mean_of_embeddings(subtopic_list, self.roadmap_nodes_df)
        else:
            return None

    def generate_embeddings_for_roadmaps(self):
        self.roadmap_nodes_df[self.emb_type] = self.roadmap_nodes_df.apply(self.embed_concepts, axis=1)
        self.roadmap_nodes_df[self.emb_type] = self.roadmap_nodes_df.apply(self.embed_topics, axis=1)

        # Column null check
        print("Number of null embedding value for Roadmap Nodes Dataframe: " + str(self.roadmap_nodes_df[self.emb_type].isnull().sum()))

        roadmap_nodes_emb_df = self.roadmap_nodes_df[[self.emb_type]].rename(columns={self.emb_type: "emb"})
        roadmap_nodes_emb_df.to_csv(self.df_path + "roadmap_nodes_{}.csv".format(self.model_name))

        return roadmap_nodes_emb_df


    

    def num_tokens_from_string(self, text: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding_name = "cl100k_base"
        encoding = tiktoken.get_encoding(encoding_name)
        num_tokens = len(encoding.encode(text))
        return num_tokens

    def return_num_of_tokens(self):

        if self.emb_type == "palm_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)
        elif self.emb_type == "voyage_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)
        elif self.emb_type == "openai_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)
        elif self.emb_type == "mistral_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)  
        
        total_token_count = self.udemy_courses_df["token_count"].sum()

        print(total_token_count)
        # udemy_courses_emb_df.to_csv(self.df_path + "udemy_courses_{}.csv".format(self.model_name))

        return total_token_count