import pandas as pd
import numpy as np
import os

import google.generativeai as genai
import voyageai
from openai import OpenAI
from mistralai.client import MistralClient
import cohere

import tiktoken


GOOGLE_MODEL = "text-embedding-004"
VOYAGE_MODEL = "voyage-large-2"
OPENAI_MODEL = "text-embedding-3-large"
MISTRAL_MODEL = "mistral-embed"
COHERE_MODEL = "embed-english-v3.0"


class EmbeddingGenerator:

    def __init__(self, udemy_courses_df: pd.DataFrame, roadmap_nodes_df: pd.DataFrame, df_path: str, model_name: str):
        self.udemy_courses_df = udemy_courses_df
        self.roadmap_nodes_df = roadmap_nodes_df
        self.model_name = model_name
        self.batch_size = 100

        if model_name == GOOGLE_MODEL:
            google_api_key = open("../api_keys/google_api_key.txt").read().strip()
            genai.configure(api_key=google_api_key)
            self.emb_type = "google_emb"
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
        elif model_name == COHERE_MODEL:
            api_key = open("../api_keys/cohere_api_key.txt").read().strip()
            self.model = cohere.Client(api_key)
            self.emb_type = "cohere_emb"
        else:
            raise Exception("System does not support this model: " + model_name)

        self.df_path = df_path + self.emb_type + "/"

        self.embed_functions = {
            "google_emb": self.google_embed_fn,
            "voyage_emb": self.voyage_embed_fn,
            "openai_emb": self.openai_embed_fn,
            "mistral_emb": self.mistral_embed_fn,
            "cohere_emb": self.cohere_embed_fn,
        }

    def google_embed_fn(self, text_list):
        return genai.embed_content(model="models/" + self.model_name, content=text_list, task_type="similarity")["embedding"]

    def voyage_embed_fn(self, texts_list):
        return self.model.embed(texts_list, model=self.model_name).embeddings

    def openai_embed_fn(self, text_list):
        text_list = [text.replace("\n", " ") for text in text_list]
        embeddings = self.model.embeddings.create(input=text_list, model=self.model_name).data
        return [embedding.embedding for embedding in embeddings]

    def mistral_embed_fn(self, text_list):
        embeddings = self.model.embeddings(model=self.model_name, input=text_list).data
        return [embedding.embedding for embedding in embeddings]

    def cohere_embed_fn(self, text_list):
        return self.model.embed(model=self.model_name, texts=text_list, input_type="search_document").embeddings

    def generate_embeddings_for_courses(self):

        courses = self.udemy_courses_df["concat_text"].tolist()
        embed_fn = self.embed_functions.get(self.emb_type, None)

        if embed_fn:
            # Process contents in batches
            batch_size = self.batch_size
            if self.emb_type == "mistral_emb":
                batch_size = 10
            embeddings = []
            for i in range(0, len(courses), batch_size):
                batch_courses = courses[i : i + batch_size]
                batch_embeddings = embed_fn(batch_courses)
                embeddings.extend(batch_embeddings)

            # Ensure the lengths match
            if len(embeddings) != len(courses):
                raise ValueError("Mismatch between number of embeddings and courses")

            # Update the DataFrame with embeddings
            # self.udemy_courses_df[self.emb_type] = pd.Series(embeddings)
            self.udemy_courses_df[self.emb_type] = embeddings
        else:
            # Handle the case where emb_type is not valid
            raise ValueError(f"Embedding type {self.emb_type} is not valid")

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

    def embed_concepts(self):

        concept_rows = self.roadmap_nodes_df[self.roadmap_nodes_df["type"] == "concept"]
        contents = concept_rows["content"].tolist()
        indices = concept_rows.index.tolist()

        embed_fn = self.embed_functions.get(self.emb_type, None)

        if embed_fn:
            # Process contents in batches
            batch_size = self.batch_size
            if self.emb_type == "mistral_emb":
                batch_size = 10
            embeddings = []
            for i in range(0, len(contents), batch_size):
                batch_contents = contents[i : i + batch_size]
                batch_embeddings = embed_fn(batch_contents)
                embeddings.extend(batch_embeddings)

            # Ensure the lengths match
            if len(embeddings) != len(concept_rows):
                raise ValueError("Mismatch between number of embeddings and concept rows")

            # Update the DataFrame with embeddings
            self.roadmap_nodes_df.loc[self.roadmap_nodes_df["type"] == "concept", self.emb_type] = pd.Series(embeddings, index=indices)

            # self.roadmap_nodes_df.loc[self.roadmap_nodes_df['type'] == 'concept', self.emb_type] = embeddings
        else:
            # Handle the case where emb_type is not valid
            raise ValueError(f"Embedding type {self.emb_type} is not valid")

    def embed_topics(self):

        topic_rows = self.roadmap_nodes_df[self.roadmap_nodes_df["type"] == "topic"]

        for idx, row in topic_rows.iterrows():
            subtopic_list = list(filter(lambda x: str(x).startswith(str(idx)), self.roadmap_nodes_df.index))
            subtopic_list.remove(idx)

            if subtopic_list:
                self.roadmap_nodes_df.at[idx, self.emb_type] = self.mean_of_embeddings(subtopic_list, self.roadmap_nodes_df)
            else:
                self.roadmap_nodes_df.at[idx, self.emb_type] = None

    def generate_embeddings_for_roadmaps(self):

        self.embed_concepts()
        self.embed_topics()

        # self.roadmap_nodes_df[self.emb_type] = self.roadmap_nodes_df.apply(self.embed_concepts, axis=1)
        # self.roadmap_nodes_df[self.emb_type] = self.roadmap_nodes_df.apply(self.embed_topics, axis=1)

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

        if self.emb_type == "google_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)
        elif self.emb_type == "voyage_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)
        elif self.emb_type == "openai_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)
        elif self.emb_type == "mistral_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)
        elif self.emb_type == "cohere_emb":
            self.udemy_courses_df["token_count"] = self.udemy_courses_df["concat_text"].apply(self.num_tokens_from_string)

        total_token_count = self.udemy_courses_df["token_count"].sum()

        print(total_token_count)
        # udemy_courses_emb_df.to_csv(self.df_path + "udemy_courses_{}.csv".format(self.model_name))

        return total_token_count
