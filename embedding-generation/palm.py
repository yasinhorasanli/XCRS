import pandas as pd
import numpy as np
import google.generativeai as palm


class PalmEmb:
    def __init__(self, udemy_courses_df: pd.DataFrame, roadmap_nodes_df: pd.DataFrame, df_path: str):
        self.udemy_courses_df = udemy_courses_df
        self.roadmap_nodes_df = roadmap_nodes_df
        self.df_path = df_path
        self.palm_model = "embedding-gecko-001"

        palm_api_key = open("palm_api_key.txt").read().strip()
        palm.configure(api_key=palm_api_key)

    def embed_courses(self, text):
        return palm.generate_embeddings(model="models/" + self.palm_model, text=text)["embedding"]

    def generate_palm_embedding_courses(self):
        self.udemy_courses_df["palm_emb"] = self.udemy_courses_df["concat_text"].apply(self.embed_courses)

        # Column null check
        print(self.udemy_courses_df[self.udemy_courses_df["palm_emb"].isnull()])

        self.udemy_courses_df.to_csv(self.df_path + "udemy_courses_{}.csv".format(self.palm_model))

        return self.udemy_courses_df

    def mean_of_embeddings(self, indices, dataframe):
        embeddings = dataframe.loc[indices]
        embeddings = embeddings[embeddings["type"] == "concept"]
        embedding_array = np.array(embeddings["palm_emb"])
        mean_embedding = np.mean(embedding_array, axis=0)
        return mean_embedding.tolist()

    def embed_concepts(self, row):
        if row["type"] == "concept":
            return palm.generate_embeddings(model="models/embedding-gecko-001", text=row["content"])["embedding"]
        else:
            return None

    def embed_topics(self, row):
        if row["type"] == "concept":
            return row["palm_emb"]
        elif row["type"] == "topic":
            subtopic_list = list(filter(lambda x: str(x).startswith(str(row.name)), self.roadmap_nodes_df.index))
            subtopic_list.remove(row.name)
            return self.mean_of_embeddings(subtopic_list, self.roadmap_nodes_df)
        else:
            return None

    def generate_palm_embedding_roadmap(self):
        self.roadmap_nodes_df["palm_emb"] = self.roadmap_nodes_df.apply(self.embed_concepts, axis=1)
        self.roadmap_nodes_df["palm_emb"] = self.roadmap_nodes_df.apply(self.embed_topics, axis=1)

        # Column null check
        print(self.roadmap_nodes_df[self.roadmap_nodes_df["palm_emb"].isnull()])

        self.roadmap_nodes_df.to_csv(self.df_path + "roadmap_nodes_{}.csv".format(self.palm_model))

        return self.roadmap_nodes_df
