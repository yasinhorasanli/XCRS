import numpy as np
import pandas as pd
import os
import json

# internal classes
import util
from embedding_generator import EmbeddingGenerator

df_path = "../data/"
roadmap_data_path = "../data/roadmaps_in_json"
roles = [
    "ai-data-scientist",
    "android",
    "backend",
    "blockchain",
    "devops",
    "frontend",
    "full-stack",
    "game-developer",
    "qa",
    "ux-design",
]


def cleanse_courses(course_df_path) -> pd.DataFrame:

    df = pd.read_csv(course_df_path, index_col=0)
    print(len(df))
    # 973

    df["name_from_site"].isnull().values.any()
    error_df = df[df["name_from_site"].isnull()]
    df = df.drop(error_df.index)
    df = df.fillna("")
    df = df[~df.index.duplicated(keep="first")]
    print(len(df))
    # 970

    """
    ####### COURSE CATEGORIES ########
    Business, Design, Development, Finance & Accounting, Health & Fitness
    IT & Software, Lifestyle, Marketing, Music, Office Productivity, Personal Development
    Photography & Video, Teaching & Academics, Udemy Free Resource Center, Vodafone
    """

    selected_df = df[df["category"].apply(util.category_matcher)]
    print(len(selected_df))
    # 453

    selected_df["concat_text"] = selected_df.apply(util.combine_columns, axis=1)

    selected_df.to_csv(df_path + "udemy_courses_{}.csv".format("cleansed"))

    return selected_df


def create_roadmap_df(roadmap_data_path) -> pd.DataFrame:

    roadmap_nodes_df = pd.DataFrame()
    for idx, role in enumerate(roles):
        roadmap_nodes_path = os.path.join(roadmap_data_path, role + ".json")

        with open(roadmap_nodes_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for node in data["children"]:
                node_df = util.traverse_nodes(node, idx + 1)
                roadmap_nodes_df = pd.concat([roadmap_nodes_df, node_df])
            roadmap_nodes_df = roadmap_nodes_df.reset_index(drop=True).dropna()

    roadmap_nodes_df.reset_index(drop=True, inplace=True)
    roadmap_nodes_df.set_index("id", inplace=True)
    roadmap_nodes_df.to_csv(df_path + "roadmap_nodes.csv")

    return roadmap_nodes_df


def main():

    # Load Raw Course and Roadmap DataFrames
    if os.path.exists(df_path + "udemy_courses_cleansed.csv"):
        udemy_courses_df = pd.read_csv(df_path + "udemy_courses_cleansed.csv", index_col=0)
        print(len(udemy_courses_df))
        # 453
    else:
        udemy_courses_df = cleanse_courses(df_path + "udemy_course_data.csv")

    if os.path.exists(df_path + "roadmap_nodes.csv"):
        roadmap_nodes_df = pd.read_csv(df_path + "roadmap_nodes.csv", index_col=0)
        print(len(roadmap_nodes_df))
        # 1104
    else:
        roadmap_nodes_df = create_roadmap_df(df_path + "roadmaps_in_json")

    roadmaps_dict = {"id": np.arange(1, len(roles) + 1), "name": roles}
    roadmaps_df = pd.DataFrame.from_dict(roadmaps_dict)
    roadmaps_df.set_index("id", inplace=True)
    # print(roadmaps_df)

    # Save Final DataFrames
    udemy_courses_df.to_csv(df_path + "udemy_courses_final.csv")
    roadmap_nodes_df.to_csv(df_path + "roadmap_nodes_final.csv")

    # EMB. GEN. for Model: "embedding-gecko-001"
    palm_embedding_generator = EmbeddingGenerator(udemy_courses_df, roadmap_nodes_df, df_path, "embedding-gecko-001")
    palm_embedding_generator.generate_embeddings_for_courses()
    palm_embedding_generator.generate_embeddings_for_roadmaps()

    # EMB. GEN. for Model: "voyage-large-2"
    voyage_embedding_generator = EmbeddingGenerator(udemy_courses_df, roadmap_nodes_df, df_path, "voyage-large-2")
    voyage_embedding_generator.generate_embeddings_for_courses()
    voyage_embedding_generator.generate_embeddings_for_roadmaps()

    return


if __name__ == "__main__":
    main()
