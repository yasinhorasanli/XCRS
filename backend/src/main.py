# API
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import logging
from pprint import pformat
import os
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as genai
import uvicorn
import voyageai
from openai import OpenAI
from mistralai.client import MistralClient
import cohere as cohereai


# internal classes
import util
from eda import ExplatoryDataAnalysis
from recom import RecommendationEngine
from models import (
    CourseRecommendation,
    RoleRecommendation,
    Recommendation,
    RecommendationResponse,
    RecommendationRequest,
    sample_rec_data,
    INSUFFICIENT_INPUT,
)
from models import GOOGLE_MODEL, VOYAGE_MODEL, OPENAI_MODEL, MISTRAL_MODEL, COHERE_MODEL

google_api_key = open("../../embedding-generation/api_keys/google_api_key.txt").read().strip()
voyage_api_key = open("../../embedding-generation/api_keys/voyage_api_key.txt").read().strip()
openai_api_key = open("../../embedding-generation/api_keys/openai_api_key.txt").read().strip()
mistral_api_key = open("../../embedding-generation/api_keys/mistral_api_key.txt").read().strip()
cohere_api_key = open("../../embedding-generation/api_keys/cohere_api_key.txt").read().strip()


voyage = voyageai.Client(api_key=voyage_api_key)
genai.configure(api_key=google_api_key)
os.environ["OPENAI_API_KEY"] = openai_api_key
openai = OpenAI()
mistral = MistralClient(api_key=mistral_api_key)
cohere = cohereai.Client(cohere_api_key)

udemy_website = "www.udemy.com"

logging.basicConfig(filename="../log/backend.log", filemode="a", format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


def google_embed_fn(text_list):
    return genai.embed_content(model="models/" + GOOGLE_MODEL, content=text_list, task_type="similarity")["embedding"]


def voyage_embed_fn(texts_list):
    return voyage.embed(texts_list, model=VOYAGE_MODEL).embeddings


def openai_embed_fn(text_list):
    text_list = [text.replace("\n", " ") for text in text_list]
    embeddings = openai.embeddings.create(input=text_list, model=OPENAI_MODEL).data
    return [embedding.embedding for embedding in embeddings]


def mistral_embed_fn(text_list):
    embeddings = mistral.embeddings(model=MISTRAL_MODEL, input=text_list).data
    return [embedding.embedding for embedding in embeddings]


def cohere_embed_fn(text_list):
    return cohere.embed(model=COHERE_MODEL, texts=text_list, input_type="search_document").embeddings


embed_functions = {
    GOOGLE_MODEL: google_embed_fn,
    VOYAGE_MODEL: voyage_embed_fn,
    OPENAI_MODEL: openai_embed_fn,
    MISTRAL_MODEL: mistral_embed_fn,
    COHERE_MODEL: cohere_embed_fn,
}


def init_data():
    global udemy_courses_df, roadmap_concepts_df, roadmaps_df

    df_path = "../../embedding-generation/data/"
    udemy_courses_file = "udemy_courses_final.csv"
    roadmap_nodes_file = "roadmap_nodes_final.csv"
    udemy_courses_df = pd.read_csv(df_path + udemy_courses_file)
    roadmap_nodes_df = pd.read_csv(df_path + roadmap_nodes_file)
    roadmap_concepts_df = roadmap_nodes_df[roadmap_nodes_df["type"] == "concept"].copy()
    roadmap_concepts_df.reset_index(inplace=True)
    # roadmap_topics_df = roadmap_nodes_df[roadmap_nodes_df["type"] == "concept"].copy()
    # roadmap_topics_df.reset_index(inplace=True)

    roles = [
        "AI Data Scientist",
        "Android Developer",
        "Backend Developer",
        "Blockchain Developer",
        "Devops Engineer",
        "Frontend Developer",
        "Full Stack Developer",
        "Game Developer",
        "QA Engineer",
        "UX Designer",
    ]

    roadmaps_dict = {"id": np.arange(1, len(roles) + 1), "name": roles}
    roadmaps_df = pd.DataFrame.from_dict(roadmaps_dict)
    roadmaps_df.set_index("id", inplace=True)

    logger.info("Data is initialized using " + udemy_courses_file + " and " + roadmap_nodes_file)
    logger.info("Total number of roadmap concepts: " + str(roadmap_concepts_df.shape[0]))
    logger.info("Total number of courses: " + str(udemy_courses_df.shape[0]))
    logger.info("Career Roles: \n" + pformat(list(zip(np.arange(1, len(roles) + 1), roles))))

    # return (udemy_courses_df, roadmap_nodes_df, roadmap_concepts_df, roadmaps_df)


def get_emb_lists(folder: str, model: str):

    df_path = "../../embedding-generation/data/" + folder + "/"
    udemy_courses_emb_df = pd.read_csv(df_path + "udemy_courses_{}.csv".format(model))
    roadmap_nodes_emb_df = pd.read_csv(df_path + "roadmap_nodes_{}.csv".format(model))
    roadmap_concepts_emb_df = roadmap_nodes_emb_df[roadmap_nodes_emb_df["id"].isin(concept_id_list)]
    # roadmap_topics_emb_df = roadmap_nodes_emb_df[roadmap_nodes_emb_df["id"].isin(topic_id_list)]

    udemy_courses_emb_df.loc[:, "emb"] = udemy_courses_emb_df.apply(util.convert_to_float, axis=1)
    course_emb_list = udemy_courses_emb_df["emb"].values
    course_emb_list = np.vstack(course_emb_list)

    roadmap_concepts_emb_df.loc[:, "emb"] = roadmap_concepts_emb_df.apply(util.convert_to_float, axis=1)
    concept_emb_list = roadmap_concepts_emb_df["emb"].values
    concept_emb_list = np.vstack(concept_emb_list)

    # roadmap_topics_emb_df.loc[:, "emb"] = roadmap_topics_emb_df.apply(util.convert_to_float, axis=1)
    # topic_emb_list = roadmap_topics_emb_df["emb"].values
    # topic_emb_list = np.vstack(topic_emb_list)

    return course_emb_list, concept_emb_list


def create_user_embeddings(took_and_liked: str, took_and_neutral: str, took_and_disliked: str, curious: str, model: str):

    category = ["TookAndLiked", "TookAndNeutral", "TookAndDisliked", "Curious"]
    user_df = pd.DataFrame(columns=category)

    user_df = pd.DataFrame.from_dict(
        [
            {
                "TookAndLiked": took_and_liked,
                "TookAndNeutral": took_and_neutral,
                "TookAndDisliked": took_and_disliked,
                "Curious": curious,
            }
        ]
    )

    result_dicts = []
    columns_to_process = category

    for column in columns_to_process:
        result_dicts.extend(user_df.apply(lambda row: util.split_and_create_dict(row, column), axis=1).to_list())

    flat_data = [item for sublist in result_dicts for item in sublist.items()]
    user_courses_df = pd.DataFrame(flat_data, columns=["course", "category"])

    # Drop rows where 'course' is null, empty, or only blank spaces
    user_courses_df = user_courses_df.dropna(subset=["course"])
    user_courses_df = user_courses_df[user_courses_df["course"].str.strip() != ""]
    user_courses_df = user_courses_df.reset_index(drop=True)

    # logger.info(user_courses_df)

    # Embedding Generation for User Data
    # TODO: If it is possible, search courses in wikipedia here.

    batch_size = 100
    user_courses = user_courses_df["course"].tolist()
    embed_fn = embed_functions.get(model, None)

    if embed_fn:
        # Process contents in batches
        embeddings = []
        for i in range(0, len(user_courses), batch_size):
            batch_user_courses = user_courses[i : i + batch_size]
            batch_embeddings = embed_fn(batch_user_courses)
            embeddings.extend(batch_embeddings)

        # Ensure the lengths match
        if len(embeddings) != len(user_courses):
            raise ValueError("Mismatch between number of embeddings and user courses")

        # Update the DataFrame with embeddings
        user_courses_df["emb"] = embeddings
    else:
        # Handle the case where emb_type is not valid
        raise ValueError(f"Embedding type {model} is not valid")

    user_emb_list = user_courses_df["emb"].values
    user_emb_list = np.vstack(user_emb_list)
    logger.info("User courses embeddings created. Shape: " + str(user_emb_list.shape))

    return (user_courses_df, user_emb_list)


def find_similar_concepts_for_courses(user_courses_df, user_emb_list, concepts_emb_list, roadmap_concepts_df, sim_thre):

    similarity_matrix = cosine_similarity(user_emb_list, concepts_emb_list)

    # Find courses with similarity score > sim_thre for each concept
    result = []
    for i, row in user_courses_df.iterrows():
        course_id = i
        course_name = row["course"]
        category = row["category"]
        similarities = similarity_matrix[i]
        for j, sim_score in enumerate(similarities):
            if sim_score > sim_thre:
                concept_id = roadmap_concepts_df.loc[j, "id"]
                role_id = util.get_role_id(concept_id)
                concept_name = roadmap_concepts_df.loc[j, "name"]
                result.append(
                    {
                        "course_id": course_id,
                        "course_name": course_name,
                        "category": category,
                        "concept_id": concept_id,
                        "concept_name": concept_name,
                        "role_id": role_id,
                        "similarity_score": sim_score,
                    }
                )

    # Create DataFrame from the result
    user_concepts_df = pd.DataFrame(result)

    logger.info("Number of similar concepts for user courses: " + str(len(user_concepts_df)))

    ####### IMPROVEMENT
    #### The decision regarding a role concept should be determined only by the most similar user course.
    # Sort the DataFrame by similarity_score in descending order
    # user_concepts_df = user_concepts_df.sort_values(by="similarity_score", ascending=False)
    # # Keep only the top record for each concept_id
    # user_concepts_df = user_concepts_df.groupby("concept_id").head(1).reset_index(drop=True)

    logger.info("Number of similar concepts for user courses after discard: " + str(len(user_concepts_df)))

    return user_concepts_df


def find_similar_courses_for_disliked_courses(user_courses_df, course_emb_list, sim_thre):

    disliked_courses = user_courses_df[user_courses_df["category"] == "TookAndDisliked"]
    disliked_course_names = disliked_courses["course"].values

    disliked_emb_list = np.vstack(disliked_courses["emb"].values)
    similarity_matrix = cosine_similarity(disliked_emb_list, course_emb_list)

    similar_ones_index = np.argwhere(similarity_matrix > sim_thre)

    disliked_similar_course_list = similar_ones_index[:, 1].tolist()
    matching_course_titles = udemy_courses_df.iloc[disliked_similar_course_list]["title"].tolist()

    # disliked_similar_course_id_list = [decoder_for_courses[idx] for idx in disliked_similar_course_list]
    course_mapping = {disliked: similar for disliked, similar in zip(disliked_course_names, matching_course_titles)}
    logger.info("Mapping of disliked courses to similar courses:\n%s", course_mapping)

    return disliked_similar_course_list


def before_recommendation(user_courses_df, decoder_for_user_courses, user_emb_list, concepts_emb_list):

    # # Multiply embeddings with coefficients based on categories
    # user_courses_df['emb'] = user_courses_df.apply(lambda row: row['emb'] * coefficients.get(row['category'], 1), axis=1)

    sim_mat_user_course_X_concept = cosine_similarity(user_emb_list, concepts_emb_list)

    # eda_for_google = ExplatoryDataAnalysis(
    #     udemy_courses_df,
    #     roadmap_concepts_df,
    #     user_courses_df,
    #     decoder_for_courses,
    #     decoder_for_concepts,
    #     sim_mat_course_X_concept,
    #     sim_mat_user_course_X_concept,
    # )

    # eda_for_google.find_course_X_concept_and_sim_scores(threshold=0.8)
    # eda_for_google.find_concept_X_course_and_sim_scores(threshold=0.8)

    # eda_for_google.find_user_course_X_concept_and_sim_scores(threshold=0.7)

    # User Courses X Concepts Matches
    max_sim_for_row_user_courses_X_concepts = util.max_element_indices(sim_mat_user_course_X_concept)
    max_sim_for_row_user_courses_X_concepts_df = pd.DataFrame(max_sim_for_row_user_courses_X_concepts)

    total_match = 0
    user_concept_id_list = []

    for i in range(max_sim_for_row_user_courses_X_concepts_df.shape[0]):
        concept_id = decoder_for_concepts[max_sim_for_row_user_courses_X_concepts_df.iloc[i]["y"]]
        max_sim = max_sim_for_row_user_courses_X_concepts_df.iloc[i]["max"]
        if max_sim > 0.7:
            total_match = total_match + 1
            user_concept_id_list.append(concept_id)

    # print(total_match)

    user_concept_id_set = set(user_concept_id_list)

    return user_concept_id_set


def main() -> None:

    logger.info("################################################### XCRS IS STARTING ###################################################")

    # LOAD DATA FOR MODELS
    init_data()

    global encoder_for_courses, decoder_for_courses, encoder_for_concepts, decoder_for_concepts
    global course_id_list, concept_id_list
    global course_emb_list_google, concept_emb_list_google, course_X_concept_google, concept_X_course_google
    global course_emb_list_voyage, concept_emb_list_voyage, course_X_concept_voyage, concept_X_course_voyage
    global course_emb_list_openai, concept_emb_list_openai, course_X_concept_openai, concept_X_course_openai
    global course_emb_list_mistral, concept_emb_list_mistral, course_X_concept_mistral, concept_X_course_mistral
    global course_emb_list_cohere, concept_emb_list_cohere, course_X_concept_cohere, concept_X_course_cohere
    global emb_thre_2sigma_dict, emb_thre_3sigma_dict, emb_thre_2_5_sigma_dict

    # Encoders/Decoders for Courses and Concepts
    course_id_list = udemy_courses_df["id"]
    encoder_for_courses = dict([(v, k) for v, k in zip(course_id_list, range(len(course_id_list)))])
    decoder_for_courses = dict([(v, k) for k, v in encoder_for_courses.items()])

    concept_id_list = roadmap_concepts_df["id"]
    encoder_for_concepts = dict([(v, k) for v, k in zip(concept_id_list, range(len(concept_id_list)))])
    decoder_for_concepts = dict([(v, k) for k, v in encoder_for_concepts.items()])

    # topic_id_list = roadmap_topics_df["id"]
    # encoder_for_topics = dict([(v, k) for v, k in zip(topic_id_list, range(len(topic_id_list)))])
    # decoder_for_topics = dict([(v, k) for k, v in encoder_for_topics.items()])

    # Udemy Courses and Concepts Embeddings List - GOOGLE
    course_emb_list_google, concept_emb_list_google = get_emb_lists(folder="google_emb", model=GOOGLE_MODEL)
    course_X_concept_google = cosine_similarity(course_emb_list_google, concept_emb_list_google)
    concept_X_course_google = course_X_concept_google.transpose()

    course_emb_list_voyage, concept_emb_list_voyage = get_emb_lists(folder="voyage_emb", model=VOYAGE_MODEL)
    course_X_concept_voyage = cosine_similarity(course_emb_list_voyage, concept_emb_list_voyage)
    concept_X_course_voyage = course_X_concept_voyage.transpose()

    course_emb_list_openai, concept_emb_list_openai = get_emb_lists(folder="openai_emb", model=OPENAI_MODEL)
    course_X_concept_openai = cosine_similarity(course_emb_list_openai, concept_emb_list_openai)
    concept_X_course_openai = course_X_concept_openai.transpose()

    course_emb_list_mistral, concept_emb_list_mistral = get_emb_lists(folder="mistral_emb", model=MISTRAL_MODEL)
    course_X_concept_mistral = cosine_similarity(course_emb_list_mistral, concept_emb_list_mistral)
    concept_X_course_mistral = course_X_concept_mistral.transpose()

    course_emb_list_cohere, concept_emb_list_cohere = get_emb_lists(folder="cohere_emb", model=COHERE_MODEL)
    course_X_concept_cohere = cosine_similarity(course_emb_list_cohere, concept_emb_list_cohere)
    concept_X_course_cohere = course_X_concept_cohere.transpose()

    models_dict = {"Google": GOOGLE_MODEL, "Voyage": VOYAGE_MODEL, "OpenAI": OPENAI_MODEL, "Mistral": MISTRAL_MODEL, "Cohere": COHERE_MODEL}
    logger.info("Similarity Matrices between Udemy Courses and Roadmap Concepts are calculated using LLMs: \n" + pformat(models_dict))

    similarity_matrices = [
        course_X_concept_google,
        course_X_concept_voyage,
        course_X_concept_openai,
        course_X_concept_mistral,
        course_X_concept_cohere,
    ]

    models_dict = {"Google": GOOGLE_MODEL, "Voyage": VOYAGE_MODEL, "OpenAI": OPENAI_MODEL, "Mistral": MISTRAL_MODEL, "Cohere": COHERE_MODEL}

    emb_thre_2sigma = [util.calculate_threshold(sim_mat=matrix, sigma_num=2) for matrix in similarity_matrices]
    emb_thre_2sigma_dict = dict(zip(models_dict.values(), emb_thre_2sigma))

    emb_thre_2_5_sigma = [util.calculate_threshold(sim_mat=matrix, sigma_num=2.5) for matrix in similarity_matrices]
    emb_thre_2_5_sigma_dict = dict(zip(models_dict.values(), emb_thre_2_5_sigma))

    emb_thre_3sigma = [util.calculate_threshold(sim_mat=matrix, sigma_num=3) for matrix in similarity_matrices]
    emb_thre_3sigma_dict = dict(zip(models_dict.values(), emb_thre_3sigma))

    logger.info("Their corresponding thresholds: \n" + pformat(list(zip(models_dict.keys(), emb_thre_2_5_sigma))))


app = FastAPI()


@app.post("/save_inputs")
async def save_inputs(request: RecommendationRequest):

    fileName = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]

    with open("../user_inputs/{}.json".format(fileName), "w") as f:
        f.write(request.model_dump_json())

    logger.info("Request saved to the file {}.json in user_inputs/".format(fileName))

    recommendations = Recommendation(
        model="", roles=[RoleRecommendation(role="", explanation="", courses=[CourseRecommendation(course="", url="", explanation="")])]
    )

    return RecommendationResponse(fileName=fileName, recommendations=[recommendations])


@app.post("/recommendations/{model_name}")
async def get_recommendations(request: RecommendationRequest, model_name: str):
    if model_name == "google":
        emb_model = GOOGLE_MODEL
        concept_emb_list = concept_emb_list_google
        course_emb_list = course_emb_list_google
        concept_X_course = concept_X_course_google
    elif model_name == "voyage":
        emb_model = VOYAGE_MODEL
        concept_emb_list = concept_emb_list_voyage
        course_emb_list = course_emb_list_voyage
        concept_X_course = concept_X_course_voyage
    elif model_name == "openai":
        emb_model = OPENAI_MODEL
        concept_emb_list = concept_emb_list_openai
        course_emb_list = course_emb_list_openai
        concept_X_course = concept_X_course_openai
    elif model_name == "mistral":
        emb_model = MISTRAL_MODEL
        concept_emb_list = concept_emb_list_mistral
        course_emb_list = course_emb_list_mistral
        concept_X_course = concept_X_course_mistral
    elif model_name == "cohere":
        emb_model = COHERE_MODEL
        concept_emb_list = concept_emb_list_cohere
        course_emb_list = course_emb_list_cohere
        concept_X_course = concept_X_course_cohere
    elif model_name == "mock":
        recommendations = Recommendation(
            model=sample_rec_data["model"],
            roles=[
                RoleRecommendation(
                    role=role["role"],
                    score=role["score"],
                    explanation=role["explanation"],
                    courses=[CourseRecommendation(**course) for course in role["courses"]],
                )
                for role in sample_rec_data["roles"]
            ],
        )
        return RecommendationResponse(fileName="", recommendations=[recommendations])
    else:
        # raise UnicornException(name=model_name)
        logger.error("Wrong Endpoint: " + model_name)
        raise HTTPException(status_code=404, detail="Embedding model not found")

    sim_thre_2sigma = emb_thre_2sigma_dict[emb_model]
    sim_thre_2_5_sigma = emb_thre_2_5_sigma_dict[emb_model]
    sim_thre_3sigma = emb_thre_3sigma_dict[emb_model]

    logger.info(util.pad_string_with_dashes("POST - recommendation request for " + model_name.upper(), length=120))
    logger.info(util.pad_string_with_dashes("", length=120))
    logger.info("TookAndLiked: " + request.took_and_liked)
    logger.info("TookAndNeutral: " + request.took_and_neutral)
    logger.info("TookAndDisliked: " + request.took_and_disliked)
    logger.info("Curious: " + request.curious)

    threshold = sim_thre_2_5_sigma
    count = (concept_X_course > sim_thre_2_5_sigma).sum().sum()
    logger.info("Total number of matches between courses and concepts: " + str(count) + " by threshold: " + str(threshold))

    ################################# 1. CREATING EMBEDDINGS FOR USER'S COURSES #######################################
    user_courses_df, user_emb_list = create_user_embeddings(
        request.took_and_liked, request.took_and_neutral, request.took_and_disliked, request.curious, emb_model
    )

    ################################# 2. FINDING SIMILAR CONCEPTS FOR USER'S COURSES ##################################
    user_concepts_df = find_similar_concepts_for_courses(user_courses_df, user_emb_list, concept_emb_list, roadmap_concepts_df, sim_thre_2_5_sigma)

    # If empty dataframe happens
    if user_concepts_df.shape[0] == 0:
        logger.error("Insufficient Input")
        return INSUFFICIENT_INPUT

    logger.info("Total number of concepts X user courses matches:" + str(user_concepts_df.shape[0]))

    ################################# 3. FINDING SIMILAR COURSES FOR USER'S DISLIKED COURSES ##########################
    disliked_similar_course_id_list = find_similar_courses_for_disliked_courses(user_courses_df, course_emb_list, sim_thre_2_5_sigma)

    recommendation = RecommendationEngine(
        udemy_courses_df,
        roadmap_concepts_df,
        concept_X_course,
        encoder_for_concepts,
        encoder_for_courses,
        roadmaps_df,
    )

    ################################# 4. ROLE RECOMMENDATION ##########################################################
    rol_rec_list = recommendation.recommend_role(user_concepts_df)

    if len(rol_rec_list) == 0:
        logger.error("Role recommendation list is empty!")
        return INSUFFICIENT_INPUT

    ################################# 5. COURSE RECOMMENDATION ##########################################################
    rol_rec_list = recommendation.recommend_courses(user_concepts_df, rol_rec_list, disliked_similar_course_id_list)

    logger.info(util.pad_string_with_dashes(model_name.upper() + " END ", length=120))
    logger.info(util.pad_string_with_dashes("", length=120))

    recommendations = Recommendation(model=emb_model, roles=rol_rec_list)

    return RecommendationResponse(fileName="", recommendations=[recommendations])


if __name__ == "__main__":
    main()
    uvicorn.run(app, host="0.0.0.0", port=8000)
