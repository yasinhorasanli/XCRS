# API
from fastapi import FastAPI, HTTPException

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as palm
import voyageai

# internal classes
import util
from eda import ExplatoryDataAnalysis
from recom import RecommendationEngine
from models import CourseRecommendation, RoleRecommendation, Recommendation, RecommendationResponse, RecommendationRequest, sample_rec_data

palm_api_key = open("../../embedding-generation/api_keys/palm_api_key.txt").read().strip()
voyage_api_key = open("../../embedding-generation/api_keys/voyage_api_key.txt").read().strip()
voyage = voyageai.Client(api_key=voyage_api_key)
palm.configure(api_key=palm_api_key)
udemy_website = "www.udemy.com"


def init_data():
    global udemy_courses_df, roadmap_concepts_df, roadmap_concepts_df, roadmaps_df

    df_path = "../../embedding-generation/data/"
    udemy_courses_df = pd.read_csv(df_path + "udemy_courses_final.csv")
    roadmap_nodes_df = pd.read_csv(df_path + "roadmap_nodes_final.csv")
    roadmap_concepts_df = roadmap_nodes_df[roadmap_nodes_df["type"] == "concept"].copy()
    roadmap_concepts_df.reset_index(inplace=True)

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

    return (udemy_courses_df, roadmap_nodes_df, roadmap_concepts_df, roadmaps_df)


def palm_embed_fn(text):
    return palm.generate_embeddings(model="models/embedding-gecko-001", text=text)["embedding"]


def voyage_embed_fn(text):
    return voyage.embed([text], model="voyage-large-2").embeddings[0]


def get_emb_lists(folder: str, model: str):

    df_path = "../../embedding-generation/data/" + folder + "/"
    udemy_courses_emb_df = pd.read_csv(df_path + "udemy_courses_{}.csv".format(model))
    roadmap_nodes_emb_df = pd.read_csv(df_path + "roadmap_nodes_{}.csv".format(model))
    roadmap_concepts_emb_df = roadmap_nodes_emb_df[roadmap_nodes_emb_df["id"].isin(concept_id_list)]
    # roadmap_concepts_emb_df = roadmap_nodes_emb_df[roadmap_nodes_emb_df["id"].isin(roadmap_concepts_df["id"])]

    udemy_courses_emb_df["emb"] = udemy_courses_emb_df.apply(util.convert_to_float, axis=1)
    course_emb_list = udemy_courses_emb_df["emb"].values
    course_emb_list = np.vstack(course_emb_list)

    roadmap_concepts_emb_df["emb"] = roadmap_concepts_emb_df.apply(util.convert_to_float, axis=1)
    concept_emb_list = roadmap_concepts_emb_df["emb"].values
    concept_emb_list = np.vstack(concept_emb_list)

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

    print(user_courses_df)

    encoder_for_user_courses = dict([(v, k) for v, k in zip(user_courses_df["course"], range(len(user_courses_df)))])
    decoder_for_user_courses = dict([(v, k) for k, v in encoder_for_user_courses.items()])

    # Embedding Generation for User Data

    # user_courses_df['palm_emb'] = user_courses_df.apply(convert_to_float, axis=1)

    # TODO: If it is possible, search courses in wikipedia here.

    if model == "embedding-gecko-001":
        user_courses_df["emb"] = user_courses_df["course"].apply(palm_embed_fn)
    elif model == "voyage-large-2":
        user_courses_df["emb"] = user_courses_df["course"].apply(voyage_embed_fn)

    user_emb_list = user_courses_df["emb"].values
    user_emb_list = np.vstack(user_emb_list)
    print(user_emb_list.shape)

    return (user_courses_df, encoder_for_user_courses, decoder_for_user_courses, user_emb_list)


def before_recommendation(user_courses_df, decoder_for_user_courses, user_emb_list, palm_concepts_emb_list):

    # # Coefficients for each category
    # coefficients = {
    #     'TookAndLiked': 1,
    #     'TookAndNeutral': 0.5,
    #     'TookAndDisliked': -1,
    #     'Curious': 1  
    # }

    # # Multiply embeddings with coefficients based on categories
    # user_courses_df['emb'] = user_courses_df.apply(lambda row: row['emb'] * coefficients.get(row['category'], 1), axis=1)


    sim_mat_user_course_X_concept = cosine_similarity(user_emb_list, palm_concepts_emb_list)

    # eda_for_palm = ExplatoryDataAnalysis(
    #     udemy_courses_df,
    #     roadmap_concepts_df,
    #     user_courses_df,
    #     decoder_for_courses,
    #     decoder_for_concepts,
    #     decoder_for_user_courses,
    #     sim_mat_course_X_concept, 
    #     sim_mat_user_course_X_concept,
    # )

    # eda_for_palm.find_course_X_concept_and_sim_scores(threshold=0.8)
    # eda_for_palm.find_concept_X_course_and_sim_scores(threshold=0.8)

    # eda_for_palm.find_user_course_X_concept_and_sim_scores(threshold=0.7)

    # User Courses X Concepts Matches
    max_sim_for_row_user_courses_X_concepts = util.max_element_indices(sim_mat_user_course_X_concept)
    max_sim_for_row_user_courses_X_concepts_df = pd.DataFrame(max_sim_for_row_user_courses_X_concepts)
    total_match = 0
    user_concept_id_list = []
    for i in range(max_sim_for_row_user_courses_X_concepts_df.shape[0]):
        # course_name = decoder_for_user_courses[max_sim_for_row_user_courses_X_concepts_df.iloc[i]["x"]]
        concept_id = decoder_for_concepts[max_sim_for_row_user_courses_X_concepts_df.iloc[i]["y"]]
        max_sim = max_sim_for_row_user_courses_X_concepts_df.iloc[i]["max"]
        if max_sim > 0.7:
            total_match = total_match + 1
            user_concept_id_list.append(concept_id)

    # print(total_match)

    user_concept_id_set = set(user_concept_id_list)

    return user_concept_id_set


def main() -> None:
    # LOAD DATA FOR PALM
    udemy_courses_df, roadmap_nodes_df, roadmap_concepts_df, roadmaps_df = init_data()

    global decoder_for_courses, encoder_for_concepts, decoder_for_concepts
    # TODO: Why did you use encoder_for_concepts here?
    global course_id_list, concept_id_list
    global course_emb_list_palm, concept_emb_list_palm, course_X_concept_palm, concept_X_course_palm
    global course_emb_list_voyage, concept_emb_list_voyage, course_X_concept_voyage, concept_X_course_voyage

    # Encoders/Decoders for Courses and Concepts
    course_id_list = udemy_courses_df["id"]
    encoder_for_courses = dict([(v, k) for v, k in zip(course_id_list, range(len(course_id_list)))])
    decoder_for_courses = dict([(v, k) for k, v in encoder_for_courses.items()])

    concept_id_list = roadmap_concepts_df["id"]
    encoder_for_concepts = dict([(v, k) for v, k in zip(concept_id_list, range(len(concept_id_list)))])
    decoder_for_concepts = dict([(v, k) for k, v in encoder_for_concepts.items()])

    # Udemy Courses and Concepts Embeddings List - PALM
    course_emb_list_palm, concept_emb_list_palm = get_emb_lists(folder="palm_emb", model="embedding-gecko-001")

    # Similarity Matrix between Courses and Concepts
    course_X_concept_palm = cosine_similarity(course_emb_list_palm, concept_emb_list_palm)
    concept_X_course_palm = course_X_concept_palm.transpose()

    # EXAMPLE
    # course_id = decoder_for_courses[max_sim_for_row_df.iloc[19]['x']]
    # concept_id = decoder_for_concepts[max_sim_for_row_df.iloc[19]['y']]

    # TODO: anotherModel

    course_emb_list_voyage, concept_emb_list_voyage = get_emb_lists(folder="voyage_emb", model="voyage-large-2")
    # get_emb_lists(udemy_courses_df, roadmap_concepts_df, model="voyage")
    course_X_concept_voyage = cosine_similarity(course_emb_list_voyage, concept_emb_list_voyage)
    concept_X_course_voyage = course_X_concept_voyage.transpose()

    # user1_took = "Physics , Intr. to Information Systems, Intr.to Comp.Eng.and Ethics, Mathematics I, Linear Algebra, Engineering Mathematics, Digital Circuits, Data Structures, Introduction to Electronics, Basics of Electrical Circuits, Object Oriented Programming, Computer Organization, Logic Circuits Laboratory, Numerical Methods, Formal Languages and Automata, Analysis of Algorithms I, Probability and Statistics, Microcomputer Lab., Database Systems, Microprocessor Systems, Computer Architecture, Computer Operating Systems, Analysis of Algorithms II, Signal&Systems for Comp.Eng."
    # user1_took_and_liked = "Digital Circuits , Data Structures , Introduction to Electronics, Microprocessor Systems , Computer Architecture"
    # user1_took_and_neutral = "Mathematics I, Linear Algebra, Engineering Mathematics, Basics of Electrical Circuits, Object Oriented Programming, Computer Organization, Logic Circuits Laboratory, Analysis of Algorithms I, Probability and Statistics, Microcomputer Lab., Database Systems, Computer Operating Systems, Analysis of Algorithms II, Signal&Systems for Comp.Eng."
    # user1_took_and_disliked = "Physics, Intr. to Information Systems, Intr.to Comp.Eng.and Ethics, Numerical Methods, Formal Languages and Automata"
    # user1_curious = "Embedded Softwares, Web Development"


# if __name__ == '__main__':

main()

app = FastAPI()


@app.post("/recommendations/palm")
async def get_recommendations(request: RecommendationRequest):

    emb_model = "embedding-gecko-001"
    user_courses_df, encoder_for_user_courses, decoder_for_user_courses, user_emb_list_palm = create_user_embeddings(
        request.took_and_liked,
        request.took_and_neutral,
        request.took_and_disliked,
        request.curious,
        emb_model
    )

    user_concept_id_set_palm = before_recommendation(user_courses_df, decoder_for_user_courses, user_emb_list_palm, concept_emb_list_palm)
    recommendation = RecommendationEngine(
        udemy_courses_df,
        roadmap_concepts_df,
        concept_X_course_palm,
        encoder_for_concepts,
        "palm_emb",
    )
    recom_role_id_palm = recommendation.recommend_role(concept_id_list, user_concept_id_set_palm)
    recom_course_id_list_palm = recommendation.recommend_courses(concept_id_list, user_concept_id_set_palm)

    recom_courses_df = udemy_courses_df[udemy_courses_df["id"].isin(recom_course_id_list_palm)]
    recom_courses_title_list = recom_courses_df["title"].tolist()
    recom_courses_url_list = [udemy_website + url for url in recom_courses_df["url"].tolist()]
    recom_courses_title_url_zipped = zip(
        recom_courses_df["title"].tolist(),
        [udemy_website + url for url in recom_courses_df["url"].tolist()],
    )

    print(recom_courses_title_list)
    print(recom_courses_url_list)

    courses = [CourseRecommendation(course=title, url=url, explanation="same for all now") for title, url in recom_courses_title_url_zipped]
    role_recommendations = [
        RoleRecommendation(
            role=roadmaps_df.loc[recom_role_id_palm]["name"],
            explanation="Explanation...",
            courses=courses,
        )
    ]
    recommendations = [Recommendation(model=emb_model, roles=role_recommendations)]

    return RecommendationResponse(recommendations=recommendations)


@app.post("/recommendations/voyage")
async def get_recommendations(request: RecommendationRequest):

    emb_model = "voyage-large-2"

    user_courses_df, encoder_for_user_courses, decoder_for_user_courses, user_emb_list_voyage = create_user_embeddings(
        request.took_and_liked,
        request.took_and_neutral,
        request.took_and_disliked,
        request.curious,
        emb_model,
    )

    user_concept_id_set_voyage = before_recommendation(user_courses_df, decoder_for_user_courses, user_emb_list_voyage, concept_emb_list_voyage)
    recommendation = RecommendationEngine(
        udemy_courses_df,
        roadmap_concepts_df,
        concept_X_course_voyage,
        encoder_for_concepts,
        "voyage_emb",
    )
    recom_role_id_voyage = recommendation.recommend_role(concept_id_list, user_concept_id_set_voyage)
    recom_course_id_list_voyage = recommendation.recommend_courses(concept_id_list, user_concept_id_set_voyage)

    recom_courses_df = udemy_courses_df[udemy_courses_df["id"].isin(recom_course_id_list_voyage)]
    recom_courses_title_list = recom_courses_df["title"].tolist()
    recom_courses_url_list = [udemy_website + url for url in recom_courses_df["url"].tolist()]
    recom_courses_title_url_zipped = zip(
        recom_courses_df["title"].tolist(),
        [udemy_website + url for url in recom_courses_df["url"].tolist()],
    )

    print(recom_courses_title_list)
    print(recom_courses_url_list)

    courses = [CourseRecommendation(course=title, url=url, explanation="same for all now") for title, url in recom_courses_title_url_zipped]

    role_recommendations = [
        RoleRecommendation(
            role=roadmaps_df.loc[recom_role_id_voyage]["name"],
            explanation="Explanation...",
            courses=courses,
        )
    ]

    recommendations = [Recommendation(model=emb_model, roles=role_recommendations)]

    return RecommendationResponse(recommendations=recommendations)


@app.post("/recommendations/mock")
async def get_recommendations(request: RecommendationRequest):

    recommendations = [
        Recommendation(
            model=sample_rec_data[0]["model"],
            roles=[
                RoleRecommendation(
                    role=role["role"], explanation=role["explanation"], courses=[CourseRecommendation(**course) for course in role["courses"]]
                )
                for role in rec["roles"]
            ],
        )
        for rec in sample_rec_data
    ]

    return RecommendationResponse(recommendations=recommendations)
