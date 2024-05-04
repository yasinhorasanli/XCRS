# API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as palm
import voyageai


# internal classes
import util
from eda import ExplatoryDataAnalysis
from recommendation import Recommendation

palm_api_key = open("../../embedding-generation/api_keys/palm_api_key.txt").read().strip()
voyage_api_key = open("../../embedding-generation/api_keys/voyage_api_key.txt").read().strip()
voyage = voyageai.Client(api_key=voyage_api_key)
palm.configure(api_key=palm_api_key)

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


def get_emb_lists(folder:str, model: str):

    df_path = "../../embedding-generation/data/" + folder + "/"
    udemy_courses_emb_df = pd.read_csv(df_path + "udemy_courses_{}.csv".format(model))
    roadmap_nodes_emb_df = pd.read_csv(df_path + "roadmap_nodes_{}.csv".format(model))
    roadmap_concepts_emb_df = roadmap_nodes_emb_df[roadmap_nodes_emb_df["id"].isin(concept_id_list)]
    #roadmap_concepts_emb_df = roadmap_nodes_emb_df[roadmap_nodes_emb_df["id"].isin(roadmap_concepts_df["id"])]

    udemy_courses_emb_df["emb"] = udemy_courses_emb_df.apply(util.convert_to_float, axis=1)
    course_emb_list = udemy_courses_emb_df["emb"].values
    course_emb_list = np.vstack(course_emb_list)

    roadmap_concepts_emb_df["emb"] = roadmap_concepts_emb_df.apply(util.convert_to_float, axis=1)
    concept_emb_list = roadmap_concepts_emb_df["emb"].values
    concept_emb_list = np.vstack(concept_emb_list)

    return course_emb_list, concept_emb_list


def create_user_embeddings(took_and_liked: str, took_and_neutral: str, took_and_disliked: str, curious: str):
    # USER EMBEDDINGS PART
    categories = ["TookAndLiked", "TookAndNeutral", "TookAndDisliked", "Curious"]
    user_df = pd.DataFrame(columns=categories)

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
    columns_to_process = categories

    for column in columns_to_process:
        result_dicts.extend(user_df.apply(lambda row: util.split_and_create_dict(row, column), axis=1).to_list())

    flat_data = [item for sublist in result_dicts for item in sublist.items()]
    user_courses_df = pd.DataFrame(flat_data, columns=["courses", "categories"])

    print(user_courses_df)

    encoder_for_user_courses = dict([(v, k) for v, k in zip(user_courses_df["courses"], range(len(user_courses_df)))])
    decoder_for_user_courses = dict([(v, k) for k, v in encoder_for_user_courses.items()])

    # Embedding Generation for User Data

    # user_courses_df['palm_emb'] = user_courses_df.apply(convert_to_float, axis=1)

    user_courses_df["palm_emb"] = user_courses_df["courses"].apply(palm_embed_fn)
    user_emb_list_palm = user_courses_df["palm_emb"].values
    user_emb_list_palm = np.vstack(user_emb_list_palm)
    print(user_emb_list_palm.shape)


    user_courses_df["voyage_emb"] = user_courses_df["courses"].apply(voyage_embed_fn)
    user_emb_list_voyage = user_courses_df["voyage_emb"].values
    user_emb_list_voyage = np.vstack(user_emb_list_voyage)
    print(user_emb_list_voyage.shape)

    return (
        user_courses_df,
        encoder_for_user_courses,
        decoder_for_user_courses,
        user_emb_list_palm,
        user_emb_list_voyage
    )


def before_recommendation(user_courses_df, decoder_for_user_courses, user_emb_list, palm_concepts_emb_list):

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
        course_name = decoder_for_user_courses[max_sim_for_row_user_courses_X_concepts_df.iloc[i]["x"]]
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
    #get_emb_lists(udemy_courses_df, roadmap_concepts_df, model="voyage")
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


# Model for course recommendation
class CourseRecommendation(BaseModel):
    course: str
    url: str
    explanation: str


# Model for role recommendation
class RoleRecommendation(BaseModel):
    role: str
    explanation: str
    courses: List[CourseRecommendation]


# Model for recommendation response
class RecommendationResponse(BaseModel):
    recommendations: List[RoleRecommendation]


# Model for recommendation request
class RecommendationRequest(BaseModel):
    took_and_liked: str
    took_and_neutral: str
    took_and_disliked: str
    curious: str


# Sample recommendation data
recommendation_data = [
    {
        "role": "Role A",
        "explanation": "Explanation for Role A",
        "courses": [
            {
                "course": "Course 1",
                "url": "Url 1",
                "explanation": "Explanation for Course 1",
            },
            {
                "course": "Course 2",
                "url": "Url 2",
                "explanation": "Explanation for Course 2",
            },
            {
                "course": "Course 3",
                "url": "Url 3",
                "explanation": "Explanation for Course 3",
            },
        ],
    },
    {
        "role": "Role B",
        "explanation": "Explanation for Role B",
        "courses": [
            {
                "course": "Course 4",
                "url": "Url 4",
                "explanation": "Explanation for Course 4",
            },
            {
                "course": "Course 5",
                "url": "Url 5",
                "explanation": "Explanation for Course 5",
            },
            {
                "course": "Course 6",
                "url": "Url 6",
                "explanation": "Explanation for Course 6",
            },
        ],
    },
    {
        "role": "Role C",
        "explanation": "Explanation for Role C",
        "courses": [
            {
                "course": "Course 7",
                "url": "Url 7",
                "explanation": "Explanation for Course 7",
            },
            {
                "course": "Course 8",
                "url": "Url 8",
                "explanation": "Explanation for Course 8",
            },
            {
                "course": "Course 9",
                "url": "Url 9",
                "explanation": "Explanation for Course 9",
            },
        ],
    },
]


@app.post("/recommendations/")
async def get_recommendations(request: RecommendationRequest):

    user_courses_df, encoder_for_user_courses, decoder_for_user_courses, user_emb_list_palm, user_emb_list_voyage = create_user_embeddings(
        request.took_and_liked,
        request.took_and_neutral,
        request.took_and_disliked,
        request.curious,
    )


    # PALM RECOMMENDATION
    user_concept_id_set_palm = before_recommendation(user_courses_df, decoder_for_user_courses, user_emb_list_palm, concept_emb_list_palm)
    recommendation = Recommendation(
        udemy_courses_df,
        roadmap_concepts_df,
        concept_X_course_palm,
        encoder_for_concepts,
        "palm_emb",
    )
    recom_role_id_palm = recommendation.recommend_role(concept_id_list, user_concept_id_set_palm)
    recom_course_id_list_palm = recommendation.recommend_courses(concept_id_list, user_concept_id_set_palm)
    #####################

    # VOYAGE RECOMMENDATION
    user_concept_id_set_voyage = before_recommendation(user_courses_df, decoder_for_user_courses, user_emb_list_voyage, concept_emb_list_voyage)
    recommendation = Recommendation(
        udemy_courses_df,
        roadmap_concepts_df,
        concept_X_course_voyage,
        encoder_for_concepts,
        "voyage_emb",
    )
    recom_role_id_voyage = recommendation.recommend_role(concept_id_list, user_concept_id_set_voyage)
    recom_course_id_list_voyage = recommendation.recommend_courses(concept_id_list, user_concept_id_set_voyage)
    ####################

    recom_courses_df = udemy_courses_df[udemy_courses_df["id"].isin(recom_course_id_list_palm)]
    recom_courses_title_list = recom_courses_df["title"].tolist()
    udemy_prefix = "www.udemy.com"
    recom_courses_url_list = [udemy_prefix + url for url in recom_courses_df["url"].tolist()]
    recom_courses_title_url_zipped = zip(
        recom_courses_df["title"].tolist(),
        [udemy_prefix + url for url in recom_courses_df["url"].tolist()],
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

    print(role_recommendations)

    # role_recommendations = [
    #     RoleRecommendation(
    #         role=rec["role"],
    #         explanation=rec["explanation"],
    #         courses=[CourseRecommendation(**course) for course in rec["courses"]],
    #     )
    #     for rec in recommendation_data
    # ]

    # Return recommendation response
    return RecommendationResponse(recommendations=role_recommendations)
