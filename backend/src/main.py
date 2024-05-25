# API
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.metrics.pairwise import cosine_similarity

import google.generativeai as palm
import uvicorn
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
    global udemy_courses_df, roadmap_concepts_df, roadmaps_df

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

    # return (udemy_courses_df, roadmap_nodes_df, roadmap_concepts_df, roadmaps_df)


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

    # Drop rows where 'course' is null, empty, or only blank spaces
    user_courses_df = user_courses_df.dropna(subset=['course'])
    user_courses_df = user_courses_df[user_courses_df['course'].str.strip() != '']
    user_courses_df = user_courses_df.reset_index(drop=True)

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


def find_similar_concepts_for_courses(user_courses_df, user_emb_list, concepts_emb_list, roadmap_concepts_df, sim_thre):

    similarity_matrix = cosine_similarity(user_emb_list, concepts_emb_list)

    # Find courses with similarity score > 0.7 for each concept
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
    result_df = pd.DataFrame(result)

    return result_df


def find_similar_courses_for_disliked_courses(user_courses_df, course_emb_list, sim_thre):

    disliked_courses = user_courses_df[user_courses_df["category"] == "TookAndDisliked"]
    disliked_emb_list = np.vstack(disliked_courses["emb"].values)
    similarity_matrix = cosine_similarity(disliked_emb_list, course_emb_list)

    similar_ones_index = np.argwhere(similarity_matrix > sim_thre)

    disliked_similar_course_list = similar_ones_index[:, 1].tolist()
    # disliked_similar_course_id_list = [decoder_for_courses[idx] for idx in disliked_similar_course_list]

    return disliked_similar_course_list


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
    init_data()

    global encoder_for_courses, decoder_for_courses, encoder_for_concepts, decoder_for_concepts
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
    course_X_concept_palm = cosine_similarity(course_emb_list_palm, concept_emb_list_palm)
    concept_X_course_palm = course_X_concept_palm.transpose()

    course_emb_list_voyage, concept_emb_list_voyage = get_emb_lists(folder="voyage_emb", model="voyage-large-2")
    course_X_concept_voyage = cosine_similarity(course_emb_list_voyage, concept_emb_list_voyage)
    concept_X_course_voyage = course_X_concept_voyage.transpose()

    # user1_took = "Physics , Intr. to Information Systems, Intr.to Comp.Eng.and Ethics, Mathematics I, Linear Algebra, Engineering Mathematics, Digital Circuits, Data Structures, Introduction to Electronics, Basics of Electrical Circuits, Object Oriented Programming, Computer Organization, Logic Circuits Laboratory, Numerical Methods, Formal Languages and Automata, Analysis of Algorithms I, Probability and Statistics, Microcomputer Lab., Database Systems, Microprocessor Systems, Computer Architecture, Computer Operating Systems, Analysis of Algorithms II, Signal&Systems for Comp.Eng."
    # user1_took_and_liked = "Digital Circuits , Data Structures , Introduction to Electronics, Microprocessor Systems , Computer Architecture"
    # user1_took_and_neutral = "Mathematics I, Linear Algebra, Engineering Mathematics, Basics of Electrical Circuits, Object Oriented Programming, Computer Organization, Logic Circuits Laboratory, Analysis of Algorithms I, Probability and Statistics, Microcomputer Lab., Database Systems, Computer Operating Systems, Analysis of Algorithms II, Signal&Systems for Comp.Eng."
    # user1_took_and_disliked = "Physics, Intr. to Information Systems, Intr.to Comp.Eng.and Ethics, Numerical Methods, Formal Languages and Automata, kotlin, Unity"
    # user1_curious = "Embedded Softwares, Web Development"

    # test_before_rec(user1_took_and_liked, user1_took_and_neutral, user1_took_and_disliked, user1_curious)


# if __name__ == '__main__':

# main()

app = FastAPI()


@app.post("/save_inputs")
async def save_inputs(request: RecommendationRequest):

    fileName = datetime.now().strftime("%Y%m%d_%H%M%S.%f")[:-3]

    print(fileName)
    print(request)

    with open("../user_inputs/{}.json".format(fileName), "w") as f:
        f.write(request.json())

    recommendations = Recommendation(
        model="", roles=[RoleRecommendation(role="", explanation="", courses=[CourseRecommendation(course="", url="", explanation="")])]
    )

    return RecommendationResponse(fileName=fileName, recommendations=[recommendations])


@app.post("/recommendations/{model_name}")
async def get_recommendations(request: RecommendationRequest, model_name: str):

    if model_name == "palm":
        emb_model = "embedding-gecko-001"
        concept_emb_list = concept_emb_list_palm
        course_emb_list = course_emb_list_palm
        sim_thre = 0.7
    elif model_name == "voyage":
        emb_model = "voyage-large-2"
        concept_emb_list = concept_emb_list_voyage
        course_emb_list = course_emb_list_voyage
        sim_thre = 0.8
    elif model_name == "mock":
        recommendations = Recommendation(
            model=sample_rec_data["model"],
            roles=[
                RoleRecommendation(
                    role=role["role"], explanation=role["explanation"], courses=[CourseRecommendation(**course) for course in role["courses"]]
                )
                for role in sample_rec_data["roles"]
            ],
        )
        return RecommendationResponse(fileName="", recommendations=[recommendations])
    else:
        # raise UnicornException(name=model_name)
        raise HTTPException(status_code=404, detail="Embedding model not found")

    user_courses_df, encoder_for_user_courses, decoder_for_user_courses, user_emb_list = create_user_embeddings(
        request.took_and_liked, request.took_and_neutral, request.took_and_disliked, request.curious, emb_model
    )

    user_concepts_df = find_similar_concepts_for_courses(user_courses_df, user_emb_list, concept_emb_list, roadmap_concepts_df, sim_thre)

    # If empty dataframe happens
    if user_concepts_df.shape[0] == 0:
        raise HTTPException(status_code=404, detail="Insufficient input.")
        # return RecommendationResponse(fileName="Insufficient input.", recommendations=[])

    disliked_similar_course_id_list = find_similar_courses_for_disliked_courses(user_courses_df, course_emb_list, sim_thre)

    print(user_concepts_df)

    recommendation = RecommendationEngine(
        udemy_courses_df,
        roadmap_concepts_df,
        concept_X_course_palm,
        encoder_for_concepts,
        encoder_for_courses,
        roadmaps_df,
        "palm_emb",
    )

    rol_rec_list = recommendation.recommend_role(user_concepts_df)
    rol_rec_list = recommendation.recommend_courses(user_concepts_df, rol_rec_list, disliked_similar_course_id_list)

    print(rol_rec_list)

    recommendations = Recommendation(model=emb_model, roles=rol_rec_list)

    return RecommendationResponse(fileName="", recommendations=[recommendations])


# class UnicornException(Exception):
#     def __init__(self, name: str):
#         self.name = name

# @app.exception_handler(UnicornException)
# async def unicorn_exception_handler(request: Request, exc: UnicornException):
#     return JSONResponse(
#         status_code=418,
#         content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
#     )


if __name__ == "__main__":
    main()
    uvicorn.run(app, host="0.0.0.0", port=8000)
    