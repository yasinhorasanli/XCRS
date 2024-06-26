import logging
from pprint import pformat
import pandas as pd
import numpy as np
from collections import Counter
import os

from openai import OpenAI


# internal classes
import util
from models import RoleRecommendation, CourseRecommendation

logging.basicConfig(filename="../log/backend.log", filemode="a", format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger()


class RecommendationEngine:
    def __init__(
        self,
        udemy_courses_df: pd.DataFrame,
        roadmap_concepts_df: pd.DataFrame,
        concept_X_course: np.ndarray,
        encoder_for_concepts: dict,
        encoder_for_courses: dict,
        roadmaps_df: pd.DataFrame,
    ):
        self.udemy_courses_df = udemy_courses_df
        self.roadmap_concepts_df = roadmap_concepts_df
        self.concept_X_course = concept_X_course
        self.encoder_for_concepts = encoder_for_concepts
        self.encoder_for_courses = encoder_for_courses
        self.roadmaps_df = roadmaps_df
        self.recom_role_id_list = []
        openai_api_key = open("../../embedding-generation/api_keys/openai_api_key.txt").read().strip()
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.exp_model = OpenAI()

    def recommend_role(self, user_concepts_df: pd.DataFrame):

        logger.info(util.pad_string_with_dashes("ROLE RECOMMENDATION ", length=100))

        concept_id_list = self.roadmap_concepts_df["id"]
        # user_concept_id_set = set(user_concepts_df["concept_id"])

        # print(user_concepts_df)
        # Coefficients for each category
        coefficients = {"TookAndLiked": 0.75, "TookAndNeutral": 0.5, "TookAndDisliked": -0.25, "Curious": 1}

        role_id_concept_counts = Counter(util.get_role_id(concept_id) for concept_id in concept_id_list)
        role_id_user_scores = Counter()
        encountered_concepts = set()

        for concept_id, role_id, category in zip(user_concepts_df["concept_id"], user_concepts_df["role_id"], user_concepts_df["category"]):
            # Check if concept_id, category, role_id has been encountered before
            concept_info = (concept_id, category, role_id)
            if concept_info not in encountered_concepts:
                # Increase counters based on category and coefficients
                score = coefficients.get(category, 0)
                role_id_user_scores[role_id] += score
                encountered_concepts.add(concept_info)

        points_dict = {role_id: role_id_user_scores[role_id] * 100 / role_id_concept_counts[role_id] for role_id in role_id_concept_counts.keys()}

        # Example Activation:   [-25,    -10,    0,      5,      10.67,  23.645, 32.5        44.6332,    50(All-N),  60.4342,    74,     75(All-L),  86,     94,     100(ALL-C)]
        #              ----->   [0.0,    0.09,   0.67,   1.8,    5.39,   43.27,  81.76,      98.07,      99.33,      99.92,      99.99,  100.0,      100.0,  100.0,  100.0]
        for key, value in points_dict.items():
            points_dict[key] = util.custom_activation(value)

        role_name_coverage_score_dict = zip(self.roadmaps_df.loc[points_dict.keys()]["name"], points_dict.values())
        role_name_coverage_score_dict = sorted(list(role_name_coverage_score_dict), key=lambda item: item[1], reverse=True)
        logger.info(role_name_coverage_score_dict)

        # Filter keys based on a threshold (0.67 is equal to having no concept for that role)
        filtered_keys = [key for key, value in points_dict.items() if value > 0.68]

        # If there are more than 3 keys with values > 0.68, return the top 3 keys with maximum values
        if len(filtered_keys) >= 3:
            recom_role_id_list = sorted(filtered_keys, key=lambda x: points_dict[x], reverse=True)[:3]
        else:
            # If there are less than 3 keys with values > 0.68, return all keys
            recom_role_id_list = sorted(filtered_keys, reverse=True)

        # print(recom_role_id_list)
        # print(sorted(role_id_concept_counts.items()))
        # print(sorted(role_id_user_scores.items()))
        # print(sorted(points_dict.items()))

        rr_list = []
        for i, recom_role_id in enumerate(recom_role_id_list):
            role = self.roadmaps_df.loc[recom_role_id]["name"]
            score = points_dict[recom_role_id]
            explanation = self.generate_explanation_for_role(recom_role_id, user_concepts_df)
            rr_list.append(RoleRecommendation(role=role, explanation=explanation, courses=[]))
            logger.info("Role-{}: {}, \tScore: {}, \tExplanation: {}".format(i + 1, role, score, explanation))

        self.recom_role_id_list = recom_role_id_list

        return rr_list

    def generate_explanation_for_role(self, recom_role_id: int, user_concepts_df: pd.DataFrame) -> str:

        role_name = self.roadmaps_df.loc[recom_role_id]["name"]
        effective_courses = user_concepts_df[user_concepts_df["role_id"] == recom_role_id]
        explanation = ""
        took_exp = "I assume that you are familiar with "
        curious_exp = "I can see that you are willing to learn "

        courses_took_set= set(effective_courses[effective_courses["category"].str.startswith("TookAnd")]["concept_name"])
        courses_curious_set = set(effective_courses[effective_courses["category"] == "Curious"]["concept_name"])

        courses_took = ", ".join(courses_took_set)
        courses_curious = ", ".join(courses_curious_set)

        if courses_took != "":
            explanation += took_exp + courses_took + ". "

        if courses_curious != "":
            if courses_took != "":
                explanation += "Also, "
            explanation += curious_exp + courses_curious + ". "
 
        response = self.exp_model.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
            "role": "system",
            "content": "You will be provided with sentences, and your task is to convert them to standard English and correct meaningless parts. Convert mostly computer science related abbreviations to their longer form. Also you can paraphrase the sentences in better way without shortening. The main objective of these sentences are to explain someone that he is okay for {} Role.".format(role_name)
            },
            {
            "role": "user",
            "content": explanation
            }
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=1
        )
        enhanced_explanation = response.choices[0].message.content

        return enhanced_explanation

    def recommend_courses(self, user_concepts_df: pd.DataFrame, rol_rec_list: list[RoleRecommendation], disliked_similar_course_id_list: list[int]):

        logger.info(util.pad_string_with_dashes("COURSE RECOMMENDATION ", length=100))

        # Number of concepts to recommend for each role
        n = 3
        udemy_website = "https://www.udemy.com"

        if len(self.recom_role_id_list) == 0:
            raise Exception("Sorry, you should get a recommendation for role first!")

        # concept_id_list = self.roadmap_concepts_df["id"]
        # user_concept_id_set = set(user_concepts_df["concept_id"])

        for recom_role_id, rol_rec in zip(self.recom_role_id_list, rol_rec_list):
            user_concepts_for_recom_role_df = user_concepts_df[user_concepts_df["role_id"] == recom_role_id]
            user_took_concepts_df = user_concepts_for_recom_role_df[user_concepts_for_recom_role_df['category'].str.startswith('TookAnd')]
            user_curious_concepts_df = user_concepts_for_recom_role_df[user_concepts_for_recom_role_df['category']== 'Curious']

            concepts_in_recom_role = {concept for concept in self.roadmap_concepts_df["id"] if util.get_role_id(concept) == recom_role_id}
            role_name = self.roadmaps_df.loc[recom_role_id]["name"]
            

            user_took_concept_id_set = set(user_took_concepts_df["concept_id"])
            user_curious_concept_id_set = set(user_curious_concepts_df["concept_id"])

            logger.info(util.pad_string_with_dashes(role_name + ' ', length=80))
            logger.info("Total number of concepts :\t" + str(len(concepts_in_recom_role)))
            logger.info("Total number of matching user took concepts:\t" + str(len(user_took_concept_id_set)))
            logger.info("Total number of matching user curious concepts:\t" + str(len(user_curious_concept_id_set)))


            remaining_concepts = concepts_in_recom_role.difference(user_took_concept_id_set)

            sorted_equalized_dict = util.equalize_digits(remaining_concepts)
            recom_concepts_df = self.roadmap_concepts_df[self.roadmap_concepts_df["id"].isin(list(sorted_equalized_dict.keys())[:n])]
            
            logger.info('Recommended concepts: \t' + str(recom_concepts_df["name"].tolist()))

            #selected_recom_concepts = list(sorted_equalized_dict.keys())[:n]
            recom_concept_id_list = recom_concepts_df["id"].tolist()

            
            recom_concept_id_list_encoded = [self.encoder_for_concepts.get(idx) for idx in recom_concept_id_list]

            concepts_in_recom_role_encoded = [self.encoder_for_concepts.get(idx) for idx in concepts_in_recom_role]

            remaining_concepts_encoded = [self.encoder_for_concepts.get(idx) for idx in remaining_concepts]


            
            # Course Recommendation Part

            # Number of courses to recommend for each concept
            m = 1

            result_for_recom_role = []
            for concept_index in recom_concept_id_list_encoded:

                top_courses = util.top_n_courses_for_concept(
                    self.udemy_courses_df, self.concept_X_course, concept_index, disliked_similar_course_id_list, m
                )
                
                concept_row = recom_concepts_df.loc[concept_index]                
                for id, course_row in top_courses.iterrows():

                    result_for_recom_role.append(
                        {
                            "role_id": recom_role_id,
                            "concept_id": concept_index,
                            "concept_name": concept_row["name"],
                            "course_id": course_row["id"],
                            "course_title": course_row["title"],
                            "course_url": udemy_website + course_row["url"],
                            "similarity_score": course_row["sim_score"],
                        }
                    )

                    explanation = self.generate_explanation_for_course(recom_role_id, self.encoder_for_courses[course_row["id"]], remaining_concepts_encoded)

                    rol_rec.courses.append(
                        CourseRecommendation(course=course_row["title"], url=udemy_website + course_row["url"], explanation=explanation)
                    )
                    logger.info('Concept: {} \t'.format(concept_row["name"]) + 'Similarity Score: {}'.format(course_row["sim_score"]))
                    logger.info("Course: {}, \tUrl: {}, \tExplanation: {}".format(course_row["title"], udemy_website + course_row["url"], explanation))                    

                    # Selected courses added to this list due to prevent reselection.
                    disliked_similar_course_id_list.append(self.encoder_for_courses[course_row["id"]])


                # print("Results: ", result_for_recom_role)
                result_df = pd.DataFrame(result_for_recom_role)
        
            logger.info(util.pad_string_with_dashes(role_name + " END ", length=80))

            

        return rol_rec_list

    def generate_explanation_for_course(self, recom_role_id: int, recom_course_id: int, remaining_concepts_encoded: set) -> str:

        top_concepts = util.top_n_concepts_for_courses(self.roadmap_concepts_df, self.concept_X_course, remaining_concepts_encoded, recom_course_id, 3)
        course_title = self.udemy_courses_df.loc[recom_course_id, "title"]
        role_name = self.roadmaps_df.loc[recom_role_id]["name"]
        explanation = ""

        # start_exp = "This course (" + course_title + ") includes the concepts of "
        start_exp = "This course includes the concepts of "

        top_concepts_set = set(top_concepts["name"])
        concepts_included = ", ".join(top_concepts_set)

        if concepts_included != "":
            explanation += start_exp + concepts_included+ " which are necessary for you to progress in the " + role_name + " role."

        return explanation
