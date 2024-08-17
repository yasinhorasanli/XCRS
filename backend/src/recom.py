import logging
from pprint import pformat
import pandas as pd
import numpy as np
from collections import Counter
import os
import json
import re

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
        roadmap_topics_df: pd.DataFrame,
        concept_X_course: np.ndarray,
        encoder_for_concepts: dict,
        encoder_for_courses: dict,
        roadmaps_df: pd.DataFrame,
    ):
        self.udemy_courses_df = udemy_courses_df
        self.roadmap_concepts_df = roadmap_concepts_df
        self.roadmap_topics_df = roadmap_topics_df
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

        # Coefficients for each category
        coefficients = {"TookAndLiked": 0.75, "TookAndNeutral": 0.5, "TookAndDisliked": -0.25, "Curious": 1}

        role_id_concept_counts = Counter(util.get_role_id(concept_id) for concept_id in concept_id_list)
        role_id_user_scores = Counter()

        category_priority = {"Curious": 4, "TookAndLiked": 3, "TookAndDisliked": 2, "TookAndNeutral": 1}
        highest_priority_concepts = {}

        role_id_user_scores = {role_id: 0 for role_id in (np.arange(1, len(self.roadmaps_df) + 1))}

        for concept_id, role_id, category in zip(user_concepts_df["concept_id"], user_concepts_df["role_id"], user_concepts_df["category"]):
            key = (concept_id, role_id)
            if key in highest_priority_concepts:
                if category_priority[category] > category_priority[highest_priority_concepts[key]]:
                    highest_priority_concepts[key] = category
            else:
                highest_priority_concepts[key] = category

        for (concept_id, role_id), category in highest_priority_concepts.items():
            score = coefficients.get(category, 0)
            role_id_user_scores[role_id] += score

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

        rr_list = []
        roadmap_concepts_id_list = self.roadmap_concepts_df["id"].values
        roadmap_topics_id_list = self.roadmap_topics_df["id"].values
        role_id_explanations_dict = self.generate_explanation_for_roles(recom_role_id_list, user_concepts_df, roadmap_concepts_id_list, roadmap_topics_id_list)

        for i, (recom_role_id, explanation) in enumerate(role_id_explanations_dict.items()):
            role = self.roadmaps_df.loc[recom_role_id]["name"]
            score = points_dict[recom_role_id]
            rr_list.append(RoleRecommendation(role=role, explanation=explanation, courses=[]))
            logger.info("Role-{}: {}, \tScore: {}, \tExplanation: {}".format(i + 1, role, score, explanation))

        self.recom_role_id_list = recom_role_id_list

        return rr_list

    def generate_explanation_for_roles(self, recom_role_id_list: list[int], user_concepts_df: pd.DataFrame, roadmap_concepts_id_list: list[int], roadmap_topics_id_list: list[int]):

        promptsArray = []
        for recom_role_id in recom_role_id_list:
            role_name = self.roadmaps_df.loc[recom_role_id]["name"]
            effective_courses = user_concepts_df[user_concepts_df["role_id"] == recom_role_id]
            
            explanation = ""
            took_exp = "I assume that you are familiar with "
            curious_exp = "I can see that you are willing to learn "

            took_concepts_set = set(effective_courses[effective_courses["category"].str.startswith("TookAnd")]["concept_id"])
            topics_took_list = util.calculate_topic_coverage(took_concepts_set, roadmap_topics_id_list, roadmap_concepts_id_list)
            topics_took_name_list = self.roadmap_topics_df[self.roadmap_topics_df['id'].isin(topics_took_list)]["name"].tolist()

            curious_concepts_set = set(effective_courses[effective_courses["category"] == "Curious"]["concept_id"])
            topics_curious_list = util.calculate_topic_coverage(curious_concepts_set, roadmap_topics_id_list, roadmap_concepts_id_list)
            topics_curious_name_list = self.roadmap_topics_df[self.roadmap_topics_df['id'].isin(topics_curious_list)]["name"].tolist()
            
            concepts_took_list = list(set(effective_courses[effective_courses["category"].str.startswith("TookAnd")]["concept_name"]))
            concepts_curious_list = list(set(effective_courses[effective_courses["category"] == "Curious"]["concept_name"]))

            courses_took = ", ".join(topics_took_name_list if len(topics_took_name_list) > 0 else concepts_took_list)
            courses_curious = ", ".join(topics_curious_name_list if len(topics_curious_name_list) > 0 else concepts_curious_list)

            if courses_took != "":
                explanation += took_exp + courses_took + ". "

            if courses_curious != "":
                if courses_took != "":
                    explanation += "Also, "
                explanation += curious_exp + courses_curious + ". "

            promptsArray.append("Role: {} \n Sentences to convert: {}".format(role_name, explanation))

        stringifiedPromptsArray = json.dumps(promptsArray)

        logger.info(pformat(stringifiedPromptsArray))

        prompts = [{"role": "user", "content": stringifiedPromptsArray}]
        batchInstruction = {
            "role": "system",
            "content": """
            You will be provided with an array of sentences. Your task is to:
            1. Convert them to standard English, correcting any meaningless parts.
            2. Expand computer science-related abbreviations to their full form.
            3. Paraphrase the sentences in a clearer way, summarizing and merging key concepts where appropriate.
            4. Ensure the sentences effectively explain why the person is suited for the specified role.

            **Important Instructions:**
            - Return the output as a valid array.
            - Do not wrap the result with '```json\n' or ('[\n' or something else.
            - Each corresponding response should not exceed 400 characters.
            - Do not include the role name in the response.
            - Ensure the final response is strictly an array of strings, with each string being the converted sentence.

            **Example Input:**
            ["Sentence 1.", "Sentence 2.", "Sentence 3."]

            **Example Output:**
            ["Converted Sentence 1.", "Converted Sentence 2.", "Converted Sentence 3."]
            """,
        }
        prompts.append(batchInstruction)
        stringifiedBatchCompletion = self.exp_model.chat.completions.create(
            model="gpt-4o", messages=prompts, temperature=0.7, max_tokens=1500, top_p=1
        )

        try:
            results = stringifiedBatchCompletion.choices[0].message.content
            # Ensure the response is in a valid JSON array format
            if not results.startswith("[") or not results.endswith("]"):
                raise ValueError("Response is not a valid JSON array.")
            
            # Load the JSON response
            batchExplanations = json.loads(results.replace("\n", " ").strip())
            logger.info(pformat(batchExplanations))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"ROLE EXPLANATION RESULT - JSON VALIDATION ERROR: {e}")
            logger.info(pformat(results))
            batchExplanations = []

        role_id_explanations_dict = {key: value for key, value in zip(recom_role_id_list, batchExplanations)}

        return role_id_explanations_dict

    def recommend_courses(self, user_concepts_df: pd.DataFrame, rol_rec_list: list[RoleRecommendation], disliked_similar_course_id_list: list[int]):

        logger.info(util.pad_string_with_dashes("COURSE RECOMMENDATION ", length=100))

        # Number of concepts to recommend for each role
        n = 3
        udemy_website = "https://www.udemy.com"

        if len(self.recom_role_id_list) == 0:
            raise Exception("Sorry, you should get a recommendation for role first!")

        for recom_role_id, rol_rec in zip(self.recom_role_id_list, rol_rec_list):
            user_concepts_for_recom_role_df = user_concepts_df[user_concepts_df["role_id"] == recom_role_id]
            user_took_concepts_df = user_concepts_for_recom_role_df[user_concepts_for_recom_role_df["category"].str.startswith("TookAnd")]
            user_curious_concepts_df = user_concepts_for_recom_role_df[user_concepts_for_recom_role_df["category"] == "Curious"]

            concepts_in_recom_role = {concept for concept in self.roadmap_concepts_df["id"] if util.get_role_id(concept) == recom_role_id}

            role_name = self.roadmaps_df.loc[recom_role_id]["name"]

            user_took_concept_id_set = set(user_took_concepts_df["concept_id"])
            user_curious_concept_id_set = set(user_curious_concepts_df["concept_id"])

            logger.info(util.pad_string_with_dashes(role_name + " ", length=80))
            logger.info("Total number of concepts :\t" + str(len(concepts_in_recom_role)))
            logger.info("Total number of matching user took concepts:\t" + str(len(user_took_concept_id_set)))
            logger.info("Total number of matching user curious concepts:\t" + str(len(user_curious_concept_id_set)))

            remaining_concepts = concepts_in_recom_role.difference(user_took_concept_id_set)


            if (len(user_took_concept_id_set)/len(concepts_in_recom_role) >= 0.3):
                num_of_remaining_concepts = len(concepts_in_recom_role)-len(user_took_concept_id_set)
                sorted_equalized_dict = util.equalize_digits(concepts_in_recom_role) 
                recom_concepts_df = self.roadmap_concepts_df[self.roadmap_concepts_df["id"].isin(list(sorted_equalized_dict.keys())[-num_of_remaining_concepts:])]
            else:
                sorted_equalized_dict = util.equalize_digits(remaining_concepts)
                recom_concepts_df = self.roadmap_concepts_df[self.roadmap_concepts_df["id"].isin(list(sorted_equalized_dict.keys()))]


            logger.info("Recommended concepts: \t" + str(recom_concepts_df["name"].tolist()))

            recom_concept_id_list = recom_concepts_df["id"].tolist()
            recom_concept_id_list_encoded = [self.encoder_for_concepts.get(idx) for idx in recom_concept_id_list]


            # Course Recommendation Part
            # Number of courses to recommend for each concept
            m = 3

            top_courses_for_each_concept = []

            for concept_index in recom_concept_id_list_encoded:
                top_courses_with_scores = util.top_n_courses_for_concept(
                    self.udemy_courses_df, self.concept_X_course, concept_index, disliked_similar_course_id_list, m
                )
                top_courses_for_each_concept.extend(top_courses_with_scores)

            self.encoder_for_courses[2602800]
            top_courses_for_each_concept_df = pd.DataFrame(top_courses_for_each_concept, columns=['course_id_enc', 'sim_score'])
            top_courses_for_each_concept_df = top_courses_for_each_concept_df[top_courses_for_each_concept_df['course_id_enc'] != self.encoder_for_courses[2602800]]
            id_counts = top_courses_for_each_concept_df['course_id_enc'].value_counts()
            top_courses_for_each_concept_df['count'] = top_courses_for_each_concept_df['course_id_enc'].map(id_counts)
            sorted_df = top_courses_for_each_concept_df.sort_values(by=['course_id_enc', 'sim_score'], ascending=[True, False])
            sorted_df = sorted_df.drop_duplicates(subset='course_id_enc', keep='first')
            sorted_df = sorted_df.sort_values(by=['count', 'sim_score'], ascending=[False, False])
            top_three_course_df = sorted_df.head(3)

            # disliked_similar_course_id_list = [decoder_for_courses[idx] for idx in disliked_similar_course_list]

            filtered_courses_df = self.udemy_courses_df.iloc[top_three_course_df['course_id_enc']].copy()
            filtered_courses_df['sim_score'] = top_three_course_df.set_index('course_id_enc')['sim_score'].values
            final_courses_df = filtered_courses_df.sort_values(by='sim_score', ascending=False)


            result_for_recom_role = []
            for id, course_row in final_courses_df.iterrows():

                result_for_recom_role.append(
                    {
                        "role_id": recom_role_id,
                        "course_id": course_row["id"],
                        "course_title": course_row["title"],
                        "course_url": udemy_website + course_row["url"],
                        "similarity_score": course_row["sim_score"],
                    }
                )
                #disliked_similar_course_id_list.append(self.encoder_for_courses[course_row["id"]])

            course_matches_df = pd.DataFrame(result_for_recom_role)

            course_id_explanations_dict = self.generate_explanation_for_courses(result_for_recom_role, remaining_concepts)

            for id, (index, concept_course_match) in enumerate(course_matches_df.iterrows()):
                explanation = course_id_explanations_dict[concept_course_match["course_id"]]
                
                rol_rec.courses.append(
                        CourseRecommendation(
                            course=concept_course_match["course_title"],
                            url=concept_course_match["course_url"],
                            explanation=str(explanation),
                        )
                    )
                # logger.info(
                #         "Concept: {} \t".format(concept_course_match["concept_name"])
                #         + "Similarity Score: {}".format(concept_course_match["similarity_score"])
                # )
                logger.info(
                        "Course-{}: {}, \tUrl: {}, \tExplanation: {}".format(
                            id + 1, concept_course_match["course_title"], concept_course_match["course_url"], explanation
                        )
                )

            logger.info(util.pad_string_with_dashes(role_name + " END ", length=80))

        return rol_rec_list

    def generate_explanation_for_courses(self, result_for_recom_role: list[dict], remaining_concepts: list):

        remaining_concepts_names = self.roadmap_concepts_df[self.roadmap_concepts_df["id"].isin(remaining_concepts)]["name"]
        remaining_concepts_name_list = ", ".join(remaining_concepts_names)

        recom_course_id_list = [item["course_id"] for item in result_for_recom_role]

        promptsArray = []
        for id, match in enumerate(result_for_recom_role):
            role_id = match["role_id"]
            course_id = match["course_id"]
            role_name = self.roadmaps_df.loc[role_id]["name"]

            course = self.udemy_courses_df[self.udemy_courses_df["id"] == course_id]

            if not course.empty:
                course_info = """Title: {} Headline: {} Description: {} What you will learn: {}""".format(
                    course.iloc[0]["title"], course.iloc[0]["headline"], course.iloc[0]["description"], course.iloc[0]["what_u_learn"]
                )
                course_info = re.sub(r"[\n\r\t]", " ", course_info)
                course_info = util.remove_emojis(course_info)
                course_info = re.sub(r"\s+", " ", course_info).strip()

                promptsArray.append(course_info)
            else:
                logger.error(f"Course with ID {course_id} not found in udemy_courses_df.")

        stringifiedPromptsArray = json.dumps(promptsArray)
        logger.info(pformat(stringifiedPromptsArray))

        prompts = [{"role": "user", "content": stringifiedPromptsArray}]
        batchInstruction = {
            "role": "system",
            "content": f"""
            You will be provided with an array of information about courses. Your task is to:
            1. Infer the key concepts from the provided course information.
            2. Explain why the user should take those courses to advance in the {role_name} role, utilizing these inferred concepts.
            3. Do not include or explicitly mention the key concepts in your explanation.
            4. Where possible, connect these explanations to the concepts the user is missing: {remaining_concepts_name_list}.

            **Important Instructions:**
            - Return the output as a valid array.
            - Do not wrap the result with '```json\n' or ('[\n' or something else.
            - Each response should be one paragraph and should not exceed 400 characters.
            - Ensure the final response is strictly an array of strings, with each string being the explanation for a course.

            **Example Input:**
            ["Course 1.", "Course 2.", "Course 3."]

            **Example Output:**
            ["Converted Text 1.", "Converted Text 2.", "Converted Text 3."]
            """
        }

        prompts.append(batchInstruction)
        stringifiedBatchCompletion = self.exp_model.chat.completions.create(
            model="gpt-4o", messages=prompts, temperature=0.7, max_tokens=1500, top_p=1
        )

        batchExplanations = []
        try:
            results = stringifiedBatchCompletion.choices[0].message.content
            # Ensure the response is in a valid JSON array format
            if not results.startswith("[") or not results.endswith("]"):
                raise ValueError("Response is not a valid JSON array.")
            
            # Load the JSON response
            batchExplanations = json.loads(results.replace("\n", " ").strip())
            logger.info(pformat(batchExplanations))
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"COURSE EXPLANATION RESULT - JSON VALIDATION ERROR: {e}")
            logger.info(pformat(results))
            batchExplanations = []

        course_id_explanations_dict = {key: value for key, value in zip(recom_course_id_list, batchExplanations)}

        return course_id_explanations_dict
