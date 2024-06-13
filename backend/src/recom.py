import pandas as pd
import numpy as np
from collections import Counter

# internal classes
import util
from models import RoleRecommendation, CourseRecommendation


class RecommendationEngine:
    def __init__(
        self,
        udemy_courses_df: pd.DataFrame,
        roadmap_concepts_df: pd.DataFrame,
        concept_X_course: np.ndarray,
        encoder_for_concepts: dict,
        encoder_for_courses: dict,
        roadmaps_df: pd.DataFrame,
        emb_type: str,
    ):
        self.udemy_courses_df = udemy_courses_df
        self.roadmap_concepts_df = roadmap_concepts_df
        self.concept_X_course = concept_X_course
        self.encoder_for_concepts = encoder_for_concepts
        self.encoder_for_courses = encoder_for_courses
        self.roadmaps_df = roadmaps_df
        self.recom_role_id_list = []

    def recommend_role(self, user_concepts_df: pd.DataFrame):

        concept_id_list = self.roadmap_concepts_df["id"]
        # user_concept_id_set = set(user_concepts_df["concept_id"])

        print(user_concepts_df)
        # Coefficients for each category
        coefficients = {"TookAndLiked": 0.75, "TookAndNeutral": 0.5, "TookAndDisliked": -0.5, "Curious": 1}

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
        # recom_role_id = max(points_dict, key=points_dict.get)

        # print(points_dict)

        # Example Activation:   [-50,   -25,    -10,    0,      5,      10.67,  23.645, 32.5        44.6332,    50(All-N),  60.4342,    74,     75(All-L),  86,     94,     100(ALL-C)]
        #              ----->   [0.0,   0.0,    0.09,   0.67,   1.8,    5.39,   43.27,  81.76,      98.07,      99.33,      99.92,      99.99,  100.0,      100.0,  100.0,  100.0]
        for key, value in points_dict.items():
            points_dict[key] = util.custom_activation(value)

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

        # NO NEED
        # rr_list = list(RoleRecommendation)
        # for recom_role_id in recom_role_id_list:
        #     role = self.roadmaps_df.loc[recom_role_id]["name"]
        #     #explanation =
        #     rr = RoleRecommendation(role=role, explanation=explanation, courses=[])
        #     rr_list.append(rr)

        rr_list = []
        for recom_role_id in recom_role_id_list:
            role = self.roadmaps_df.loc[recom_role_id]["name"]
            score = points_dict[recom_role_id]
            explanation = self.generate_explanation_for_role(recom_role_id, user_concepts_df)
            # rr = RoleRecommendation(role=role, score=score, explanation=explanation, courses=[])
            rr_list.append(RoleRecommendation(role=role, score=score, explanation=explanation, courses=[]))

        # print("Recommended Role-{} with points: {}".format(recom_role_id, points_dict[recom_role_id]))
        # print("Recommended Role: " + self.roadmaps_df.loc[recom_role_id]["name"])

        self.recom_role_id_list = recom_role_id_list

        return rr_list

    def generate_explanation_for_role(self, recom_role_id: int, user_concepts_df: pd.DataFrame) -> str:

        role = self.roadmaps_df.loc[recom_role_id]["name"]
        effective_courses = user_concepts_df[user_concepts_df["role_id"] == recom_role_id]
        explanation = ""
        took_exp = "I assume that you are familiar with "
        curious_exp = "I can see that you are willing to learn "

        courses_took = ""
        courses_curious = ""

        for i, course in effective_courses.iterrows():
            if course["category"].startswith("TookAnd"):
                courses_took += course["concept_name"] + ", "
            if course["category"] == "Curious":
                courses_curious += course["concept_name"] + ", "

        if courses_took != "":
            explanation += took_exp + courses_took[:-2] + ". "

        if courses_curious != "":
            if courses_took != "":
                explanation += "Also, "
            explanation += curious_exp + courses_curious[:-2] + ". "

        print(explanation)

        return explanation

    def recommend_courses(self, user_concepts_df: pd.DataFrame, rol_rec_list: list[RoleRecommendation], disliked_similar_course_id_list: list[int]):

        # Number of concepts to recommend for each role
        n = 3
        udemy_website = "https://www.udemy.com"

        if len(self.recom_role_id_list) == 0:
            raise Exception("Sorry, you should get a recommendation for role first!")

        concept_id_list = self.roadmap_concepts_df["id"]
        user_concept_id_set = set(user_concepts_df["concept_id"])

        for recom_role_id, rol_rec in zip(self.recom_role_id_list, rol_rec_list):
            recom_role_user_concepts = user_concepts_df[user_concepts_df["role_id"] == recom_role_id]
            recom_role_concepts = {concept for concept in self.roadmap_concepts_df["id"] if util.get_role_id(concept) == recom_role_id}

            print("Number of role-" + str(recom_role_id) + " concepts : " + str(len(recom_role_concepts)))
            print("Number of matching user concepts for role-" + str(recom_role_id) + ": " + str(recom_role_user_concepts.shape[0]))

            recom_concepts = recom_role_concepts.difference(recom_role_user_concepts["concept_id"])

            original_dict, sorted_equalized_dict = util.equalize_digits(recom_concepts)

            # print("Original Dict:", original_dict)
            # print("Sorted equalized Dict:", sorted_equalized_dict)

            # Explanation for Course recommendation --- Concept recommendation and familarity

            # familiar_concepts = self.roadmap_concepts_df[self.roadmap_concepts_df["id"].isin(user_concepts_for_recom_role)]
            # print("Concepts you are already familiar: ")
            # print(familiar_concepts[["id", "name"]])

            recom_concepts_df = self.roadmap_concepts_df[self.roadmap_concepts_df["id"].isin(list(sorted_equalized_dict.keys())[:n])]
            print()
            print("Recommended concepts: ")
            print(recom_concepts_df[["id", "name"]])

            selected_recom_concepts = list(sorted_equalized_dict.keys())[:n]
            selected_recom_concepts = [self.encoder_for_concepts.get(idx) for idx in selected_recom_concepts]
            print(selected_recom_concepts)

            # Course Recommendation Part

            # Number of courses to recommend for each concept
            m = 1

            result_for_recom_role = []
            for concept_index in selected_recom_concepts:

                top_courses = util.top_n_courses_for_concept(
                    self.udemy_courses_df, self.concept_X_course, concept_index, disliked_similar_course_id_list, m
                )
                # TODO: Check if the concept is true
                rec_concept_row = self.roadmap_concepts_df.loc[self.roadmap_concepts_df["id"] == concept_index]
                for id, course_row in top_courses.iterrows():

                    result_for_recom_role.append(
                        {
                            "role_id": recom_role_id,
                            "concept_id": concept_index,
                            "concept_name": rec_concept_row["name"],
                            "course_id": course_row["id"],
                            "course_title": course_row["title"],
                            "course_url": udemy_website + course_row["url"],
                            "similarity_score": course_row["sim_score"],
                        }
                    )

                    explanation = self.generate_explanation_for_course(recom_role_id, self.encoder_for_courses[course_row["id"]])

                    rol_rec.courses.append(CourseRecommendation(course=course_row["title"], url=udemy_website + course_row["url"], explanation=explanation))
                    # Selected courses added to this list due to prevent reselection.
                    disliked_similar_course_id_list.append(self.encoder_for_courses[course_row["id"]])

                # Display the result
                print(f"Top {m} Similarity Scores: ", top_courses["sim_score"])
                print(f"Corresponding Courses: ", top_courses[["id", "title"]])

                print("Results: ", result_for_recom_role)
                result_df = pd.DataFrame(result_for_recom_role)

        return rol_rec_list


    def generate_explanation_for_course(self, recom_role_id: int, recom_course_id: int) -> str:

        top_concepts = util.top_n_concepts_for_courses(self.roadmap_concepts_df, self.concept_X_course, recom_role_id, recom_course_id, 3)
        course_title = self.udemy_courses_df.loc[recom_course_id, "title"]
        role_name = self.roadmaps_df.loc[recom_role_id]["name"]
        explanation = ""
        
        # start_exp = "This course (" + course_title + ") includes the concepts of " 
        start_exp = "This course includes the concepts of " 

        concepts_included = ""
        seen_concepts = set()

        for id, concept_row in top_concepts.iterrows():
            concept_name = concept_row["name"]            
            if concept_name not in seen_concepts:
                seen_concepts.add(concept_name)
                concepts_included += concept_name + ", "
        
        if concepts_included != "":
            explanation += start_exp + concepts_included[:-2] + " which are necessary for you to progress in the " + role_name + " role."
            
        print(explanation)

        return explanation
