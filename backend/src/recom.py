import pandas as pd
import numpy as np

# internal classes
import util

class RecommendationEngine:
    def __init__(
        self,
        udemy_courses_df: pd.DataFrame,
        roadmap_concepts_df: pd.DataFrame,
        concept_X_course: np.ndarray,
        encoder_for_concepts: dict,
        emb_type: str
    ):
        self.udemy_courses_df = udemy_courses_df
        self.roadmap_concepts_df = roadmap_concepts_df
        self.concept_X_course = concept_X_course
        self.encoder_for_concepts = encoder_for_concepts
        self.recom_role_id = None
        

    def recommend_role(self, concept_id_list, user_concept_id_set):

        # Role Recommendation for User
        # Necessary to understand how many concepts belongs to the each role.
        role_id_concept_counts = {digit: 0 for digit in range(1, 11)}
        for concept_id in concept_id_list:
            role_id = util.get_role_id(concept_id)
            role_id_concept_counts[role_id] += 1

        user_role_id_concept_counts = {digit: 0 for digit in range(1, 11)}
        for concept_id in user_concept_id_set:
            role_id = util.get_role_id(concept_id)
            user_role_id_concept_counts[role_id] += 1

        ratio_dict = {
            digit: user_role_id_concept_counts[digit] / role_id_concept_counts[digit]
            for digit in role_id_concept_counts.keys()
        }
        recom_role_id = max(ratio_dict, key=ratio_dict.get)

        print(role_id_concept_counts)
        print(user_role_id_concept_counts)
        print(ratio_dict)
        print("Role id with the maximum ratio:", recom_role_id)

        print("Recommended Role-{} with percentage: {}".format(recom_role_id, ratio_dict[recom_role_id] * 100))
        print("Recommended Role: " + self.roadmap_concepts_df.loc[recom_role_id]["name"])

        self.recom_role_id = recom_role_id

        return recom_role_id

    def recommend_courses(self, concept_id_list, user_concept_id_set):

        if self.recom_role_id is None:
            raise Exception("Sorry, you should get a recommendation for role first!")

        # Course Recommendation for User
        user_concepts_for_recom_role = set()
        for concept_id in user_concept_id_set:
            if util.get_role_id(concept_id) == self.recom_role_id:
                user_concepts_for_recom_role.add(concept_id)

        recom_role_concepts = set()
        for concept in concept_id_list:
            if util.get_role_id(concept) == self.recom_role_id:
                recom_role_concepts.add(concept)

        
        print("Number of role-" + str(self.recom_role_id) + " concepts : " + str(len(recom_role_concepts)))
        print(
            "Number of matching user concepts for role-"
            + str(self.recom_role_id)
            + ": "
            + str(len(user_concepts_for_recom_role))
        )

        recom_concepts = recom_role_concepts.difference(user_concepts_for_recom_role)

        original_dict, sorted_equalized_dict = util.equalize_digits(recom_concepts)

        print("Original Dict:", original_dict)
        print("Sorted equalized Dict:", sorted_equalized_dict)

        # Explanation for Course recommendation --- Concept recommendation and familarity

        familiar_concepts = self.roadmap_concepts_df[self.roadmap_concepts_df["id"].isin(user_concepts_for_recom_role)]
        recom_concepts_df = self.roadmap_concepts_df[
            self.roadmap_concepts_df["id"].isin(list(sorted_equalized_dict.keys())[:3])
        ]
        print("Concepts you are already familiar: ")
        print(familiar_concepts[["id", "name"]])
        print()
        print("Recommended 3 concepts: ")
        print(recom_concepts_df[["id", "name"]])

        selected_recom_concepts = list(sorted_equalized_dict.keys())[:3]
        selected_recom_concepts = [self.encoder_for_concepts.get(idx) for idx in selected_recom_concepts]
        print(selected_recom_concepts)

        # Course Recommendation Part

        n = 3

        for concept_index in selected_recom_concepts:
            top_scores, top_courses = util.top_n_similarity_scores_for_concept(
                self.udemy_courses_df, self.concept_X_course, concept_index, n
            )

            # Display the result
            print(f"Top {n} Similarity Scores:", top_scores)
            print(f"Corresponding Courses:", top_courses[["id", "title"]])
            print()

        return top_courses["id"]
