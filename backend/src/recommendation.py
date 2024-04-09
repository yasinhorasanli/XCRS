import pandas as pd
import numpy as np

# internal classes
import util

class Recommendation:
    def __init__(
        self,
        udemy_courses_df: pd.DataFrame,
        roadmaps_df: pd.DataFrame
    ):
        self.udemy_courses_df = udemy_courses_df 
        self.roadmaps_df = roadmaps_df 


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

        ratio_dict = {digit: user_role_id_concept_counts[digit] / role_id_concept_counts[digit] for digit in role_id_concept_counts.keys()}
        recom_role_id = max(ratio_dict, key=ratio_dict.get)

        print(role_id_concept_counts)
        print(user_role_id_concept_counts)
        print(ratio_dict)
        print("Role id with the maximum ratio:", recom_role_id)

        print("Recommended Role-{} with percentage: {}".format(recom_role_id, ratio_dict[recom_role_id] * 100))
        print("Recommended Role: " + self.roadmaps_df.loc[recom_role_id]["name"])

        return recom_role_id