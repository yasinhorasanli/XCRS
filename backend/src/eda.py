import pandas as pd
import numpy as np

# internal classes
import util

class ExplatoryDataAnalysis:
    def __init__(
        self,
        udemy_courses_df: pd.DataFrame, 
        roadmap_concepts_df: pd.DataFrame,
        user_courses_df: pd.DataFrame,
        decoder_for_courses: dict,
        decoder_for_concepts: dict,
        decoder_for_user_courses: dict,
        sim_mat_course_X_concept: np.ndarray,
        sim_mat_user_course_X_concept: np.ndarray
    ):
        self.udemy_courses_df = udemy_courses_df
        self.roadmap_concepts_df = roadmap_concepts_df
        self.user_courses_df = user_courses_df
        self.decoder_for_courses = decoder_for_courses
        self.decoder_for_concepts = decoder_for_concepts
        self.decoder_for_user_courses = decoder_for_user_courses
        self.sim_mat_course_X_concept = sim_mat_course_X_concept
        self.sim_mat_user_course_X_concept = sim_mat_user_course_X_concept

    def find_course_X_concept_and_sim_scores(self, threshold):

        # Matched concept for each course
        max_sim_for_row_courses_X_concepts = util.max_element_indices(self.sim_mat_course_X_concept)
        max_sim_for_row_courses_X_concepts_df = pd.DataFrame(max_sim_for_row_courses_X_concepts)

        # Courses X Concepts Matches
        total_match = 0
        for i in range(max_sim_for_row_courses_X_concepts_df.shape[0]):
            course_id = self.decoder_for_courses[max_sim_for_row_courses_X_concepts_df.iloc[i]["x"]]
            concept_id = self.decoder_for_concepts[max_sim_for_row_courses_X_concepts_df.iloc[i]["y"]]
            max_sim = max_sim_for_row_courses_X_concepts_df.iloc[i]["max"]
            if max_sim > threshold:
                total_match = total_match + 1
                print("Course: " + self.udemy_courses_df.loc[self.udemy_courses_df['id'] == course_id]['title'].to_string())
                print("Concept: " + self.roadmap_concepts_df.loc[self.roadmap_concepts_df['id'] == concept_id]['name'].to_string())
                print("Sim Score: " + str(max_sim))
                print("-------------------------------------------------------------------------------------------------")
        print(total_match)

    def find_concept_X_course_and_sim_scores(self, threshold):

        # Matched course for each concept
        sim_mat_concept_X_course = self.sim_mat_course_X_concept.transpose()
        max_sim_for_column_courses_X_concepts = util.max_element_indices(sim_mat_concept_X_course)
        max_sim_for_column_courses_X_concepts_df = pd.DataFrame(max_sim_for_column_courses_X_concepts)

        # Concepts X Courses Matches
        total_match = 0
        for i in range(max_sim_for_column_courses_X_concepts_df.shape[0]):
            concept_id = self.decoder_for_concepts[max_sim_for_column_courses_X_concepts_df.iloc[i]["x"]]
            course_id = self.decoder_for_courses[max_sim_for_column_courses_X_concepts_df.iloc[i]["y"]]
            max_sim = max_sim_for_column_courses_X_concepts_df.iloc[i]["max"]
            if max_sim > threshold:
                total_match = total_match + 1
                print("Concept: " + self.roadmap_concepts_df.loc[self.roadmap_concepts_df['id'] == concept_id]['name'].to_string())
                print("Course: " + self.udemy_courses_df.loc[self.udemy_courses_df['id'] == course_id]['title'].to_string())
                print("Sim Score: " + str(max_sim))
                print("-------------------------------------------------------------------------------------------------")
        print(total_match)


    def find_user_course_X_concept_and_sim_scores(self, threshold):
        
        max_sim_for_row_user_courses_X_concepts = util.max_element_indices(self.sim_mat_user_course_X_concept)
        max_sim_for_row_user_courses_X_concepts_df = pd.DataFrame(max_sim_for_row_user_courses_X_concepts)

        # User Courses X Concepts Matches
        total_match = 0
        user_concept_id_list = []
        for i in range(max_sim_for_row_user_courses_X_concepts_df.shape[0]):
            course_name = self.decoder_for_user_courses[max_sim_for_row_user_courses_X_concepts_df.iloc[i]["x"]]
            concept_id = self.decoder_for_concepts[max_sim_for_row_user_courses_X_concepts_df.iloc[i]["y"]]
            max_sim = max_sim_for_row_user_courses_X_concepts_df.iloc[i]["max"]
            if max_sim > threshold:
                total_match = total_match + 1
                print("User Course: " + course_name)
                print("Category: " + self.user_courses_df.loc[i]['categories'])
                print("Concept id: " + str(concept_id))
                user_concept_id_list.append(concept_id)
                print("Concept: " + self.roadmap_concepts_df.loc[self.roadmap_concepts_df['id'] == concept_id]['name'].to_string())
                print("Sim Score: " + str(max_sim))
                print("-------------------------------------------------------------------------------------------------")
        print(total_match)


    def find_concept_X_user_course_and_sim_scores(self, threshold):
        
        sim_mat_concept_X_user_course = self.sim_mat_user_course_X_concept.transpose()
        max_sim_for_column_concepts_X_user_courses = util.max_element_indices(sim_mat_concept_X_user_course)
        max_sim_for_column_concepts_X_user_courses_df = pd.DataFrame(max_sim_for_column_concepts_X_user_courses)

        # Concepts X User Courses Matches
        total_match = 0
        user_concept_id_list = []
        for i in range(max_sim_for_column_concepts_X_user_courses_df.shape[0]):
            concept_id = self.decoder_for_concepts[max_sim_for_column_concepts_X_user_courses_df.iloc[i]["x"]]
            course_name = self.decoder_for_user_courses[max_sim_for_column_concepts_X_user_courses_df.iloc[i]["y"]]
            max_sim = max_sim_for_column_concepts_X_user_courses_df.iloc[i]["max"]
            if max_sim > threshold:
                total_match = total_match + 1
                print("Concept: " + self.roadmap_concepts_df.loc[self.roadmap_concepts_df['id'] == concept_id]['name'].to_string())
                print("User Course: " + course_name)
                print("Category: " + self.user_courses_df.loc[i]['categories'])
                print("Sim Score: " + str(max_sim))
                print("-------------------------------------------------------------------------------------------------")
        print(total_match)
