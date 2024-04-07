# API
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict

import textwrap
import numpy as np
import pandas as pd
#import tensorflow as tf
import os
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import google.generativeai as palm

# internal classes
import util


def init_palm_data():
    save_dir = "../data/"
    emb_model_name="embedding-gecko-001"
    udemy_courses_df = pd.read_csv(save_dir + 'udemy_courses_{}.csv'.format(emb_model_name))
    roadmap_nodes_df = pd.read_csv(save_dir + 'roadmap_nodes_{}.csv'.format(emb_model_name))
    roadmap_concepts_df = roadmap_nodes_df[roadmap_nodes_df['type']=="concept"].copy()
    roadmap_concepts_df.reset_index(inplace=True)

    roles = ['ai-data-scientist', 'android', 'backend', 'blockchain', 'devops', 'frontend', 'full-stack', 'game-developer', 'qa', 'ux-design']

    roadmaps_dict = {'id': np.arange(1, len(roles)+1), 'name': roles}
    roadmaps_df = pd.DataFrame.from_dict(roadmaps_dict)
    roadmaps_df.set_index('id', inplace=True)
    
    return (udemy_courses_df, roadmap_nodes_df, roadmap_concepts_df, roadmaps_df)


def embed_fn(text):
  return palm.generate_embeddings(model="models/embedding-gecko-001", text=text)['embedding']

# LOAD DATA FOR PALM 
udemy_courses_df, roadmap_nodes_df, roadmap_concepts_df, roadmaps_df = init_palm_data()

# Encoders/Decoders for Courses and Concepts
course_id_list = udemy_courses_df['id']
encoder_for_courses = dict([(v, k) for v, k in zip(course_id_list, range(len(course_id_list)))])
decoder_for_courses = dict([(v, k) for k, v in encoder_for_courses.items()])

concept_id_list = roadmap_concepts_df['id']
encoder_for_concepts = dict([(v, k) for v, k in zip(concept_id_list, range(len(concept_id_list)))])
decoder_for_concepts = dict([(v, k) for k, v in encoder_for_concepts.items()])

# Necessary to understand which role the concept belongs to
digit_counts = {digit: 0 for digit in range(1,11)}
for concept_id in concept_id_list:
  role_id = util.get_role_id(concept_id)
  digit_counts[role_id] += 1

def get_emb_lists(courses_df:pd.DataFrame, concepts_df:pd.DataFrame, model:str):
  column_name = model + '_emb'

  courses_df[column_name] = courses_df.apply(util.convert_to_float, axis=1)
  course_emb_list = courses_df[column_name].values
  course_emb_list = np.vstack(course_emb_list)

  concepts_df[column_name] = concepts_df.apply(util.convert_to_float, axis=1)
  concept_emb_list = concepts_df[column_name].values
  concept_emb_list = np.vstack(concept_emb_list)
  
  return course_emb_list, concept_emb_list


# Udemy Courses and Concepts Embeddings List - PALM
palm_course_emb_list, palm_concepts_emb_list = get_emb_lists(udemy_courses_df, roadmap_concepts_df, model='palm')

# Similarity Matrix between Courses and Concepts
sim_mat_course_x_roadmap = cosine_similarity(palm_course_emb_list, palm_concepts_emb_list)
sim_mat_roadmap_x_course = sim_mat_course_x_roadmap.transpose()

# Matched concept for each course
max_sim_for_row = util.max_element_indices(sim_mat_course_x_roadmap)
max_sim_for_row_df = pd.DataFrame(max_sim_for_row)
# Matched course for each concept
max_sim_for_column = util.max_element_indices(sim_mat_roadmap_x_course)
max_sim_for_column_df = pd.DataFrame(max_sim_for_column)

# EXAMPLE
# course_id = decoder_for_courses[max_sim_for_row_df.iloc[19]['x']]
# concept_id = decoder_for_concepts[max_sim_for_row_df.iloc[19]['y']]



# TODO: anotherModel
# anotherModel_course_emb_list,  anotherModel_concepts_emb_list = get_emb_lists(udemy_courses_df, roadmap_concepts_df, model='anotherModel')



# Udemy Courses X Concepts Matches
# By setting the threshold as 0.8
total_match = 0
for i in range(len(max_sim_for_row)):
  course_id = decoder_for_courses[max_sim_for_row_df.iloc[i]['x']]
  concept_id = decoder_for_concepts[max_sim_for_row_df.iloc[i]['y']]
  max_sim = max_sim_for_row_df.iloc[i]['max']
  if (max_sim > 0.8):
    total_match = total_match + 1
    # print("Course: " + udemy_courses_df.loc[udemy_courses_df['id'] == course_id]['title'].to_string())
    # print("Concept: " + roadmap_concepts_df.loc[roadmap_concepts_df['id'] == concept_id]['name'].to_string())
    # print("Sim Score: " + str(max_sim))
    # print("-------------------------------------------------------------------------------------------------")


# Concepts X Udemy Courses Matches
# By setting the threshold as 0.8
total_match2 = 0
for i in range(len(max_sim_for_column)):
  concept_id = decoder_for_concepts[max_sim_for_column_df.iloc[i]['x']]
  course_id = decoder_for_courses[max_sim_for_column_df.iloc[i]['y']]
  max_sim = max_sim_for_column_df.iloc[i]['max']
  if (max_sim > 0.8):
    total_match2 = total_match2 + 1
    # print("Concept: " + roadmap_concepts_df.loc[roadmap_concepts_df['id'] == concept_id]['name'].to_string())
    # print("Course: " + udemy_courses_df.loc[udemy_courses_df['id'] == course_id]['title'].to_string())
    # print("Sim Score: " + str(max_sim))
    # print("-------------------------------------------------------------------------------------------------")

# USER EMBEDDINGS PART
categories = ['TookAndLiked', 'TookAndNeutral', 'TookAndDisliked', 'Curious']
user_df = pd.DataFrame(columns=categories)
user1_took = "Physics , Intr. to Information Systems, Intr.to Comp.Eng.and Ethics, Mathematics I, Linear Algebra, Engineering Mathematics, Digital Circuits, Data Structures, Introduction to Electronics, Basics of Electrical Circuits, Object Oriented Programming, Computer Organization, Logic Circuits Laboratory, Numerical Methods, Formal Languages and Automata, Analysis of Algorithms I, Probability and Statistics, Microcomputer Lab., Database Systems, Microprocessor Systems, Computer Architecture, Computer Operating Systems, Analysis of Algorithms II, Signal&Systems for Comp.Eng."
user1_took_and_liked = "Digital Circuits , Data Structures , Introduction to Electronics, Microprocessor Systems , Computer Architecture"
user1_took_and_neutral = "Mathematics I, Linear Algebra, Engineering Mathematics, Basics of Electrical Circuits, Object Oriented Programming, Computer Organization, Logic Circuits Laboratory, Analysis of Algorithms I, Probability and Statistics, Microcomputer Lab., Database Systems, Computer Operating Systems, Analysis of Algorithms II, Signal&Systems for Comp.Eng."
user1_took_and_disliked = "Physics, Intr. to Information Systems, Intr.to Comp.Eng.and Ethics, Numerical Methods, Formal Languages and Automata"
user1_curious = "Embedded Softwares, Web Development"
user_df = pd.DataFrame.from_dict([{'TookAndLiked': user1_took_and_liked, 'TookAndNeutral': user1_took_and_neutral, 'TookAndDisliked': user1_took_and_disliked, 'Curious': user1_curious}])

result_dicts = []
columns_to_process = categories

for column in columns_to_process:
    result_dicts.extend(user_df.apply(lambda row: util.split_and_create_dict(row, column), axis=1).to_list())

flat_data = [item for sublist in result_dicts for item in sublist.items()]
user_courses_df = pd.DataFrame(flat_data, columns=['courses', 'categories'])

print(user_courses_df)

# Embedding Generation for User Data 

palm_api_key = "AIzaSyDJNptL0HcFZEIXhJu3wUSiR1FHvEgOvw0"

palm.configure(api_key=palm_api_key)
model = "models/embedding-gecko-001"

user_courses_df['palm_emb'] = user_courses_df['courses'].apply(embed_fn)

# User Courses X Roadmap Concepts Similarity Calculation

encoder_for_user_courses = dict([(v, k) for v, k in zip(user_courses_df['courses'], range(len(user_courses_df)))])
decoder_for_user_courses = dict([(v, k) for k, v in encoder_for_user_courses.items()])

# user_courses_df['palm_emb'] = user_courses_df.apply(convert_to_float, axis=1)
user_emb_list = user_courses_df['palm_emb'].values
user_emb_list = np.vstack(user_emb_list)

print(user_emb_list.shape)

sim_mat_user_x_roadmap = cosine_similarity(user_emb_list, palm_concepts_emb_list)
max_sim_for_row2 = util.max_element_indices(sim_mat_user_x_roadmap)
max_sim_for_column2 = util.max_element_indices(sim_mat_user_x_roadmap.transpose())
max_sim_for_row2_df = pd.DataFrame(max_sim_for_row2)
max_sim_for_column2_df = pd.DataFrame(max_sim_for_column2)

# User Courses X Concepts Matches
total_match = 0
user_concept_id_list = []
for i in range(len(max_sim_for_row2)):
  course_name = decoder_for_user_courses[max_sim_for_row2_df.iloc[i]['x']]
  concept_id = decoder_for_concepts[max_sim_for_row2_df.iloc[i]['y']]
  max_sim = max_sim_for_row2_df.iloc[i]['max']
  if (max_sim > 0.7):
    total_match = total_match + 1
    # print("User Course: " + course_name)
    # print("Category: " + user_courses_df.loc[i]['categories'])
    # print("Concept id: " + str(concept_id))
    user_concept_id_list.append(concept_id)
    # print("Concept: " + roadmap_concepts_df.loc[roadmap_concepts_df['id'] == concept_id]['name'].to_string())
    # print("Sim Score: " + str(max_sim))
    # print("-------------------------------------------------------------------------------------------------")


# Role Recommendation for User
    
user_concept_id_set = set(user_concept_id_list)
user_digit_counts = {digit: 0 for digit in range(1,11)}
for concept_id in user_concept_id_set:
  role_id = util.get_role_id(concept_id)
  user_digit_counts[role_id] += 1

ratio_dict = {digit: user_digit_counts[digit] / digit_counts[digit] for digit in digit_counts.keys()}
recom_role_id = max(ratio_dict, key=ratio_dict.get)

print(digit_counts)
print(user_digit_counts)
print(ratio_dict)
print("Role id with the maximum ratio:", recom_role_id)

print("Recommended Role-{} with percentage: {}".format(recom_role_id, ratio_dict[recom_role_id]*100))
print("Recommended Role: " + roadmaps_df.loc[recom_role_id]['name'])


# Course Recommendation for User

user_concepts_for_recom_role = set()
for concept_id in user_concept_id_set:
  if (util.get_role_id(concept_id) == recom_role_id):
    user_concepts_for_recom_role.add(concept_id)

recom_role_concepts = set()
for concept in concept_id_list:
  if (util.get_role_id(concept) == recom_role_id):
    recom_role_concepts.add(concept)

print("Number of role-" + str(recom_role_id) + " concepts : " + str(len(recom_role_concepts)))
print("Number of matching user concepts for role-" + str(recom_role_id) + ": " + str(len(user_concepts_for_recom_role)))

recom_concepts = recom_role_concepts.difference(user_concepts_for_recom_role)

original_dict, sorted_equalized_dict = util.equalize_digits(recom_concepts)

print("Original Dict:", original_dict)
print("Sorted equalized Dict:", sorted_equalized_dict)


# Explanation for Course recommendation --- Concept recommendation and familarity

familiar_concepts = roadmap_concepts_df[roadmap_concepts_df['id'].isin(user_concepts_for_recom_role)]
recom_concepts_df = roadmap_concepts_df[roadmap_concepts_df['id'].isin(list(sorted_equalized_dict.keys())[:3])]
print("Concepts you are already familiar: ")
print(familiar_concepts[['id', 'name']])
print()
print("Recommended 3 concepts: ")
print(recom_concepts_df[['id', 'name']])

selected_recom_concepts = list(sorted_equalized_dict.keys())[:3]
selected_recom_concepts = [encoder_for_concepts.get(idx) for idx in selected_recom_concepts]
print(selected_recom_concepts)


# Course Recommendation Part

n = 3

for concept_index in selected_recom_concepts:
  top_scores, top_courses = util.top_n_similarity_scores_for_concept(udemy_courses_df, sim_mat_roadmap_x_course, concept_index, n)

  # Display the result
  print(f"Top {n} Similarity Scores:", top_scores)
  print(f"Corresponding Courses:", top_courses[['id', 'title']])
  print()


# Concepts X User Courses Matches
  
total_match2 = 0
for i in range(len(max_sim_for_column2)):
  concept_id = decoder_for_concepts[max_sim_for_column2_df.iloc[i]['x']]
  course_name = decoder_for_user_courses[max_sim_for_column2_df.iloc[i]['y']]
  max_sim = max_sim_for_column2_df.iloc[i]['max']
  if (max_sim > 0.7):
    total_match2 = total_match2 + 1
    # print("Concept: " + roadmap_concepts_df.loc[roadmap_concepts_df['id'] == concept_id]['name'].to_string())
    # print("User Course: " + course_name)
    # print("Category: " + user_courses_df.loc[user_courses_df['courses'] == course_name]['categories'].to_string())
    # print("Sim Score: " + str(max_sim))
    # print("-------------------------------------------------------------------------------------------------")

# Prints number of total matches
print(total_match)
print(total_match2)



app = FastAPI()

# Model for course recommendation
class CourseRecommendation(BaseModel):
    course: str

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
        "courses": [{"course": "Course 1"}, {"course": "Course 2"}, {"course": "Course 3"}]
    },
    {
        "role": "Role B",
        "explanation": "Explanation for Role B",
        "courses": [{"course": "Course 4"}, {"course": "Course 5"}, {"course": "Course 6"}]
    },
    {
        "role": "Role C",
        "explanation": "Explanation for Role C",
        "courses": [{"course": "Course 7"}, {"course": "Course 8"}, {"course": "Course 9"}]
    }
]




@app.post("/recommendations/")
async def get_recommendations(request: RecommendationRequest):
    # Extract role and course recommendations from sample data
    role_recommendations = [
        RoleRecommendation(
            role=rec["role"],
            explanation=rec["explanation"],
            courses=[CourseRecommendation(**course) for course in rec["courses"]]
        )
        for rec in recommendation_data
    ]

    # Return recommendation response
    return RecommendationResponse(recommendations=role_recommendations)