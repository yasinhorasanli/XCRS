import numpy as np
import re
import pandas as pd

def get_role_id(concept_id):
    while concept_id % 100 != concept_id:
        concept_id = concept_id / 100
    return int(concept_id)

def get_parent_topics(concept_id):
    parent_topic_list = []
    while concept_id % 100 != concept_id:
        concept_id = int(concept_id / 100)
        if concept_id < 100:
            break
        parent_topic_list.append(concept_id)             
    return parent_topic_list

def calculate_topic_coverage(concept_id_list, roadmap_topics_id_list, roadmap_concepts_id_list):
    topic_concept_count = {topic_id: 0 for topic_id in roadmap_topics_id_list}
    topic_coverage_count = topic_concept_count.copy()

    for concept_id in roadmap_concepts_id_list:
        parent_topics = get_parent_topics(concept_id)
        for topic_id in parent_topics:
            if topic_id in topic_concept_count:
                topic_concept_count[topic_id] += 1

    for concept_id in concept_id_list:
        parent_topics = get_parent_topics(concept_id)
        for topic_id in parent_topics:
            if topic_id in topic_coverage_count:
                topic_coverage_count[topic_id] += 1

    # Calculate the coverage percentage
    coverage_percentage = {topic_id: (count / topic_concept_count[topic_id]) * 100 for topic_id, count in topic_coverage_count.items() if topic_concept_count[topic_id] > 0}
    
    # Filter topics with at least 40% coverage and sort them
    covered_topics = {topic_id: coverage for topic_id, coverage in coverage_percentage.items() if coverage >= 40}
    sorted_covered_topics = sorted(covered_topics.items(), key=lambda x: x[1], reverse=True)
    
    return [topic_id for topic_id, coverage in sorted_covered_topics]

def custom_activation(x):
    k = 0.2  # scaling factor
    b = 25  # shift
    # Apply shifted and scaled sigmoid function and round to 2 decimal places
    return np.round(100 / (1 + np.exp(-k * (x - b))), 2)


def convert_to_float(row):
    element = row["emb"]
    element = np.fromstring(element[1:-1], dtype=float, sep=",")
    return element


def max_element_indices(arr):
    no_of_rows = len(arr)
    no_of_column = len(arr[0])
    max_sim_for_row = []
    for i in range(no_of_rows):
        max = 0
        for j in range(no_of_column):
            if arr[i][j] > max:
                max = arr[i][j]
                x = i
                y = j
        max_sim_for_row.append({"x": x, "y": y, "max": max})
    return max_sim_for_row


def split_and_create_dict(row, column_name):
    values = [value.strip() for value in row[column_name].split(",")]
    return {value: column_name for value in values}


# def create_lists_from_dataframe(df, columns):
#     result_lists = []

#     for index, row in df.iterrows():
#         row_values = [row[column] for column in columns]
#         result_lists.append(row_values)

#     return result_lists


def equalize_digits(original_list):

    # TODO: ValueError: max() arg is an empty sequence
    # Find the maximum number of digits
    max_digits = max(map(lambda x: len(str(x)), original_list))
    # Calculate the necessary power of 10 for each integer
    powers_of_10 = [10 ** (max_digits - len(str(x))) for x in original_list]
    # Equalize the number of digits by multiplying with the calculated powers of 10
    equalized_list = [x * power for x, power in zip(original_list, powers_of_10)]
    # Create dictionaries for original and equalized/sorted lists
    # original_dict = dict(zip(original_list, equalized_list))
    sorted_equalized_dict = dict(sorted(zip(original_list, equalized_list), key=lambda x: x[1]))

    return sorted_equalized_dict


def top_n_courses_for_concept(udemy_courses_df, similarity_matrix, concept_index, disliked_course_id_list, n):
    # Get the row corresponding to the specified concept index
    row = similarity_matrix[concept_index, :]

    # Reduce the score of disliked courses
    row_reduced = row.copy()
    row_reduced[disliked_course_id_list] = row_reduced[disliked_course_id_list] / 2

    # Get the indices of the top N similarity scores for the specified concept
    top_indices = np.argsort(row_reduced)[-n:][::-1]

    # Get the top N scores and their corresponding courses
    top_scores = row[top_indices]
    top_courses_with_scores = [(index, score) for index, score in zip(top_indices, top_scores)]

    return top_courses_with_scores

def top_n_concepts_for_courses(roadmap_concepts_df, similarity_matrix, remaining_concepts_encoded, course_id, n):

    column = similarity_matrix[:, course_id]
    filtered_column = column[remaining_concepts_encoded]
    top_filtered_indices = np.argsort(filtered_column)[-n:][::-1]
    top_indices = [remaining_concepts_encoded[idx] for idx in top_filtered_indices]
    top_scores = column[top_indices]
    top_concepts = roadmap_concepts_df.iloc[top_indices].copy()
    top_concepts.loc[:, "sim_score"] = top_scores

    return top_concepts



def calculate_threshold(sim_mat, sigma_num):
    flattened_scores = sim_mat.flatten()
    mean = np.mean(flattened_scores)
    std_dev = np.std(flattened_scores)

    return mean + sigma_num * std_dev


def calculate_mean(sim_mat):
    flattened_scores = sim_mat.flatten()
    mean = np.mean(flattened_scores)

    return mean


def calculate_sigma(sim_mat):
    flattened_scores = sim_mat.flatten()
    std_dev = np.std(flattened_scores)

    return std_dev


def pad_string_with_dashes(input_string, length=120):
    max_length = length
    if len(input_string) > max_length:
        raise ValueError("Input string is longer than 120 characters")

    num_dashes = max_length - len(input_string)
    padded_string = input_string + "-" * num_dashes

    return padded_string


def remove_emojis(text):
    # Remove emojis using the emoji library
    emoji_pattern = re.compile(
        pattern="["
                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                u"\U0001F600-\U0001F64F"  # emoticons
                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                u"\U0001F700-\U0001F77F"  # alchemical symbols
                u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                u"\U00002702-\U000027B0"  # Dingbats
                u"\U000024C2-\U0001F251" 
                "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)