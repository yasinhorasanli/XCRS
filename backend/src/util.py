import numpy as np


def get_role_id(concept_id):
    while concept_id % 100 != concept_id:
        concept_id = concept_id / 100
    return int(concept_id)


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
    original_dict = dict(zip(original_list, equalized_list))
    sorted_equalized_dict = dict(sorted(zip(original_list, equalized_list), key=lambda x: x[1]))

    return original_dict, sorted_equalized_dict


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
    top_courses = udemy_courses_df.iloc[top_indices]
    top_courses["sim_score"] = top_scores

    return top_courses


def top_n_concepts_for_courses(roadmap_concepts_df, similarity_matrix, recom_role_id, course_id, n):
    
    column = similarity_matrix[:, course_id]
    top_indices = np.argsort(column)[-n:][::-1]
    top_scores = column[top_indices]
    top_courses = roadmap_concepts_df.iloc[top_indices]
    top_courses["sim_score"] = top_scores

    return top_courses