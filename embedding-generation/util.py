import pandas as pd
import numpy as np
import os


def category_matcher(x):
    categoryToBeSelected = ["Development", "IT & Software", "Office Productivity"]
    return any(x.startswith(i) for i in categoryToBeSelected)


def combine_columns(row):
    return (
        row["title"]
        + ". \n "
        + row["headline"]
        + ". \n "
        + row["category"]
        + ". \n "
        + row["what_u_learn"]
        + ". \n "
        + row["description"]
    )


def list_files(startpath):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        indent = " " * 4 * (level)
        print("{}{}/".format(indent, os.path.basename(root)))
        subindent = " " * 4 * (level + 1)
        for f in files:
            print("{}{}".format(subindent, f))


def traverse_nodes(node, id):
    node_df = pd.DataFrame()
    if "id" in node:
        # if children key exists, then it is a topic
        if "children" in node:
            # id generation e.g.: 6, 102 --> 602
            idx = str(id) + node["id"][-2:]
            # print(node['fileName'] + " " + idx)
            topic_element = {"id": idx, "name": node["name"], "content": node["name"], "type": "topic"}
            node_df = pd.concat([node_df, pd.DataFrame([topic_element])])
            children = node["children"]
            for child in children:
                node_df2 = traverse_nodes(child, idx)
                node_df = pd.concat([node_df, node_df2])
            return node_df
        else:
            idx = str(id) + node["id"][-2:]
            concept_element = {
                "id": idx,
                "name": node["name"],
                "content": (node["content"][1:] if node["content"].startswith("\n") else node["content"]),
                "type": "concept",
            }
            # print("\t" + node['fileName'] + " " + idx)
            return pd.DataFrame([concept_element])
    else:
        # print(node['fileName'])
        return None
