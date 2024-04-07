import textwrap
import numpy as np
import pandas as pd
#import tensorflow as tf
import os
from pathlib import Path
import json

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt




save_dir = "./data/"
emb_model_name="embedding-gecko-001"
udemy_courses_df = pd.read_csv(save_dir + 'udemy_courses_{}.csv'.format(emb_model_name))
roadmap_nodes_df = pd.read_csv(save_dir + 'roadmap_nodes_{}.csv'.format(emb_model_name))
roadmap_concepts_df = roadmap_nodes_df[roadmap_nodes_df['type']=="concept"].copy()
roadmap_concepts_df.reset_index(inplace=True)

roles = ['ai-data-scientist', 'android', 'backend', 'blockchain', 'devops', 'frontend', 'full-stack', 'game-developer', 'qa', 'ux-design']

roadmaps_dict = {'id': np.arange(1, len(roles)+1), 'name': roles}
roadmaps_df = pd.DataFrame.from_dict(roadmaps_dict)
roadmaps_df.set_index('id', inplace=True)