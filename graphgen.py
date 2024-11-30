import networkx as nx
import matplotlib.pyplot as plt
from random import sample
from extension import load_meta
from itertools import combinations

metadata = load_meta("./data/question_meta.csv")
metadata = {k:metadata[k] for k in metadata if k < 300}
subject_to_questions = {}

for question_id, subjects in metadata.items():
    for subject in subjects:
        if subject not in subject_to_questions:
            subject_to_questions[subject] = []
        subject_to_questions[subject].append(question_id)

# Generate pairs of question IDs for each subject
question_pairs = set()

# Iterate through all pairs of question IDs
for q1, q2 in combinations(metadata.keys(), 2):
    # Find the intersection of their subjects
    common_subjects = set(metadata[q1]) & set(metadata[q2])
    # Check if they share at least two subjects
    if len(common_subjects) >= 3:
        question_pairs.add((q1, q2))

print(question_pairs)

n = len(metadata.keys())
Q = [i for i in range(n)]
E = question_pairs
G = nx.Graph()
# Add vertices and edges
G.add_nodes_from(Q)
G.add_edges_from(E)

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(G, seed=42, scale=2, k=0.1)
nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=700)
nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.5, width=0.8)
nx.draw_networkx_labels(G, pos, font_size=6)
plt.title("Improved Graph Visualization", fontsize=16)
plt.axis("off")
plt.show()