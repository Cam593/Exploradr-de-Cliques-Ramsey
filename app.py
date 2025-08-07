# app.py
import random
from itertools import combinations
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt

# ---------- UI ----------
st.title("Explorador de cliques monocromáticos y arcoíris")
n_vertices  = st.slider("Número de vértices (n)", 3, 30, 6)
num_colors  = st.slider("Colores de arista",        2, 10, 5)
edge_prob   = st.slider("Prob. de arista p",        0.1, 1.0, 1.0, 0.05)
k_mono      = st.number_input("k monocromático",    2, n_vertices, 3)
k_rain      = st.number_input("k arcoíris",         2, n_vertices, 3)
seed        = st.number_input("Seed (opcional)",    value=7, step=1)
max_show    = st.number_input("Máx. subgrafos a mostrar", 1, 10, 1)

# ---------- Lógica ----------
@st.cache_resource
def random_colored_graph(n, p, k, seed):
    if seed is not None:
        random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() <= p:
                G.add_edge(u, v, color=random.randint(0, k - 1))
    return G

def is_mono(G, verts):
    cols = {G[u][v]["color"] for u, v in combinations(verts, 2)}
    return len(cols) == 1

def is_rainbow(G, verts):
    cols = [G[u][v]["color"] for u, v in combinations(verts, 2)]
    return len(cols) == len(set(cols))

def find(G, k, predicate, lim):
    out = []
    for verts in combinations(G.nodes, k):
        if all(G.has_edge(u, v) for u, v in combinations(verts, 2)):
            if predicate(G, verts):
                out.append(verts)
                if len(out) >= lim:
                    break
    return out

G = random_colored_graph(n_vertices, edge_prob, num_colors, seed)
mono = find(G, k_mono, is_mono, max_show)
rain = find(G, k_rain, is_rainbow, max_show)

# ---------- Dibujo ----------
def draw_subgraph(sub, title):
    H = G.subgraph(sub)
    cmap = plt.get_cmap("tab10")
    pos = nx.circular_layout(H)
    edge_colors = [cmap(G[u][v]["color"]) for u, v in H.edges()]
    nx.draw(H, pos, edge_color=edge_colors, node_color="white",
            with_labels=True, node_size=500, width=3)
    plt.title(title)
    plt.axis("off")

tab1, tab2 = st.tabs(["Grafo completo", "Subgrafos"])

with tab1:
    plt.figure(figsize=(4,4))
    cmap = plt.get_cmap("tab10")
    pos = nx.spring_layout(G, seed=1)
    nx.draw(G, pos,
            edge_color=[cmap(G[u][v]['color']) for u,v in G.edges()],
            node_color="lightsteelblue", with_labels=True,
            node_size=600, width=2)
    plt.axis("off")
    st.pyplot(plt.gcf())

with tab2:
    for sub in mono:
        plt.figure()
        draw_subgraph(sub, f"Mono-K{k_mono}: {sub}")
        st.pyplot(plt.gcf())
    for sub in rain:
        plt.figure()
        draw_subgraph(sub, f"Rainbow-K{k_rain}: {sub}")
        st.pyplot(plt.gcf())
