"""
app.py — Explorador interactivo de cliques monocromáticos y arcoíris

Lanza con
    streamlit run app.py
"""
import random
from itertools import combinations

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl  # acceso a mpl.colormaps

# ------------------------------------------------------------
# 🖌️  Configuración global de la página
# ------------------------------------------------------------
st.set_page_config(
    page_title="Explorador de Cliques Ramsey",
    page_icon="🌈",
    layout="wide",
)

st.title("🌈 Explorador de cliques monocromáticos y arcoíris")
st.markdown(
    """
    Ajusta los parámetros en la barra lateral para generar un grafo aleatorio, después
    analiza si contiene cliques **monocromáticos** (todas sus aristas del mismo color) o
    **arcoíris** (todas las aristas de colores distintos).  
    Los colores de arista se eligen de la paleta *tab10* de Matplotlib.
    """
)

# ------------------------------------------------------------
# ⚙️  Controles en la barra lateral
# ------------------------------------------------------------
with st.sidebar:
    st.header("Ajustes del grafo")
    n_vertices = st.slider("Número de vértices (n)", 3, 40, 8)
    num_colors = st.slider("Colores de arista", 2, 10, 5)
    edge_prob = st.slider("Probabilidad de arista p", 0.1, 1.0, 1.0, 0.05)

    st.header("Búsqueda de cliques")
    k_mono = st.number_input("k monocromático", 2, n_vertices, 3)
    k_rain = st.number_input("k arcoíris", 2, n_vertices, 3)
    max_show = st.number_input("Máx. subgrafos a mostrar", 1, 20, 4)

    with st.expander("Opciones avanzadas"):
        seed_option = st.toggle("Usar semilla fija", value=True)
        seed = st.number_input("Seed", value=7, step=1) if seed_option else None
        layout_option = st.radio("Layout del grafo principal", ["spring", "circular"], index=0)

    regenerate = st.button("🔄 Generar grafo")

# ------------------------------------------------------------
# 📊  Generación del grafo coloreado
# ------------------------------------------------------------

def random_colored_graph(n: int, p: float, k: int, seed: int | None = None) -> nx.Graph:
    """Genera un grafo G(n,p) y colorea sus aristas aleatoriamente con k colores."""
    if seed is not None:
        random.seed(seed)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() <= p:
                G.add_edge(u, v, color=random.randint(0, k - 1))
    return G

@st.cache_resource(show_spinner=False)
def get_graph(n, p, k, seed):
    return random_colored_graph(n, p, k, seed)

# Si el usuario pulsó el botón o es la primera carga, obtenemos el grafo
if regenerate or "G" not in st.session_state:
    st.session_state["G"] = get_graph(n_vertices, edge_prob, num_colors, seed)

G: nx.Graph = st.session_state["G"]

# ------------------------------------------------------------
# 🔎  Predicados y búsqueda de cliques
# ------------------------------------------------------------

def is_monochromatic_clique(G: nx.Graph, verts: tuple[int, ...]) -> bool:
    colors = {G[u][v]["color"] for u, v in combinations(verts, 2)}
    return len(colors) == 1

def is_rainbow_clique(G: nx.Graph, verts: tuple[int, ...]) -> bool:
    colors = [G[u][v]["color"] for u, v in combinations(verts, 2)]
    return len(colors) == len(set(colors))

def find_cliques(G: nx.Graph, k: int, predicate, limit: int):
    found: list[tuple[int, ...]] = []
    for verts in combinations(G.nodes, k):
        if all(G.has_edge(u, v) for u, v in combinations(verts, 2)) and predicate(G, verts):
            found.append(verts)
            if len(found) >= limit:
                break
    return found

mono_cliques = find_cliques(G, k_mono, is_monochromatic_clique, max_show)
rain_cliques = find_cliques(G, k_rain, is_rainbow_clique, max_show)

# ------------------------------------------------------------
# 🖼️  Visualización
# ------------------------------------------------------------

def _get_cmap(name: str, n: int):
    """Devuelve una *ListedColormap* con `n` colores.

    Se adapta a versiones viejas ⇄ nuevas de Matplotlib:
    - En Matplotlib ≥3.8 existe `mpl.colormaps`, pero su `.get_cmap` solo admite 1 parámetro.
    - En versiones anteriores podemos usar `plt.get_cmap(name, lut=n)`.
    """
    # Intento 1 – API clásica (Matplotlib ≤3.7)
    try:
        return plt.get_cmap(name, n)
    except TypeError:
        pass  # Firma no acepta 2 args o no existe
    # Intento 2 – API nueva (Matplotlib ≥3.8)
    try:
        return mpl.colormaps.get_cmap(name).resampled(n)
    except Exception:
        # Último recurso: colormap original sin re‑muestrear
        return plt.get_cmap(name)


def draw_graph(graph: nx.Graph, layout: str):
    pos = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
    }[layout](graph, seed=1)

    cmap = _get_cmap("tab10", num_colors)
    edge_colors = [cmap(graph[u][v]["color"]) for u, v in graph.edges]

    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw(
        graph,
        pos,
        edge_color=edge_colors,
        node_color="lightsteelblue",
        with_labels=True,
        node_size=650,
        width=2,
        ax=ax,
    )
    ax.axis("off")
    return fig


def draw_subgraph(parent: nx.Graph, verts: tuple[int, ...], title: str):
    H = parent.subgraph(verts)
    cmap = _get_cmap("tab10", num_colors)
    edge_colors = [cmap(parent[u][v]["color"]) for u, v in H.edges]
    pos = nx.circular_layout(H)

    fig, ax = plt.subplots(figsize=(3, 3))
    nx.draw(
        H,
        pos,
        edge_color=edge_colors,
        node_color="white",
        with_labels=True,
        node_size=500,
        width=3,
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")
    return fig

# ----  Distribución en la página ----
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Grafo completo")
    fig_main = draw_graph(G, layout_option)
    st.pyplot(fig_main)

with col_right:
    st.subheader("Resumen")
    st.metric("Vértices", len(G.nodes))
    st.metric("Aristas", len(G.edges))
    st.metric(
        f"Cliques mono K{k_mono}",
        len(mono_cliques),
        delta="✔️" if mono_cliques else "✖️",
    )
    st.metric(
        f"Cliques arcoíris K{k_rain}",
        len(rain_cliques),
        delta="✔️" if rain_cliques else "✖️",
    )

# ----  Subgrafos ----
if mono_cliques or rain_cliques:
    st.divider()
    st.subheader("Subgrafos destacados")

    tabs = st.tabs([f"Monocromáticos ({len(mono_cliques)})", f"Arcoíris ({len(rain_cliques)})"])

    # Monocromáticos
    with tabs[0]:
        if mono_cliques:
            cols = st.columns(min(len(mono_cliques), 4))
            for i, verts in enumerate(mono_cliques):
                with cols[i % 4]:
                    fig = draw_subgraph(G, verts, f"Mono-K{k_mono}: {verts}")
                    st.pyplot(fig)
        else:
            st.info("No se encontraron cliques monocromáticos.")

    # Arcoíris
    with tabs[1]:
        if rain_cliques:
            cols = st.columns(min(len(rain_cliques), 4))
            for i, verts in enumerate(rain_cliques):
                with cols[i % 4]:
                    fig = draw_subgraph(G, verts, f"Rainbow-K{k_rain}: {verts}")
                    st.pyplot(fig)
        else:
            st.info("No se encontraron cliques arcoíris.")
else:
    st.warning("No se encontraron subgrafos que cumplan los criterios seleccionados.")

# ------------------------------------------------------------
# 📌  Nota al pie
# ------------------------------------------------------------
with st.expander("Detalles de implementación"):
    st.write(
        "El grafo se genera con un modelo Erdős–Rényi \(G(n, p)\). "
        "Las combinaciones de \(k\) vértices se exploran de forma exhaustiva "
        "hasta encontrar el número máximo de ejemplos indicado."
    )
