"""
app.py — Explorador interactivo de cliques monocromáticos y arcoíris

Ejecuta:
    streamlit run app.py
"""
import random
from itertools import combinations
import io

import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl  # acceso a mpl.colormaps

# ------------------------------------------------------------
# 📚  Módulo de conocimiento: números de Ramsey clásicos (2 colores)
# ------------------------------------------------------------
# Tabla mínima de valores exactos conocidos y algunas cotas.
# Claves simétricas: (min(s,t), max(s,t)).
EXACT_R: dict[tuple[int, int], int] = {
    (3, 3): 6,
    (3, 4): 9,
    (3, 5): 14,
    (3, 6): 18,
    (4, 4): 18,
    (4, 5): 25,
    (3, 7): 23,
}
BOUNDS_R: dict[tuple[int, int], tuple[int, int]] = {
    # Ejemplo conocido: R(5,5) aún abierto
    (5, 5): (43, 49),
}

def ramsey_lookup(s: int, t: int):
    """Devuelve info de R(s,t) para el caso clásico de 2 colores.

    Retorna un dict con llaves {type, value} (exact) o {type, lower, upper} (bounds),
    o None si no tenemos el dato en esta tabla mínima.
    """
    a, b = sorted((s, t))
    if (a, b) in EXACT_R:
        return {"type": "exact", "value": EXACT_R[(a, b)]}
    if (a, b) in BOUNDS_R:
        L, U = BOUNDS_R[(a, b)]
        return {"type": "bounds", "lower": L, "upper": U}
    return None

# ------------------------------------------------------------
# 🖌️  Configuración global
# ------------------------------------------------------------
st.set_page_config(
    page_title="Explorador de Cliques Ramsey",
    page_icon="🌈",
    layout="wide",
)

st.title("🌈 Explorador de cliques monocromáticos y arcoíris")
st.markdown(
    """
    Ajusta los parámetros en la barra lateral para generar un grafo aleatorio, luego
    inspecciona si contiene cliques **monocromáticos** (todas las aristas del mismo color)
    o **arcoíris** (todas las aristas con colores distintos).
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

    # --- Modo exploración (experimental) ---
    with st.expander("Modo exploración (experimental)"):
        explore_on = st.toggle("Activar modo exploración", value=False)
        if explore_on:
            target = st.selectbox("Patrón objetivo", ["monocromático", "arcoíris", "mono o arcoíris"], index=2)
            k_target = st.number_input("k objetivo", 2, 10, min(k_mono, 5))
            n_min, n_max = st.slider("Rango de n", 3, 80, (max(3, k_target), max(3, k_target + 10)))
            trials = st.number_input("Simulaciones por n", 5, 500, 50, step=5)
            threshold = st.slider("Umbral de probabilidad", 0.1, 1.0, 0.5, 0.05)
            go_explore = st.button("▶ Ejecutar exploración")
            st.caption("Nota: la complejidad crece rápido con k; mantén k pequeño (≤5).")

# ------------------------------------------------------------
# 📊  Generación del grafo
# ------------------------------------------------------------

def random_colored_graph(n: int, p: float, k: int, seed: int | None = None) -> nx.Graph:
    """Genera un grafo Erdős–Rényi G(n,p) y colorea sus aristas con k colores."""
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

if regenerate or "G" not in st.session_state:
    st.session_state["G"] = get_graph(n_vertices, edge_prob, num_colors, seed)

G: nx.Graph = st.session_state["G"]

# ------------------------------------------------------------
# 🔎  Utilidades de cliques
# ------------------------------------------------------------

def is_monochromatic_clique(G: nx.Graph, verts: tuple[int, ...]) -> bool:
    return len({G[u][v]["color"] for u, v in combinations(verts, 2)}) == 1

def is_rainbow_clique(G: nx.Graph, verts: tuple[int, ...]) -> bool:
    colors = [G[u][v]["color"] for u, v in combinations(verts, 2)]
    return len(colors) == len(set(colors))

def find_cliques(G: nx.Graph, k: int, predicate, limit: int):
    found = []
    for verts in combinations(G.nodes, k):
        if all(G.has_edge(u, v) for u, v in combinations(verts, 2)) and predicate(G, verts):
            found.append(verts)
            if len(found) >= limit:
                break
    return found

mono_cliques = find_cliques(G, k_mono, is_monochromatic_clique, max_show)
rain_cliques = find_cliques(G, k_rain, is_rainbow_clique, max_show)

# ------------------------------------------------------------
# 🔬  Modo exploración: barrido de n con múltiples simulaciones
# ------------------------------------------------------------

def _exists_pattern(G: nx.Graph, k: int, target: str) -> bool:
    if target == "monocromático":
        return len(find_cliques(G, k, is_monochromatic_clique, 1)) > 0
    if target == "arcoíris":
        return len(find_cliques(G, k, is_rainbow_clique, 1)) > 0
    # mono o arcoíris
    return (
        len(find_cliques(G, k, is_monochromatic_clique, 1)) > 0
        or len(find_cliques(G, k, is_rainbow_clique, 1)) > 0
    )


def run_exploration(target: str, k: int, n_min: int, n_max: int, trials: int, p: float, colors: int):
    n_values = list(range(n_min, n_max + 1))
    successes = []
    progress = st.progress(0, text="Simulando…")
    total = len(n_values) * trials
    done = 0
    for n in n_values:
        s = 0
        for _ in range(trials):
            Gtmp = random_colored_graph(n, p, colors, seed=None)
            if _exists_pattern(Gtmp, k, target):
                s += 1
            done += 1
            if done % max(1, total // 100) == 0:
                progress.progress(done / total, text=f"n={n}  ({done}/{total})")
        successes.append(s)
    progress.empty()
    probs = [s / trials for s in successes]
    return n_values, successes, probs

# ------------------------------------------------------------
# 🎨  Colormaps compatibles (Matplotlib 3.7 ↔ 3.8)
# ------------------------------------------------------------

def _get_cmap(name: str, n: int):
    try:
        return plt.get_cmap(name, n)  # Matplotlib ≤3.7
    except TypeError:
        try:
            return mpl.colormaps.get_cmap(name).resampled(n)  # Matplotlib ≥3.8
        except Exception:
            return plt.get_cmap(name)

# ------------------------------------------------------------
# 🖼️  Dibujado de grafos
# ------------------------------------------------------------

def _layout_pos(graph: nx.Graph, layout: str):
    if layout == "spring":
        return nx.spring_layout(graph, seed=1)
    # circular_layout no acepta 'seed'
    return nx.circular_layout(graph)

def draw_graph(graph: nx.Graph, layout: str):
    pos = _layout_pos(graph, layout)
    cmap = _get_cmap("tab10", num_colors)
    edge_colors = [cmap(graph[u][v]["color"]) for u, v in graph.edges]

    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw(
        graph,
        pos,
        edge_color=edge_colors,
        node_color="lightsteelblue",
        node_size=650,
        width=2,
        with_labels=True,
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
        node_size=500,
        width=3,
        with_labels=True,
        ax=ax,
    )
    ax.set_title(title)
    ax.axis("off")
    return fig

# ------------------------------------------------------------
# 📐  Layout de la página
# ------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Grafo completo")
    st.pyplot(draw_graph(G, layout_option))

with col_right:
    st.subheader("Resumen")
    st.metric("Vértices", len(G.nodes))
    st.metric("Aristas", len(G.edges))
    st.metric(f"Cliques mono K{k_mono}", len(mono_cliques), delta="✔️" if mono_cliques else "✖️")
    st.metric(f"Cliques arcoíris K{k_rain}", len(rain_cliques), delta="✔️" if rain_cliques else "✖️")

    # --- Conocimiento teórico: Ramsey clásico 2 colores ---
    st.markdown("---")
    st.caption("Conocimiento teórico: números de Ramsey (2 colores)")
    info = ramsey_lookup(k_mono, k_mono)
    if info is None:
        st.info(
            "Valores exactos sólo se conocen para casos pequeños. "
            "Para k mayores mostramos resultados **experimentales** de tu grafo.")
    elif info["type"] == "exact":
        R = info["value"]
        st.success(f"R({k_mono},{k_mono}) = {R} (exacto conocido)")
        if n_vertices >= R:
            st.write(
                f"Con n = {n_vertices} ≥ R, **cualquier 2-coloración** de K_n contiene "
                f"un K_{k_mono} monocromático (teorema de Ramsey).")
        else:
            st.warning(
                f"Con n = {n_vertices} < R, no hay garantía teórica en 2 colores; "
                "esta app explora ejemplos aleatorios.")
    else:  # bounds
        st.warning(
            f"Para R({k_mono},{k_mono}) sólo se conocen cotas: "
            f"{info['lower']} ≤ R ≤ {info['upper']}.")
        st.caption("Tu simulación aporta intuición pero no prueba el valor exacto.")

# --- Resultados del modo exploración ---
if 'explore_payload' not in st.session_state:
    st.session_state['explore_payload'] = None

if 'go_explore' not in st.session_state:
    st.session_state['go_explore'] = False

# Disparador desde la barra lateral
try:
    if explore_on and go_explore:
        st.session_state['go_explore'] = True
        st.session_state['explore_payload'] = dict(target=target, k=k_target, n_min=n_min, n_max=n_max, trials=trials, threshold=threshold)
except NameError:
    pass

if st.session_state['go_explore'] and st.session_state['explore_payload'] is not None:
    st.divider()
    st.subheader("Modo exploración (experimental)")
    p = edge_prob
    colors = num_colors
    params = st.session_state['explore_payload']
    n_vals, succ, probs = run_exploration(params['target'], params['k'], params['n_min'], params['n_max'], params['trials'], p, colors)

    # Gráfica
    fig = plt.figure(figsize=(6,3.5))
    plt.plot(n_vals, probs, marker='o')
    plt.axhline(params['threshold'], linestyle='--')
    plt.ylim(0,1)
    plt.xlabel('n')
    plt.ylabel('Probabilidad de éxito')
    plt.title(f"Patrón: {params['target']}  |  k={params['k']}  |  p={p}")
    st.pyplot(fig)

    # Primer n que cruza el umbral
    n_star = next((n for n,pr in zip(n_vals, probs) if pr >= params['threshold']), None)
    if n_star is not None:
        st.success(f"Primer n con probabilidad ≥ {params['threshold']:.0%}: **n = {n_star}**")
    else:
        st.warning("Ningún n del rango alcanzó el umbral seleccionado.")

    # Descargar CSV
    csv_lines = ["n,successes,trials,prob"]
    for n, s, pr in zip(n_vals, succ, probs):
        csv_lines.append(f"{n},{s},{params['trials']},{pr:.6f}")
    csv = "\n".join(csv_lines)

    st.download_button(
        "⬇️ Descargar resultados (CSV)",
        data=csv.encode("utf-8"),
        file_name="exploracion_ramsey.csv",
        mime="text/csv",
    )

    # Consumimos el disparador para evitar repetir automáticamente
    st.session_state['go_explore'] = False

# Subgrafos
if mono_cliques or rain_cliques:
    st.divider()
    st.subheader("Subgrafos destacados")
    tabs = st.tabs([f"Monocromáticos ({len(mono_cliques)})", f"Arcoíris ({len(rain_cliques)})"])

    with tabs[0]:
        if mono_cliques:
            cols = st.columns(min(len(mono_cliques), 4))
            for i, verts in enumerate(mono_cliques):
                with cols[i % 4]:
                    st.pyplot(draw_subgraph(G, verts, f"Mono-K{k_mono}: {verts}"))
        else:
            st.info("No se encontraron cliques monocromáticos.")

    with tabs[1]:
        if rain_cliques:
            cols = st.columns(min(len(rain_cliques), 4))
            for i, verts in enumerate(rain_cliques):
                with cols[i % 4]:
                    st.pyplot(draw_subgraph(G, verts, f"Rainbow-K{k_rain}: {verts}"))
        else:
            st.info("No se encontraron cliques arcoíris.")
else:
    st.warning("No se encontraron subgrafos que cumplan los criterios seleccionados.")

# ------------------------------------------------------------
# ℹ️  Nota al pie y material extra
# ------------------------------------------------------------
with st.expander("Detalles de implementación"):
    st.write(
        "El grafo se genera con un modelo Erdős–Rényi \(G(n,p)\). "
        "Para cada conjunto de \(k\) vértices se comprueba si forman un clique y si "
        "cumplen la propiedad monocromática o arcoíris."
    )

with st.expander("Conceptos clave ⋯", expanded=False):
    st.markdown(
        """
        ### ¿Qué es un grafo?
        Un grafo es un par \(G = (V,E)\) formado por un conjunto de **vértices** \(V\)
        y un conjunto de **aristas** \(E\) que unen pares de vértices. En este proyecto
        trabajamos con **grafos simples** (sin lazos ni aristas múltiples) donde, además,
        cada arista recibe un **color** entero de `0` a `num_colors−1`.

        ### Clique
        Un **clique** \(K_k\) es un subconjunto de \(k\) vértices donde **todas** las
        aristas posibles entre ellos están presentes. Es decir, forman un subgrafo
        completo.

        - *Clique monocromático*: todas esas aristas comparten **el mismo color**.
        - *Clique arcoíris*: cada arista tiene un **color distinto**.

        ### (Rainbow) Ramsey numbers
        El número de Ramsey clásico \(R(s,t)\) es el mínimo \(n\) tal que **cualquier**
        coloreo rojo/azul de las aristas de un \(K_n\) contiene un clique rojo de tamaño
        \(s\) **o** un clique azul de tamaño \(t\).  
        En la variante **arcoíris** (rainbow Ramsey), se pregunta por el mínimo \(n\)
        para garantizar un clique arcoíris \(K_k\) o un clique monocromático \(K_k\)
        cuando las aristas se colorean con varios colores.

        > En esta app no calculamos ese número de forma teórica (lo cual es muy difícil),
        > sino que **exploramos experimentalmente**: generamos muchos grafos aleatorios y
        buscamos cliques que cumplan alguna de las dos condiciones.

        ---
        **Para saber más**: consulta [Graham, Rothschild & Spencer, *Ramsey Theory*]
        o la reciente survey de rainbow Ramsey numbers de J. Fox.
        """
    )
