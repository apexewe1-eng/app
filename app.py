# app.py
import random
import time
import statistics
import streamlit as st

from solvers import (
    GOAL,
    manhattan,
    a_star_baseline_metrics,
    NFSACOSolver,
    apply_move,
)

# ---------- Helpers ----------

def make_solvable_shuffle(steps: int):
    s = GOAL
    for _ in range(steps):
        moves = []
        for d in ["Up", "Down", "Left", "Right"]:
            ns = apply_move(s, d)
            if ns != s:
                moves.append(d)
        s = apply_move(s, random.choice(moves))
    return s

def moved_tile_pos(prev_state, curr_state):
    if prev_state == curr_state:
        return None
    bp = prev_state.index(0)
    bc = curr_state.index(0)
    if bp == bc:
        return None
    return bp  # moved tile ends at previous blank position

def board_css(tile_px, font_px, gap_px=12):
    cls = f"puz_{tile_px}_{font_px}"
    css = f"""
    <style>
    .{cls} {{
        display: grid;
        grid-template-columns: repeat(3, {tile_px}px);
        gap: {gap_px}px;
        padding: {gap_px}px;
        background: #050914;
        border-radius: 22px;
        width: fit-content;
        border: 2px solid #111827;
    }}
    .{cls} .cell {{
        width: {tile_px}px;
        height: {tile_px}px;
        border-radius: 16px;
        background: #0b1220;
        border: 3px solid #334155;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: {font_px}px;
        font-weight: 800;
        color: #e5e7eb;
        user-select: none;
        box-sizing: border-box;
    }}
    .{cls} .blank {{
        border-color: #1f2937;
        color: transparent;
    }}
    .{cls} .moved {{
        border-color: #22c55e !important;
        box-shadow: 0 0 0 3px rgba(34,197,94,.25),
                    0 0 24px rgba(34,197,94,.25);
    }}
    </style>
    """
    return cls, css

def render_board(state, tile_px, font_px, highlight=None):
    cls, css = board_css(tile_px, font_px)
    html = css + f'<div class="{cls}">'
    for i, v in enumerate(state):
        extra = " moved" if (highlight == i and v != 0) else ""
        if v == 0:
            html += f'<div class="cell blank{extra}">.</div>'
        else:
            html += f'<div class="cell{extra}">{v}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def winner_text(metric, a_val, n_val, lower_is_better=True):
    if lower_is_better:
        if a_val < n_val:
            return f"🏁 **{metric} winner: A*** (A*={a_val} < NFS-ACO={n_val})"
        if n_val < a_val:
            return f"🏁 **{metric} winner: NFS-ACO** (NFS-ACO={n_val} < A*={a_val})"
        return f"🏁 **{metric}: Tie** (A*={a_val}, NFS-ACO={n_val})"
    else:
        if a_val > n_val:
            return f"🏁 **{metric} winner: A*** (A*={a_val} > NFS-ACO={n_val})"
        if n_val > a_val:
            return f"🏁 **{metric} winner: NFS-ACO** (NFS-ACO={n_val} > A*={a_val})"
        return f"🏁 **{metric}: Tie** (A*={a_val}, NFS-ACO={n_val})"

def safe_mean_std(vals):
    if not vals:
        return 0.0, 0.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    return statistics.mean(vals), statistics.stdev(vals)

def make_frontier_chart_data(a, nfs):
    # returns list of dict rows for st.line_chart
    L = max(len(a["frontier_sizes"]), len(nfs["frontier_sizes"]))
    rows = []
    for i in range(L):
        a_y = a["frontier_sizes"][i] if i < len(a["frontier_sizes"]) else a["frontier_sizes"][-1]
        n_y = nfs["frontier_sizes"][i] if i < len(nfs["frontier_sizes"]) else nfs["frontier_sizes"][-1]
        rows.append({"expansion_index": i, "A*": a_y, "NFS-ACO": n_y})
    return rows

# ---------- UI ----------

st.set_page_config(page_title="Ganesh Pokharel - 34138027", layout="wide")
st.markdown("# Ganesh Pokharel - 34138027")
st.markdown("### **Efficiency Analysis of A* and NFS-ACO search Algorithm**")
st.divider()

# Controls
c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    shuffle_steps = st.slider("Shuffle", 5, 200, 60)
with c2:
    anim_speed = st.slider("Speed (sec/move)", 0.10, 1.50, 0.60, 0.05)
with c3:
    tile_px = st.slider("Tile size", 60, 140, 90, 5)
with c4:
    font_px = st.slider("Font size", 24, 90, 44, 2)
with c5:
    st.write("")
    gen_btn = st.button("Generate new puzzle")
    solve_btn = st.button("Solve")

# Session state
if "start" not in st.session_state:
    st.session_state.start = make_solvable_shuffle(60)
if "a" not in st.session_state:
    st.session_state.a = None
if "nfs" not in st.session_state:
    st.session_state.nfs = None
if "step" not in st.session_state:
    st.session_state.step = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

# Actions
if gen_btn:
    st.session_state.start = make_solvable_shuffle(shuffle_steps)
    st.session_state.a = None
    st.session_state.nfs = None
    st.session_state.step = 0
    st.session_state.playing = False
    st.rerun()

if solve_btn:
    s = st.session_state.start
    a = a_star_baseline_metrics(s, GOAL)
    nfs = NFSACOSolver(GOAL, manhattan(s, GOAL)).solve_metrics(s)
    st.session_state.a = a
    st.session_state.nfs = nfs
    st.session_state.step = 0
    st.session_state.playing = False
    st.rerun()

# Playback controls
if st.session_state.a and st.session_state.nfs:
    st.divider()
    a_states = st.session_state.a["states"]
    n_states = st.session_state.nfs["states"]
    max_steps = max(len(a_states), len(n_states)) - 1

    p1, p2, p3, p4 = st.columns([1, 1, 1, 3])
    with p1:
        if st.button("⏮ Prev"):
            st.session_state.playing = False
            st.session_state.step = clamp(st.session_state.step - 1, 0, max_steps)
            st.rerun()
    with p2:
        if st.button("⏭ Next"):
            st.session_state.playing = False
            st.session_state.step = clamp(st.session_state.step + 1, 0, max_steps)
            st.rerun()
    with p3:
        if st.session_state.playing:
            if st.button("⏸ Pause"):
                st.session_state.playing = False
                st.rerun()
        else:
            if st.button("▶ Play"):
                st.session_state.playing = True
                st.rerun()
    with p4:
        idx = st.slider("Move", 0, max_steps, st.session_state.step)
        if idx != st.session_state.step:
            st.session_state.playing = False
            st.session_state.step = idx
            st.rerun()

    # autoplay (no autorefresh)
    if st.session_state.playing:
        if st.session_state.step < max_steps:
            time.sleep(anim_speed)
            st.session_state.step += 1
            st.rerun()
        else:
            st.session_state.playing = False

# Boards
colA, colN = st.columns(2, gap="large")
with colA:
    st.subheader("A*")
    a_board = st.empty()
    a_info = st.empty()
with colN:
    st.subheader("NFS-ACO")
    n_board = st.empty()
    n_info = st.empty()

start = st.session_state.start

if not st.session_state.a:
    with a_board:
        render_board(start, tile_px, font_px)
    with n_board:
        render_board(start, tile_px, font_px)
    a_info.info("Click **Solve** to compute paths, winners, frontier growth and experiments.")
    n_info.info("Click **Solve** to compute paths, winners, frontier growth and experiments.")
else:
    a = st.session_state.a
    nfs = st.session_state.nfs
    i = st.session_state.step

    a_states = a["states"]
    n_states = nfs["states"]
    ai = min(i, len(a_states) - 1)
    ni = min(i, len(n_states) - 1)

    a_hi = moved_tile_pos(a_states[ai - 1], a_states[ai]) if ai > 0 else None
    n_hi = moved_tile_pos(n_states[ni - 1], n_states[ni]) if ni > 0 else None

    with a_board:
        render_board(a_states[ai], tile_px, font_px, a_hi)
    with n_board:
        render_board(n_states[ni], tile_px, font_px, n_hi)

    a_info.success(
        f"Steps: {a['steps']} | Time: {a['time_ms']:.3f} ms | "
        f"Nodes: {a['nodes_expanded']} | Max frontier: {a['max_frontier']} | "
        f"Visited: {a['visited_count']}"
    )
    n_info.success(
        f"Steps: {nfs['steps']} | Time: {nfs['time_ms']:.3f} ms | "
        f"Nodes: {nfs['nodes_expanded']} | Max frontier: {nfs['max_frontier']} | "
        f"Visited: {nfs['visited_count']} | "
        f"Thermal: {nfs.get('thermal_limit', 0):.2f} | Evap: {nfs.get('evaporation', 0):.3f}"
    )

    # Winner banners
    st.divider()
    st.subheader("🏁 Winners (per metric)")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.success(winner_text("Time (ms)", round(a["time_ms"], 3), round(nfs["time_ms"], 3), True))
    with w2:
        st.success(winner_text("Nodes expanded", a["nodes_expanded"], nfs["nodes_expanded"], True))
    with w3:
        st.success(winner_text("Max frontier (space)", a["max_frontier"], nfs["max_frontier"], True))

    # Frontier growth chart
    st.divider()
    st.subheader("🔥 Frontier growth (frontier size as search progresses)")
    chart_rows = make_frontier_chart_data(a, nfs)
    st.line_chart(chart_rows, x="expansion_index", y=["A*", "NFS-ACO"])

# ---------- Multi-run experiment mode ----------
st.divider()
st.subheader("📊 Multi-run average plots (research mode)")

e1, e2, e3 = st.columns([1.3, 2.2, 1.3])
with e1:
    trials = st.slider("Trials per difficulty", 3, 50, 10)
with e2:
    depths = st.multiselect(
        "Shuffle difficulties",
        [20, 40, 60, 80, 100, 120, 150, 180, 200],
        default=[40, 60, 100, 150]
    )
with e3:
    run_exp = st.button("Run experiment")

if run_exp and depths:
    rows = []

    for d in depths:
        a_times, a_nodes, a_space = [], [], []
        n_times, n_nodes, n_space = [], [], []

        for _ in range(trials):
            s = make_solvable_shuffle(d)
            a = a_star_baseline_metrics(s, GOAL)
            nfs = NFSACOSolver(GOAL, manhattan(s, GOAL)).solve_metrics(s)

            a_times.append(a["time_ms"])
            a_nodes.append(a["nodes_expanded"])
            a_space.append(a["max_frontier"])

            n_times.append(nfs["time_ms"])
            n_nodes.append(nfs["nodes_expanded"])
            n_space.append(nfs["max_frontier"])

        a_tm, a_ts = safe_mean_std(a_times)
        a_nm, a_ns = safe_mean_std(a_nodes)
        a_sm, a_ss = safe_mean_std(a_space)

        n_tm, n_ts = safe_mean_std(n_times)
        n_nm, n_ns = safe_mean_std(n_nodes)
        n_sm, n_ss = safe_mean_std(n_space)

        rows.append({
            "shuffle": d,
            "A*_time_mean(ms)": a_tm, "A*_time_std": a_ts,
            "A*_nodes_mean": a_nm, "A*_nodes_std": a_ns,
            "A*_space_mean(max_frontier)": a_sm, "A*_space_std": a_ss,
            "NFS_time_mean(ms)": n_tm, "NFS_time_std": n_ts,
            "NFS_nodes_mean": n_nm, "NFS_nodes_std": n_ns,
            "NFS_space_mean(max_frontier)": n_sm, "NFS_space_std": n_ss,
        })

    st.write("**Experiment summary (mean ± std)**")
    st.dataframe(rows, use_container_width=True)

    # Charts (no matplotlib)
    st.subheader("Average time vs difficulty")
    st.line_chart(rows, x="shuffle", y=["A*_time_mean(ms)", "NFS_time_mean(ms)"])

    st.subheader("Average nodes expanded vs difficulty")
    st.line_chart(rows, x="shuffle", y=["A*_nodes_mean", "NFS_nodes_mean"])

    st.subheader("Average space (max frontier) vs difficulty")
    st.line_chart(rows, x="shuffle", y=["A*_space_mean(max_frontier)", "NFS_space_mean(max_frontier)"])
