import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

N = 4
SIZE = N * N
EMPTY = 0
SOLVED_BONUS = 10_000


# ----------------------------
# Board helpers
# ----------------------------
def goal_board() -> List[int]:
    return list(range(1, SIZE)) + [EMPTY]


def is_solved(board: List[int]) -> bool:
    return board == goal_board()


def idx_to_rc(i: int) -> Tuple[int, int]:
    return i // N, i % N


def rc_to_idx(r: int, c: int) -> int:
    return r * N + c


def find_empty(board: List[int]) -> int:
    return board.index(EMPTY)


def neighbors_of_empty(empty_idx: int) -> List[int]:
    r, c = idx_to_rc(empty_idx)
    out = []
    if r > 0:
        out.append(rc_to_idx(r - 1, c))
    if r < N - 1:
        out.append(rc_to_idx(r + 1, c))
    if c > 0:
        out.append(rc_to_idx(r, c - 1))
    if c < N - 1:
        out.append(rc_to_idx(r, c + 1))
    return out


def swap(board: List[int], i: int, j: int) -> List[int]:
    b = board[:]
    b[i], b[j] = b[j], b[i]
    return b


def successors(board: List[int]) -> List[Tuple[int, List[int]]]:
    """Return list of (tile_index_to_move, next_board)."""
    e = find_empty(board)
    return [(t_idx, swap(board, e, t_idx)) for t_idx in neighbors_of_empty(e)]


# ----------------------------
# Solvability (15-puzzle parity)
# ----------------------------
def inversion_count(arr: List[int]) -> int:
    a = [x for x in arr if x != EMPTY]
    inv = 0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] > a[j]:
                inv += 1
    return inv


def is_solvable(board: List[int]) -> bool:
    # For even width (4):
    # solvable iff (blankRowFromBottom is even) XOR (inversions is even)
    inv_even = inversion_count(board) % 2 == 0
    empty_idx = find_empty(board)
    r, _ = idx_to_rc(empty_idx)
    blank_row_from_bottom = N - r  # 1..N
    blank_even = blank_row_from_bottom % 2 == 0
    return blank_even != inv_even


def shuffled_solvable_board() -> List[int]:
    base = goal_board()
    for _ in range(5000):
        b = base[:]
        random.shuffle(b)
        if b != base and is_solvable(b):
            return b
    # fallback: random legal moves from goal (always solvable)
    b = base
    for _ in range(80):
        _, b = random.choice(successors(b))
    return b


# ----------------------------
# Heuristic + utility
# ----------------------------
def manhattan(board: List[int]) -> int:
    s = 0
    for i, t in enumerate(board):
        if t == EMPTY:
            continue
        goal_idx = t - 1
        r1, c1 = idx_to_rc(i)
        r2, c2 = idx_to_rc(goal_idx)
        s += abs(r1 - r2) + abs(c1 - c2)
    return s


def misplaced(board: List[int]) -> int:
    m = 0
    for i, t in enumerate(board):
        if t == EMPTY:
            continue
        if t != i + 1:
            m += 1
    return m


def utility(board: List[int], depth_remaining: int) -> float:
    # Higher is better for MAX
    if is_solved(board):
        return SOLVED_BONUS + depth_remaining
    # negate distance so MAX wants larger utility
    return -(manhattan(board) + 0.25 * misplaced(board))


# ----------------------------
# Expectiminimax (MAX / CHANCE / MIN)
# Gremlin (MIN) picks a move that hurts MAX.
# CHANCE: with prob glitchProb -> MIN acts, else nothing happens.
# ----------------------------
@dataclass
class EIParams:
    glitch_prob: float


def expectiminimax(board: List[int], depth: int, node_type: str, p: EIParams) -> float:
    if depth == 0 or is_solved(board):
        return utility(board, depth)

    if node_type == "MAX":
        best = -float("inf")
        for _, nb in successors(board):
            v = expectiminimax(nb, depth - 1, "CHANCE", p)
            best = max(best, v)
        return best

    if node_type == "MIN":
        best = float("inf")
        for _, nb in successors(board):
            v = expectiminimax(nb, depth - 1, "MAX", p)
            best = min(best, v)
        return best

    # CHANCE
    v_no = expectiminimax(board, depth - 1, "MAX", p)
    v_glitch = expectiminimax(board, depth - 1, "MIN", p)
    return (1 - p.glitch_prob) * v_no + p.glitch_prob * v_glitch


def best_move(board: List[int], depth: int, p: EIParams) -> Tuple[Optional[int], float]:
    best_score = -float("inf")
    best_idx = None
    for t_idx, nb in successors(board):
        score = expectiminimax(nb, depth - 1, "CHANCE", p)
        if score > best_score:
            best_score = score
            best_idx = t_idx
    return best_idx, best_score


# ----------------------------
# Image slicing
# ----------------------------
def slice_image(img: Image.Image, tile_px: int = 96) -> List[Image.Image]:
    # Center-crop to square then resize to NxN tiles
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((tile_px * N, tile_px * N))

    tiles = []
    for r in range(N):
        for c in range(N):
            tile = img.crop((c * tile_px, r * tile_px, (c + 1) * tile_px, (r + 1) * tile_px))
            tiles.append(tile)
    return tiles  # index corresponds to goal position (tile number t -> tiles[t-1])


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="4×4 Puzzle + Expectiminimax", layout="centered")

st.title("4×4 Sliding Puzzle (Image Upload) + Expectiminimax Hints")

with st.sidebar:
    st.header("Settings")
    glitch_prob = st.slider("Glitch probability (chance node)", 0.0, 0.6, 0.2, 0.05)
    depth = st.slider("Hint search depth", 1, 7, 4, 1)
    tile_px = st.slider("Tile size (px)", 64, 140, 96, 8)

    st.caption(
        "Gameplay twist: after your move, with some probability a 'gremlin' (MIN) makes one move that hurts you. "
        "Hint uses expectiminimax: MAX → CHANCE → MIN."
    )

# session state init
if "board" not in st.session_state:
    st.session_state.board = shuffled_solvable_board()
if "moves" not in st.session_state:
    st.session_state.moves = 0
if "tiles" not in st.session_state:
    st.session_state.tiles = None  # list of PIL tiles or None

uploaded = st.file_uploader("Upload an image (optional)", type=["png", "jpg", "jpeg", "webp"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.session_state.tiles = slice_image(img, tile_px=tile_px)
    st.image(img, caption="Uploaded image (used as texture)", use_container_width=True)

colA, colB, colC = st.columns(3)
with colA:
    if st.button("Shuffle"):
        st.session_state.board = shuffled_solvable_board()
        st.session_state.moves = 0
with colB:
    if st.button("Reset to Goal"):
        st.session_state.board = goal_board()
        st.session_state.moves = 0
with colC:
    if st.button("Hint (Expectiminimax)"):
        t_idx, score = best_move(st.session_state.board, depth, EIParams(glitch_prob))
        st.session_state.hint_idx = t_idx
        st.toast(f"Hint: move tile at index {t_idx} (score {score:.2f})")

hint_idx = st.session_state.get("hint_idx", None)

board = st.session_state.board
moves = st.session_state.moves

st.subheader(f"Moves: {moves}")
if is_solved(board):
    st.success("Solved! 🎉")
else:
    st.info(f"Manhattan distance: {manhattan(board)}")

# Render grid:
# Each cell shows the tile image/number and a Move button if it can slide into empty.
empty_idx = find_empty(board)
legal = set(neighbors_of_empty(empty_idx))

# Optional: show the recommended tile highlight
def cell_border(i: int) -> str:
    if hint_idx is not None and i == hint_idx:
        return "border: 4px solid #2ea44f; border-radius: 12px; padding: 6px;"
    return "border: 1px solid #ddd; border-radius: 12px; padding: 6px;"

# UI grid
for r in range(N):
    cols = st.columns(N)
    for c in range(N):
        i = rc_to_idx(r, c)
        t = board[i]
        with cols[c]:
            st.markdown(f"<div style='{cell_border(i)}'>", unsafe_allow_html=True)

            if t == EMPTY:
                st.markdown(
                    f"<div style='height:{tile_px}px; display:flex; align-items:center; justify-content:center; "
                    f"background:#fafafa; border-radius:12px; color:#777;'>empty</div>",
                    unsafe_allow_html=True,
                )
            else:
                if st.session_state.tiles is not None:
                    st.image(st.session_state.tiles[t - 1], use_container_width=True)
                else:
                    st.markdown(
                        f"<div style='height:{tile_px}px; display:flex; align-items:center; justify-content:center; "
                        f"font-size:28px; font-weight:700; background:#f3f3f3; border-radius:12px;'>{t}</div>",
                        unsafe_allow_html=True,
                    )

            can_move = (i in legal) and (not is_solved(board))
            if st.button("Move", key=f"move_{i}", disabled=not can_move):
                # Player move: swap tile i with empty
                next_board = swap(board, i, empty_idx)
                st.session_state.board = next_board
                st.session_state.moves += 1
                st.session_state.hint_idx = None  # clear hint after move

                # CHANCE event: glitch -> gremlin makes one bad move
                if (not is_solved(next_board)) and random.random() < glitch_prob:
                    succ = successors(next_board)
                    # Gremlin chooses successor with lowest utility for MAX
                    worst_u = float("inf")
                    pick = next_board
                    for _, nb in succ:
                        u = utility(nb, 0)
                        if u < worst_u:
                            worst_u = u
                            pick = nb
                    st.session_state.board = pick
                    st.toast("⚠️ Glitch! Gremlin moved one tile.", icon="⚠️")

                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

st.caption(
    "Tip: If you set glitch probability to 0%, it becomes a normal deterministic 15-puzzle. "
    "With glitches on, the hint is genuinely 'best in expectation' (expectiminimax)."
)
