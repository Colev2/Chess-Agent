#!/usr/bin/env python3

# ============================================================
# eval_checkpoints.py
#
# External evaluator for the current/latest archived RL checkpoint.
#
# Main goal:
#   Periodically answer whether the current agent is improving.
#
# Default match plan with --step 5:
#   latest model_iter_N      vs model_after_sl
#   latest model_iter_N      vs model_iter_(N-5), if available
#
# Evaluation style:
#   - deterministic MCTS: no Dirichlet noise, no temperature
#   - many starting FENs; by default the script generates enough legal
#     deterministic opening-like FENs so each FEN is used once with each color
#   - batched/vectorized game evaluation in one process, with both models
#     loaded once on the same device
#   - batched NN inference inside each MCTS simulation across active games
#
# Example quick check:
#   python -u eval_checkpoints.py --games 100 --sims 100 --step 5 --no-ordo
#
# Example stronger check:
#   python -u eval_checkpoints.py --games 500 --sims 300 --step 5 --batch-games 64 --no-ordo
#
# Example with curated FENs:
#   python -u eval_checkpoints.py --games 500 --sims 300 --fen-file eval_fens.txt
#
# FEN file format:
#   Either plain FEN per line, or:
#       Name | fen-string
#   Lines beginning with # are ignored.
# ============================================================

from __future__ import annotations

import argparse
import csv
import math
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import chess
import chess.pgn
import torch
import torch.nn.functional as F

import chess_agent


# ============================================================
# Imports from main project file
# ============================================================

ChessNet = chess_agent.ChessNet
board_to_tensor = chess_agent.board_to_tensor
NUM_MOVES = getattr(chess_agent, "NUM_MOVES", 4672)

OUTPUT_DIR = getattr(chess_agent, "OUTPUT_DIR", Path("outputs"))
RL_MODEL_DIR = getattr(
    chess_agent,
    "RL_MODEL_DIR",
    OUTPUT_DIR / "rl_models",
)

RL_CHECKPOINT_DIR = getattr(
    chess_agent,
    "RL_CHECKPOINT_DIR",
    OUTPUT_DIR / "rl_checkpoints",
)

MODEL_AFTER_SL_PATH = getattr(
    chess_agent,
    "MODEL_AFTER_SL_PATH",
    RL_CHECKPOINT_DIR / "model_after_sl.pt",
)

SELFPLAY_C_PUCT = getattr(chess_agent, "SELFPLAY_C_PUCT", 1.5)
MAX_GAME_MOVES = getattr(chess_agent, "MAX_GAME_MOVES", 250)


# ============================================================
# Evaluation output paths
# ============================================================

EVAL_DIR = OUTPUT_DIR / "eval"
EVAL_RESULTS_CSV = EVAL_DIR / "eval_latest_results.csv"
EVAL_GAMES_PGN = EVAL_DIR / "eval_latest_games.pgn"
EVAL_ORDO_RATINGS_TXT = EVAL_DIR / "ordo_latest_ratings.txt"


# ============================================================
# Base opening / starting positions
# ============================================================

BASE_STARTING_POSITIONS = [
    (
        "Ruy Lopez",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    ),
    (
        "Italian Game",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    ),
    (
        "Sicilian Najdorf Structure",
        "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6",
    ),
    (
        "French Advance",
        "rnbqkbnr/ppp2ppp/4p3/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    ),
    (
        "Caro-Kann Advance",
        "rnbqkbnr/pp2pppp/2p5/3pP3/3P4/8/PPP2PPP/RNBQKBNR b KQkq - 0 3",
    ),
    (
        "Queen's Gambit Declined",
        "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    ),
    (
        "King's Indian Defense",
        "rnbqk2r/ppppppbp/5np1/8/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 2 4",
    ),
    (
        "English Symmetrical",
        "r1bqkb1r/pp1ppppp/2n2n2/2p5/2P5/2N2N2/PP1PPPPP/R1BQKB1R w KQkq - 4 4",
    ),
    (
        "London System",
        "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/5N2/PPP1PPPP/RN1QKB1R b KQkq - 3 3",
    ),
    (
        "Slav Defense",
        "rnbqkb1r/pp2pppp/2p2n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 2 4",
    ),
]


# ============================================================
# Data structures
# ============================================================

@dataclass(frozen=True)
class EvalCheckpoint:
    name: str
    path: Path
    iteration: Optional[int] = None


class EvalMCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[chess.Move, "EvalMCTSNode"] = {}

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def expand(self, priors: dict[chess.Move, float]) -> None:
        self.children = {
            move: EvalMCTSNode(prior)
            for move, prior in priors.items()
        }


@dataclass
class ActiveEvalGame:
    game_idx: int
    board: chess.Board
    played_moves: list[chess.Move]
    checkpoint_is_white: bool
    white_name: str
    black_name: str
    opening_name: Optional[str]
    initial_fen: Optional[str]
    done: bool = False
    result: Optional[str] = None
    termination: Optional[str] = None


# ============================================================
# Checkpoint loading
# ============================================================

def strip_module_prefix(state_dict: dict) -> dict:
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict

    return {
        k.removeprefix("module."): v
        for k, v in state_dict.items()
    }


def extract_model_state_dict(checkpoint_obj) -> dict:
    """
    Handles common checkpoint formats:
      - {"model": state_dict, ...}
      - {"model_state_dict": state_dict, ...}
      - {"state_dict": state_dict, ...}
      - raw state_dict
    """

    if isinstance(checkpoint_obj, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint_obj and isinstance(checkpoint_obj[key], dict):
                return strip_module_prefix(checkpoint_obj[key])

        if all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return strip_module_prefix(checkpoint_obj)

    raise ValueError("Could not find model state_dict inside checkpoint.")


def create_eval_model(device: torch.device):
    try:
        model = ChessNet()
    except TypeError as exc:
        raise TypeError(
            "ChessNet() could not be created with no arguments. "
            "Edit create_eval_model() in eval_checkpoints.py to match your "
            "ChessNet constructor."
        ) from exc

    model = model.to(device)
    model.eval()
    return model


def load_eval_model(checkpoint_path: Path, device: torch.device):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    obj = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = create_eval_model(device)
    state_dict = extract_model_state_dict(obj)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model


# ============================================================
# Move encoding / model output helpers
# ============================================================

def move_to_policy_index(board: chess.Board, move: chess.Move) -> int:
    return int(chess_agent.move_to_id(move, board))


def split_model_output(output):
    if isinstance(output, dict):
        policy = output.get("policy", None)
        value = output.get("value", None)

        if policy is None:
            policy = output.get("policy_logits", None)

        if policy is None or value is None:
            raise RuntimeError(
                "Model returned dict, but could not find policy/value keys."
            )

        return policy, value

    if isinstance(output, (tuple, list)) and len(output) == 2:
        a, b = output

        a_flat = a.view(a.shape[0], -1) if torch.is_tensor(a) and a.ndim >= 2 else a
        b_flat = b.view(b.shape[0], -1) if torch.is_tensor(b) and b.ndim >= 2 else b

        if torch.is_tensor(a_flat) and a_flat.shape[-1] == NUM_MOVES:
            return a, b

        if torch.is_tensor(b_flat) and b_flat.shape[-1] == NUM_MOVES:
            return b, a

        return a, b

    raise RuntimeError(
        "Model output format not understood. Expected tuple/list(policy, value) "
        "or dict with policy/value."
    )


# ============================================================
# Terminal values and batched NN evaluation
# ============================================================

def terminal_value_from_side_to_move(board: chess.Board) -> float:
    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        return 0.0

    if outcome.winner is None:
        return 0.0

    return 1.0 if outcome.winner == board.turn else -1.0


@torch.no_grad()
def eval_policy_value_batch(
    model,
    boards: list[chess.Board],
    device: torch.device,
) -> list[tuple[dict[chess.Move, float], float]]:
    """
    Batched NN evaluation for many leaf boards belonging to the same model.

    Returns one (legal-move priors, value) pair per board.
    Value is from the board's side-to-move perspective.
    """

    if len(boards) == 0:
        return []

    results: list[Optional[tuple[dict[chess.Move, float], float]]] = [None] * len(boards)
    eval_indices: list[int] = []
    tensors: list[np.ndarray] = []
    legal_indices_batch: list[list[int]] = []
    legal_moves_batch: list[list[chess.Move]] = []

    for i, board in enumerate(boards):
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            results[i] = ({}, terminal_value_from_side_to_move(board))
            continue

        legal_indices = [move_to_policy_index(board, move) for move in legal_moves]

        x = board_to_tensor(board)
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()

        eval_indices.append(i)
        tensors.append(np.asarray(x, dtype=np.float32))
        legal_indices_batch.append(legal_indices)
        legal_moves_batch.append(legal_moves)

    if len(eval_indices) == 0:
        return [r for r in results if r is not None]

    x_np = np.stack(tensors).astype(np.float32, copy=False)
    x = torch.from_numpy(x_np).to(device, non_blocking=True)

    use_cuda = device.type == "cuda"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
        output = model(x)
        policy_logits_batch, value_batch = split_model_output(output)

    policy_logits_batch = policy_logits_batch.detach().float()
    value_np = value_batch.detach().float().view(-1).cpu().numpy()

    for j, original_idx in enumerate(eval_indices):
        legal_indices = legal_indices_batch[j]
        legal_moves = legal_moves_batch[j]

        legal_idx_tensor = torch.tensor(
            legal_indices,
            dtype=torch.long,
            device=policy_logits_batch.device,
        )

        legal_logits = policy_logits_batch[j, legal_idx_tensor]
        legal_probs = F.softmax(legal_logits, dim=0)

        if torch.isnan(legal_probs).any() or torch.isinf(legal_probs).any():
            uniform = 1.0 / len(legal_moves)
            priors = {move: uniform for move in legal_moves}
        else:
            probs_cpu = legal_probs.detach().cpu().tolist()
            priors = {
                move: float(prob)
                for move, prob in zip(legal_moves, probs_cpu)
            }

        results[original_idx] = (priors, float(value_np[j]))

    final = []
    for r in results:
        if r is None:
            raise RuntimeError("Internal error: missing batched eval result.")
        final.append(r)

    return final


# ============================================================
# Batched eval-only MCTS
# ============================================================

def select_child(
    node: EvalMCTSNode,
    c_puct: float,
) -> tuple[chess.Move, EvalMCTSNode]:
    best_score = -1e30
    best_move = None
    best_child = None

    parent_sqrt_visits = math.sqrt(max(1, node.visit_count))

    for move, child in node.children.items():
        q = -child.value
        u = c_puct * child.prior * parent_sqrt_visits / (1 + child.visit_count)
        score = q + u

        if score > best_score:
            best_score = score
            best_move = move
            best_child = child

    if best_move is None or best_child is None:
        raise RuntimeError("select_child called on a node with no children.")

    return best_move, best_child


def backup_path(search_path: list[EvalMCTSNode], value: float) -> None:
    for path_node in reversed(search_path):
        path_node.visit_count += 1
        path_node.value_sum += value
        value = -value


def run_eval_mcts_batch(
    model,
    root_boards: list[chess.Board],
    device: torch.device,
    num_simulations: int,
    c_puct: float,
) -> list[chess.Move]:
    """
    Vectorized/batched eval MCTS for many independent root boards using one model.

    Important:
      - no Dirichlet noise
      - no temperature sampling
      - one leaf per root per simulation round
      - NN leaf evaluation is batched across all active roots
      - final move is highest visit count
    """

    if len(root_boards) == 0:
        return []

    roots = [EvalMCTSNode(prior=1.0) for _ in root_boards]

    # Fast exits for forced-move positions.
    forced_moves: dict[int, chess.Move] = {}
    active_root_indices: list[int] = []

    for i, board in enumerate(root_boards):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise RuntimeError("MCTS called on board with no legal moves.")
        if len(legal_moves) == 1:
            forced_moves[i] = legal_moves[0]
        else:
            active_root_indices.append(i)

    for _ in range(num_simulations):
        leaf_boards: list[chess.Board] = []
        leaf_paths: list[list[EvalMCTSNode]] = []

        # Selection for one leaf per active root.
        for root_idx in active_root_indices:
            board = root_boards[root_idx].copy(stack=False)
            node = roots[root_idx]
            search_path = [node]

            while node.expanded and not board.is_game_over(claim_draw=True):
                move, node = select_child(node, c_puct=c_puct)
                board.push(move)
                search_path.append(node)

            if board.is_game_over(claim_draw=True):
                value = terminal_value_from_side_to_move(board)
                backup_path(search_path, value)
            else:
                leaf_boards.append(board)
                leaf_paths.append(search_path)

        if leaf_boards:
            eval_results = eval_policy_value_batch(model, leaf_boards, device)

            for search_path, (priors, value) in zip(leaf_paths, eval_results):
                leaf_node = search_path[-1]
                if not leaf_node.expanded:
                    leaf_node.expand(priors)
                backup_path(search_path, value)

    chosen_moves: list[chess.Move] = []

    for i, board in enumerate(root_boards):
        if i in forced_moves:
            chosen_moves.append(forced_moves[i])
            continue

        root = roots[i]
        legal_moves = list(board.legal_moves)

        if not root.children:
            # Extremely defensive fallback; should not happen except num_simulations=0.
            chosen_moves.append(legal_moves[0])
            continue

        best_move, _ = max(
            root.children.items(),
            key=lambda item: (item[1].visit_count, item[1].prior),
        )
        chosen_moves.append(best_move)

    return chosen_moves


# ============================================================
# Playing games
# ============================================================

def result_from_board(board: chess.Board) -> tuple[str, str]:
    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        return "1/2-1/2", "MaxMoves"

    if outcome.winner is None:
        return "1/2-1/2", str(outcome.termination)

    if outcome.winner == chess.WHITE:
        return "1-0", str(outcome.termination)

    return "0-1", str(outcome.termination)


def score_for_checkpoint(result: str, checkpoint_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5

    if checkpoint_is_white:
        return 1.0 if result == "1-0" else 0.0

    return 1.0 if result == "0-1" else 0.0


def append_game_to_pgn(
    pgn_path: Path,
    white_name: str,
    black_name: str,
    result: str,
    moves: list[chess.Move],
    round_name: str,
    termination: str,
    num_simulations: int,
    initial_fen: Optional[str] = None,
    opening_name: Optional[str] = None,
) -> None:
    pgn_path.parent.mkdir(parents=True, exist_ok=True)

    board = chess.Board(initial_fen) if initial_fen else chess.Board()

    game = chess.pgn.Game()
    game.headers["Event"] = "ChessAgent latest checkpoint evaluation"
    game.headers["Site"] = "Aristotle/Nefeli"
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["Round"] = round_name
    game.headers["White"] = white_name
    game.headers["Black"] = black_name
    game.headers["Result"] = result
    game.headers["Termination"] = termination
    game.headers["EvalSims"] = str(num_simulations)

    if opening_name:
        game.headers["Opening"] = opening_name

    if initial_fen:
        game.headers["SetUp"] = "1"
        game.headers["FEN"] = initial_fen

    node = game

    for move in moves:
        if move not in board.legal_moves:
            raise RuntimeError(f"Illegal move while writing PGN: {move}")

        node = node.add_variation(move)
        board.push(move)

    with open(pgn_path, "a", encoding="utf-8") as f:
        print(game, file=f, end="\n\n")


# ============================================================
# Starting FEN suite
# ============================================================

def validate_fen(fen: str) -> bool:
    try:
        board = chess.Board(fen)
        return not board.is_game_over(claim_draw=True) and len(list(board.legal_moves)) > 0
    except Exception:
        return False


def read_fen_file(path: Path) -> list[tuple[str, str]]:
    positions: list[tuple[str, str]] = []
    seen = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_num, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line or line.startswith("#"):
                continue

            if "|" in line:
                name, fen = [part.strip() for part in line.split("|", 1)]
            else:
                name = f"FEN_{line_num}"
                fen = line

            if not validate_fen(fen):
                print(f"[eval] skipped invalid/non-playable FEN line {line_num}: {fen}", flush=True)
                continue

            key = chess.Board(fen).board_fen() + " " + ("w" if chess.Board(fen).turn else "b")
            if key in seen:
                continue
            seen.add(key)
            positions.append((name, fen))

    return positions


def opening_move_score(board: chess.Board, move: chess.Move) -> float:
    """
    Lightweight deterministic heuristic for generating legal, opening-like FENs.
    This is NOT used for evaluation play. It only builds a fixed diverse FEN suite.
    """

    piece = board.piece_at(move.from_square)
    if piece is None:
        return 0.0

    score = 1.0
    to_file = chess.square_file(move.to_square)
    to_rank = chess.square_rank(move.to_square)
    to_center_distance = abs(to_file - 3.5) + abs(to_rank - 3.5)

    score += max(0.0, 4.0 - to_center_distance) * 0.25

    if board.is_capture(move):
        score += 0.20

    if board.is_castling(move):
        score += 1.30

    if piece.piece_type == chess.PAWN:
        # Prefer central pawn moves in the opening.
        if chess.square_file(move.from_square) in (3, 4):
            score += 1.10
        elif chess.square_file(move.from_square) in (2, 5):
            score += 0.45
        else:
            score += 0.05

    elif piece.piece_type in (chess.KNIGHT, chess.BISHOP):
        # Prefer developing minor pieces from the back rank.
        from_rank = chess.square_rank(move.from_square)
        home_rank = 0 if piece.color == chess.WHITE else 7
        if from_rank == home_rank:
            score += 1.00
        else:
            score += 0.15

    elif piece.piece_type == chess.QUEEN:
        # Avoid too many early queen moves when generating opening FENs.
        score -= 0.60

    elif piece.piece_type == chess.KING and not board.is_castling(move):
        score -= 0.80

    return max(0.01, score)


def generate_opening_fens(
    target_count: int,
    seed: int,
    min_plies: int,
    max_plies: int,
) -> list[tuple[str, str]]:
    """
    Generates a deterministic legal FEN suite.

    This avoids repeating deterministic eval games when --games is large.
    For a more chess-theoretically curated suite, pass --fen-file later.
    """

    rng = random.Random(seed)
    positions: list[tuple[str, str]] = []
    seen = set()
    attempts = 0
    max_attempts = max(10_000, target_count * 200)

    while len(positions) < target_count and attempts < max_attempts:
        attempts += 1
        board = chess.Board()
        target_plies = rng.randint(min_plies, max_plies)

        for _ in range(target_plies):
            if board.is_game_over(claim_draw=True):
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            scored = []
            for move in legal_moves:
                score = opening_move_score(board, move)
                # Deterministic randomness from rng; helps generate a diverse suite.
                score *= rng.uniform(0.50, 1.50)
                scored.append((score, move))

            scored.sort(key=lambda item: item[0], reverse=True)
            top_k = min(len(scored), 8)
            top_scores = [max(0.01, s) for s, _ in scored[:top_k]]
            top_moves = [m for _, m in scored[:top_k]]
            move = rng.choices(top_moves, weights=top_scores, k=1)[0]
            board.push(move)

        if board.is_game_over(claim_draw=True) or not list(board.legal_moves):
            continue

        fen = board.fen()
        key = board.board_fen() + " " + ("w" if board.turn else "b")
        if key in seen:
            continue

        seen.add(key)
        positions.append((f"GeneratedOpening_{len(positions) + 1}", fen))

    if len(positions) < target_count:
        print(
            f"[eval] WARNING: generated only {len(positions)}/{target_count} FENs.",
            flush=True,
        )

    return positions


def build_starting_positions(args) -> list[tuple[Optional[str], Optional[str]]]:
    if args.no_starting_fens:
        return [(None, None)]

    needed_pairs = max(1, math.ceil(args.games / 2))

    positions: list[tuple[str, str]] = []
    seen = set()

    def add_position(name: str, fen: str) -> None:
        if not validate_fen(fen):
            return
        board = chess.Board(fen)
        key = board.board_fen() + " " + ("w" if board.turn else "b")
        if key in seen:
            return
        seen.add(key)
        positions.append((name, fen))

    if args.fen_file is not None:
        for name, fen in read_fen_file(Path(args.fen_file)):
            add_position(name, fen)
    else:
        for name, fen in BASE_STARTING_POSITIONS:
            add_position(name, fen)

    if args.auto_generate_fens and len(positions) < needed_pairs:
        generated = generate_opening_fens(
            target_count=needed_pairs - len(positions),
            seed=args.fen_seed,
            min_plies=args.generated_min_plies,
            max_plies=args.generated_max_plies,
        )
        for name, fen in generated:
            add_position(name, fen)

    if len(positions) == 0:
        print("[eval] WARNING: no valid starting FENs available; using normal start position.", flush=True)
        return [(None, None)]

    if len(positions) < needed_pairs:
        print(
            f"[eval] WARNING: only {len(positions)} unique FENs for {args.games} games/match. "
            f"Deterministic games may repeat after {2 * len(positions)} games/match.",
            flush=True,
        )
    else:
        print(
            f"[eval] using {needed_pairs} FEN pairs for {args.games} games/match "
            f"from {len(positions)} available positions.",
            flush=True,
        )

    return positions[:needed_pairs]


def select_starting_position(
    game_idx: int,
    starting_positions: list[tuple[Optional[str], Optional[str]]],
) -> tuple[Optional[str], Optional[str]]:
    if len(starting_positions) == 0:
        return None, None

    # Games are paired by FEN:
    #   game 0 -> FEN 0 with checkpoint as White
    #   game 1 -> FEN 0 with checkpoint as Black
    #   game 2 -> FEN 1 with checkpoint as White
    #   game 3 -> FEN 1 with checkpoint as Black
    fen_pair_idx = game_idx // 2
    return starting_positions[fen_pair_idx % len(starting_positions)]


# ============================================================
# CSV summary
# ============================================================

CSV_FIELDS = [
    "checkpoint",
    "opponent",
    "games",
    "wins",
    "draws",
    "losses",
    "score",
    "elo_delta",
    "sims",
    "step",
    "fen_count",
    "updated_at",
]


def elo_delta_from_score(score: float, eps: float = 1e-4) -> float:
    score = max(eps, min(1.0 - eps, score))
    return 400.0 * math.log10(score / (1.0 - score))


def load_existing_results(csv_path: Path) -> dict[tuple[str, str, int], dict]:
    rows = {}

    if not csv_path.exists():
        return rows

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            try:
                checkpoint = row["checkpoint"]
                opponent = row["opponent"]
                sims = int(row.get("sims", 0))
                rows[(checkpoint, opponent, sims)] = row
            except Exception:
                continue

    return rows


def write_results_csv(
    csv_path: Path,
    rows: dict[tuple[str, str, int], dict],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(
        rows.values(),
        key=lambda r: (
            r.get("checkpoint", ""),
            r.get("opponent", ""),
            int(r.get("sims", 0)),
        ),
    )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for row in sorted_rows:
            writer.writerow({
                field: row.get(field, "")
                for field in CSV_FIELDS
            })


def make_result_row(
    checkpoint_name: str,
    opponent_name: str,
    wins: int,
    draws: int,
    losses: int,
    sims: int,
    step: int,
    fen_count: int,
) -> dict:
    games = wins + draws + losses

    if games > 0:
        score = (wins + 0.5 * draws) / games
        elo_delta = elo_delta_from_score(score)
    else:
        score = 0.0
        elo_delta = 0.0

    return {
        "checkpoint": checkpoint_name,
        "opponent": opponent_name,
        "games": str(games),
        "wins": str(wins),
        "draws": str(draws),
        "losses": str(losses),
        "score": f"{score:.6f}",
        "elo_delta": f"{elo_delta:.2f}",
        "sims": str(sims),
        "step": str(step),
        "fen_count": str(fen_count),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


# ============================================================
# Finding checkpoints and building latest-only match graph
# ============================================================

def parse_iteration_from_path(path: Path) -> Optional[int]:
    match = re.match(r"model_iter_(\d+)\.pt$", path.name)

    if not match:
        return None

    return int(match.group(1))


def find_model_after_sl() -> Path:
    candidates = [
        Path(MODEL_AFTER_SL_PATH),
        Path(RL_MODEL_DIR) / "model_after_sl.pt",
        Path(OUTPUT_DIR) / "rl_models" / "model_after_sl.pt",

        # Legacy fallback from old layout.
        Path(RL_CHECKPOINT_DIR) / "model_after_sl.pt",
        Path(OUTPUT_DIR) / "rl_checkpoints" / "model_after_sl.pt",
    ]

    for path in candidates:
        if path.exists():
            return path

    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError("Could not find model_after_sl.pt. Checked:\n" f"{checked}")


def find_eval_checkpoints(step: int) -> tuple[EvalCheckpoint, list[EvalCheckpoint]]:
    baseline_path = find_model_after_sl()

    baseline = EvalCheckpoint(
        name="model_after_sl",
        path=baseline_path,
        iteration=None,
    )

    iter_paths = sorted(
        Path(RL_MODEL_DIR).glob("model_iter_*.pt"),
        key=lambda p: parse_iteration_from_path(p)
        if parse_iteration_from_path(p) is not None
        else -1,
    )

    checkpoints: list[EvalCheckpoint] = []

    for path in iter_paths:
        iteration = parse_iteration_from_path(path)

        if iteration is None:
            continue

        if step > 0 and iteration % step != 0:
            continue

        checkpoints.append(
            EvalCheckpoint(
                name=f"model_iter_{iteration}",
                path=path,
                iteration=iteration,
            )
        )

    return baseline, checkpoints


def build_latest_match_plan(
    baseline: EvalCheckpoint,
    checkpoints: list[EvalCheckpoint],
    step: int,
) -> list[tuple[EvalCheckpoint, EvalCheckpoint]]:
    """
    Returns only the monitoring matches the user asked for:

        latest model_iter_N vs model_after_sl
        latest model_iter_N vs model_iter_(N-step), if available

    N is the largest available archived checkpoint iteration divisible by step.
    """

    if not checkpoints:
        return []

    latest = max(
        checkpoints,
        key=lambda ckpt: -1 if ckpt.iteration is None else ckpt.iteration,
    )

    matches = [(latest, baseline)]

    if latest.iteration is not None:
        previous_iteration = latest.iteration - step
        previous = next(
            (ckpt for ckpt in checkpoints if ckpt.iteration == previous_iteration),
            None,
        )

        if previous is not None:
            matches.append((latest, previous))
        else:
            print(
                f"[eval] previous checkpoint model_iter_{previous_iteration}.pt not found; "
                f"latest-vs-previous match will be skipped.",
                flush=True,
            )

    return matches


# ============================================================
# Running batched matches
# ============================================================

def init_active_game(
    checkpoint: EvalCheckpoint,
    opponent: EvalCheckpoint,
    game_idx: int,
    starting_positions: list[tuple[Optional[str], Optional[str]]],
) -> ActiveEvalGame:
    opening_name, initial_fen = select_starting_position(game_idx, starting_positions)
    board = chess.Board(initial_fen) if initial_fen else chess.Board()

    checkpoint_is_white = (game_idx % 2 == 0)

    if checkpoint_is_white:
        white_name = checkpoint.name
        black_name = opponent.name
    else:
        white_name = opponent.name
        black_name = checkpoint.name

    return ActiveEvalGame(
        game_idx=game_idx,
        board=board,
        played_moves=[],
        checkpoint_is_white=checkpoint_is_white,
        white_name=white_name,
        black_name=black_name,
        opening_name=opening_name,
        initial_fen=initial_fen,
    )


def finalize_active_game(game: ActiveEvalGame) -> None:
    if game.done:
        return

    result, termination = result_from_board(game.board)
    game.result = result
    game.termination = termination
    game.done = True


def play_eval_game_batch(
    checkpoint: EvalCheckpoint,
    opponent: EvalCheckpoint,
    model_checkpoint,
    model_opponent,
    game_indices: list[int],
    starting_positions: list[tuple[Optional[str], Optional[str]]],
    num_simulations: int,
    c_puct: float,
    max_moves: int,
    device: torch.device,
) -> list[ActiveEvalGame]:
    games = [
        init_active_game(
            checkpoint=checkpoint,
            opponent=opponent,
            game_idx=game_idx,
            starting_positions=starting_positions,
        )
        for game_idx in game_indices
    ]

    for _ply in range(max_moves):
        active_games = [
            game for game in games
            if not game.done and not game.board.is_game_over(claim_draw=True)
        ]

        if not active_games:
            break

        checkpoint_games = []
        opponent_games = []

        for game in active_games:
            # If checkpoint is White, it moves on white turns; otherwise on black turns.
            checkpoint_to_move = (game.board.turn == chess.WHITE) == game.checkpoint_is_white

            if checkpoint_to_move:
                checkpoint_games.append(game)
            else:
                opponent_games.append(game)

        for group_games, model in ((checkpoint_games, model_checkpoint), (opponent_games, model_opponent)):
            if not group_games:
                continue

            boards = [game.board for game in group_games]
            moves = run_eval_mcts_batch(
                model=model,
                root_boards=boards,
                device=device,
                num_simulations=num_simulations,
                c_puct=c_puct,
            )

            for game, move in zip(group_games, moves):
                game.board.push(move)
                game.played_moves.append(move)

                if game.board.is_game_over(claim_draw=True):
                    finalize_active_game(game)

    for game in games:
        finalize_active_game(game)

    return games


def play_eval_match(
    checkpoint: EvalCheckpoint,
    opponent: EvalCheckpoint,
    existing_row: Optional[dict],
    target_games: int,
    num_simulations: int,
    c_puct: float,
    max_moves: int,
    device: torch.device,
    pgn_path: Path,
    step: int,
    starting_positions: list[tuple[Optional[str], Optional[str]]],
    batch_games: int,
) -> dict:
    """
    Plays missing games only.

    wins/draws/losses are always from `checkpoint` perspective.
    Games are evaluated in batches; within each move, boards are grouped by
    model and each model's MCTS leaf evaluations are batched.
    """

    wins = int(existing_row.get("wins", 0)) if existing_row else 0
    draws = int(existing_row.get("draws", 0)) if existing_row else 0
    losses = int(existing_row.get("losses", 0)) if existing_row else 0

    already_done = wins + draws + losses
    remaining = target_games - already_done

    if remaining <= 0:
        return existing_row

    print(
        f"[eval] {checkpoint.name} vs {opponent.name}: "
        f"{already_done}/{target_games} already done, playing {remaining} games.",
        flush=True,
    )

    model_checkpoint = load_eval_model(checkpoint.path, device)
    model_opponent = load_eval_model(opponent.path, device)

    t0 = time.time()
    completed_now = 0

    try:
        for batch_start in range(already_done, target_games, batch_games):
            batch_end = min(target_games, batch_start + batch_games)
            game_indices = list(range(batch_start, batch_end))

            batch_results = play_eval_game_batch(
                checkpoint=checkpoint,
                opponent=opponent,
                model_checkpoint=model_checkpoint,
                model_opponent=model_opponent,
                game_indices=game_indices,
                starting_positions=starting_positions,
                num_simulations=num_simulations,
                c_puct=c_puct,
                max_moves=max_moves,
                device=device,
            )

            for game in batch_results:
                if game.result is None or game.termination is None:
                    raise RuntimeError("Internal error: active game was not finalized.")

                checkpoint_score = score_for_checkpoint(
                    result=game.result,
                    checkpoint_is_white=game.checkpoint_is_white,
                )

                if checkpoint_score == 1.0:
                    wins += 1
                elif checkpoint_score == 0.5:
                    draws += 1
                else:
                    losses += 1

                opening_tag = (
                    game.opening_name.replace(" ", "_").replace("'", "")
                    if game.opening_name else "startpos"
                )

                round_name = (
                    f"{checkpoint.name}_vs_{opponent.name}_"
                    f"{game.game_idx + 1}_{opening_tag}"
                )

                append_game_to_pgn(
                    pgn_path=pgn_path,
                    white_name=game.white_name,
                    black_name=game.black_name,
                    result=game.result,
                    moves=game.played_moves,
                    round_name=round_name,
                    termination=game.termination,
                    num_simulations=num_simulations,
                    initial_fen=game.initial_fen,
                    opening_name=game.opening_name,
                )

            completed_now += len(batch_results)
            total_done = already_done + completed_now
            elapsed = max(1e-9, time.time() - t0)
            games_per_min = completed_now / elapsed * 60.0

            print(
                f"[eval] {checkpoint.name} vs {opponent.name}: "
                f"{total_done}/{target_games} games | "
                f"W/D/L={wins}/{draws}/{losses} | "
                f"{games_per_min:.2f} games/min",
                flush=True,
            )

    finally:
        del model_checkpoint
        del model_opponent

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return make_result_row(
        checkpoint_name=checkpoint.name,
        opponent_name=opponent.name,
        wins=wins,
        draws=draws,
        losses=losses,
        sims=num_simulations,
        step=step,
        fen_count=len(starting_positions),
    )


# ============================================================
# Optional Ordo
# ============================================================

def run_ordo_if_available(
    pgn_path: Path,
    output_path: Path,
    ordo_bin: str = "ordo",
    average: int = 0,
) -> None:
    exe = shutil.which(ordo_bin)

    if exe is None:
        print("Ordo not available; skipped rating estimation.")
        return

    cmd = [
        exe,
        "-a",
        str(average),
        "-p",
        str(pgn_path),
        "-o",
        str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        print(f"[eval] Ordo ratings saved to {output_path}")

        if result.stdout.strip():
            print(result.stdout)

    except Exception as exc:
        print(f"Ordo crashed or failed; skipped rating estimation. Error: {exc}")


# ============================================================
# Main eval command
# ============================================================

def eval_checkpoints(args) -> None:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset_output:
        for path in (EVAL_RESULTS_CSV, EVAL_GAMES_PGN, EVAL_ORDO_RATINGS_TXT):
            if path.exists():
                path.unlink()
                print(f"[eval] deleted old output: {path}", flush=True)

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    starting_positions = build_starting_positions(args)

    print("=" * 80)
    print("[eval] Starting latest checkpoint evaluation")
    print(f"[eval] device: {device}")
    print(f"[eval] games per match: {args.games}")
    print(f"[eval] sims per move: {args.sims}")
    print(f"[eval] step: {args.step}")
    print(f"[eval] batch games: {args.batch_games}")
    print(f"[eval] starting FEN pairs available: {len(starting_positions)}")
    print(f"[eval] checkpoint dir: {RL_CHECKPOINT_DIR}")
    print(f"[eval] model dir: {RL_MODEL_DIR}")
    print(f"[eval] output dir: {EVAL_DIR}")
    print("=" * 80)

    baseline, checkpoints = find_eval_checkpoints(step=args.step)

    if not checkpoints:
        print("[eval] No model_iter_*.pt checkpoints found.")
        return

    print(f"[eval] baseline: {baseline.name} -> {baseline.path}")
    print("[eval] found archived checkpoints:")

    for ckpt in checkpoints:
        print(f"  - {ckpt.name}: {ckpt.path}")

    match_plan = build_latest_match_plan(
        baseline=baseline,
        checkpoints=checkpoints,
        step=args.step,
    )

    if not match_plan:
        print("[eval] No valid matches to play.")
        return

    print("[eval] latest-only match plan:")
    for checkpoint, opponent in match_plan:
        print(f"  - {checkpoint.name} vs {opponent.name}")

    result_rows = load_existing_results(EVAL_RESULTS_CSV)

    for checkpoint, opponent in match_plan:
        key = (checkpoint.name, opponent.name, args.sims)
        existing_row = result_rows.get(key)

        if args.force:
            existing_row = None

        current_games = int(existing_row.get("games", 0)) if existing_row else 0

        if current_games >= args.games and not args.force:
            print(
                f"[eval] skipping {checkpoint.name} vs {opponent.name}: "
                f"{current_games}/{args.games} games already done.",
                flush=True,
            )
            continue

        row = play_eval_match(
            checkpoint=checkpoint,
            opponent=opponent,
            existing_row=existing_row,
            target_games=args.games,
            num_simulations=args.sims,
            c_puct=args.c_puct,
            max_moves=args.max_moves,
            device=device,
            pgn_path=EVAL_GAMES_PGN,
            step=args.step,
            starting_positions=starting_positions,
            batch_games=args.batch_games,
        )

        result_rows[key] = row
        write_results_csv(EVAL_RESULTS_CSV, result_rows)

    print("=" * 80)
    print(f"[eval] CSV summary saved to: {EVAL_RESULTS_CSV}")
    print(f"[eval] PGN database saved to: {EVAL_GAMES_PGN}")
    print("=" * 80)

    if args.no_ordo:
        print("[eval] Ordo skipped because --no-ordo was set.")
    else:
        run_ordo_if_available(
            pgn_path=EVAL_GAMES_PGN,
            output_path=EVAL_ORDO_RATINGS_TXT,
            ordo_bin=args.ordo_bin,
            average=args.ordo_average,
        )


# ============================================================
# CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the latest archived AlphaZero-style chess checkpoint."
    )

    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Games per match. Total games are usually 2x this, because latest plays SL and previous.",
    )
    parser.add_argument("--sims", type=int, default=200, help="MCTS simulations per move.")
    parser.add_argument("--step", type=int, default=5, help="Archived checkpoint interval, usually 5.")

    parser.add_argument("--c-puct", type=float, default=SELFPLAY_C_PUCT)
    parser.add_argument("--max-moves", type=int, default=MAX_GAME_MOVES)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--batch-games",
        type=int,
        default=32,
        help="How many eval games to keep active at once. Increase for better batching; decrease if RAM is high.",
    )

    parser.add_argument(
        "--no-starting-fens",
        action="store_true",
        help="Disable starting FENs and start every evaluation game from the normal initial chess position.",
    )
    parser.add_argument(
        "--fen-file",
        type=str,
        default=None,
        help="Optional file with curated FENs. Format: either plain FEN or Name | FEN per line.",
    )
    parser.add_argument(
        "--auto-generate-fens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-generate deterministic legal opening FENs if the built-in/fen-file suite is too small.",
    )
    parser.add_argument("--fen-seed", type=int, default=12345)
    parser.add_argument("--generated-min-plies", type=int, default=6)
    parser.add_argument("--generated-max-plies", type=int, default=14)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replay matches from scratch in CSV perspective. PGN will still append unless --reset-output is also set.",
    )
    parser.add_argument(
        "--reset-output",
        action="store_true",
        help="Delete eval_latest_results.csv, eval_latest_games.pgn and ordo_latest_ratings.txt before running.",
    )

    parser.add_argument(
        "--no-ordo",
        action="store_true",
        help="Skip Ordo even if it is installed.",
    )
    parser.add_argument("--ordo-bin", type=str, default="ordo")
    parser.add_argument("--ordo-average", type=int, default=0)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.games <= 0:
        raise ValueError("--games must be positive.")
    if args.sims < 0:
        raise ValueError("--sims must be non-negative.")
    if args.step <= 0:
        raise ValueError("--step must be positive.")
    if args.batch_games <= 0:
        raise ValueError("--batch-games must be positive.")
    if args.generated_min_plies < 0 or args.generated_max_plies < args.generated_min_plies:
        raise ValueError("Invalid generated ply range.")

    eval_checkpoints(args)


if __name__ == "__main__":
    main()
