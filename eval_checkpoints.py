#!/usr/bin/env python3

# ============================================================
# eval_checkpoints.py
#
# External checkpoint evaluator for AlphaZero-style chess agent.
#
# Usage:
#   python -u eval_checkpoints.py --games 40 --sims 100 --step 5
#
# This script:
#   - imports model/encoding/path definitions from chess_agent.py
#   - scans model_iter_*.pt checkpoints
#   - matches each checkpoint against model_after_sl.pt
#   - also matches model_iter_N against model_iter_(N-step)
#   - writes CSV summary
#   - appends PGN games
#   - optionally runs Ordo if installed
# ============================================================

import argparse
import csv
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

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
EVAL_RESULTS_CSV = EVAL_DIR / "eval_results.csv"
EVAL_GAMES_PGN = EVAL_DIR / "eval_games.pgn"
EVAL_ORDO_RATINGS_TXT = EVAL_DIR / "ordo_ratings.txt"


# ============================================================
# Opening / starting positions for evaluation
# ============================================================
#
# These are not used in training. They are only for external evaluation.
#
# Scheduling rule:
#   game 0: FEN 0, checkpoint White
#   game 1: FEN 0, checkpoint Black
#   game 2: FEN 1, checkpoint White
#   game 3: FEN 1, checkpoint Black
#   ...
#
# With default --games 40 and 10 FENs:
#   each FEN is played 4 times per match:
#     - 2 times with checkpoint as White
#     - 2 times with checkpoint as Black

STARTING_POSITIONS = [
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

@dataclass
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

        # Raw state_dict case.
        if all(torch.is_tensor(v) for v in checkpoint_obj.values()):
            return strip_module_prefix(checkpoint_obj)

    raise ValueError("Could not find model state_dict inside checkpoint.")


def create_eval_model(device: torch.device):
    """
    Creates the model architecture.

    If your ChessNet constructor requires arguments, edit this function.
    For your current code, ChessNet() should probably be enough.
    """

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
# Move encoding / board tensor wrappers
# ============================================================

def move_to_policy_index(board: chess.Board, move: chess.Move) -> int:
    return int(chess_agent.move_to_id(move, board))

def board_to_tensor_batch(board: chess.Board, device: torch.device) -> torch.Tensor:
    """
    Uses your project's board_to_tensor(board).

    Expected single-board output:
      [C, 8, 8]

    Returns:
      [1, C, 8, 8]
    """

    x = board_to_tensor(board)

    if not torch.is_tensor(x):
        x = torch.tensor(x)

    x = x.float()

    if x.ndim == 3:
        x = x.unsqueeze(0)

    if x.ndim != 4:
        raise RuntimeError(
            f"board_to_tensor returned unexpected shape: {tuple(x.shape)}"
        )

    return x.to(device, non_blocking=True)


def split_model_output(output):
    """
    Supports:
      - model(x) -> (policy_logits, value)
      - model(x) -> {"policy": ..., "value": ...}

    Also tries to detect tuple order by shape.
    """

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

        # Policy should have 4672 outputs.
        if torch.is_tensor(a_flat) and a_flat.shape[-1] == 4672:
            return a, b

        if torch.is_tensor(b_flat) and b_flat.shape[-1] == 4672:
            return b, a

        # Fallback: assume normal order.
        return a, b

    raise RuntimeError(
        "Model output format not understood. Expected tuple/list(policy, value) "
        "or dict with policy/value."
    )


# ============================================================
# Evaluation neural network call
# ============================================================

def terminal_value_from_side_to_move(board: chess.Board) -> float:
    """
    Returns terminal value from the perspective of the side to move.

    win for side to move: +1
    loss for side to move: -1
    draw: 0
    """

    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        return 0.0

    if outcome.winner is None:
        return 0.0

    return 1.0 if outcome.winner == board.turn else -1.0


@torch.no_grad()
def eval_policy_value(
    model,
    board: chess.Board,
    device: torch.device,
) -> tuple[dict[chess.Move, float], float]:
    """
    Returns:
      priors: legal move -> probability
      value: scalar from side-to-move perspective
    """

    legal_moves = list(board.legal_moves)

    if not legal_moves:
        return {}, terminal_value_from_side_to_move(board)

    x = board_to_tensor_batch(board, device)

    output = model(x)
    policy_logits, value = split_model_output(output)

    policy_logits = policy_logits.view(-1)
    value = float(value.view(-1)[0].item())

    legal_indices = []
    valid_moves = []

    for move in legal_moves:
        idx = move_to_policy_index(board, move)
        legal_indices.append(idx)
        valid_moves.append(move)

    legal_idx_tensor = torch.tensor(
        legal_indices,
        dtype=torch.long,
        device=policy_logits.device,
    )

    legal_logits = policy_logits[legal_idx_tensor]
    legal_probs = F.softmax(legal_logits, dim=0)

    if torch.isnan(legal_probs).any() or torch.isinf(legal_probs).any():
        uniform = 1.0 / len(valid_moves)
        priors = {move: uniform for move in valid_moves}
    else:
        probs_cpu = legal_probs.detach().cpu().tolist()
        priors = {
            move: float(prob)
            for move, prob in zip(valid_moves, probs_cpu)
        }

    return priors, value


# ============================================================
# Eval-only MCTS
# ============================================================

def select_child(
    node: EvalMCTSNode,
    c_puct: float,
) -> tuple[chess.Move, EvalMCTSNode]:
    """
    PUCT selection.

    child.value is from the child node's side-to-move perspective.
    From the parent perspective, that value is -child.value.
    """

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


def run_eval_mcts(
    model,
    root_board: chess.Board,
    device: torch.device,
    num_simulations: int,
    c_puct: float,
) -> chess.Move:
    """
    Eval-only MCTS.

    Important:
      - no Dirichlet noise
      - no temperature sampling
      - final move is highest visit count
    """

    legal_moves = list(root_board.legal_moves)

    if not legal_moves:
        raise RuntimeError("MCTS called on board with no legal moves.")

    if len(legal_moves) == 1:
        return legal_moves[0]

    root = EvalMCTSNode(prior=1.0)

    for _ in range(num_simulations):
        board = root_board.copy(stack=False)
        node = root
        search_path = [node]

        # Selection.
        while node.expanded and not board.is_game_over(claim_draw=True):
            move, node = select_child(node, c_puct=c_puct)
            board.push(move)
            search_path.append(node)

        # Expansion/evaluation.
        if board.is_game_over(claim_draw=True):
            value = terminal_value_from_side_to_move(board)
        else:
            priors, value = eval_policy_value(model, board, device)
            node.expand(priors)

        # Backup.
        # value is from current node side-to-move perspective.
        # Each step upward flips perspective.
        for path_node in reversed(search_path):
            path_node.visit_count += 1
            path_node.value_sum += value
            value = -value

    if not root.children:
        return legal_moves[0]

    best_move, best_child = max(
        root.children.items(),
        key=lambda item: (item[1].visit_count, item[1].prior),
    )

    return best_move


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


def play_single_eval_game(
    white_name: str,
    white_model,
    black_name: str,
    black_model,
    device: torch.device,
    num_simulations: int,
    c_puct: float,
    max_moves: int,
    initial_fen: Optional[str] = None,
) -> tuple[str, list[chess.Move], str]:
    board = chess.Board(initial_fen) if initial_fen else chess.Board()
    played_moves: list[chess.Move] = []

    while not board.is_game_over(claim_draw=True) and len(played_moves) < max_moves:
        model = white_model if board.turn == chess.WHITE else black_model

        move = run_eval_mcts(
            model=model,
            root_board=board,
            device=device,
            num_simulations=num_simulations,
            c_puct=c_puct,
        )

        board.push(move)
        played_moves.append(move)

    result, termination = result_from_board(board)
    return result, played_moves, termination


# ============================================================
# PGN writing
# ============================================================

def append_game_to_pgn(pgn_path: Path, white_name: str, black_name: str, result: str, moves: list[chess.Move], round_name: str, termination: str,
    num_simulations: int, initial_fen: Optional[str] = None, opening_name: Optional[str] = None) -> None:
    pgn_path.parent.mkdir(parents=True, exist_ok=True)

    board = chess.Board(initial_fen) if initial_fen else chess.Board()

    game = chess.pgn.Game()
    game.headers["Event"] = "ChessAgent checkpoint evaluation"
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
    "updated_at",
]


def elo_delta_from_score(score: float, eps: float = 1e-4) -> float:
    """
    Approximate two-player Elo delta:

        400 * log10(score / (1 - score))

    Clamped to avoid infinite values for 0% or 100%.
    """

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
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


# ============================================================
# Finding checkpoints and building match graph
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

    # Legacy fallback from old layout
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


def build_match_plan(
    baseline: EvalCheckpoint,
    checkpoints: list[EvalCheckpoint],
    step: int,
) -> list[tuple[EvalCheckpoint, EvalCheckpoint]]:
    """
    Returns matches as:

        checkpoint vs opponent

    CSV is always from checkpoint's perspective.

    Example with step=5:

        model_iter_5  vs model_after_sl
        model_iter_10 vs model_after_sl
        model_iter_10 vs model_iter_5
        model_iter_15 vs model_after_sl
        model_iter_15 vs model_iter_10
    """

    by_iter = {
        ckpt.iteration: ckpt
        for ckpt in checkpoints
        if ckpt.iteration is not None
    }

    matches: list[tuple[EvalCheckpoint, EvalCheckpoint]] = []

    for ckpt in checkpoints:
        # Fixed baseline match.
        matches.append((ckpt, baseline))

        # Adjacent checkpoint match.
        if ckpt.iteration is not None:
            previous_iteration = ckpt.iteration - step
            previous_ckpt = by_iter.get(previous_iteration)

            if previous_ckpt is not None:
                matches.append((ckpt, previous_ckpt))

    # Remove accidental duplicates.
    seen = set()
    unique_matches = []

    for checkpoint, opponent in matches:
        key = (checkpoint.name, opponent.name)

        if key in seen:
            continue

        if checkpoint.path.resolve() == opponent.path.resolve():
            continue

        seen.add(key)
        unique_matches.append((checkpoint, opponent))

    return unique_matches


def select_starting_position(
    game_idx: int,
    use_starting_fens: bool,
) -> tuple[Optional[str], Optional[str]]:
    """
    Returns (opening_name, fen).

    We pair games by FEN:
      game 0 and 1 use FEN 0,
      game 2 and 3 use FEN 1,
      etc.

    This guarantees that each FEN is played once with the checkpoint as White
    and once with the checkpoint as Black before moving to the next FEN.
    """

    if not use_starting_fens or len(STARTING_POSITIONS) == 0:
        return None, None

    fen_pair_idx = game_idx // 2
    opening_name, fen = STARTING_POSITIONS[
        fen_pair_idx % len(STARTING_POSITIONS)
    ]

    return opening_name, fen

# ============================================================
# Running matches
# ============================================================

def play_eval_match(checkpoint: EvalCheckpoint, opponent: EvalCheckpoint, existing_row: Optional[dict], target_games: int, num_simulations: int, c_puct: float,
    max_moves: int, device: torch.device, pgn_path: Path, step: int, use_starting_fens: bool) -> dict:
    """
    Plays missing games only.

    wins/draws/losses are always from `checkpoint` perspective.
    """

    wins = int(existing_row["wins"]) if existing_row else 0
    draws = int(existing_row["draws"]) if existing_row else 0
    losses = int(existing_row["losses"]) if existing_row else 0

    already_done = wins + draws + losses
    remaining = target_games - already_done

    if remaining <= 0:
        return existing_row

    print(f"[eval] {checkpoint.name} vs {opponent.name}: " f"{already_done}/{target_games} already done, " f"playing {remaining} games.", flush=True)

    model_checkpoint = load_eval_model(checkpoint.path, device)
    model_opponent = load_eval_model(opponent.path, device)

    for local_game_idx in range(remaining):
        absolute_game_idx = already_done + local_game_idx

        opening_name, initial_fen = select_starting_position(game_idx=absolute_game_idx, use_starting_fens=use_starting_fens)

        # Alternate colors.
        checkpoint_is_white = (absolute_game_idx % 2 == 0)

        if checkpoint_is_white:
            white_name = checkpoint.name
            white_model = model_checkpoint
            black_name = opponent.name
            black_model = model_opponent
        else:
            white_name = opponent.name
            white_model = model_opponent
            black_name = checkpoint.name
            black_model = model_checkpoint

        result, moves, termination = play_single_eval_game(white_name=white_name, white_model=white_model, black_name=black_name, black_model=black_model,
            device=device, num_simulations=num_simulations, c_puct=c_puct, max_moves=max_moves, initial_fen=initial_fen)

        checkpoint_score = score_for_checkpoint(result=result, checkpoint_is_white=checkpoint_is_white)

        if checkpoint_score == 1.0:
            wins += 1
        elif checkpoint_score == 0.5:
            draws += 1
        else:
            losses += 1

        opening_tag = (opening_name.replace(" ", "_").replace("'", "") if opening_name else "startpos")

        round_name = (f"{checkpoint.name}_vs_{opponent.name}_" f"{absolute_game_idx + 1}_{opening_tag}")

        append_game_to_pgn(pgn_path=pgn_path, white_name=white_name, black_name=black_name, result=result, moves=moves, round_name=round_name,
            termination=termination, num_simulations=num_simulations, initial_fen=initial_fen, opening_name=opening_name)

        print(
            f"[eval] game {absolute_game_idx + 1}/{target_games}: "
            f"{white_name} vs {black_name} {result} | "
            f"opening={opening_name or 'startpos'} | "
            f"{checkpoint.name} W/D/L = {wins}/{draws}/{losses}",
            flush=True,
        )

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

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    print("=" * 80)
    print("[eval] Starting checkpoint evaluation")
    print(f"[eval] device: {device}")
    print(f"[eval] games per match: {args.games}")
    print(f"[eval] sims per move: {args.sims}")
    print(f"[eval] step: {args.step}")
    if args.no_starting_fens:
        print("[eval] starting FENs: disabled")
    else:
        print(f"[eval] starting FENs: {len(STARTING_POSITIONS)}")
    print(f"[eval] checkpoint dir: {RL_CHECKPOINT_DIR}")
    print(f"[eval] output dir: {EVAL_DIR}")
    print("=" * 80)

    baseline, checkpoints = find_eval_checkpoints(step=args.step)

    if not checkpoints:
        print("[eval] No model_iter_*.pt checkpoints found.")
        return

    print(f"[eval] baseline: {baseline.name} -> {baseline.path}")
    print("[eval] found checkpoints:")

    for ckpt in checkpoints:
        print(f"  - {ckpt.name}: {ckpt.path}")

    match_plan = build_match_plan(
        baseline=baseline,
        checkpoints=checkpoints,
        step=args.step,
    )

    if not match_plan:
        print("[eval] No valid matches to play.")
        return

    print("[eval] match plan:")

    for checkpoint, opponent in match_plan:
        print(f"  - {checkpoint.name} vs {opponent.name}")

    result_rows = load_existing_results(EVAL_RESULTS_CSV)

    for checkpoint, opponent in match_plan:
        key = (checkpoint.name, opponent.name, args.sims)

        existing_row = result_rows.get(key)

        if args.force:
            existing_row = None

        current_games = int(existing_row["games"]) if existing_row else 0

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
            use_starting_fens=not args.no_starting_fens,
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate AlphaZero-style chess checkpoints externally.")

    parser.add_argument("--no-starting-fens", action="store_true", help="Disable opening FENs and start every evaluation game from the normal initial chess position.")

    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--step", type=int, default=5)

    parser.add_argument("--c-puct", type=float, default=SELFPLAY_C_PUCT)
    parser.add_argument("--max-moves", type=int, default=MAX_GAME_MOVES)

    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--force",
        action="store_true",
        help="Replay matches from scratch in CSV perspective. PGN will still append.",
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
    eval_checkpoints(args)


if __name__ == "__main__":
    main()