import os
import math
import time
import random
import argparse
import multiprocessing as mp
from pathlib import Path
from collections import deque, Counter

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sys
import queue as pyqueue
import traceback
import torch.multiprocessing as mp


# ============================================================
# Nefeli Runtime Config
# ============================================================

PROJECT_DIR = Path.home() / "chess-agent-aristotle"

# ============================================================
# HOME (persistent)
# ============================================================

HOME_OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR = HOME_OUTPUT_DIR

CHECKPOINT_DIR = HOME_OUTPUT_DIR / "checkpoints"
RL_MODEL_DIR = HOME_OUTPUT_DIR / "rl_models"

# ============================================================
# SCRATCH (large / temporary)
# ============================================================

SCRATCH_OUTPUT_DIR = Path(
    os.environ.get(
        "CHESS_SCRATCH_OUTPUT_DIR",
        str(HOME_OUTPUT_DIR),
    )
)

RL_CHECKPOINT_DIR = SCRATCH_OUTPUT_DIR / "rl_checkpoints"

# ============================================================
# Other paths
# ============================================================

DATA_DIR = PROJECT_DIR / "data"

SL_SHARD_DIR = Path(
    os.environ.get(
        "SL_SHARD_DIR",
        str(SCRATCH_OUTPUT_DIR / "sl_shards")
    )
)

for d in [
    CHECKPOINT_DIR,
    RL_MODEL_DIR,
    RL_CHECKPOINT_DIR,
    DATA_DIR,
    SL_SHARD_DIR,
]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# Persistent checkpoints/models (HOME)
# ============================================================

SL_LATEST_PATH = CHECKPOINT_DIR / "latest_sl.pt"

MODEL_LATEST_PATH = RL_MODEL_DIR / "model_latest.pt"
MODEL_BEST_PATH = RL_MODEL_DIR / "model_best.pt"
MODEL_AFTER_SL_PATH = RL_MODEL_DIR / "model_after_sl.pt"

# ============================================================
# Large working checkpoints (SCRATCH)
# ============================================================

RL_LATEST_PATH = RL_CHECKPOINT_DIR / "rl_latest.pt"
SELFPLAY_PARTIAL_PATH = RL_CHECKPOINT_DIR / "selfplay_partial.pt"

SELFPLAY_PARTIAL_SAVE_EVERY_GAMES = 16
SELFPLAY_PARTIAL_SAVE_EVERY_SECONDS = 300
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Slurm gives the job a fixed CPU allocation. Keep PyTorch training threads modest,
# because self-play uses separate worker processes.
SLURM_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", "32"))
torch.set_num_threads(min(8, max(1, SLURM_CPUS // 4)))

# ============================================================
# Hyperparameters - Nefeli RL version
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True

# SL values are kept only for checkpoint compatibility / shared model config.
WEIGHT_DECAY = 1e-4
VALUE_LOSS_WEIGHT = 0.5

# Nefeli RL settings.
# These are intentionally larger than Colab, but still conservative for a first stable run.
RL_ITERATIONS = 3
REPLAY_BUFFER_SIZE = 200_000
GAMES_PER_ITERATION = 128
SELFPLAY_SIMS = 150
RL_BATCH_SIZE = 512
RL_LR = 1e-4
SELFPLAY_C_PUCT = 1.5
MAX_GAME_MOVES = 250
TEMP_PLIES = 24
ROOT_DIRICHLET_ALPHA = 0.3
ROOT_DIRICHLET_EPSILON = 0.25

# Parallel self-play with inference server
PARALLEL_SELFPLAY = True

SELFPLAY_WORKERS = 8          # test with 4 first, then 8/16/24
WORKER_VECTOR_GAMES = 2        # active games per worker
INFER_SERVER_BATCH_SIZE = 16  # target GPU inference batch
INFER_SERVER_TIMEOUT_MS = 5    # max wait before flushing partial batch

# Policy training. When enabled, the policy loss softmax is computed only
# over legal moves, matching the legal-move masking used by MCTS.
MASKED_POLICY_LOSS = True
POLICY_MASK_VALUE = -1e4

# Dynamic fixed-step RL training schedule based on new self-play samples.
RL_REUSE_FACTOR = 4
RL_TRAIN_MIN_STEPS = 10
RL_TRAIN_MAX_STEPS = 200

NUM_WORKERS = 2     # For DataLoaders

DEFAULT_PGN_PATH = DATA_DIR / "lichess_elite_2024-07.pgn"

SL_EPOCHS = 7
SL_BATCH_SIZE = 512
SL_LR = 1e-4
SL_SHARD_SIZE = 50_000
SL_VALUE_MIN_PLY = 10
SL_VALUE_POSITIONS_PER_GAME = 5
SL_VALUE_LOSS_WEIGHT = 0.5



# ============================================================
# Board To Tensor
# ============================================================

PIECE_TYPE_TO_IDX = {
    chess.PAWN: 0,
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
    chess.QUEEN: 4,
    chess.KING: 5,
}


def orient_square(square: chess.Square, turn: chess.Color) -> chess.Square:
    return square if turn == chess.WHITE else chess.square_mirror(square)


def square_to_row_col_canonical(square: chess.Square, turn: chess.Color):
    sq = orient_square(square, turn)
    row = 7 - chess.square_rank(sq)
    col = chess.square_file(sq)
    return row, col


def board_to_tensor(board: chess.Board) -> np.ndarray:
    x = np.zeros((17, 8, 8), dtype=np.float32)
    turn = board.turn

    for square, piece in board.piece_map().items():
        row, col = square_to_row_col_canonical(square, turn)
        base = 0 if piece.color == turn else 6
        plane = base + PIECE_TYPE_TO_IDX[piece.piece_type]
        x[plane, row, col] = 1.0

    if board.ep_square is not None:
        row, col = square_to_row_col_canonical(board.ep_square, turn)
        x[12, row, col] = 1.0

    opponent = not turn
    if board.has_kingside_castling_rights(turn):
        x[13, :, :] = 1.0
    if board.has_queenside_castling_rights(turn):
        x[14, :, :] = 1.0
    if board.has_kingside_castling_rights(opponent):
        x[15, :, :] = 1.0
    if board.has_queenside_castling_rights(opponent):
        x[16, :, :] = 1.0

    return x


# ============================================================
# Move Encoding
# ============================================================

QUEEN_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
KNIGHT_DIRS = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
UNDERPROMOTIONS = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

NUM_MOVE_PLANES = 73
NUM_MOVES = 8 * 8 * NUM_MOVE_PLANES


def canonical_square(square: chess.Square, turn: chess.Color):
    return square if turn == chess.WHITE else chess.square_mirror(square)


def square_to_row_col_from_oriented_square(square: chess.Square):
    row = 7 - chess.square_rank(square)
    col = chess.square_file(square)
    return row, col


def move_to_id(move: chess.Move, board: chess.Board) -> int:
    turn = board.turn

    from_sq = canonical_square(move.from_square, turn)
    to_sq = canonical_square(move.to_square, turn)

    from_row, from_col = square_to_row_col_from_oriented_square(from_sq)
    to_row, to_col = square_to_row_col_from_oriented_square(to_sq)

    dr = to_row - from_row
    dc = to_col - from_col

    if move.promotion in [None, chess.QUEEN]:
        for dir_idx, (step_r, step_c) in enumerate(QUEEN_DIRS):
            for dist in range(1, 8):
                if dr == step_r * dist and dc == step_c * dist:
                    plane = dir_idx * 7 + (dist - 1)
                    return (from_row * 8 + from_col) * NUM_MOVE_PLANES + plane

        for knight_idx, (step_r, step_c) in enumerate(KNIGHT_DIRS):
            if dr == step_r and dc == step_c:
                plane = 56 + knight_idx
                return (from_row * 8 + from_col) * NUM_MOVE_PLANES + plane

    if move.promotion in UNDERPROMOTIONS:
        promotion_piece_idx = UNDERPROMOTIONS.index(move.promotion)

        if dc == -1:
            direction_idx = 0
        elif dc == 0:
            direction_idx = 1
        elif dc == 1:
            direction_idx = 2
        else:
            raise ValueError(f"Invalid underpromotion move: {move}")

        plane = 64 + promotion_piece_idx * 3 + direction_idx
        return (from_row * 8 + from_col) * NUM_MOVE_PLANES + plane

    raise ValueError(f"Move cannot be encoded: {move}")


def legal_moves_mask(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_MOVES, dtype=np.bool_)

    for legal_move in board.legal_moves:
        move_id = move_to_id(legal_move, board)
        mask[move_id] = True

    return mask


# ============================================================
# Result Encoding
# ============================================================


def result_to_value(result: str) -> float:
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    if result == "1/2-1/2":
        return 0.0
    return 0.0


def value_for_side_to_move(board: chess.Board, game_result: str) -> float:
    white_value = result_to_value(game_result)
    return white_value if board.turn == chess.WHITE else -white_value


# ============================================================
# Neural Network
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class ChessNet(nn.Module):
    def __init__(self, in_channels=17, channels=128, num_blocks=8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, NUM_MOVES),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        policy_logits = self.policy_head(x)
        value = self.value_head(x).squeeze(1)
        return policy_logits, value


# ============================================================
# Checkpoint Functions
# ============================================================


def save_model_only(model, path, extra=None):
    payload = {
        "model_state_dict": model.state_dict(),
        "config": {
            "channels": 128,
            "num_blocks": 8,
            "num_moves": NUM_MOVES,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print("Saved model-only:", path, flush=True)


def load_model_checkpoint(path, device):
    model = ChessNet().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("Loaded model:", path, flush=True)
    return model, ckpt


# ============================================================
# Dataset
# ============================================================

class SelfPlayDataset(Dataset):
    def __init__(self, replay_buffer):
        self.data = list(replay_buffer)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if len(sample) == 4:
            x, pi, z, legal_mask = sample
        elif len(sample) == 3:
            # Backward compatibility for older replay samples without masks.
            x, pi, z = sample
            legal_mask = np.ones(NUM_MOVES, dtype=np.bool_)
        else:
            raise ValueError(f"Unexpected self-play sample format: len={len(sample)}")

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(pi, dtype=torch.float32),
            torch.tensor(z, dtype=torch.float32),
            torch.tensor(legal_mask, dtype=torch.bool),
        )



# ============================================================
# Supervised Learning From PGN
# ============================================================

class SupervisedShardDataset(Dataset):
    def __init__(self, shard_path):
        data = np.load(shard_path)
        self.x = data["x"]
        self.move_id = data["move_id"]
        self.z = data["z"]
        self.value_weight = data["value_weight"]
        self.legal_mask = data["legal_mask"]

    def __len__(self):
        return len(self.move_id)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.x[idx], dtype=torch.float32),
            torch.tensor(self.move_id[idx], dtype=torch.long),
            torch.tensor(self.z[idx], dtype=torch.float32),
            torch.tensor(self.value_weight[idx], dtype=torch.float32),
            torch.tensor(self.legal_mask[idx], dtype=torch.bool),
        )


def clear_sl_shards():
    for path in SL_SHARD_DIR.glob("sl_shard_*.npz"):
        path.unlink()

    meta_path = SL_SHARD_DIR / "metadata.pt"
    if meta_path.exists():
        meta_path.unlink()


def save_sl_shard(shard_idx, xs, move_ids, zs, value_weights, legal_masks):
    if len(xs) == 0:
        return None

    shard_path = SL_SHARD_DIR / f"sl_shard_{shard_idx:05d}.npz"

    np.savez(shard_path, x=np.stack(xs).astype(np.uint8), move_id=np.array(move_ids, dtype=np.int64), z=np.array(zs, dtype=np.float32),
                value_weight=np.array(value_weights, dtype=np.float32), legal_mask=np.stack(legal_masks).astype(np.bool_))


    print(f"Saved SL shard {shard_idx}: {shard_path} samples={len(move_ids)}", flush=True)
    return shard_path

def sample_value_plies(num_plies, min_ply, positions_per_game):
    """
    Stratified SL value sampling.

    Keeps policy training on every PGN position, but gives value loss weight
    only to a small number of positions per game.

    Important: ply is zero-based, and min_ply is strict here.
    With min_ply=10, eligible value positions are ply 11, 12, ...
    """
    if positions_per_game is None or positions_per_game <= 0:
        return set()

    # Strictly after min_ply, as requested: ply > value_min_ply.
    eligible = list(range(int(min_ply) + 1, num_plies))
    if not eligible:
        return set()

    k = min(int(positions_per_game), len(eligible))
    if k == len(eligible):
        return set(eligible)

    selected = []

    # Split the eligible part of the game into k temporal buckets and
    # sample one random ply from each bucket. This gives early/mid/late
    # coverage instead of accidentally picking all value targets from one stage.
    boundaries = np.linspace(0, len(eligible), num=k + 1, dtype=int)

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        bucket = eligible[start:end]
        if bucket:
            selected.append(random.choice(bucket))

    # Safety fallback; normally not needed when k <= len(eligible).
    while len(selected) < k:
        ply = random.choice(eligible)
        if ply not in selected:
            selected.append(ply)

    return set(selected)


def prepare_sl_shards_from_pgn(pgn_path, shard_size=SL_SHARD_SIZE, value_min_ply=SL_VALUE_MIN_PLY, value_positions_per_game=SL_VALUE_POSITIONS_PER_GAME):
    pgn_path = Path(pgn_path)

    if not pgn_path.exists():
        raise FileNotFoundError(f"PGN file not found: {pgn_path}")

    print("Preparing SL shards from PGN:", pgn_path, flush=True)
    print("Output shard dir:", SL_SHARD_DIR, flush=True)
    print("Shard size:", shard_size, flush=True)
    print("Value min ply:", value_min_ply, flush=True)
    print("Value positions per game:", value_positions_per_game, flush=True)

    clear_sl_shards()

    xs = []
    move_ids = []
    zs = []
    value_weights = []
    legal_masks = []

    shard_idx = 0
    total_games = 0
    used_games = 0
    skipped_games = 0
    total_samples = 0
    total_value_weighted_samples = 0
    shard_paths = []
    result_counter = Counter()
    start_time = time.time()

    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        pbar = tqdm(desc="Parsing PGN games", file=sys.stdout, dynamic_ncols=True)

        while True:
            game = chess.pgn.read_game(f)

            if game is None:
                break

            total_games += 1
            pbar.update(1)

            result = game.headers.get("Result", "*")

            if result not in ["1-0", "0-1", "1/2-1/2"]:
                skipped_games += 1
                continue

            moves = list(game.mainline_moves())
            value_plies = sample_value_plies(num_plies=len(moves), min_ply=value_min_ply, positions_per_game=value_positions_per_game)

            board = game.board()
            bad_game = False

            game_xs = []
            game_move_ids = []
            game_zs = []
            game_value_weights = []
            game_legal_masks = []

            try:
                for ply, move in enumerate(moves):
                    value_weight = np.float32(1.0 if ply in value_plies else 0.0)

                    move_id = move_to_id(move, board)
                    legal_mask = legal_moves_mask(board)

                    if not legal_mask[move_id]:
                        raise RuntimeError(f"Target move_id {move_id} is not marked legal for move {move}")

                    game_xs.append(board_to_tensor(board).astype(np.uint8))
                    game_move_ids.append(move_id)
                    game_legal_masks.append(legal_mask)
                    game_zs.append(np.float32(value_for_side_to_move(board, result)))
                    game_value_weights.append(value_weight)

                    board.push(move)

            except Exception as e:
                skipped_games += 1
                print(f"Skipped game {total_games} due to error: {e}", flush=True)
                continue

            if bad_game:
                skipped_games += 1
                continue

            for x, move_id, z, value_weight, legal_mask in zip(game_xs, game_move_ids, game_zs, game_value_weights, game_legal_masks):
                xs.append(x)
                move_ids.append(move_id)
                zs.append(z)
                value_weights.append(value_weight)
                legal_masks.append(legal_mask)

                total_samples += 1
                total_value_weighted_samples += int(value_weight > 0.0)

                if len(xs) >= shard_size:
                    shard_path = save_sl_shard(shard_idx, xs, move_ids, zs, value_weights, legal_masks)
                    shard_paths.append(str(shard_path))

                    shard_idx += 1
                    xs.clear()
                    move_ids.clear()
                    zs.clear()
                    value_weights.clear()
                    legal_masks.clear()

            used_games += 1
            result_counter[result] += 1

            if total_games % 1000 == 0:
                elapsed = time.time() - start_time
                pbar.set_postfix({
                    "used": used_games,
                    "skipped": skipped_games,
                    "samples": total_samples,
                    "value_samples": total_value_weighted_samples,
                    "shards": shard_idx,
                    "elapsed_min": f"{elapsed / 60:.1f}",
                })

        pbar.close()

    if len(xs) > 0:
        shard_path = save_sl_shard(shard_idx, xs, move_ids, zs, value_weights, legal_masks)
        shard_paths.append(str(shard_path))

    metadata = {
        "pgn_path": str(pgn_path),
        "total_games": total_games,
        "used_games": used_games,
        "skipped_games": skipped_games,
        "total_samples": total_samples,
        "total_value_weighted_samples": total_value_weighted_samples,
        "num_shards": len(shard_paths),
        "shard_paths": shard_paths,
        "result_counter": dict(result_counter),
        "shard_size": shard_size,
        "value_min_ply": value_min_ply,
        "value_positions_per_game": value_positions_per_game,
        "num_moves": NUM_MOVES,
    }

    meta_path = SL_SHARD_DIR / "metadata.pt"
    torch.save(metadata, meta_path)

    print()
    print("=" * 70, flush=True)
    print("Finished preparing SL shards", flush=True)
    print("Metadata:", metadata, flush=True)
    print("Saved metadata:", meta_path, flush=True)
    print("=" * 70, flush=True)

    return metadata


def list_sl_shards():
    return sorted(SL_SHARD_DIR.glob("sl_shard_*.npz"))


def train_sl_one_epoch(model, optimizer, device, shard_paths, epoch_idx, num_epochs):
    model.train()
    random.shuffle(shard_paths)

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_value_weighted_samples = 0.0
    total_samples = 0

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    print()
    print("=" * 70, flush=True)
    print(f"SL Epoch {epoch_idx}/{num_epochs}", flush=True)
    print("=" * 70, flush=True)

    for shard_num, shard_path in enumerate(shard_paths, start=1):
        dataset = SupervisedShardDataset(shard_path)

        loader = DataLoader(dataset, batch_size=SL_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available(), drop_last=False, 
        persistent_workers=True if NUM_WORKERS > 0 else False, prefetch_factor=2 if NUM_WORKERS > 0 else None)

        pbar = tqdm(loader, desc=f"SL shard {shard_num}/{len(shard_paths)}", file=sys.stdout, dynamic_ncols=True)

        for X, move_id, z, value_weight, legal_mask in pbar:
            X = X.to(device, non_blocking=True)
            move_id = move_id.to(device, non_blocking=True)
            z = z.to(device, non_blocking=True)
            value_weight = value_weight.to(device, non_blocking=True)
            legal_mask = legal_mask.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
                policy_logits, value_pred = model(X)

                masked_logits = policy_logits.masked_fill(~legal_mask, -1e4)
                policy_loss = F.cross_entropy(masked_logits, move_id)
                raw_value_loss = (value_pred - z) ** 2

                if value_weight.sum() > 0:
                    value_loss = (raw_value_loss * value_weight).sum() / value_weight.sum()
                else:
                    value_loss = raw_value_loss.mean() * 0.0

                loss = policy_loss + SL_VALUE_LOSS_WEIGHT * value_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            n = X.size(0)
            weighted_n = float(value_weight.sum().item())

            total_loss += loss.item() * n
            total_policy_loss += policy_loss.item() * n
            total_value_loss += value_loss.item() * weighted_n
            total_value_weighted_samples += weighted_n
            total_samples += n

            pbar.set_postfix({
                "loss": f"{total_loss / max(1, total_samples):.4f}",
                "pol": f"{total_policy_loss / max(1, total_samples):.4f}",
                "val": f"{total_value_loss / max(1.0, total_value_weighted_samples):.4f}",
                "samples": total_samples,
            })

        del loader, dataset

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "loss": total_loss / max(1, total_samples),
        "policy_loss": total_policy_loss / max(1, total_samples),
        "value_loss": total_value_loss / max(1.0, total_value_weighted_samples),
        "samples": total_samples,
        "value_weighted_samples": total_value_weighted_samples,
    }


def train_sl(pgn_path=DEFAULT_PGN_PATH, sl_epochs=SL_EPOCHS, value_min_ply=SL_VALUE_MIN_PLY, value_positions_per_game=SL_VALUE_POSITIONS_PER_GAME, force_prepare=False, resume=True):
    print("Device:", DEVICE, flush=True)
    print("Project:", PROJECT_DIR, flush=True)
    print("PGN:", pgn_path, flush=True)
    print("SL epochs:", sl_epochs, flush=True)
    print("SL batch size:", SL_BATCH_SIZE, flush=True)
    print("SL LR:", SL_LR, flush=True)
    print("SL value loss weight:", SL_VALUE_LOSS_WEIGHT, flush=True)
    print("SL value min ply:", value_min_ply, flush=True)
    print("SL value positions per game:", value_positions_per_game, flush=True)
    print("SL resume:", resume, flush=True)

    shard_paths = list_sl_shards()

    if force_prepare or len(shard_paths) == 0:
        prepare_sl_shards_from_pgn(pgn_path=pgn_path, shard_size=SL_SHARD_SIZE, value_min_ply=value_min_ply, value_positions_per_game=value_positions_per_game)
        shard_paths = list_sl_shards()
    if len(shard_paths) == 0:
        raise RuntimeError("No SL shards found. Could not start supervised training.")

    print("Found SL shards:", len(shard_paths), flush=True)

    model = ChessNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=SL_LR, weight_decay=WEIGHT_DECAY)

    all_metrics = []
    start_epoch = 1

    if resume and SL_LATEST_PATH.exists():
        print("Resuming SL from:", SL_LATEST_PATH, flush=True)

        ckpt = torch.load(SL_LATEST_PATH, map_location=DEVICE, weights_only=False)

        model.load_state_dict(ckpt["model_state_dict"])

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Loaded SL optimizer state.", flush=True)

        completed_epoch = int(ckpt.get("epoch", 0))
        all_metrics = ckpt.get("all_metrics", [])

        start_epoch = completed_epoch + 1

        print(f"Resumed SL after epoch {completed_epoch}. Continuing from epoch {start_epoch}.", flush=True)

    if start_epoch > sl_epochs:
        print(f"SL already completed: checkpoint epoch={start_epoch - 1}, target sl_epochs={sl_epochs}", flush=True)
        return

    for epoch in range(start_epoch, sl_epochs + 1):
        metrics = train_sl_one_epoch(model=model, optimizer=optimizer, device=DEVICE, shard_paths=shard_paths, epoch_idx=epoch, num_epochs=sl_epochs)

        all_metrics.append(metrics)
        print("SL train metrics:", metrics, flush=True)

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "all_metrics": all_metrics,
            "config": {
                "source": "supervised_policy_value_from_pgn",
                "pgn_path": str(pgn_path),
                "sl_epochs": sl_epochs,
                "sl_batch_size": SL_BATCH_SIZE,
                "sl_lr": SL_LR,
                "sl_value_loss_weight": SL_VALUE_LOSS_WEIGHT,
                "sl_value_min_ply": value_min_ply,
                "sl_value_positions_per_game": value_positions_per_game,
                "num_moves": NUM_MOVES,
            },
        }

        torch.save(checkpoint, SL_LATEST_PATH)
        print("Saved SL checkpoint:", SL_LATEST_PATH, flush=True)

        save_model_only( model, MODEL_AFTER_SL_PATH,
            extra={
                "source": "after_supervised_policy_value",
                "epoch": epoch,
                "metrics": metrics,
            },
        )

        save_model_only(model, MODEL_LATEST_PATH,
            extra={
                "source": "after_supervised_policy_value",
                "epoch": epoch,
                "metrics": metrics,
            },
        )

    save_model_only(model, MODEL_BEST_PATH,
        extra={
            "source": "initial_best_after_supervised_policy_value",
            "epoch": sl_epochs,
            "metrics": all_metrics[-1] if all_metrics else None,
        },
    )

    print()
    print("=" * 70, flush=True)
    print("Finished SL training", flush=True)
    print("Saved latest SL:", SL_LATEST_PATH, flush=True)
    print("Saved model_after_sl:", MODEL_AFTER_SL_PATH, flush=True)
    print("Saved model_latest:", MODEL_LATEST_PATH, flush=True)
    print("Saved initial model_best:", MODEL_BEST_PATH, flush=True)
    print("=" * 70, flush=True)



# ============================================================
# MCTS
# ============================================================


def terminal_value_for_side_to_move(board: chess.Board) -> float:
    # Full/slow terminal helper. Use outside the hot MCTS loop.
    if board.is_checkmate():
        return -1.0

    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.can_claim_fifty_moves()
        or board.can_claim_threefold_repetition()
    ):
        return 0.0

    result = board.result(claim_draw=True)

    if result == "1-0":
        return 1.0 if board.turn == chess.WHITE else -1.0
    if result == "0-1":
        return 1.0 if board.turn == chess.BLACK else -1.0

    return 0.0


def mcts_terminal_value_for_side_to_move(board: chess.Board):
    # Hot-loop MCTS terminal check. Do not use claim_draw=True here.
    if board.is_checkmate():
        return -1.0
    if board.is_stalemate() or board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 100:
        return 0.0
    return None


class MCTSNode:
    __slots__ = ("board", "parent", "move", "prior", "children", "N", "W", "terminal_value")

    def __init__(self, board=None, parent=None, move=None, prior=0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.prior = float(prior)
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.terminal_value = (mcts_terminal_value_for_side_to_move(board) if board is not None else None)

    @property
    def Q(self):
        return 0.0 if self.N == 0 else self.W / self.N

    def is_expanded(self):
        return len(self.children) > 0


def materialize_node_board(node: MCTSNode) -> chess.Board:
    if node.board is None:
        if node.parent is None or node.move is None:
            raise RuntimeError("Cannot materialize MCTS node without parent/move.")

        parent_board = materialize_node_board(node.parent)
        board = parent_board.copy(stack=False)
        board.push(node.move)

        node.board = board
        node.terminal_value = mcts_terminal_value_for_side_to_move(board)

    return node.board


def select_child_puct(node, c_puct):
    best_score = -1e9
    best_child = None
    sqrt_N = math.sqrt(max(1, node.N))

    for child in node.children.values():
        q_value = -child.Q
        u_value = c_puct * child.prior * sqrt_N / (1 + child.N)
        score = q_value + u_value

        if score > best_score:
            best_score = score
            best_child = child

    return best_child


def select_leaf(root, c_puct):
    node = root

    while node.is_expanded() and node.terminal_value is None:
        node = select_child_puct(node, c_puct=c_puct)
        materialize_node_board(node)

    return node


def backup(node, value):
    while node is not None:
        node.N += 1
        node.W += value
        value = -value
        node = node.parent


def add_dirichlet_noise_to_root(root, alpha, epsilon):
    moves = list(root.children.keys())
    if len(moves) == 0:
        return

    noise = np.random.dirichlet([alpha] * len(moves))

    for move, n in zip(moves, noise):
        child = root.children[move]
        child.prior = (1.0 - epsilon) * child.prior + epsilon * float(n)


def softmax_np(logits):
    logits = logits.astype(np.float32, copy=False)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    total = exp_logits.sum()

    if total <= 0 or not np.isfinite(total):
        return np.ones_like(logits, dtype=np.float32) / len(logits)

    return (exp_logits / total).astype(np.float32, copy=False)


def expand_node_with_policy_logits(node, policy_logits_np):
    board = materialize_node_board(node)

    if node.terminal_value is not None or node.is_expanded():
        return

    legal_moves = list(board.legal_moves)

    if len(legal_moves) == 0:
        node.terminal_value = 0.0
        return

    legal_indices = np.fromiter((move_to_id(move, board) for move in legal_moves), dtype=np.int64, count=len(legal_moves))

    priors = softmax_np(policy_logits_np[legal_indices])

    # Lazy children: no child Board copies here.
    for move, prior in zip(legal_moves, priors):
        node.children[move] = MCTSNode(board=None, parent=node, move=move, prior=float(prior))



def root_visit_policy(root):
    board = root.board
    pi = np.zeros(NUM_MOVES, dtype=np.float32)

    moves = list(root.children.keys())
    visits = np.array([root.children[m].N for m in moves], dtype=np.float32)

    if len(moves) == 0:
        return pi, [], np.array([], dtype=np.float32)

    if visits.sum() <= 0:
        probs = np.ones(len(moves), dtype=np.float32) / len(moves)
    else:
        probs = visits / visits.sum()

    for move, p in zip(moves, probs):
        pi[move_to_id(move, board)] = p

    return pi, moves, probs


def legal_mask_from_board(board: chess.Board) -> np.ndarray:
    mask = np.zeros(NUM_MOVES, dtype=np.bool_)

    for move in board.legal_moves:
        mask[move_to_id(move, board)] = True

    return mask


def sample_move_from_probs(moves, probs, temperature):
    if len(moves) == 0:
        return None

    if temperature == 0.0:
        return moves[int(np.argmax(probs))]

    probs = probs ** (1.0 / temperature)

    if probs.sum() <= 0 or np.isnan(probs).any():
        probs = np.ones(len(moves), dtype=np.float32) / len(moves)
    else:
        probs = probs / probs.sum()

    idx = np.random.choice(len(moves), p=probs)
    return moves[idx]


# ============================================================
# Parallel Inference Server / Remote MCTS
# ============================================================


def expand_node_with_legal_priors(node, legal_moves, priors):
    """
    Worker-side expansion when the inference server returns already-normalized
    priors over this node's legal moves.

    legal_moves and priors must have the same order.
    """
    board = materialize_node_board(node)

    if node.terminal_value is not None or node.is_expanded():
        return

    if len(legal_moves) == 0:
        node.terminal_value = 0.0
        return

    for move, prior in zip(legal_moves, priors):
        node.children[move] = MCTSNode(board=None, parent=node, move=move, prior=float(prior))


class RemoteInferenceClient:
    """
    CPU-worker side client.

    Sends leaf tensors + legal move indices to the GPU inference server.
    Receives legal-move priors + value predictions.
    """

    def __init__(self, worker_id, request_q, response_q):
        self.worker_id = int(worker_id)
        self.request_q = request_q
        self.response_q = response_q
        self.next_req_id = 0
        self.pending = {}

    def _get_response(self, req_id):
        if req_id in self.pending:
            return self.pending.pop(req_id)

        while True:
            msg = self.response_q.get()

            tag = msg[0]

            if tag == "result":
                _, got_req_id, priors_batch, value_batch = msg

                if got_req_id == req_id:
                    return priors_batch, value_batch

                self.pending[got_req_id] = (priors_batch, value_batch)

            elif tag == "error":
                _, got_req_id, err_text = msg
                raise RuntimeError(
                    f"Inference server error for worker={self.worker_id}, "
                    f"req={got_req_id}:\n{err_text}"
                )

            else:
                raise RuntimeError(f"Unexpected response tag: {tag}")

    def evaluate_and_expand(self, nodes):
        if len(nodes) == 0:
            return []

        values = [None] * len(nodes)

        eval_indices = []
        eval_tensors = []
        legal_moves_batch = []
        legal_indices_batch = []

        for idx, node in enumerate(nodes):
            board = materialize_node_board(node)

            if node.terminal_value is not None:
                values[idx] = float(node.terminal_value)
                continue

            legal_moves = list(board.legal_moves)

            if len(legal_moves) == 0:
                node.terminal_value = 0.0
                values[idx] = 0.0
                continue

            legal_indices = np.fromiter((move_to_id(move, board) for move in legal_moves), dtype=np.int64, count=len(legal_moves))

            eval_indices.append(idx)
            eval_tensors.append(board_to_tensor(board))
            legal_moves_batch.append(legal_moves)
            legal_indices_batch.append(legal_indices)

        if len(eval_indices) == 0:
            return values

        x_np = np.stack(eval_tensors).astype(np.float32, copy=False)

        req_id = self.next_req_id
        self.next_req_id += 1

        self.request_q.put(("eval", self.worker_id, req_id, x_np, legal_indices_batch))

        priors_batch, value_batch = self._get_response(req_id)

        for batch_idx, node_idx in enumerate(eval_indices):
            node = nodes[node_idx]

            expand_node_with_legal_priors(node=node, legal_moves=legal_moves_batch[batch_idx], priors=priors_batch[batch_idx])

            values[node_idx] = float(value_batch[batch_idx])

        return values


def run_mcts_batch_remote(boards, infer_client, num_simulations, c_puct, add_noise=False, dirichlet_alpha=0.3, dirichlet_epsilon=0.25):
    """
    Same logical MCTS as run_mcts_batch, but NN evaluation is sent to
    the inference server instead of using a local model.
    """
    roots = [MCTSNode(board.copy(stack=False)) for board in boards]

    infer_client.evaluate_and_expand(roots)

    if add_noise:
        for root in roots:
            add_dirichlet_noise_to_root(root, alpha=dirichlet_alpha, epsilon=dirichlet_epsilon)

    for _ in range(num_simulations):
        leaves = [select_leaf(root, c_puct=c_puct) for root in roots]
        values = infer_client.evaluate_and_expand(leaves)

        for leaf, value in zip(leaves, values):
            backup(leaf, value)

    return roots

@torch.no_grad()
def inference_server_loop(model_state_dict_cpu, request_q, response_queues, device, server_batch_size, timeout_ms):
    """
    Owns the CUDA model.

    Receives requests:
        ("eval", worker_id, req_id, x_np, legal_indices_batch)

    Sends responses:
        ("result", req_id, priors_batch, value_batch)
    """
    torch.set_num_threads(1)

    device = torch.device(device)

    model = ChessNet().to(device)
    model.load_state_dict(model_state_dict_cpu)
    model.eval()

    timeout_s = float(timeout_ms) / 1000.0

    pending = []

    def flush_pending():
        nonlocal pending

        if len(pending) == 0:
            return

        x_parts = [item["x_np"] for item in pending]
        x_np = np.concatenate(x_parts, axis=0).astype(np.float32, copy=False)

        x = torch.from_numpy(x_np).to(device, non_blocking=True)

        use_cuda = device.type == "cuda"

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            policy_logits_batch, value_batch = model(x)

        policy_logits_np = policy_logits_batch.detach().float().cpu().numpy()
        value_np = value_batch.detach().float().view(-1).cpu().numpy()

        offset = 0

        for item in pending:
            worker_id = item["worker_id"]
            req_id = item["req_id"]
            legal_indices_batch = item["legal_indices_batch"]
            n = item["n"]

            priors_batch = []

            for j in range(n):
                legal_indices = legal_indices_batch[j]
                legal_logits = policy_logits_np[offset + j, legal_indices]
                priors = softmax_np(legal_logits)
                priors_batch.append(priors)

            values = value_np[offset:offset + n].copy()

            response_queues[worker_id].put((
                "result",
                req_id,
                priors_batch,
                values,
            ))

            offset += n

        pending = []

    while True:
        try:
            if len(pending) == 0:
                msg = request_q.get()
            else:
                msg = request_q.get(timeout=timeout_s)
        except pyqueue.Empty:
            flush_pending()
            continue

        tag = msg[0]

        if tag == "stop":
            flush_pending()
            break

        if tag != "eval":
            continue

        _, worker_id, req_id, x_np, legal_indices_batch = msg

        pending.append({
            "worker_id": int(worker_id),
            "req_id": int(req_id),
            "x_np": x_np,
            "legal_indices_batch": legal_indices_batch,
            "n": int(x_np.shape[0]),
        })

        total_pending = sum(item["n"] for item in pending)

        if total_pending >= server_batch_size:
            flush_pending()


# ============================================================
# Self-Play
# ============================================================


def finalize_self_play_data(game_history, result):
    final_data = []

    for item in game_history:
        z = value_for_side_to_move(item["board"], result)
        final_data.append((item["x"], item["pi"], np.float32(z), item["legal_mask"]))

    return final_data


def save_selfplay_partial(path, iteration, num_games, num_simulations, completed_games, worker_vector_games=None):
    payload = {
        "iteration": int(iteration),
        "num_games": int(num_games),
        "num_simulations": int(num_simulations),
        "worker_vector_games": int(worker_vector_games) if worker_vector_games is not None else None,
        "mode": "parallel_inference",
        "completed_games": completed_games,
        "saved_at": time.time(),
    }

    tmp_path = Path(str(path) + ".tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_matching_selfplay_partial(path, iteration, num_games, num_simulations, worker_vector_games=None):
    path = Path(path)

    if not path.exists():
        return []

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Could not load partial self-play checkpoint {path}: {e}", flush=True)
        return []

    if (ckpt.get("iteration") != int(iteration) or ckpt.get("num_games") != int(num_games) or ckpt.get("num_simulations") != int(num_simulations) 
            or ckpt.get("mode") != "parallel_inference" or ckpt.get("worker_vector_games") != (int(worker_vector_games) if worker_vector_games is not None else None)):
        return []

    completed = ckpt.get("completed_games", [])

    print(f"Loaded partial parallel self-play: {len(completed)}/{num_games} games", flush=True)

    return completed



def selfplay_worker_loop(worker_id, game_indices, request_q, response_q, result_q, num_simulations, worker_vector_games, iteration):
    """
    CPU-only self-play worker.

    It owns MCTS trees and chess.Board objects.
    It does NOT own a CUDA model.
    """
    try:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["NUMEXPR_NUM_THREADS"] = "1"

        torch.set_num_threads(1)

        seed = (int(time.time() * 1000) + 100003 * int(worker_id) + 1009 * int(iteration)) % (2**32 - 1)

        random.seed(seed)
        np.random.seed(seed)

        infer_client = RemoteInferenceClient(worker_id=worker_id, request_q=request_q, response_q=response_q)

        for batch_start in range(0, len(game_indices), worker_vector_games):
            batch_game_indices = game_indices[batch_start:batch_start + worker_vector_games]

            boards = [chess.Board() for _ in batch_game_indices]
            game_histories = [[] for _ in batch_game_indices]
            done = [False for _ in batch_game_indices]

            def register_completed_game(local_idx, result):
                data = finalize_self_play_data(game_histories[local_idx], result)

                payload = {
                    "game_idx": int(batch_game_indices[local_idx]),
                    "data": data,
                    "result": result,
                }

                result_q.put(("game", int(worker_id), payload))

            for ply in range(MAX_GAME_MOVES):
                active_local_indices = []

                for local_idx, board in enumerate(boards):
                    if done[local_idx]:
                        continue

                    if board.is_game_over(claim_draw=True):
                        result = board.result(claim_draw=True)
                        done[local_idx] = True
                        register_completed_game(local_idx, result)
                    else:
                        active_local_indices.append(local_idx)

                if len(active_local_indices) == 0:
                    break

                active_boards = [boards[local_idx] for local_idx in active_local_indices]

                roots = run_mcts_batch_remote(boards=active_boards, infer_client=infer_client, num_simulations=num_simulations, c_puct=SELFPLAY_C_PUCT, add_noise=True,
                    dirichlet_alpha=ROOT_DIRICHLET_ALPHA, dirichlet_epsilon=ROOT_DIRICHLET_EPSILON)

                for local_idx, root in zip(active_local_indices, roots):
                    if done[local_idx]:
                        continue

                    board = boards[local_idx]
                    pi, moves, probs = root_visit_policy(root)

                    if len(moves) == 0:
                        result = (board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2")
                        done[local_idx] = True
                        register_completed_game(local_idx, result)
                        continue

                    game_histories[local_idx].append({
                        "x": board_to_tensor(board),
                        "pi": pi,
                        "legal_mask": legal_mask_from_board(board),
                        "board": board.copy(stack=False),
                    })

                    temperature = 0.8 if ply < TEMP_PLIES else 0.0
                    move = sample_move_from_probs(moves=moves, probs=probs, temperature=temperature)

                    if move is None:
                        result = "1/2-1/2"
                        done[local_idx] = True
                        register_completed_game(local_idx, result)
                        continue

                    board.push(move)

                    if board.is_game_over(claim_draw=True):
                        result = board.result(claim_draw=True)
                        done[local_idx] = True
                        register_completed_game(local_idx, result)

            for local_idx, board in enumerate(boards):
                if done[local_idx]:
                    continue

                if board.is_game_over(claim_draw=True):
                    result = board.result(claim_draw=True)
                else:
                    result = "1/2-1/2"

                done[local_idx] = True
                register_completed_game(local_idx, result)

        result_q.put(("worker_done", int(worker_id)))

    except Exception:
        result_q.put((
            "worker_error",
            int(worker_id),
            traceback.format_exc(),
        ))


def generate_self_play_games_parallel_inference(model, replay_buffer, num_games, num_simulations, iteration=None, num_workers=SELFPLAY_WORKERS,
    worker_vector_games=WORKER_VECTOR_GAMES, server_batch_size=INFER_SERVER_BATCH_SIZE, server_timeout_ms=INFER_SERVER_TIMEOUT_MS):
    """
    Parallel CPU self-play workers + one GPU inference server.

    This is still synchronous at the RL level:
        generate all games -> return -> train
    """
    if iteration is None:
        iteration = -1

    num_workers = int(min(num_workers, num_games))
    num_workers = max(1, num_workers)

    print(
        f"Parallel self-play + inference server: "
        f"games={num_games}, workers={num_workers}, "
        f"worker_vector_games={worker_vector_games}, sims={num_simulations}, "
        f"server_batch={server_batch_size}, timeout_ms={server_timeout_ms}, "
        f"device={DEVICE}",
        flush=True,
    )

     # ---------------------------------------------------------
    # Load partial completed games for this iteration, if any.
    # These are finalized games from a previous interrupted run.
    # ---------------------------------------------------------

    completed_games = load_matching_selfplay_partial(SELFPLAY_PARTIAL_PATH, iteration=iteration, num_games=num_games, num_simulations=num_simulations,
        worker_vector_games=worker_vector_games)

    completed_indices = set()
    total_positions = 0
    new_positions = 0

    for item in completed_games:
        game_idx = int(item["game_idx"])

        if game_idx in completed_indices:
            continue

        completed_indices.add(game_idx)

        data = item["data"]
        replay_buffer.extend(data)

        positions = len(data)
        total_positions += positions
        new_positions += positions

    if len(completed_games) >= num_games:
        print("All self-play games were already completed in partial checkpoint.", flush=True)

        completed_games = sorted(completed_games, key=lambda x: x["game_idx"])
        results = [item["result"] for item in completed_games]

        counts = Counter(results)

        print("Parallel self-play result counts:", {
            "1-0": counts.get("1-0", 0),
            "0-1": counts.get("0-1", 0),
            "1/2-1/2": counts.get("1/2-1/2", 0),
        }, flush=True)

        print("New positions this iteration:", new_positions, flush=True)
        print("Buffer size:", len(replay_buffer), flush=True)

        return results, new_positions

    ctx = mp.get_context("spawn")

    request_q = ctx.Queue(maxsize=max(64, num_workers * 8))
    result_q = ctx.Queue()
    response_queues = [ctx.Queue(maxsize=32) for _ in range(num_workers)]

    model_state_dict_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    server_proc = ctx.Process(target=inference_server_loop, args=(model_state_dict_cpu, request_q, response_queues, str(DEVICE), int(server_batch_size), int(server_timeout_ms)))

    server_proc.start()

    remaining_indices = [i for i in range(num_games) if i not in completed_indices]

    if len(remaining_indices) == 0:
        game_chunks = [[] for _ in range(num_workers)]
    else:
        num_workers = int(min(num_workers, len(remaining_indices)))
        num_workers = max(1, num_workers)
        game_chunks = [remaining_indices[i::num_workers] for i in range(num_workers)]

    workers = []

    for worker_id in range(num_workers):
        p = ctx.Process(target=selfplay_worker_loop, args=(worker_id, game_chunks[worker_id], request_q, response_queues[worker_id], result_q, int(num_simulations),
                int(worker_vector_games), int(iteration)))
        p.start()
        workers.append(p)

    workers_done = 0
    last_partial_save_time = time.time()
    last_partial_save_games = len(completed_games)

    pbar = tqdm(total=num_games, initial=len(completed_games), desc="Parallel self-play", file=sys.stdout, dynamic_ncols=True)

    start_time = time.time()

    try:
        while len(completed_games) < num_games:
            msg = result_q.get()
            tag = msg[0]

            if tag == "game":
                _, worker_id, item = msg

                game_idx = int(item["game_idx"])

                if game_idx in completed_indices:
                    continue

                completed_indices.add(game_idx)
                completed_games.append(item)

                data = item["data"]
                replay_buffer.extend(data)

                positions = len(data)
                total_positions += positions
                new_positions += positions

                now = time.time()

                should_save_partial = (len(completed_games) - last_partial_save_games >= SELFPLAY_PARTIAL_SAVE_EVERY_GAMES or now - last_partial_save_time >= SELFPLAY_PARTIAL_SAVE_EVERY_SECONDS
                    or len(completed_games) >= num_games)

                if should_save_partial:
                    save_selfplay_partial(SELFPLAY_PARTIAL_PATH, iteration=iteration, num_games=num_games, num_simulations=num_simulations, completed_games=completed_games,
                        worker_vector_games=worker_vector_games)

                    print(f"\nSaved partial self-play checkpoint: " f"{len(completed_games)}/{num_games} games -> {SELFPLAY_PARTIAL_PATH}", flush=True)

                    last_partial_save_time = now
                    last_partial_save_games = len(completed_games)

                results_so_far = [x["result"] for x in completed_games]
                counts = Counter(results_so_far)

                elapsed = max(1e-9, time.time() - start_time)
                pps = total_positions / elapsed

                pbar.update(1)
                pbar.set_postfix({
                    "pos": total_positions,
                    "pps": f"{pps:.1f}",
                    "1-0": counts.get("1-0", 0),
                    "0-1": counts.get("0-1", 0),
                    "draw": counts.get("1/2-1/2", 0),
                    "buffer": len(replay_buffer),
                })

            elif tag == "worker_done":
                workers_done += 1

                if workers_done == num_workers and len(completed_games) < num_games:
                    raise RuntimeError(f"All workers finished but only got " f"{len(completed_games)}/{num_games} games.")

            elif tag == "worker_error":
                _, worker_id, err_text = msg
                raise RuntimeError(f"Self-play worker {worker_id} failed:\n{err_text}")

            else:
                raise RuntimeError(f"Unexpected result_q message: {tag}")

    finally:
        pbar.close()

        for p in workers:
            p.join(timeout=5)

        for p in workers:
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)

        request_q.put(("stop",))

        server_proc.join(timeout=20)

        if server_proc.is_alive():
            server_proc.terminate()
            server_proc.join(timeout=5)

    save_selfplay_partial(SELFPLAY_PARTIAL_PATH, iteration=iteration, num_games=num_games, num_simulations=num_simulations, completed_games=completed_games,
        worker_vector_games=worker_vector_games)

    print(f"Saved final partial self-play checkpoint before training: " f"{len(completed_games)}/{num_games} games -> {SELFPLAY_PARTIAL_PATH}", flush=True)

    completed_games = sorted(completed_games, key=lambda x: x["game_idx"])
    results = [item["result"] for item in completed_games]

    counts = Counter(results)

    print("Parallel self-play result counts:", {
        "1-0": counts.get("1-0", 0),
        "0-1": counts.get("0-1", 0),
        "1/2-1/2": counts.get("1/2-1/2", 0),
    }, flush=True)

    print("New positions this iteration:", new_positions, flush=True)
    print("Buffer size:", len(replay_buffer), flush=True)

    return results, new_positions


# ============================================================
# RL Training
# ============================================================


def get_rl_train_steps(new_positions):
    """
    Training-ratio schedule.

    Instead of training for 1.5 epochs over the whole replay buffer, train for
    a number of random mini-batch updates proportional to the number of new
    self-play positions generated in the current iteration.
    """
    if new_positions <= 0:
        return 0

    steps = int(math.ceil(RL_REUSE_FACTOR * new_positions / RL_BATCH_SIZE))
    steps = max(RL_TRAIN_MIN_STEPS, steps)
    steps = min(RL_TRAIN_MAX_STEPS, steps)

    return steps


def sample_batch_from_replay_buffer(replay_buffer_list, batch_size):
    """
    Sample one random mini-batch from replay buffer.

    New masked-policy samples have format (x, pi, z, legal_mask). Older
    samples with format (x, pi, z) are still accepted and receive an all-True
    mask, which exactly preserves the old full-softmax behavior for them.
    """
    if len(replay_buffer_list) >= batch_size:
        batch = random.sample(replay_buffer_list, batch_size)
    else:
        batch = random.choices(replay_buffer_list, k=batch_size)

    xs = []
    pis = []
    zs = []
    legal_masks = []

    for sample in batch:
        if len(sample) == 4:
            x, pi, z, legal_mask = sample
        elif len(sample) == 3:
            # Backward compatibility for older replay buffers/checkpoints.
            # All moves are treated as allowed, reproducing the old loss.
            x, pi, z = sample
            legal_mask = np.ones(NUM_MOVES, dtype=np.bool_)
        else:
            raise ValueError(f"Unexpected self-play sample format: len={len(sample)}")

        xs.append(x)
        pis.append(pi)
        zs.append(z)
        legal_masks.append(legal_mask)

    X = torch.from_numpy(np.stack(xs).astype(np.float32))
    pi = torch.from_numpy(np.stack(pis).astype(np.float32))
    z = torch.tensor(zs, dtype=torch.float32)
    legal_mask = torch.from_numpy(np.stack(legal_masks).astype(np.bool_))

    return X, pi, z, legal_mask


def train_rl_fixed_steps(model, replay_buffer, optimizer, device, batch_size, new_positions):
    model.train()

    if len(replay_buffer) == 0:
        raise RuntimeError("Replay buffer is empty. Generate self-play games first.")

    train_steps = get_rl_train_steps(new_positions)

    if train_steps <= 0:
        raise RuntimeError("Train steps is 0. No new self-play positions were generated.")

    # Convert deque to list once. This avoids repeatedly indexing the deque.
    # The list stores references to existing samples, not deep copies of arrays.
    replay_buffer_list = list(replay_buffer)

    print(
        f"RL training fixed steps: steps={train_steps}, "
        f"batch_size={batch_size}, buffer={len(replay_buffer)}, "
        f"new_positions={new_positions}, reuse_factor={RL_REUSE_FACTOR}",
        flush=True,
    )

    total_loss = 0.0
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0

    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    pbar = tqdm(range(train_steps), desc="RL training", file=sys.stdout, dynamic_ncols=True)

    for _ in pbar:
        X, pi, z, legal_mask = sample_batch_from_replay_buffer(replay_buffer_list, batch_size)

        X = X.to(device, non_blocking=True)
        pi = pi.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)
        legal_mask = legal_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            policy_logits, value_pred = model(X)

            if MASKED_POLICY_LOSS:
                # Match MCTS inference: normalize policy only over legal moves.
                # -1e4 is safer than -1e9 under fp16/autocast and still gives
                # practically zero probability to illegal moves.
                masked_logits = policy_logits.masked_fill(~legal_mask, POLICY_MASK_VALUE)
                log_probs = F.log_softmax(masked_logits, dim=1)
            else:
                log_probs = F.log_softmax(policy_logits, dim=1)

            policy_loss = -(pi * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_pred, z)
            loss = policy_loss + VALUE_LOSS_WEIGHT * value_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        n = X.size(0)

        total_loss += loss.item() * n
        total_policy_loss += policy_loss.item() * n
        total_value_loss += value_loss.item() * n
        total_samples += n

        pbar.set_postfix({
            "loss": f"{total_loss / max(1, total_samples):.4f}",
            "pol": f"{total_policy_loss / max(1, total_samples):.4f}",
            "val": f"{total_value_loss / max(1, total_samples):.4f}",
            "samples": total_samples,
        })

    return {
        "loss": total_loss / max(1, total_samples),
        "policy_loss": total_policy_loss / max(1, total_samples),
        "value_loss": total_value_loss / max(1, total_samples),
        "samples": total_samples,
        "train_steps": train_steps,
        "buffer_size": len(replay_buffer),
        "new_positions": new_positions,
        "reuse_factor": RL_REUSE_FACTOR,
        "masked_policy_loss": MASKED_POLICY_LOSS,
    }


def train_rl(fresh_rl_from_sl=False):
    print("Device:", DEVICE, flush=True)
    print("Project:", PROJECT_DIR, flush=True)
    print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID", "not_in_slurm"), flush=True)
    print("SLURM_CPUS_PER_TASK:", os.environ.get("SLURM_CPUS_PER_TASK", "unset"), flush=True)
    print("SLURM_GPUS:", os.environ.get("SLURM_GPUS", "unset"), flush=True)
    print("Worker active games:", WORKER_VECTOR_GAMES, flush=True)
    print("DataLoader workers:", NUM_WORKERS, flush=True)
    print("Arena gate: disabled", flush=True)

    model = ChessNet().to(DEVICE)

    replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
    selfplay_history = []
    train_history = []
    start_iteration = 0

    # =========================================================
    # Resume RL if full RL checkpoint exists
    # =========================================================

    if RL_LATEST_PATH.exists() and not fresh_rl_from_sl:
        print("Resuming RL from:", RL_LATEST_PATH, flush=True)

        ckpt = torch.load(RL_LATEST_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        replay_buffer.extend(ckpt.get("replay_buffer", []))
        selfplay_history = ckpt.get("selfplay_history", [])
        train_history = ckpt.get("train_history", [])
        start_iteration = int(ckpt.get("iteration", 0))

        optimizer = torch.optim.AdamW(model.parameters(), lr=RL_LR, weight_decay=WEIGHT_DECAY)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("Loaded optimizer state.", flush=True)

        print("Resumed from iteration:", start_iteration, flush=True)
        print("Loaded replay buffer size:", len(replay_buffer), flush=True)

    # =========================================================
    # Otherwise start RL from latest/SL model
    # =========================================================

    else:
        if fresh_rl_from_sl:
            print("Fresh RL requested. Ignoring old rl_latest.pt if it exists.", flush=True)

        if fresh_rl_from_sl:
            candidate_paths = [MODEL_AFTER_SL_PATH, SL_LATEST_PATH, MODEL_LATEST_PATH]
        else:
            candidate_paths = [MODEL_LATEST_PATH, MODEL_AFTER_SL_PATH, SL_LATEST_PATH]

        ckpt = None
        start_path = None

        for path in candidate_paths:
            if path.exists():
                start_path = path
                print("Starting RL from:", start_path, flush=True)
                ckpt = torch.load(start_path, map_location=DEVICE, weights_only=False)
                break

        if ckpt is None:
            raise RuntimeError("No checkpoint found. Run supervised training first or copy a model checkpoint.")

        model.load_state_dict(ckpt["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=RL_LR, weight_decay=WEIGHT_DECAY)

    # =========================================================
    # RL Loop
    # =========================================================

    for iteration in range(start_iteration, start_iteration + RL_ITERATIONS):
        current_iter = iteration + 1

        print()
        print("=" * 70, flush=True)
        print(f"RL Iteration {current_iter}", flush=True)
        print(f"This run target: {start_iteration + RL_ITERATIONS}", flush=True)
        print("=" * 70, flush=True)

        # -----------------------------------------------------
        # 1. Vectorized self-play from the current/latest model
        # -----------------------------------------------------

        model.eval()

        selfplay_results, new_positions = generate_self_play_games_parallel_inference(model=model, replay_buffer=replay_buffer, num_games=GAMES_PER_ITERATION,
            num_simulations=SELFPLAY_SIMS, iteration=current_iter, num_workers=SELFPLAY_WORKERS, worker_vector_games=WORKER_VECTOR_GAMES,
            server_batch_size=INFER_SERVER_BATCH_SIZE, server_timeout_ms=INFER_SERVER_TIMEOUT_MS)

        result_counts = Counter(selfplay_results)

        selfplay_summary = {
            "iteration": current_iter,
            "new_positions": new_positions,
            "games": len(selfplay_results),
            "results": {
                "1-0": result_counts.get("1-0", 0),
                "0-1": result_counts.get("0-1", 0),
                "1/2-1/2": result_counts.get("1/2-1/2", 0),
            },
            "buffer_size": len(replay_buffer),
        }
        selfplay_history.append(selfplay_summary)

        print("Self-play summary:", selfplay_summary, flush=True)

        # -----------------------------------------------------
        # 2. Train current model on replay buffer
        # -----------------------------------------------------

        metrics = train_rl_fixed_steps(model=model, replay_buffer=replay_buffer, optimizer=optimizer, device=DEVICE, batch_size=RL_BATCH_SIZE, new_positions=new_positions)

        train_history.append({
            "iteration": current_iter,
            "metrics": metrics,
        })

        print("Train metrics:", metrics, flush=True)

        # -----------------------------------------------------
        # 3. Save full latest RL checkpoint WITH replay buffer
        # -----------------------------------------------------

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": current_iter,
            "metrics": metrics,
            "selfplay_summary": selfplay_summary,
            "buffer_size": len(replay_buffer),
            "replay_buffer": list(replay_buffer),
            "selfplay_history": selfplay_history,
            "train_history": train_history,
            "config": {
                "cluster": "nefeli",
                "partition_recommendation": "a100",
                "gpus_recommendation": 1,
                "cpus_recommendation": 32,
                "replay_buffer_size": REPLAY_BUFFER_SIZE,
                "games_per_iteration": GAMES_PER_ITERATION,
                "selfplay_sims": SELFPLAY_SIMS,
                "worker_vector_games": WORKER_VECTOR_GAMES,
                "batch_size": RL_BATCH_SIZE,
                "rl_lr": RL_LR,
                "rl_reuse_factor": RL_REUSE_FACTOR,
                "rl_train_min_steps": RL_TRAIN_MIN_STEPS,
                "rl_train_max_steps": RL_TRAIN_MAX_STEPS,
                "masked_policy_loss": MASKED_POLICY_LOSS,
                "policy_mask_value": POLICY_MASK_VALUE,
                "selfplay_c_puct": SELFPLAY_C_PUCT,
                "max_game_moves": MAX_GAME_MOVES,
                "temp_plies": TEMP_PLIES,
                "root_dirichlet_alpha": ROOT_DIRICHLET_ALPHA,
                "root_dirichlet_epsilon": ROOT_DIRICHLET_EPSILON,
                "arena_gate": False,
                "parallel_selfplay": PARALLEL_SELFPLAY,
                "selfplay_workers": SELFPLAY_WORKERS,
                "worker_vector_games": WORKER_VECTOR_GAMES,
                "infer_server_batch_size": INFER_SERVER_BATCH_SIZE,
                "infer_server_timeout_ms": INFER_SERVER_TIMEOUT_MS,
                "vectorized_selfplay": not PARALLEL_SELFPLAY,
                "dataloader_workers": NUM_WORKERS,
            },
        }

        torch.save(checkpoint, RL_LATEST_PATH)
        print("Saved full RL checkpoint:", RL_LATEST_PATH, flush=True)

        if SELFPLAY_PARTIAL_PATH.exists():
            SELFPLAY_PARTIAL_PATH.unlink()
            print("Deleted completed partial self-play checkpoint:", SELFPLAY_PARTIAL_PATH, flush=True)

        # -----------------------------------------------------
        # 4. Save archival checkpoint WITHOUT replay buffer
        # -----------------------------------------------------

        if current_iter % 5 == 0:
            eval_path = RL_MODEL_DIR / f"model_iter_{current_iter}.pt"

            save_model_only(model, eval_path,
                extra={
                    "source": "rl_eval_checkpoint_vectorized_no_arena",
                    "iteration": current_iter,
                    "metrics": metrics,
                    "selfplay_summary": selfplay_summary,
                    "config": checkpoint["config"],
                },
            )

            print("Saved lightweight eval checkpoint:", eval_path, flush=True)
        # -----------------------------------------------------
        # 5. Save latest NN-only checkpoint
        # -----------------------------------------------------

        save_model_only(model, MODEL_LATEST_PATH,
            extra={
                "source": "rl_latest_vectorized_no_arena",
                "iteration": current_iter,
                "metrics": metrics,
                "selfplay_summary": selfplay_summary,
            },
        )

        print("Saved latest NN:", MODEL_LATEST_PATH, flush=True)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# CLI
# ============================================================


def main(args):
    if args.command == "prepare_sl":
        prepare_sl_shards_from_pgn(pgn_path=args.pgn, shard_size=args.sl_shard_size, value_min_ply=args.sl_value_min_ply, value_positions_per_game=args.sl_value_positions_per_game)

    elif args.command == "train_sl":
        train_sl(pgn_path=args.pgn, sl_epochs=args.sl_epochs, value_min_ply=args.sl_value_min_ply, value_positions_per_game=args.sl_value_positions_per_game, force_prepare=args.force_prepare)

    elif args.command == "train_rl":
        train_rl(fresh_rl_from_sl=args.fresh_rl_from_sl)
    else:
        raise ValueError("Unknown command. Use: prepare_sl, train_sl, or train_rl")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()

    parser.add_argument("command", choices=["prepare_sl", "train_sl", "train_rl"])

    parser.add_argument("--pgn", type=str, default=str(DEFAULT_PGN_PATH))

    parser.add_argument("--sl-epochs", type=int, default=SL_EPOCHS)

    parser.add_argument("--sl-shard-size", type=int, default=SL_SHARD_SIZE)

    parser.add_argument("--sl-value-min-ply", type=int, default=SL_VALUE_MIN_PLY)

    parser.add_argument("--sl-value-positions-per-game", type=int, default=SL_VALUE_POSITIONS_PER_GAME)

    parser.add_argument("--force-prepare", action="store_true")

    parser.add_argument("--fresh-rl-from-sl", action="store_true")

    args = parser.parse_args()
    main(args)



