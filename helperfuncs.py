# Chess position reading and conversion to tensor
import chess
import re
import torch
import torch.nn as nn
import time

nodes = 0
eval_time = 0
outcome_time = 0
depth = 0

# Reconfigure device
try:
    device = xm.xla_device()
    print("Running on the TPU")
except:
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        print('Running on the GPU')
        torch.cuda.synchronize()
    else:
        device = torch.device('cpu')
        print('Running on the CPU')
TRANSFORM = {0.5: 0, 1.0: 1.0, 0.9167: 0.8333, 0.8333: 0.6667, 0.75: 0.5, 0.6667: 0.3333, 0.5833: 0.1667, 0.0: -1.0, 0.0833: -0.8333, 0.1667: -0.6667, 0.25: -0.5, 0.3333: -0.3333, 0.4167: -0.1667}

chr_to_num = {"k": -1, "q": -0.8333, "r": -0.6667, "b": -0.5, "n": -0.3333, "p": -0.1667, "P": 0.1667, "N": 0.3333, "B": 0.5, "R": 0.6667, "Q": 0.8333, "K": 1}

def square_to_int(sq):
    return (ord(sq[0]) - 97) * 8 + int(sq[1]) - 1

def squareint_to_square(sqint):
    return (sqint // 8, sqint % 8)

def int_to_bin(anint, pad=4):
    return [int(_) for _ in "0" * (pad - len(bin(anint)[2:])) + bin(anint)[2:]]

def fast_board_to_boardmap(board):
    # Slower than piece_map() when there are less pieces on the board, but faster (~2x) in most cases.
    boards = [[0 for _ in range(8)] for _ in range(8)]
    for square in board.pieces(chess.PAWN, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["P"]
    for square in board.pieces(chess.PAWN, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["p"]
    for square in board.pieces(chess.KNIGHT, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["N"]
    for square in board.pieces(chess.KNIGHT, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["n"]
    for square in board.pieces(chess.BISHOP, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["B"]
    for square in board.pieces(chess.BISHOP, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["b"]
    for square in board.pieces(chess.ROOK, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["R"]
    for square in board.pieces(chess.ROOK, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["r"]
    for square in board.pieces(chess.QUEEN, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["Q"]
    for square in board.pieces(chess.QUEEN, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["q"]
    for square in board.pieces(chess.KING, chess.WHITE):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["K"]
    for square in board.pieces(chess.KING, chess.BLACK):
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = chr_to_num["k"]
    return [boards]

def fast_board_to_feature(board):
    whosemove = [int(board.turn)]
    enpassqnum = board.ep_square
    can_enpassant = [0]
    if (enpassqnum is not None) and (board.has_legal_en_passant()):
        enpassqnum = (enpassqnum % 8) * 8 + (enpassqnum // 8)
        enpassqnum = int_to_bin(enpassqnum, pad=6)
        can_enpassant = [1]
    else:
        enpassqnum = int_to_bin(0, pad=6)
    castling_rights = [int(board.has_kingside_castling_rights(chess.WHITE)), int(board.has_queenside_castling_rights(chess.WHITE)), int(board.has_kingside_castling_rights(chess.BLACK)), int(board.has_queenside_castling_rights(chess.BLACK))]
    return whosemove + can_enpassant + enpassqnum + castling_rights

def lt5(board):
    # Are there less then 5 pieces on the board? If so, go to tablebase probing.
    p = 0
    p += len(board.pieces(chess.PAWN, chess.WHITE))
    if p > 3: return False
    p += len(board.pieces(chess.PAWN, chess.BLACK))
    if p > 3: return False
    p += len(board.pieces(chess.KNIGHT, chess.WHITE))
    p += len(board.pieces(chess.KNIGHT, chess.BLACK))
    p += len(board.pieces(chess.BISHOP, chess.WHITE))
    p += len(board.pieces(chess.BISHOP, chess.BLACK))
    p += len(board.pieces(chess.ROOK, chess.WHITE))
    if p > 3: return False
    p += len(board.pieces(chess.ROOK, chess.BLACK))
    if p > 3: return False
    p += len(board.pieces(chess.QUEEN, chess.WHITE))
    p += len(board.pieces(chess.QUEEN, chess.BLACK))
    return p <= 3