# Chess position reading and conversion to tensor
import chess
import re
import torch
import torch.nn as nn
import time

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

chr_to_num = {"k": 0, "q": 1, "r": 2, "b": 3, "n": 4, "p": 5, "P": 7, "N": 8, "B": 9, "R": 10, "Q": 11, "K": 12}

def square_to_int(sq):
    return (ord(sq[0]) - 97) * 8 + int(sq[1]) - 1

def squareint_to_square(sqint):
    return (sqint // 8, sqint % 8)

def int_to_bin(anint, pad=4):
    return [int(_) for _ in "0" * (pad - len(bin(anint)[2:])) + bin(anint)[2:]]

def board_to_boardlist(board):
    boards = [[[0 for _ in range(8)] for _ in range(8)] for _ in range(12)]
    for square, piece in board.piece_map().items():
        idx = squareint_to_square(square)
        boards[chr_to_num[piece.symbol()]][idx[0]][idx[1]] = 1
    return boards

def board_to_boardmap(board):
    boards = [[0.5 for _ in range(8)] for _ in range(8)]
    for square, piece in board.piece_map().items():
        idx = squareint_to_square(square)
        boards[idx[0]][idx[1]] = round((chr_to_num[piece.symbol()]) / 12, 4)
    return [boards]


def fen_to_feature_wboard_list(fen_str):
    board = re.split(" |/", fen_str)
    r = 0
    whosemove = []
    enpassqnum = board[10]
    can_enpassant = [0]
    castling_rights = [0, 0, 0, 0]
    if enpassqnum != "-":
        enpassqnum = int_to_bin(square_to_int(enpassqnum), pad=6)
        can_enpassant = [1]
    else:
        enpassqnum = int_to_bin(0, pad=6)
    for entry in board:
        if entry == "w":
            whosemove = [1]
        elif entry == "b":
            whosemove = [0]
        elif r == 9:
            if entry.find("K") != -1:
                castling_rights[0] = 1
            if entry.find("Q") != -1:
                castling_rights[1] = 1
            if entry.find("k") != -1:
                castling_rights[2] = 1
            if entry.find("q") != -1:
                castling_rights[3] = 1
        r += 1
    return whosemove + can_enpassant + enpassqnum + castling_rights

