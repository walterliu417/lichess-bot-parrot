from os import ttyname
from typing_extensions import Self
import helperfuncs
from helperfuncs import *
import numpy as np
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

try:
    TABLEBASE = chess.syzygy.open_tablebase("/content/drive/MyDrive/parrot/tablebase_5pc")
    print("5 piece Syzygy endgame tablebase found.")
except:
    print("Could not find tablebase")
    TABLEBASE = None

TIMES_UP = -999
EXACT = 0
LOWERBOUND = 1
UPPERBOUND = 2

class Node:
    pass

class Node:

    def __init__(self, board: chess.Board, move: chess.Move | None, net: nn.Module, parent: Node | None, table: dict | None, depth=0):
        self.board = board
        self.move = move
        self.value = None
        self.parent = parent
        self.visits = 0
        self.depth = depth

        self.net = net
        self.children = []
        self.flag = None
        self.table = table

    def ucb(self, c=1.4):
        try:
            if not self.board.turn:
                return (self.value / (self.visits + 1)) + c * (np.log(self.parent.visits) / (self.visits + 1))
            elif self.board.turn:
                return (self.value / (self.visits + 1)) - c * (np.log(self.parent.visits) / (self.visits + 1))
        except:
            return self.value / (self.visits + 1)
        
    def evaluate_nn(self):
        boardlist = fast_board_to_boardmap(self.board)
        if not self.board.turn:
            boardlist = np.rot90(boardlist, 2) * -1
            boardlist = boardlist.tolist()
        pos = torch.tensor(boardlist, device=device, dtype=torch.float).reshape(1, 1, 8, 8)
        with torch.no_grad():
            return self.net.forward(pos)
    
    def evaluate_position(self):
        if TABLEBASE and lt5(self.board):
            result = TABLEBASE.probe_wdl(self.board)
            if result == 2:
                return 1
            elif result == -2:
                return 0
            elif result in [-1, 0, 1]:
                return 0.5
        outcome = self.board.result(claim_draw=True)
        if outcome != "*":
            if (outcome == "1-0") and self.board.turn:
                return 1 + max((10 - self.depth) / 10, 0)
            elif (outcome == "0-1") and self.board.turn:
                return 0 - max((10 - self.depth) / 10, 0)
            elif (outcome == "1-0") and not self.board.turn:
                return 0 - max((10 - self.depth) / 10, 0)
            elif (outcome == "0-1") and not self.board.turn:
                return 1 + max((10 - self.depth) / 10, 0)
            elif outcome == "1/2-1/2":
                return 0.5
        return None
    
    def generate_children(self):
        all_positions = []

        evaled = []
        not_evaled = []
        helperfuncs.depth = max(helperfuncs.depth, self.depth + 1)
        blm = self.board.legal_moves
        helperfuncs.nodes += blm.count()
        for move in blm:
            newboard = self.board.copy()
            newboard.push(move)
            newnode = Node(newboard, move, self.net, self, self.table, depth=self.depth + 1)
            score = newnode.evaluate_position()
            if score is not None:
                newnode.value = score
                evaled.append(newnode)
            else:
                boardlist = fast_board_to_boardmap(newboard)
                if not newboard.turn:
                    boardlist = np.rot90(boardlist, 2) * -1
                    boardlist = boardlist.tolist()
                all_positions.append(boardlist)
                not_evaled.append(newnode)
            newnode.flag = EXACT

        pos = torch.tensor(all_positions, device=device, dtype=torch.float).reshape(len(not_evaled), 1, 8, 8)
        result = self.net.forward(pos)
        for i in range(len(not_evaled)):
            not_evaled[i].value = float(result[i])
            evaled.append(not_evaled[i])
        
        self.children = evaled

    def pns(self, start_time, time_for_this_move):
        while time.time() - start_time < time_for_this_move:

            # 1. Traverse tree
            target_node = self
            while target_node.children != []:
                target_node.visits += 1
                target_node.children = sorted(target_node.children, key=lambda child: child.value)
                target_node = target_node.children[0]
            
            # 2. Expansion and simulation
            target_node.generate_children()
            target_node.visits += 1

            # 3. Backpropagation
            while True:
                if target_node.children == []:
                    target_node.value = target_node.evaluate_position()
                    if target_node.value is None:
                        target_node.value = target_node.evaluate_nn()
                else:
                    target_node.value = min(target_node.children, key=lambda child: child.value).value
                if target_node.parent is not None:
                    target_node = target_node.parent
                else:
                    break

        # 4. Select move - UBFMS
        max_visits = max(self.children, key=lambda child: child.visits)
        print(max_visits.visits)
        selected_child = max(self.children, key=lambda child: 0.3 * child.value + 0.7 * child.visits / max_visits.visits)
        print(selected_child.visits, selected_child.value)
        return selected_child
    
    def negamax(self, depth, alpha, beta, color, start_time, time_for_this_move):
        if time.time() - start_time > time_for_this_move:
            return TIMES_UP, None
        alpha_original = alpha
        try:
            ttvalue, ttmove, flag, ttdepth = self.table[self.board.fen()]
            if ttdepth >= depth:
                if flag == EXACT:
                    return ttvalue, ttmove
                elif flag == LOWERBOUND:
                    alpha = max(alpha, ttvalue)
                elif self.flag == UPPERBOUND:
                    beta = min(beta, ttvalue)

                if alpha >= beta:
                    return ttvalue, ttmove
        except:
            pass
        helperfuncs.nodes += 1
        s1 = time.time()
        self.value = self.evaluate_position()
        helperfuncs.outcome_time += (time.time() - s1)
        if self.value is not None:
            if self.value <= alpha_original:
                self.flag = UPPERBOUND
            elif self.value >= beta:
                self.flag = LOWERBOUND
            else:
                self.flag = EXACT
            return self.value * color, self.move
        if depth == 0:
            s2 = time.time()
            self.value = self.evaluate_nn()
            helperfuncs.eval_time += (time.time() - s2)

            if self.value <= alpha_original:
                self.flag = UPPERBOUND
            elif self.value >= beta:
                self.flag = LOWERBOUND
            else:
                self.flag = EXACT
            return self.value * color, self.move
        
        if self.children != []:
            evaled, not_evaled = [], []
            for child in self.children:
                if child.value is not None:
                    evaled.append(child)
                else:
                    not_evaled.append(child)
            if self.board.turn:
                evaled = sorted(evaled, key=lambda x: x.value, reverse=True)
            else:
                evaled = sorted(evaled, key=lambda x: x.value)
            self.children = evaled + not_evaled

        elif self.children == []:
            captures = []
            checks = []
            others = []
            for move in self.board.legal_moves:
                newboard = self.board.copy()
                newboard.push(move)
                newnode = Node(newboard, move, self.net, self, self.table, self.depth + 1)
                if self.board.is_capture(move):
                    captures.append(newnode)
                elif self.board.gives_check(move):
                    checks.append(newnode)
                else:
                    others.append(newnode)
            self.children = captures + checks + others

        value = -10000
        best_child = None
        for child in self.children:
            newvalue, newmove = child.negamax(depth - 1, -beta, -alpha, -color, start_time, time_for_this_move)
            if newvalue == TIMES_UP:
                return TIMES_UP, None
            if -newvalue > value:
                value = -newvalue
                best_child = child
            alpha = max(alpha, value)
            if alpha >= beta:
                break

        self.value = value
        if value <= alpha_original:
            flag = UPPERBOUND
        elif value >= beta:
            flag = LOWERBOUND
        else:
            flag = EXACT
        self.table[self.board.fen()] = (self.value, self.move, flag, depth)
        return value, best_child.move
