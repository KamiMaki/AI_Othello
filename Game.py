import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class Othello_vec():
    def __init__(self,env_nums):
        self.env_nums = env_nums
        self.row_count = 8
        self.column_count = 8
        self.board = np.zeros((self.env_nums,self.row_count,self.column_count)) #　紀錄盤面資訊
        self.illegal = [(0,0),(0,7),(7,0),(7,7)]#[0,7,56,63] # 角落禁止落子 
        self.directions = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(-1,-1)]

        # reward shaping: 佔領特定位置給予對應reward
        self.position_reward = np.array([[-100,8,5,4,4,5,8,-100],\
                                           [8,-10,-10,-6,-6,-10,-10,8],\
                                           [5,-10,0,0,0,0,-10,5],\
                                           [4,-6,0,0,0,0,-6,4],\
                                           [4,-6,0,0,0,0,-6,4],\
                                           [5,-10,0,0,0,0,-10,5],\
                                           [8,-10,-10,-6,-6,-10,-10,8],\
                                           [-100,8,5,4,4,5,8,-100]])
        
        self.round = np.zeros(self.env_nums)  # 目前回合
        self.color = np.ones(self.env_nums)*-1 # 紀錄當前下棋方的顏色
        self.action_size = 64 # 棋盤大小
        self.done = np.zeros(self.env_nums,dtype = np.bool_)  #是否結束
        
    # 重設環境，return 初始 state
    def reset(self):
        self.board = np.zeros((self.env_nums,8,8))
        self.round = np.zeros(self.env_nums)  # 目前回合
        self.color = np.ones(self.env_nums)*-1 # 紀錄當前下棋方的顏色
        self.done = np.zeros(self.env_nums,dtype = np.bool_)  #是否結束
        return self.get_state(),self.round

    # 取得當前盤面&下棋方的顏色
    def get_state(self):
        return self.board*self.color[:,None,None]
        
    # 判斷位置是否在棋盤內
    def is_in_board(self,x,y):
        if (x,y) in self.illegal:
            return False
        elif 0 <= x <= 7 and 0 <= y <= 7:
            return True
        else:
            return False
    
    # 取得當前回合&黑白各自的棋數
    def get_info(self):
        black = np.sum(self.board == -1,axis=(1,2))
        white = np.sum(self.board == 1,axis=(1,2))
        return black,white,self.round
        
    # 取得目前玩家顏色
    def get_player(self):
        return self.color

    # 如果有不能落子的盤面，變換盤面視角，更新對應的player
    def skip_round(self,skip_idx):
        self.color[skip_idx] *= -1
        
    # 輸入action，計算reward，取得next state  
    def step(self,action): 
        # 更新回合數，如果有沒成功落子的player則要再等待對手玩一回合
        self.round += 1

        # 當前回合reward 
        reward = np.zeros(self.env_nums)

        # 將對手棋子進行翻轉
        def flip(env_idx,flip):
            for p in flip:
                self.board[env_idx][p[0]][p[1]]*=-1

        # 落子後翻轉對方的棋子
        for i in np.arange(self.env_nums):
            row = action[i] // 8
            col = action[i] % 8

            self.board[i][row][col] = self.color[i]
            to_flip = []
            for d in self.directions: # 檢查每個方向
                r = row+d[0]
                c = col+d[1]
                temp_flip = []
                # 沿著同一方向將對手位置存入temp_flip
                while (self.is_in_board(r, c) and self.board[i][r][c] == self.color[i]*-1):
                    temp_flip.append((r,c))
                    r = r+d[0]
                    c = c+d[1]
                # 如果末端有自己的棋子，且有待被翻的對手棋子
                if self.is_in_board(r, c) and self.board[i][r][c] == self.color[i] and temp_flip:
                    to_flip += temp_flip
            flip(i,to_flip)   

            
            # 給予positional reward 
            reward[i] += self.position_reward[row][col]
            
            # 根據翻了對手幾顆棋給予reward
            # reward[i] += len(to_flip)
            
        # 如果會讓對手攻佔角落則扣分
        lose_corner_idx = self.get_legal_move(color=self.color*-1)[:,[1,6,8,15,48,55,57,62]].any(axis=1)
        reward[lose_corner_idx] -= 20

        # 遊戲超過16回合時如果場面小於等於5顆，給予懲罰
        if self.round[0] >=16:
            score_black,score_white,_ = self.get_info() 
            penalty_idx = ((score_black<=5) & (self.color == -1)) | ((score_white<=5) & (self.color == 1))
            reward[penalty_idx] -= 15
        
        # 如果回合過半，根據分數差給予reward
        if self.round[0] >=32:
            score_black,score_white,_ = self.get_info() 
            leading_idx = ((score_black > score_white) & (self.color == -1)) | ((score_white > score_black) & (self.color == 1))
            reward += (score_white-score_black)*self.color
        
        # 結束回合，換對方
        self.color *= -1

        # 如果結束比賽，根據勝負給予reward
        done, winner = self.get_done_score()
        if done.any():
            reward[done] += winner[done] * self.color[done] * 50 # reward of winning the game
            
        return done, self.board, reward,self.round, winner, self.color

    # 兩方都不能下的情況，提早結束遊戲
    def set_done(self,done_idx):
        self.done[done_idx] = True

    # 取得遊戲是否結束以及目前贏家
    def get_done_score(self):
        black_score = np.sum(self.board == -1,axis=(1,2))
        white_score = np.sum(self.board ==  1,axis=(1,2))
        total_score = black_score + white_score

        self.done = (total_score == 60) | self.done
        winner = np.ones(self.env_nums) # white(player 1) wins
        winner[black_score >  white_score] = -1 # black (player -1) wins
        winner[black_score == white_score] = 0 # even
        # print(f'black_score {black_score}')
        # print(f'white_score {white_score}')
        return self.done, winner
        
    # 取得當前可以下的位置
    def get_legal_move(self, color = None):
        if color is None:
            color = self.color
        
        legal_moves = np.empty((self.env_nums,self.action_size),dtype=np.bool_)
        for i in range(self.env_nums):
            
            # 棋盤6*6可落子範圍
            legal_move = np.zeros((self.column_count,self.row_count),dtype=np.bool_)
            legal_move[1:-1,1:-1] = True

            # 新增邊界位置
            border = np.where((legal_move == False) & (self.board[i] == 0))
            border = zip(list(border[0]),list(border[1]))
            
            # 移除已經落子的位置
            legal_move[self.board[i] != 0] = False

            # 找尋邊界是否存在合法位置
            for p in border: 
                if p in self.illegal:
                    continue
                for d in self.directions:
                    r = p[0] + d[0]
                    c = p[1] + d[1]
                    flag = False
                    while self.is_in_board(r, c) and self.board[i][r][c] == color[i]*-1:
                       r = r+d[0]
                       c = c+d[1]
                       flag = True
                    if self.is_in_board(r, c) and self.board[i][r][c] == color[i] and flag:
                         legal_move[p[0],p[1]] = True
            
            legal_moves[i] = legal_move.flatten()
        return legal_moves


