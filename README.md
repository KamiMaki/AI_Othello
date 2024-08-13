
# AI Othello

## 簡介
AI Othello 是一個基於強化學習技術的黑白棋（Othello）策略專案。此專案使用雙重深度 Q 網絡（DDQN）演算法來訓練 AI 代理，並設計了多種 reward shaping 技術來幫助代理更有效地學習策略。針對傳統的 Othello 遊戲，我們引入了特殊規則，並提供了一個互動式環境，讓使用者能夠與 AI 代理對弈或觀察 AI 之間的對弈。

### 特殊規則
1. **角落限制**：遊戲中，四個角落（A1、A8、H1、H8）禁止落子。
2. **中心起始區域**：遊戲開始時，僅允許在中心 6x6 的區域內落子。
3. **邊界落子限制**：僅當落子可以翻轉對手的棋子時，才允許在邊界上落子。


### 專案檔案結構
- `Agent.py`：定義了基於 DDQN 演算法的 AI 代理行為邏輯，包含決策機制、reward shaping 策略和與環境的互動。
- `Game.py`：實現了包含上述特殊規則的 Othello 遊戲邏輯，負責處理遊戲狀態、規則和合法步數檢查。
- `Network.py`：定義了神經網路模型，用於動作價值（Q 值）預測。
- `ReplayBuffer.py`：實現經驗回放緩衝區，用於存儲和取樣訓練資料。
- `agent_black.pth`：訓練好的黑方 AI 代理的權重檔案。
- `agent_white.pth`：訓練好的白方 AI 代理的權重檔案。
- `render.ipynb`：互動式 Jupyter Notebook，允許用戶進行 AI 對弈或人機對弈。
- `train.ipynb`：包含使用 DDQN 演算法和 reward shaping 技術訓練 AI 代理的程式碼，展示如何應用強化學習技術進行模型訓練。

### 如何使用
1. **運行訓練**：使用 `train.ipynb` 來訓練 AI 代理，該 Notebook 包含了完整的訓練流程，包括環境設置、模型初始化、reward shaping 策略的應用、以及訓練過程的細節。
2. **互動式對弈**：打開 `render.ipynb`，您可以與 AI 進行互動，與 AI 進行對局，或觀察兩個 AI 之間的對弈。

### 要求
- Python 3.x
- 需要安裝以下主要依賴項：
  - `torch`
  - `numpy`
  - `matplotlib`
  - `jupyter`

---

# AI Othello

## Introduction
AI Othello is a project that builds an Othello (Reversi) strategy using reinforcement learning techniques. This project utilizes the Double Deep Q-Network (DDQN) algorithm to train AI agents and incorporates various reward shaping techniques to enhance the learning process. We have introduced special rules to the traditional Othello game and provide an interactive environment where users can play against AI or watch AI compete against each other.

### Special Rules
1. **Corner Restrictions**: The four corners (A1, A8, H1, H8) are not allowed for moves.
2. **Central Starting Area**: At the start of the game, moves are only allowed within the central 6x6 area.
3. **Boundary Move Restriction**: Moves on the boundary are only permitted if they result in flipping the opponent's pieces.


### Project Structure
- `Agent.py`: Defines the behavior logic of the AI agent based on the DDQN algorithm, including decision-making, reward shaping strategies, and interactions with the environment.
- `Game.py`: Implements the Othello game logic with the special rules mentioned above, handling game states, rules, and legal move validation.
- `Network.py`: Defines the neural network model used for action-value (Q-value) prediction.
- `ReplayBuffer.py`: Implements the replay buffer for storing and sampling training data.
- `agent_black.pth`: Pre-trained model weights for the black-side AI agent.
- `agent_white.pth`: Pre-trained model weights for the white-side AI agent.
- `render.ipynb`: Interactive Jupyter Notebook that allows users to experiment with AI vs AI or human vs AI matches.
- `train.ipynb`: Contains the code for training the AI agent using the DDQN algorithm and reward shaping techniques, demonstrating how to apply reinforcement learning techniques for model training.

### How to Use
1. **Training**: Use `train.ipynb` to train the AI agents. This notebook includes the entire training process, from environment setup and model initialization to applying reward shaping strategies and the training loop details.
2. **Interactive Play**: Open `render.ipynb` to interact with the AI. You can choose to play against the AI or watch two AI agents play against each other.

### Requirements
- Python 3.x
- Required dependencies:
  - `torch`
  - `numpy`
  - `matplotlib`
  - `jupyter`

