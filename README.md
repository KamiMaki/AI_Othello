
# AI Othello

## Introduction
AI Othello is a project that builds an Othello (Reversi) strategy using reinforcement learning techniques. This project provides a complete pipeline for training and testing AI agents, and offers an interactive environment where users can play against AI or watch AI compete against each other.

### Project Structure
- `Agent.py`: Defines the behavior logic of the AI agent, including decision-making and interactions with the environment.
- `Game.py`: Implements the Othello game logic, handling game states, rules, and legal move validation.
- `Network.py`: Defines the neural network model used for action-value prediction or strategy inference.
- `ReplayBuffer.py`: Implements the replay buffer for storing and sampling training data.
- `agent_black.pth`: Pre-trained model weights for the black-side AI agent.
- `agent_white.pth`: Pre-trained model weights for the white-side AI agent.
- `render.ipynb`: Interactive Jupyter Notebook that allows users to experiment with AI vs AI or human vs AI matches.
- `train.ipynb`: Contains the code for training the AI agent, showcasing how to apply reinforcement learning techniques for model training.

### How to Use
1. **Training**: Use `train.ipynb` to train the AI agents. This notebook includes the entire training process, from environment setup and model initialization to training loop details.
2. **Interactive Play**: Open `render.ipynb` to interact with the AI. You can choose to play as black or white against the AI or watch two AI agents play against each other.

### Requirements
- Python 3.x
- Required dependencies:
  - `torch`
  - `numpy`
  - `matplotlib`
  - `jupyter`

---

# AI Othello

## 簡介
AI Othello 是一個使用強化學習技術建構的黑白棋（Othello）策略專案。此專案包含了訓練和測試 AI 代理（Agent）的完整流程，並提供了一個互動式的環境讓用戶可以與 AI 對弈或觀察 AI 之間的對弈。

### 專案檔案結構
- `Agent.py`：定義了 AI 代理的行為邏輯，包含決策機制和與環境的互動。
- `Game.py`：實現了 Othello 遊戲邏輯，負責處理遊戲狀態、規則和合法步數檢查。
- `Network.py`：定義了神經網路模型，用於預測動作價值或策略。
- `ReplayBuffer.py`：實現經驗回放緩衝區，用於存儲和取樣訓練資料。
- `agent_black.pth`：訓練好的黑方 AI 代理的權重檔案。
- `agent_white.pth`：訓練好的白方 AI 代理的權重檔案。
- `render.ipynb`：互動式 Jupyter Notebook，允許用戶進行 AI 對弈或人機對弈。
- `train.ipynb`：包含訓練 AI 代理的程式碼，展示如何使用強化學習技術進行模型訓練。

### 如何使用
1. **運行訓練**：使用 `train.ipynb` 來訓練 AI 代理，該 Notebook 包含了整個訓練流程，包括環境設置、模型初始化、以及訓練過程的細節。
2. **互動式對弈**：打開 `render.ipynb`，您可以與 AI 進行互動，選擇黑方或白方，與 AI 進行對局，或觀察兩個 AI 之間的對弈。

### 要求
- Python 3.x
- 需要安裝以下主要依賴項：
  - `torch`
  - `numpy`
  - `matplotlib`
  - `jupyter`

---
