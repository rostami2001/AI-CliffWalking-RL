# Reinforcement Learning: Cliff Walking - Q-Learning vs SARSA  

## Description  
This repository contains a **Reinforcement Learning (RL)** implementation comparing **Q-Learning** and **SARSA** algorithms. Developed as part of an *Artificial Intelligence* course (2024), the project demonstrates:  
- **Q-Learning** (off-policy TD control)  
- **SARSA** (on-policy TD control)  
- Performance analysis via **reward convergence**, **step efficiency**, and **TD error**.  

Key features:  
- **Visualizations**: GIFs of agent trajectories, Q-table action maps, and comparative plots.  
- **Hyperparameters**: Epsilon decay, learning rate (`ALPHA`), and discount factor (`GAMMA`).  
- **Metrics**: Cumulative rewards, steps per episode, and temporal difference (TD) error analysis.  

---

## Results  
### Key Findings  
1. **Q-Learning** converges faster to the optimal path (along the cliff edge) due to its *off-policy* nature.  
2. **SARSA** adopts a safer but longer path (away from the cliff) due to *on-policy* updates.  
3. **TD Error**: Both algorithms show convergence to near-zero error, with Q-Learning having higher initial exploration variance.  
