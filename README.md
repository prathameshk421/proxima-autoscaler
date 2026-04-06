title: Cloud AutoScaling RL (OpenEnv)
emoji: ☁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Proxima Auto-Scaler ☁️
*Developed for the Meta-PyTorch Cloud Hackathon.*

This project showcases a production-ready, highly robust Reinforcement Learning pipeline attempting to dynamically autoscale simulated cloud infrastructure footprints using a custom Multi-Headed Proximal Policy Optimization (PPO) architecture coupled dynamically via SB3 logic arrays. 

The simulated backend physically represents a true `M/M/c` Queueing Theory system natively processing robust MMPP variations spanning three difficulty workloads natively ("easy", "medium", "hard"). Additionally, it dynamically parses standard scale-up delays through isolated execution queues (simulating real EC2 provisioning lags), forcing our PPO agent to mathematically map delayed provisioning horizons instead of blindly assuming linear capacity integrations! 

### Hugging Face API Endpoints

Our REST execution service provides isolated, un-batched inference calls directly from the Docker instance. 
- **`GET /health`**: Tests basic server operation logic mapping and ensures the `ppo_cloud_final` configuration graph loaded into RAM.
- **`GET /evaluate?difficulty=<diff>&seed=<int>`**: Computes natively up to 200 simulation timestamps across our compiled PPO variants against designated testing variables, returning exact trace vectors alongside calculated normalization benchmarks.
- **`GET /baselines?difficulty=<diff>&seed=<int>`**: Iterates the identical simulated environments physically across standard rule-based heuristics (`Threshold`, `Target Tracking`, and `Step Scaling`) allowing 1:1 metric correlations against arbitrary workloads.
