with open("logs/hyperparameter_results.csv", "r") as f:
    lines = f.readlines()

kept_lines = [line for line in lines if not line.startswith("DQN,")]

with open("logs/hyperparameter_results.csv", "w") as f:
    f.writelines(kept_lines)

print(f"Removed {len(lines) - len(kept_lines)} DQN rows, {len(kept_lines)} rows remaining")