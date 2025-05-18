# No normalisation

❯ python experiment.py --task "MLP-TINY" --dataset "./dataset/exported/data_176_cleaned" --seed 42 --device cuda
Saving model...
Model saved to saved_runs/MLP-TINY-1747143497.pth
Predicting...
Average error: `21.649776458740234`
❯ python experiment.py --task "MLP-TINY" --dataset "./dataset/exported/data_176_aged" --load "./saved_runs/MLP-TINY-1747143497.pth" --seed 42 --device cuda
Loading model...
Model loaded from ./saved_runs/MLP-TINY-1747143497.pth
Predicting...
Average error: `129.3238067626953`
❯ python experiment.py --task "MLP-TINY" --dataset "./dataset/exported/data_176_really_aged" --load "./saved_runs/MLP-TINY-1747143497.pth" --seed 42 --device cuda
Loading model...
Model loaded from ./saved_runs/MLP-TINY-1747143497.pth
Predicting...
Average error: `961.4111938476562`

# Normalisation
❯ python experiment.py --task "MLP-TINY" --dataset "./dataset/exported/data_176_cleaned" --seed 42 --device cuda
Saving model...
Model saved to saved_runs/MLP-TINY-1747143321.pth
Predicting...
Average error: `13.878569602966309`

❯ python experiment.py --task "MLP-TINY" --dataset "./dataset/exported/data_176_aged" --load "./saved_runs/MLP-TINY-NORMALISED-1747143321.pth" --seed 42 --device cuda
Loading model...
Model loaded from ./saved_runs/MLP-TINY-NORMALISED-1747143321.pth
Predicting...
Average error: `13.78288459777832`

❯ python experiment.py --task "MLP-TINY" --dataset "./dataset/exported/data_176_really_aged" --load "./saved_runs/MLP-TINY-NORMALISED-1747143321.pth" --seed 42 --device cuda
Loading model...
Model loaded from ./saved_runs/MLP-TINY-NORMALISED-1747143321.pth
Predicting...
Average error: `24.692312240600586`

# Table

| **Dataset**                 | **No Normalization Error (mm)** | **Normalized Input Error (mm)** | **Improvement in %** | **Notes**                                      |
|-|-|-|-|-|
| data_176_cleaned (~0 hours aged)            | 21.65                       | 13.88                      | -35.88% | Trained on clean data                          |
| data_176_aged (~50,000 hours aged simulated)               | 129.32                      | 13.78                      | -89.34% | Huge improvement in generalization with normalization |
| data_176_really_aged (~500,000 hours aged simulated)        | 961.41                      | 24.69                      | -97.43% | Massive improvement with normalization, even on extreme data although absolute is getting unacceptable (if it were fresh data) |
