# IdeNet
This is the official implement code of paper  IdeNet: Identifying camouflaged objects like creatures.
# Note
The results and trained model will be available if the paper is accepted.
# Train
1. Change **--train_path** and **--test-path** in Train.py\
run **python Train.py**\
The trained model will be save in **./checkpoints/**\

2.change the --pth_path as your save model path, and change  the value of **data_path** and **save_path**.\
run **python MyTesting.py** to get test results.\

3.Navigate to **./evaltools**, change the value of **--pred_root** (the path of **save_path** in step 2), **--GT_root** (the path of test datasets), and **model_name** (the model name your saved, like "IdeNet" in our paper) in **eval.py**.\
run **eval.py** to get metrics.

