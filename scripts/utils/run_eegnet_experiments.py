import subprocess
import re
import numpy as np
import sys

seeds = [
    1,
    2,
    3,
    4,
    5
]


results = []


for seed in seeds:

    print("================")
    print("Running seed:", seed)


    output = subprocess.check_output(
        [
            sys.executable,
            "scripts/models/deep_learning/train_eegnet.py",
            str(seed)
        ],
        text=True
    )


    print(output)


    match = re.search(
        r"EEGNet Accuracy:\s*\n([0-9.]+)",
        output
    )


    if match:

        acc = float(
            match.group(1)
        )

        results.append(acc)



print("================")
print("All Results")

for i,acc in enumerate(results):

    print(
        f"Run {i+1}: {acc:.4f}"
    )



print("----------------")

print(
    "Mean:",
    np.mean(results)
)


print(
    "Std:",
    np.std(results)
)