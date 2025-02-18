import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    # Read jailbroken numbers resulting dataset
    all_r = [4, 8, 16, 32, 64]
    all_jailbroken = []

    for r in all_r:
        result_df = pd.read_csv(f"../../outputs/wordgame_harmbench/wordgame_r{r}.csv")
        print(result_df)
        # Count how many "True" value there is in "jailbroken" column
        jailbroken_count = result_df["jailbroken"].value_counts().get(True, 0)
        all_jailbroken.append(jailbroken_count)

    # Plot the jailbroken numbers with respect to r values
    fig, ax = plt.subplots(figsize=(8, 6))  # Increase figure size for readability
    ax.plot(
        all_r,
        all_jailbroken, 
        marker='o', 
        linestyle='-', 
        color='#6FA3EF',  # Lighter blue color
        markersize=8       # Bigger dots
    )

    # Set logarithmic scale for x-axis to make spacing equal
    ax.set_xscale('log')

    # Ensure only original x values are used as ticks, but spaced equally due to log scale
    ax.set_xticks(all_r)
    ax.set_xticklabels([str(r) for r in all_r], fontsize=14)  # Make x labels bigger

    # Ensure y-axis starts from 0
    ax.set_ylim(bottom=0)

    # Lighter gray grid lines
    ax.grid(True, linestyle='--', linewidth=0.3, color='lightgray')

    # Labels and title with increased font size
    ax.set_xlabel("Recurrent steps", fontsize=16)
    ax.set_ylabel("No. Jailbroken prompts", fontsize=16)
    ax.set_title("Jailbroken prompts with respect to recurrent steps", fontsize=18)

    # Show the plot and save to ../plots/raw_jb_r.png, with dpi=500
    plt.show()
    fig.savefig("../../plots/wordgame_jb_r.png", dpi=500)


if __name__ == "__main__":
    main()
