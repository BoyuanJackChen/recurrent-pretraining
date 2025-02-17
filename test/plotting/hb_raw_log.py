import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap



def plot_outputs(prompt, df_list):
    # Create figure with a controlled size
    num_answers = len(df_list)
    fig, ax = plt.subplots(figsize=(min(14, num_answers * 3), 8))  # Slightly larger height for better spacing
    ax.set_axis_off()

    # Move prompt higher
    ax.text(
        0.5, 0.92, prompt, ha='center', va='center', fontsize=16, fontweight='bold', wrap=True,
        transform=ax.transAxes
    )

    # Define horizontal positions based on number of answers
    x_positions = np.linspace(0.1, 0.9, num_answers)  # Evenly space across width

    # Align text at the top
    y_position = 0.5  

    # Place each answer in its respective column, ensuring alignment
    for i, df in enumerate(df_list):
        answer_text = df["answer"].replace("\\n", "\n")
        wrapped_text = "\n".join(textwrap.wrap(answer_text, width=32))
        ax.text(
            x_positions[i], y_position, wrapped_text,
            ha='center', va='top', fontsize=14, wrap=True,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", lw=1),
            transform=ax.transAxes
        )

    # Show the plot without forcing full screen
    plt.show()




def main():
    ### Read jailbroken numbers resulting dataset with pandas
    all_r = [4, 8, 16, 32, 64]
    df_4 = pd.read_csv("../../outputs/raw_harmbench/raw_r4.csv")
    df_8 = pd.read_csv("../../outputs/raw_harmbench/raw_r8.csv")
    df_16 = pd.read_csv("../../outputs/raw_harmbench/raw_r16.csv")
    df_32 = pd.read_csv("../../outputs/raw_harmbench/raw_r32.csv")
    df_64 = pd.read_csv("../../outputs/raw_harmbench/raw_r64.csv")

    ### Loop through each row of these dataframes in parallel, and check if ANY of the dataframe's "jailbroken" column is True
    for i, (row_4, row_8, row_16, row_32, row_64) in enumerate(zip(df_4.iterrows(), df_8.iterrows(), df_16.iterrows(), df_32.iterrows(), df_64.iterrows())):
        if any([row_4[1]["jailbroken"], row_8[1]["jailbroken"], row_16[1]["jailbroken"], row_32[1]["jailbroken"], row_64[1]["jailbroken"]]):
           plot_outputs(row_4[1]["question"], [row_4[1], row_8[1], row_16[1], row_32[1], row_64[1]])
           break



if __name__ == "__main__":
    main()
