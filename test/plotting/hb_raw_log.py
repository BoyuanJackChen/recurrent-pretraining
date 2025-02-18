import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import os




def plot_outputs(prompt, df_list, all_r, all_jailbroken, output_name):
    # Create figure with an increased size (1.5x larger)
    num_answers = len(df_list)
    fig_width = min(14, num_answers * 3) * 1.5  # 1.5x width
    fig_height = 8 * 1.5  # 1.5x height
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  
    ax.set_axis_off()

    # Move prompt higher
    ax.text(
        0.5, 1.05, prompt, ha='center', va='center', fontsize=16, fontweight='bold', wrap=True,
        transform=ax.transAxes
    )

    # Align second row positions
    x_positions = np.linspace(0.01, 0.99, num_answers)  # Evenly space across width
    y_position = 0.9

    # Dynamically adjust text wrapping width based on figure width
    text_block_width = max(20, int(fig_width * 8.5 / num_answers))  # Ensuring reasonable minimum width

    # Calculate offset for text alignment
    text_offset = 0.11  # Adjusts positioning slightly to the left

    # Place each answer in its respective column, ensuring alignment
    for i, (df, r_value, is_jailbroken) in enumerate(zip(df_list, all_r, all_jailbroken)):
        answer_text = df["answer"].replace("\\n", "\n")

        # Wrap text dynamically based on computed width
        wrapped_text = "\n".join(textwrap.wrap(answer_text, width=text_block_width))

        # Determine text color for "r={r}" label
        r_text_color = "red" if is_jailbroken else "black"

        # Display the "r={r}" label above each text block (centered)
        ax.text(
            x_positions[i], y_position + 0.05, f"r={r_value}",
            ha='center', va='bottom', fontsize=14, fontweight='bold', color=r_text_color,
            transform=ax.transAxes
        )

        # Display the actual answer text (inside box, left-aligned)
        ax.text(
            x_positions[i] - text_offset, y_position, wrapped_text,
            ha='left', va='top', fontsize=14, wrap=True,
            bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", lw=1),
            transform=ax.transAxes
        )

    # Save the plot
    plt.savefig(output_name, dpi=150)

    # Explicitly close the figure to free memory
    plt.close(fig)





def main():
    ### Read jailbroken numbers resulting dataset with pandas
    all_r = [4, 8, 16, 32, 64]
    df_4 = pd.read_csv("../../outputs/raw_harmbench/raw_r4.csv")
    df_8 = pd.read_csv("../../outputs/raw_harmbench/raw_r8.csv")
    df_16 = pd.read_csv("../../outputs/raw_harmbench/raw_r16.csv")
    df_32 = pd.read_csv("../../outputs/raw_harmbench/raw_r32.csv")
    df_64 = pd.read_csv("../../outputs/raw_harmbench/raw_r64.csv")

    ### mkdir if it doesnt exist
    output_dir = "../../outputs/raw_harmbench/jailbrokens"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ### Loop through each row of these dataframes in parallel, and check if ANY of the dataframe's "jailbroken" column is True
    for i, (row_4, row_8, row_16, row_32, row_64) in enumerate(zip(df_4.iterrows(), df_8.iterrows(), df_16.iterrows(), df_32.iterrows(), df_64.iterrows())):
        if any([row_8[1]["jailbroken"], row_16[1]["jailbroken"], row_32[1]["jailbroken"], row_64[1]["jailbroken"]]):
            ### Check in the output dir if there is already a .png file with filename starting with str(i+1). If there is, skip this iteration
            if any([f.startswith(f"{str(i+1)}_") for f in os.listdir(output_dir)]):
                continue

            all_jailbroken = [row_4[1]["jailbroken"], row_8[1]["jailbroken"], row_16[1]["jailbroken"], row_32[1]["jailbroken"], row_64[1]["jailbroken"]]
            indices = [i for i, x in enumerate(all_jailbroken) if x]
            jailbroken_rs = [all_r[i] for i in indices]
            jailbroken_rs = ", ".join(map(str, jailbroken_rs))
            filename = os.path.join(output_dir, f"{str(i+1)}_{jailbroken_rs}.png")
            plot_outputs(row_4[1]["question"], [row_4[1], row_8[1], row_16[1], row_32[1], row_64[1]], all_r, all_jailbroken, filename)



if __name__ == "__main__":
    main()
