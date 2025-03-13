# main.py

from simulation import run_simulation
from constants import *
import pandas as pd
from utils import plot_combined_results, combine_sim_fig, plot_avg_wound_closure_with_std, calculate_avg_migration_count_per_senescence, calculate_avg_division_count_per_senescence, div_mig_perm_slope_avg_calculation

if __name__ == "__main__":

    #  # Directory where Excel files are saved
    input_dir = '/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/EMT Model/Final_Data'  # Current directory

    # try:
    #     for senescent_prob in constant_senescence_probability:
    #         run_simulation(senescent_prob, NUM_STEPS)

    #     # Generate the combined plots for all senescence probabilities
    #     plot_combined_results(input_dir)
    
    # except Exception as e:
    #     print(f"An error occurred: {e}")

    # plot_avg_wound_closure_with_std(input_dir)
    # plot_combined_results(input_dir)

    # # Calculate average migration counts and std after the wound closure
    # input_directory = "/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/EMT Model/Final_Data"
    # avg_migration_counts, avg_fluctuation_stds = calculate_avg_migration_count_per_senescence(input_directory)
    # print("Corrected Average Migration Counts:", avg_migration_counts)
    # print("Average Fluctuation Standard Deviations:", avg_fluctuation_stds)

    # # Display the results
    # df_results_corrected = pd.DataFrame(
    #     list(avg_migration_counts.items()), columns=["Senescence Probability", "Avg Migration Count"]
    # )
    # df_fluctuation_stds = pd.DataFrame(
    #     list(avg_fluctuation_stds.items()), columns=["Senescence Probability", "Fluctuation Std Dev"]
    # )

    # # Calculate average division counts before wound closure
    # input_directory = "/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/EMT Model/Final_Data"
    # corrected_avg_division_counts = calculate_avg_division_count_per_senescence(input_directory)
    # print(corrected_avg_division_counts)

    # # Display the results
    # df_results_corrected = pd.DataFrame(
    #     list(corrected_avg_division_counts.items()), columns=["Senescence Probability", "Corrected Avg Division Count"]
    # )

    # # Combine the simulation figures
    # fig_dir = "/Users/jihopark/Desktop/Jiho_IS/Lung_Epithelial_Simulation/EMT Model/Image_Creation"
    # combine_sim_fig(fig_dir)

    # calculate the division, migration, and permeability drop slope.
    div_mig_perm_slope_avg_calculation(input_dir)
