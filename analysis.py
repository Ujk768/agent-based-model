import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_data(file_path):
    """Loads the CSV file, handles the unnamed step column, and filters for the final step."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Ensure 'dropout_abm_simulation_results.csv' is in the same directory.")
        return None

    # --- FIX: Identify and rename the Step column ---
    # The step column is incorrectly read as 'Unnamed: 0' when the CSV is saved.
    step_column = None
    if 'Unnamed: 0' in df.columns:
        df.rename(columns={'Unnamed: 0': 'Step'}, inplace=True)
        step_column = 'Step'
    elif 'Step' in df.columns:
        step_column = 'Step'
    elif 'step' in df.columns:
        step_column = 'step'
    else:
        print(f"Error: Could not find the 'Step' column in the CSV. Available columns: {list(df.columns)}")
        return None
        
    # Filter for the final step of the simulation (the overall result for each trial)
    final_step = df[step_column].max()
    final_results_df = df[df[step_column] == final_step].copy() # Use .copy() to avoid SettingWithCopyWarning
    
    # Drop unnecessary columns (like the now-used 'Step' and redundant parameter columns)
    columns_to_drop = ['AgentID', step_column, 'Financial Aid Policy', 'Peer Influence Weight']
    final_results_df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    
    return final_results_df

def calculate_means(df):
    """Calculates mean dropout rates by scenario."""
    
    # Define the columns that hold the dropout rate results
    analysis_cols = [
        'Total Dropout Rate',
        'Low SES Dropout Rate',
        'Medium SES Dropout Rate',
        'High SES Dropout Rate'
    ]

    # Group by scenario and calculate the mean for each metric across all trials
    mean_dropout_rates = df.groupby('Scenario').mean() # Group by 'Scenario' after dropping the parameters/trial number
    
    print("\n--- Mean Dropout Rates Across 30 Trials ---")
    print(mean_dropout_rates[analysis_cols].round(2))
    print("------------------------------------------")
    
    return mean_dropout_rates[analysis_cols]

def visualize_results(mean_df):
    """Generates two plots: Total Dropout Rate and SES Grouped Dropout Rates."""

    sns.set_style("whitegrid")

    # --- Plot 1: Total Dropout Rate Comparison ---
    
    # Prepare data for plotting Total Dropout Rate
    total_dropout_plot = mean_df['Total Dropout Rate'].reset_index()

    plt.figure(figsize=(8, 6))
    sns.barplot(x='Scenario', y='Total Dropout Rate', data=total_dropout_plot, palette='viridis')
    plt.title('Figure 1: Mean Total Dropout Rate Comparison by Scenario')
    plt.ylabel('Mean Total Dropout Rate (%)')
    plt.xlabel('Scenario')
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.savefig('Total_Dropout_Comparison.png')
    plt.close() # Use close instead of show to prevent blocking

    # --- Plot 2: Dropout Rate by SES Group ---

    # Reshape the data for a grouped bar plot (long format)
    ses_dropout_long = mean_df.stack().reset_index()
    ses_dropout_long.columns = ['Scenario', 'SES Group', 'Dropout Rate (%)']

    # Clean up SES Group names for better plotting
    ses_dropout_long['SES Group'] = ses_dropout_long['SES Group'].str.replace(' Dropout Rate', '')

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='Scenario', 
        y='Dropout Rate (%)', 
        hue='SES Group', 
        data=ses_dropout_long, 
        palette='Set1'
    )

    plt.title('Figure 2: Mean Dropout Rate by Socioeconomic Status (SES) Group')
    plt.ylabel('Mean Dropout Rate (%)')
    plt.xlabel('Scenario')
    plt.xticks(rotation=10)
    plt.legend(title='SES Group')
    plt.tight_layout()
    plt.savefig('SES_Dropout_Comparison.png')
    plt.close() # Use close instead of show to prevent blocking


if __name__ == "__main__":
    
    # 1. Load Data
    data_file = 'dropout_abm_simulation_results.csv'
    final_data = load_data(data_file)
    
    if final_data is not None:
        
        # 2. Calculate Means
        mean_results = calculate_means(final_data)
        
        # 3. Visualize Results (Plots are saved as PNG files)
        visualize_results(mean_results)
        
        print("\nAnalysis and visualizations complete!")
        print("Results saved to: Total_Dropout_Comparison.png and SES_Dropout_Comparison.png")