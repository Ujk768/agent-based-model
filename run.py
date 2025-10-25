import pandas as pd
from school_model import SchoolModel

# --- SIMULATION CONFIGURATION ---
NUM_STEPS = 20 # Number of semesters to run the simulation (Reduced from 50)
NUM_TRIALS = 30 # Number of runs for statistical robustness
N_AGENTS = 200 # Number of students
K_DEGREE = 4 # Average number of initial connections per student
REWIRING_PROB = 0.2 # Small-world network parameter (chance of rewiring a link)

# --- BASELINE PARAMETERS ---
BASE_PARAMS = {
    "N": N_AGENTS,
    "k_degree": K_DEGREE,
    "rewiring_prob": REWIRING_PROB,
    "base_dropout_rate": 0.001,         # Very small inherent risk
    "performance_volatility": 1.5,      # How much performance scores fluctuate
    "peer_influence_weight": 0.5,       # Equal weight between academic risk and peer risk
    "financial_aid_policy": False       # The default state: NO financial aid
}

# --- EXPERIMENTAL SCENARIOS ---
scenarios = {
    "1_Baseline": {
        **BASE_PARAMS,
        "financial_aid_policy": False,
    },
    "2_Intervention_FinancialAid": {
        **BASE_PARAMS,
        "financial_aid_policy": True, # The intervention: Financial aid is implemented
    },
    "3_Contagion_Check_HighPI": {
        **BASE_PARAMS,
        "financial_aid_policy": False,
        "peer_influence_weight": 0.8, # Test the effect of strong peer contagion
    }
}

# --- RUNNER FUNCTION ---
def run_simulation(scenario_name, params, num_trials, num_steps):
    """Runs a single scenario for multiple trials and returns the data."""
    print(f"\n--- Running Scenario: {scenario_name} ---")
    
    # List to hold the model data (one DataFrame per trial)
    all_trials_data = []

    for i in range(num_trials):
        # Create and run the model
        model = SchoolModel(**params)
        for _ in range(num_steps):
            model.step()
        
        # Get the collected model data for this trial
        df = model.datacollector.get_model_vars_dataframe()
        df['Trial'] = i + 1
        df['Scenario'] = scenario_name
        all_trials_data.append(df)
        
        if (i + 1) % 10 == 0:
            print(f"  Trial {i + 1}/{num_trials} complete.")

    # Combine all trials into one DataFrame
    return pd.concat(all_trials_data)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Run all scenarios
    all_results = []
    for name, params in scenarios.items():
        results = run_simulation(name, params, NUM_TRIALS, NUM_STEPS)
        all_results.append(results)

    # Combine all scenario results
    final_df = pd.concat(all_results)
    
    # Save the aggregated data to a CSV file for Part 3 analysis
    output_filename = "dropout_abm_simulation_results.csv"
    final_df.to_csv(output_filename, index=True)
    
    print(f"\n✅ All simulations complete.")
    print(f"Results saved to: {output_filename}")
    print("Use this CSV file for Part 3 (Analysis & Visualization).")