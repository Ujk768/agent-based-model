import matplotlib
# FIX: Set the backend to 'agg' (Non-interactive) for server use
matplotlib.use('agg')

import panel as pn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from school_model import SchoolModel # Assumes your model class is defined

# Initialize Panel extension
pn.extension(sizing_mode="stretch_width") 

# --- 1. Simulation and Analysis Function (The Core Logic) ---
def run_and_analyze_abm(
    num_steps,
    num_trials,
    base_dropout_rate,
    peer_influence_weight,
    financial_aid_enabled
):
    """
    Runs the simulation for three scenarios using the input parameters, 
    calculates mean dropout rates, and returns the table and plots.
    """
    
    # ⚠️ NOTE: The core parameters for the scenarios are set here.
    BASE_PARAMS = {
        "N": 200,
        "k_degree": 4,
        "rewiring_prob": 0.2,
        "performance_volatility": 1.5,
        "base_dropout_rate": base_dropout_rate,
        "peer_influence_weight": peer_influence_weight,
        "financial_aid_policy": False,
    }

    scenarios = {
        # Baseline uses the current base parameters
        "1_Baseline": {**BASE_PARAMS, "financial_aid_policy": False},
        
        # Intervention enables financial aid based on the toggle
        "2_Intervention_FinancialAid": {**BASE_PARAMS, "financial_aid_policy": financial_aid_enabled},
        
        # Contagion Check uses the higher peer influence weight from the slider
        "3_Contagion_Check_HighPI": {**BASE_PARAMS, "peer_influence_weight": peer_influence_weight * 1.5}
    }
    
    all_results = []
    
    for name, params in scenarios.items():
        all_trials_data = []
        for i in range(num_trials):
            model = SchoolModel(**params)
            for _ in range(num_steps):
                model.step()
            
            # Get data from the final step of the model run
            df = model.datacollector.get_model_vars_dataframe().iloc[[-1]] 
            df['Trial'] = i + 1
            df['Scenario'] = name
            all_trials_data.append(df)
            
        all_results.append(pd.concat(all_trials_data))

    final_df = pd.concat(all_results)

    # Calculate Mean Dropout Rates across all trials and scenarios
    mean_results = final_df.groupby('Scenario')[
        ['Total Dropout Rate', 'Low SES Dropout Rate', 'Medium SES Dropout Rate', 'High SES Dropout Rate']
    ].mean().reset_index()
    
    # --- 2. Create Output Components (Table and Plots) ---

    # A. Table Output (Formatted like your console output)
    table_output = mean_results.round(2).set_index('Scenario')
    
    # B. Plot 1: Total Dropout Rate Comparison
    total_dropout_plot_df = mean_results[['Scenario', 'Total Dropout Rate']]
    
    # We use a fixed figure size for consistency in the layout
    fig1, ax1 = plt.subplots(figsize=(7, 4)) 
    sns.barplot(
        x='Scenario', 
        y='Total Dropout Rate', 
        data=total_dropout_plot_df, 
        palette='viridis', 
        ax=ax1,
        hue='Scenario',
        legend=False
    )
    ax1.set_title(f'Total Dropout Comparison (Steps={num_steps}, Trials={num_trials})', fontsize=12)
    ax1.set_ylabel('Mean Total Dropout Rate (%)')
    ax1.set_xlabel('')
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', rotation=15)
    plt.tight_layout()
    
    # C. Plot 2: Dropout Rate by SES Group 
    # Prepare data for the grouped bar chart
    ses_dropout_long = pd.melt(
        mean_results, 
        id_vars='Scenario', 
        value_vars=['Low SES Dropout Rate', 'Medium SES Dropout Rate', 'High SES Dropout Rate'],
        var_name='SES Group', 
        value_name='Dropout Rate (%)'
    )
    # Clean up the SES group names for better legend/label readability
    ses_dropout_long['SES Group'] = ses_dropout_long['SES Group'].str.replace(' Dropout Rate', '').str.replace(' SES', '')

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    sns.barplot(
        x='Scenario',
        y='Dropout Rate (%)',
        hue='SES Group',
        data=ses_dropout_long,
        palette='Set1',
        ax=ax2
    )

    ax2.set_title('Dropout by Socioeconomic Status Group', fontsize=12)
    ax2.set_ylabel('Mean Dropout Rate (%)')
    ax2.set_xlabel('')
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=15)
    ax2.legend(title='SES Group', loc='upper left')
    plt.tight_layout()

    # Crucial: Close plots to free up memory (Prevents web app from crashing over time)
    plt.close(fig1)
    plt.close(fig2)
    
    # Return the table and the plots wrapped in a Panel Column
    return pn.Column(
        pn.pane.Markdown("### Mean Dropout Rates Across Scenarios (Table View)"),
        # Use pn.pane.DataFrame to render the summary table
        pn.pane.DataFrame(table_output, name="Results Table", height=150),
        pn.Row(pn.pane.Matplotlib(fig1), pn.pane.Matplotlib(fig2))
    )

# --- 3. Define Widgets (Controls) ---
steps_slider = pn.widgets.IntSlider(name='1. Simulation Steps', start=10, end=40, step=5, value=20)
trials_slider = pn.widgets.IntSlider(name='2. Number of Trials (Max 10 for speed)', start=1, end=10, step=1, value=5)
base_rate_slider = pn.widgets.FloatSlider(name='3. Base Dropout Risk (Initial Risk)', start=0.005, end=0.02, step=0.001, value=0.01)
peer_weight_slider = pn.widgets.FloatSlider(name='4. Peer Influence Base Weight', start=0.1, end=1.0, step=0.1, value=0.5)
financial_aid_toggle = pn.widgets.Toggle(name='5. Enable Financial Aid Intervention', button_type='success', value=True)

# --- 4. Create the Panel Layout ---
# Bind the function to the widgets for automatic updates
interactive_output = pn.bind(
    run_and_analyze_abm,
    num_steps=steps_slider,
    num_trials=trials_slider,
    base_dropout_rate=base_rate_slider,
    peer_influence_weight=peer_weight_slider,
    financial_aid_enabled=financial_aid_toggle
)

# Build the sidebar and main content
sidebar = pn.Column(
    "## ⚙️ Model Controls",
    pn.layout.Divider(),
    steps_slider,
    trials_slider,
    base_rate_slider,
    peer_weight_slider,
    pn.layout.Divider(),
    "## 💡 Intervention Toggle",
    financial_aid_toggle,
    sizing_mode="fixed", 
    width=300
)

# Use a Panel Template for a professional look
dashboard = pn.template.FastListTemplate(
    title="Agent-Based Model: Dropout Intervention Explorer",
    sidebar=[sidebar],
    main=[
        pn.Column(
            "## Dynamic Simulation Analysis",
            "Adjust the controls on the left to re-run the 3 scenarios and instantly update the results below.",
            interactive_output
        )
    ]
)

# --- 5. Serve the Application ---
dashboard.servable()