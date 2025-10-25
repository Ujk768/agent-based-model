import mesa
import networkx as nx
import numpy as np
import random
from student_agent import StudentAgent, LOW_SES, MEDIUM_SES, HIGH_SES

class SchoolModel(mesa.Model):
    """
    The main model for the school, managing students and the network.
    
    Class-level constants define fixed parameters for all scenarios.
    """
    # Fixed Model Parameters 
    PERFORMANCE_THRESHOLD = 60  # Performance score below this increases risk
    INITIAL_PERFORMANCE_MEAN = 75
    INITIAL_PERFORMANCE_STD = 10
    FINANCIAL_AID_EFFECTIVENESS = 0.4 # Reduces P_drop by 40% for Low SES students

    def __init__(self, N, k_degree, rewiring_prob, base_dropout_rate, 
                 peer_influence_weight, performance_volatility, financial_aid_policy):
        
        self.running = True
        self.n_agents = N
        self.steps = 0
        # --- 0. Initialize Random Seed for Reproducibility ---
        # FIX: Random object must be initialized before the scheduler.
        self.random = random.Random(42) 
        
        # --- 1. Assign Experimental Parameters ---
        self.base_dropout_rate = base_dropout_rate
        self.peer_influence_weight = peer_influence_weight
        self.performance_volatility = performance_volatility
        self.financial_aid_policy = financial_aid_policy # The main intervention

        # --- 2. Setup the NetworkX Graph ---
        # FIX: The graph must be created before the NetworkGrid.
        self.G = nx.watts_strogatz_graph(n=N, k=k_degree, p=rewiring_prob)
        
        # --- 3. Initialize NetworkGrid ---
        # FIX: Pass the graph (self.G) directly to the NetworkGrid
        self.grid = mesa.space.NetworkGrid(self.G) 
        
        # --- 4. Initialize Scheduler ---
        self.schedule = mesa.time.RandomActivation(self)

        # --- 5. Final Setup ---
        self.make_agents()
        self.setup_datacollector()


    def make_agents(self):
        """
        Create students, assign attributes, and add them to the model.
        """
        # Distribute SES: 40% Low, 40% Medium, 20% High
        ses_distribution = [LOW_SES] * int(self.n_agents * 0.4) + \
                           [MEDIUM_SES] * int(self.n_agents * 0.4) + \
                           [HIGH_SES] * int(self.n_agents * 0.2)
        # We use numpy.random.shuffle here as it's separate from Mesa's model.random
        np.random.shuffle(ses_distribution) 

        for i, node in enumerate(self.G.nodes()):
            # Initialize performance with some variance
            initial_performance = np.random.normal(
                # FIX: Access class constants via SchoolModel.CONSTANT
                loc=SchoolModel.INITIAL_PERFORMANCE_MEAN, 
                scale=SchoolModel.INITIAL_PERFORMANCE_STD 
            )
            initial_performance = np.clip(initial_performance, 50, 95)

            # Create agent
            a = StudentAgent(i, self, initial_performance, ses_distribution[i])
            self.schedule.add(a)
            self.grid.place_agent(a, node) # Place agent on the network node
    
    
    def get_dropout_rate(self):
        """Returns the current percentage of dropped out students."""
        dropout_count = sum(1 for a in self.schedule.agents if a.status == 'Dropped Out')
        return (dropout_count / self.n_agents) * 100
        
    def get_dropout_by_ses(self, ses_level):
        """Returns the dropout rate for a specific SES level."""
        
        ses_agents = [a for a in self.schedule.agents if a.ses == ses_level]
        if not ses_agents:
            return 0
        
        dropout_count = sum(1 for a in ses_agents if a.status == 'Dropped Out')
        return (dropout_count / len(ses_agents)) * 100

    def setup_datacollector(self):
        """Sets up the Mesa DataCollector for model and agent data."""
        
        model_reporters = {
            "Total Dropout Rate": self.get_dropout_rate,
            "Low SES Dropout Rate": lambda m: self.get_dropout_by_ses(LOW_SES),
            "Medium SES Dropout Rate": lambda m: self.get_dropout_by_ses(MEDIUM_SES),
            "High SES Dropout Rate": lambda m: self.get_dropout_by_ses(HIGH_SES),
            "Financial Aid Policy": lambda m: m.financial_aid_policy, 
            "Peer Influence Weight": lambda m: m.peer_influence_weight,
        }
        
        agent_reporters = {
            "Performance": "performance",
            "Status": "status",
            "SES": "ses"
        }

        self.datacollector = mesa.DataCollector(
            model_reporters=model_reporters, 
            agent_reporters=agent_reporters
        )

    def step(self):
        """Advance the model by one step."""
        self.steps += 1 # <--- CRITICAL FIX: Increment step counter
        self.datacollector.collect(self)
        self.schedule.step()