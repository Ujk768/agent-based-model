import mesa
import numpy as np

# Define SES levels as constants for clarity
LOW_SES, MEDIUM_SES, HIGH_SES = 0, 1, 2

class StudentAgent(mesa.Agent):
    """
    A student agent in the school network.
    Attributes:
        performance (float): Academic performance score (0-100).
        ses (int): Socioeconomic Status (0=Low, 1=Medium, 2=High).
        status (str): Current status ('Attending', 'Dropped Out').
    """
    def __init__(self, unique_id, model, initial_performance, ses):
        # *** INITIALIZATION WORKAROUND ***
        # Manually assign attributes required by Mesa's inheritance to avoid 
        # the persistent "object.__init__() takes exactly one argument" error.
        self.unique_id = unique_id
        self.model = model
        self.pos = None # CRITICAL: Grid placement requires 'pos' to be initialized to None
        
        # Now assign your custom attributes
        self.performance = initial_performance
        self.ses = ses
        self.status = 'Attending'

    def step(self):
        """
        Agent's decision process for a single time step (semester).
        """
        if self.status == 'Dropped Out':
            return # Dropped out agents do nothing

        # --- 1. Update Performance ---
        # Performance fluctuates slightly each semester
        fluctuation = np.random.normal(loc=0, scale=self.model.performance_volatility)
        self.performance = np.clip(
            self.performance + fluctuation,
            a_min=0, a_max=100
        )

        # --- 2. Calculate Peer Influence ---
        # 1. Get the list of neighboring Node IDs (integers 0 to N-1)

        neighbor_list = self.model.grid.get_neighbors(self.pos, include_center=False)


        neighbor_nodes = [agent.unique_id for agent in neighbor_list]
        
        if not neighbor_nodes:
            peer_influence_score = 0
        else:
            dropped_out_count = 0
            
            # 2. Directly access the agent stored in the NetworkX node attribute.
            # This is the most robust way to get the agent on a NetworkGrid.
            # The agent is stored in the "agent" key of the node's attribute dictionary.
            G = self.model.G # Get the raw NetworkX graph object
            
            for node_id in neighbor_nodes:
                # Retrieve the agent object from the graph's node attributes
                neighbor_agent_container = G.nodes[node_id]["agent"]
                
                if isinstance(neighbor_agent_container, list) and neighbor_agent_container:
                    neighbor_agent = neighbor_agent_container[0]
                else:
                    neighbor_agent = neighbor_agent_container

                # Check the status
                if neighbor_agent.status == 'Dropped Out':
                    dropped_out_count += 1
            
            dropped_out_neighbors = dropped_out_count
            peer_influence_score = dropped_out_neighbors / len(neighbor_nodes)


        # --- 3. Calculate Base Dropout Probability (P_drop) ---
        # ... (rest of the code remains the same)
        performance_risk = max(0, self.model.PERFORMANCE_THRESHOLD - self.performance) / self.model.PERFORMANCE_THRESHOLD
        
        # Combined risk factors: P_drop is a function of both
        p_drop = (
            self.model.base_dropout_rate + 
            (performance_risk * (1 - self.model.peer_influence_weight)) + 
            (peer_influence_score * self.model.peer_influence_weight)     
        )
        
        # --- 4. Apply Intervention (Financial Aid) ---
        if self.model.financial_aid_policy and self.ses == LOW_SES:
            p_drop *= (1 - self.model.FINANCIAL_AID_EFFECTIVENESS) 

        # --- 5. Make Dropout Decision ---
        if np.random.rand() < p_drop:
            self.status = 'Dropped Out'