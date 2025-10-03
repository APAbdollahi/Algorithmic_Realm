import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from pyvis.network import Network
import random
import copy

# --- Core Simulation Class (V4 - designed for state snapshots) ---

class SocialMediaSimulatorV4:
    """
    Final version enabling on-demand graph generation for any cycle.
    The core logic is the same, but it's structured to be loaded with a state.
    """
    def __init__(self, params):
        # Store all parameters from a dictionary
        self.params = params
        self.N = params['N']
        
        # State variables that change over time
        self.opinions = None
        self.conviction_scores = None
        self.post_likes = None
        self.last_cycle_edges = []
        
        self._initialize_population()

    def _initialize_population(self):
        # Opinions
        num_plus_one = int(self.N * self.params['original_opinion_pct'])
        num_minus_one = self.N - num_plus_one
        opinions = np.array([1] * num_plus_one + [-1] * num_minus_one)
        np.random.shuffle(opinions)
        self.opinions = opinions
        
        # Conviction Scores (drawn from a clipped normal distribution)
        conviction = np.random.normal(self.params['conviction_mean'], self.params['conviction_std'], self.N)
        self.conviction_scores = np.clip(conviction, 0, None) # Ensure conviction is not negative
        
        # Likes count
        self.post_likes = {1: 1, -1: 1} # Start with 1 to avoid division by zero

    def load_state(self, opinions, post_likes):
        """Loads a previous state to run a simulation from that point."""
        self.opinions = opinions
        self.post_likes = post_likes

    def run_single_cycle(self, record_edges=False):
        """Simulates one cycle. If record_edges is True, it stores the communication graph."""
        poster_indices = np.random.permutation(self.N)
        if record_edges:
            self.last_cycle_edges = []

        for poster_id in poster_indices:
            if np.random.rand() < self.params['ct_post_probability']:
                post_type = self.opinions[poster_id]
            else:
                post_type = 0

            amplification = self.params['amplification_ratios'][post_type]
            num_viewers = int(self.N * self.params['base_broadcast_ratio'] * amplification)
            
            viewer_indices = self._select_viewers(poster_id, num_viewers, post_type)

            if record_edges and post_type != 0:
                for viewer_id in viewer_indices:
                    self.last_cycle_edges.append((poster_id, viewer_id))

            if post_type != 0:
                for viewer_id in viewer_indices:
                    # Liking mechanism
                    if self.opinions[viewer_id] == post_type:
                        prob_of_liking = self.params['like_ratio'] * self.conviction_scores[viewer_id]
                        if np.random.rand() < prob_of_liking:
                            self.post_likes[post_type] += 1
                    # Flipping mechanism
                    else: 
                        if np.random.rand() < self.params['flip_probability']:
                            self.opinions[viewer_id] = post_type

    def _select_viewers(self, poster_id, num_viewers, post_type):
        """Selects viewers based on a mix of random broadcast and like-based targeting."""
        possible_viewer_indices = np.array([i for i in range(self.N) if i != poster_id])
        if num_viewers > len(possible_viewer_indices) or num_viewers == 0:
            return []
        if post_type == 0:
            return np.random.choice(possible_viewer_indices, size=num_viewers, replace=False)

        total_likes = self.post_likes[1] + self.post_likes[-1]
        
        prob_for_aligned_viewers = self.post_likes[post_type] / total_likes
        
        aligned_viewers = possible_viewer_indices[self.opinions[possible_viewer_indices] == post_type]
        unaligned_viewers = possible_viewer_indices[self.opinions[possible_viewer_indices] != post_type]
        
        num_targeted = int(num_viewers * self.params['like_feedback_strength'])
        num_random = num_viewers - num_targeted

        selected_viewers = []
        if num_targeted > 0 and len(aligned_viewers) > 0 and len(unaligned_viewers) > 0:
            num_to_aligned = int(num_targeted * prob_for_aligned_viewers)
            if num_to_aligned > len(aligned_viewers): num_to_aligned = len(aligned_viewers)

            num_to_unaligned = num_targeted - num_to_aligned
            if num_to_unaligned > len(unaligned_viewers): num_to_unaligned = len(unaligned_viewers)

            if num_to_aligned > 0:
                selected_viewers.extend(np.random.choice(aligned_viewers, size=num_to_aligned, replace=False))
            if num_to_unaligned > 0:
                selected_viewers.extend(np.random.choice(unaligned_viewers, size=num_to_unaligned, replace=False))
        
        remaining_viewers = np.setdiff1d(possible_viewer_indices, selected_viewers)
        if num_random > len(remaining_viewers): num_random = len(remaining_viewers)
        
        if num_random > 0:
             selected_viewers.extend(np.random.choice(remaining_viewers, size=num_random, replace=False))
        
        return selected_viewers

    def get_opinion_stats(self):
        return {
            'plus_one_pct': np.mean(self.opinions == 1),
            'plus_one_likes': self.post_likes[1],
            'minus_one_likes': self.post_likes[-1],
        }

# --- Main Simulation Runner Function ---
@st.cache_data
def run_full_simulation(params, num_cycles):
    """Runs the full simulation and caches the results, returning stats and state histories."""
    sim = SocialMediaSimulatorV4(params)
    
    history_stats = [sim.get_opinion_stats()]
    history_states = [{'opinions': copy.deepcopy(sim.opinions), 'post_likes': copy.deepcopy(sim.post_likes)}]

    for i in range(num_cycles):
        sim.run_single_cycle()
        history_stats.append(sim.get_opinion_stats())
        history_states.append({'opinions': copy.deepcopy(sim.opinions), 'post_likes': copy.deepcopy(sim.post_likes)})
        
    return pd.DataFrame(history_stats), history_states

# --- Streamlit GUI ---
st.set_page_config(layout="wide")
st.title("The Algorithmic Realm")

# --- Sidebar Controls ---
st.sidebar.header("Define Parameters")
params = {}
params['N'] = st.sidebar.slider('N (Participants)', 100, 5000, 1000, 100)
num_cycles = st.sidebar.slider('Number of Cycles', 5, 100, 30, 1)
params['original_opinion_pct'] = st.sidebar.slider('Original Opinion (+1 %)', 0.0, 1.0, 0.8, 0.01)
params['flip_probability'] = st.sidebar.slider('Flip Probability', 0.0, 0.1, 0.01, 0.001, format='%.3f')
st.sidebar.markdown("---")
st.sidebar.subheader("User Behavior")
params['like_ratio'] = st.sidebar.slider('Base Like Ratio', 0.0, 1.0, 0.05, 0.01)
params['conviction_mean'] = st.sidebar.slider('Mean Conviction', 0.1, 2.0, 1.0, 0.1)
params['conviction_std'] = st.sidebar.slider('Std Dev of Conviction', 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")
st.sidebar.subheader("The 'Secret Law'")
params['like_feedback_strength'] = st.sidebar.slider('Like Feedback Strength (Echo Chamber)', 0.0, 1.0, 0.9, 0.05)
amp_plus_one = st.sidebar.slider('Global Amplify (+1 View)', 0.0, 2.0, 0.2, 0.05)
amp_minus_one = st.sidebar.slider('Global Amplify (-1 View)', 0.0, 2.0, 0.8, 0.05)
params['amplification_ratios'] = {1: amp_plus_one, -1: amp_minus_one, 0: 0.5}

# Set fixed internal parameters
params['base_broadcast_ratio'] = 0.1
params['ct_post_probability'] = 0.5

run_button = st.sidebar.button("Run Simulation", type="primary")

# --- Main Page Logic ---

# Initialize session state if it doesn't exist
if 'simulation_has_run' not in st.session_state:
    st.session_state.simulation_has_run = False
    st.session_state.history_df = None
    st.session_state.history_states = None
    st.session_state.params = {}
    st.session_state.num_cycles = 0

# The button click updates the state and triggers the simulation run
if run_button:
    st.session_state.simulation_has_run = True
    st.session_state.params = params
    st.session_state.num_cycles = num_cycles
    
    with st.spinner('The Algorithmic Realm is shaping reality...'):
        df, states = run_full_simulation(st.session_state.params, st.session_state.num_cycles)
        st.session_state.history_df = df
        st.session_state.history_states = states

# The display logic only executes if a simulation has been run
if st.session_state.simulation_has_run:
    df = st.session_state.history_df
    history_states = st.session_state.history_states
    
    st.header("Evolution of Public Opinion and Engagement")
    col1, col2 = st.columns(2)
    with col1: # Opinion Chart
        st.subheader("Opinion Distribution")
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.plot(df.index, df['plus_one_pct'], marker='o', linestyle='-', label='Evolution of +1 Opinion')
        ax1.axhline(y=st.session_state.params['original_opinion_pct'], color='r', linestyle='--', label=f"Original Opinion ({st.session_state.params['original_opinion_pct']:.0%})")
        ax1.set_title('Public Opinion in the Digital Space')
        ax1.set_xlabel('Cycle (Day)'); ax1.set_ylabel('Percentage of Population with +1 Opinion')
        ax1.set_ylim(0, 1); ax1.grid(True); ax1.legend()
        st.pyplot(fig1)

    with col2: # Engagement Chart
        st.subheader("Engagement (Likes)")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        total_likes = df['plus_one_likes'] + df['minus_one_likes']
        ax2.plot(df.index, df['plus_one_likes'] / total_likes, marker='.', linestyle='-', color='blue', label='% of Likes for +1 Posts')
        ax2.plot(df.index, df['minus_one_likes'] / total_likes, marker='.', linestyle='-', color='orange', label='% of Likes for -1 Posts')
        ax2.set_title('Share of Voice (Likes)')
        ax2.set_xlabel('Cycle (Day)'); ax2.set_ylabel('Percentage of Total Likes')
        ax2.set_ylim(0, 1); ax2.grid(True); ax2.legend()
        st.pyplot(fig2)

    st.header("Visualizing the Social Fabric: The Time Machine")
    
    day_to_visualize = st.slider("Select Day to Visualize Network", min_value=1, max_value=st.session_state.num_cycles, value=st.session_state.num_cycles, step=1)
    max_nodes_slider = st.slider('Max Nodes to Display in Graph', 50, 500, 200, 10)

    with st.spinner(f"Generating communication graph for Day {day_to_visualize}..."):
        previous_state = history_states[day_to_visualize - 1]
        temp_sim = SocialMediaSimulatorV4(st.session_state.params)
        temp_sim.load_state(previous_state['opinions'], previous_state['post_likes'])
        temp_sim.run_single_cycle(record_edges=True)
        current_opinions = history_states[day_to_visualize]['opinions']
        
        num_nodes_to_display = min(temp_sim.N, max_nodes_slider)
        sampled_nodes = random.sample(range(temp_sim.N), num_nodes_to_display)
        
        net = Network(height='750px', width='100%', bgcolor='#222222', font_color='white', notebook=True, cdn_resources='in_line')
        net.force_atlas_2based()

        node_colors = {1: '#007bff', -1: '#ffa500'}
        for node_id in sampled_nodes:
            opinion = int(current_opinions[node_id])
            net.add_node(int(node_id), label=str(node_id), color=node_colors[opinion], title=f"Opinion: {opinion}")
        
        for u, v in temp_sim.last_cycle_edges:
            py_u, py_v = int(u), int(v)
            if py_u in sampled_nodes and py_v in sampled_nodes:
                poster_opinion = int(current_opinions[py_u])
                edge_color = node_colors[poster_opinion]
                net.add_edge(py_u, py_v, color=edge_color, width=0.5, title=f"Influence from {py_u} ({poster_opinion})")
        
        try:
            path = '/tmp' # A temporary path for the HTML file
            net.save_graph(f'{path}/pyvis_graph.html')
            HtmlFile = open(f'{path}/pyvis_graph.html', 'r', encoding='utf-8')
            st.components.v1.html(HtmlFile.read(), height=800, scrolling=True)
        except Exception as e:
            st.error(f"Could not generate graph: {e}")
else:
    st.info("Welcome to the Algorithmic Realm. Please adjust the parameters in the sidebar and click 'Run Simulation' to begin.")
