
import plotly.graph_objects as go

# Define the nodes and their positions for the ML pipeline flowchart
nodes = {
    'Data Generation': (0, 10),
    'Data Preprocessing': (0, 8.5),
    'EDA': (0, 7),
    'Data Splitting': (0, 5.5),
    'Train Set': (-2, 4),
    'Test Set': (2, 4),
    'Model Training': (-2, 2.5),
    'Hyperparameter Tuning': (-2, 1),
    'Optimized Model': (-2, -0.5),
    'Model Evaluation': (0, -2),
    'Output Visualizations': (0, -3.5),
    'Performance Metrics': (0, -5)
}

# Define edges (connections between nodes)
edges = [
    ('Data Generation', 'Data Preprocessing'),
    ('Data Preprocessing', 'EDA'),
    ('EDA', 'Data Splitting'),
    ('Data Splitting', 'Train Set'),
    ('Data Splitting', 'Test Set'),
    ('Train Set', 'Model Training'),
    ('Model Training', 'Hyperparameter Tuning'),
    ('Hyperparameter Tuning', 'Optimized Model'),
    ('Optimized Model', 'Model Evaluation'),
    ('Test Set', 'Model Evaluation'),
    ('Model Evaluation', 'Output Visualizations'),
    ('Output Visualizations', 'Performance Metrics')
]

# Create edge traces
edge_x = []
edge_y = []
for edge in edges:
    x0, y0 = nodes[edge[0]]
    x1, y1 = nodes[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])

# Create node traces
node_x = [pos[0] for pos in nodes.values()]
node_y = [pos[1] for pos in nodes.values()]
node_text = list(nodes.keys())

# Abbreviate text to meet 15 character limit
node_text_display = [
    'Data Gen',
    'Data Preproc',
    'EDA',
    'Data Split',
    'Train Set',
    'Test Set',
    'Model Train',
    'Hyperparam Tune',
    'Optimized Mdl',
    'Model Eval',
    'Output Visual',
    'Perf Metrics'
]

# Create figure
fig = go.Figure()

# Add edges
fig.add_trace(go.Scatter(
    x=edge_x, y=edge_y,
    mode='lines',
    line=dict(width=2, color='#21808d'),
    hoverinfo='none',
    showlegend=False
))

# Add nodes
fig.add_trace(go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    marker=dict(size=30, color='#1FB8CD', line=dict(width=2, color='#21808d')),
    text=node_text_display,
    textposition='middle center',
    textfont=dict(size=10, color='#13343b'),
    hoverinfo='text',
    hovertext=node_text,
    showlegend=False
))

# Update layout
fig.update_layout(
    title='ML Pipeline for Churn Prediction',
    showlegend=False,
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    plot_bgcolor='#F3F3EE',
    paper_bgcolor='#F3F3EE'
)

fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('ml_pipeline_flowchart.png')
fig.write_image('ml_pipeline_flowchart.svg', format='svg')

print("Chart saved successfully")
