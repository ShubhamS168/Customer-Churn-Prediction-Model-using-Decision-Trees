
import plotly.graph_objects as go

# Data
metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
scores = [0.82, 0.79, 0.68, 0.73]

# Brand colors for each bar
colors = ['#1FB8CD', '#DB4545', '#2E8B57', '#5D878F']

# Create bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    x=metrics,
    y=scores,
    marker=dict(color=colors),
    text=[f'{score:.2f}' for score in scores],
    textposition='outside',
    showlegend=False
))

# Update layout
fig.update_layout(
    title='Model Performance Metrics (Test Set)',
    yaxis_title='Score',
    yaxis=dict(range=[0, 1])
)

# Update traces
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('model_performance.png')
fig.write_image('model_performance.svg', format='svg')
