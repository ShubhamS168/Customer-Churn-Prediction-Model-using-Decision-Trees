
import plotly.graph_objects as go
import json

# Data from the provided JSON
data = {"features": ["Tenure", "TotalWatchHours", "SupportTickets", "MonthlyCharges", "SubscriptionType", "Age", "PaymentMethod", "Gender"], "importance": [0.235, 0.182, 0.157, 0.121, 0.104, 0.098, 0.065, 0.038]}

features = data['features']
importance = data['importance']

# Create horizontal bar chart
fig = go.Figure()

fig.add_trace(go.Bar(
    y=features,
    x=importance,
    orientation='h',
    marker=dict(color='#1FB8CD'),  # Blue/teal color from brand colors
    text=[f'{val:.3f}' for val in importance],
    textposition='outside',
    textfont=dict(size=12),
    cliponaxis=False
))

# Update layout with title and axis labels
# Title shortened to meet <40 char requirement: "Feature Importance - Decision Tree" (34 chars)
# X-axis shortened to meet 15 char limit: "Import. Score" (13 chars)
fig.update_layout(
    title='Feature Importance - Decision Tree',
    xaxis_title='Import. Score',
    yaxis_title='',
    showlegend=False
)

# Reverse the y-axis to show highest importance at top
fig.update_yaxes(autorange="reversed")

# Save as both PNG and SVG
fig.write_image('feature_importance.png')
fig.write_image('feature_importance.svg', format='svg')
