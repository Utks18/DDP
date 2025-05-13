import plotly.graph_objects as go

def create_architecture_diagram():
    # Create figure with custom styling
    fig = go.Figure()
    
    # Define colors
    colors = {
        'background': '#f8f9fa',
        'box': '#ffffff',
        'border': '#2c3e50',
        'arrow': '#34495e',
        'text': '#2c3e50'
    }
    
    # Define positions for boxes (using a grid layout)
    positions = {
        'input': (0.1, 0.85),
        'clip_visual': (0.3, 0.85),
        'pool': (0.5, 0.85),
        'norm': (0.7, 0.85),
        'prompts': (0.1, 0.55),
        'tokenize': (0.3, 0.55),
        'embed': (0.5, 0.55),
        'clip_text': (0.7, 0.55),
        'sim_pos': (0.5, 0.25),
        'sim_neg': (0.7, 0.25),
        'logits': (0.6, 0.05)
    }
    
    # Define box texts
    texts = {
        'input': 'Input Images\n[B, 3, 256, 256]',
        'clip_visual': 'CLIP Visual Encoder\n(ResNet-50)',
        'pool': 'Average Pooling\n[B, D]',
        'norm': 'Feature Normalization',
        'prompts': 'Text Prompts\n(46 classes)',
        'tokenize': 'Tokenization',
        'embed': 'Text Embedding',
        'clip_text': 'CLIP Text Encoder\n(Transformer)',
        'sim_pos': 'Positive Similarity\n[B, C]',
        'sim_neg': 'Negative Similarity\n[B, C]',
        'logits': 'Final Logits\n[B, C]'
    }
    
    # Add boxes and text in a single loop
    for name, pos in positions.items():
        # Add box
        fig.add_shape(
            type="rect",
            x0=pos[0]-0.08, y0=pos[1]-0.04,
            x1=pos[0]+0.08, y1=pos[1]+0.04,
            fillcolor=colors['box'],
            line=dict(color=colors['border'], width=1),
            opacity=0.9
        )
        # Add text
        fig.add_annotation(
            x=pos[0], y=pos[1],
            text=texts[name],
            showarrow=False,
            font=dict(size=10, color=colors['text']),
            align='center'
        )
    
    # Add arrows with straight paths
    arrows = [
        ('input', 'clip_visual'),
        ('clip_visual', 'pool'),
        ('pool', 'norm'),
        ('prompts', 'tokenize'),
        ('tokenize', 'embed'),
        ('embed', 'clip_text'),
        ('norm', 'sim_pos'),
        ('norm', 'sim_neg'),
        ('clip_text', 'sim_pos'),
        ('clip_text', 'sim_neg'),
        ('sim_pos', 'logits'),
        ('sim_neg', 'logits')
    ]
    
    for start, end in arrows:
        start_pos = positions[start]
        end_pos = positions[end]
        fig.add_shape(
            type="line",
            x0=start_pos[0], y0=start_pos[1],
            x1=end_pos[0], y1=end_pos[1],
            line=dict(color=colors['arrow'], width=1),
            opacity=0.8
        )
    
    # Update layout with minimal settings
    fig.update_layout(
        title='MLRSNet Model Architecture',
        showlegend=False,
        plot_bgcolor=colors['background'],
        paper_bgcolor=colors['background'],
        width=800,
        height=600,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # Save as PNG only (faster than HTML)
    fig.write_image('model_architecture.png', scale=1)
    print("Architecture diagram saved as 'model_architecture.png'")

if __name__ == "__main__":
    create_architecture_diagram() 