"""
Meta-Agent Graph Visualization
==============================
Creates a visual flowchart of the LangGraph workflow (Registry-Based v2.0).
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_graph_visualization():
    """Create a beautiful visualization of the registry-based meta-agent graph."""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 14))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Colors - Dark theme inspired
    colors = {
        'node': '#1e3a5f',           # Deep blue
        'node_text': '#ffffff',
        'decision': '#7c3aed',       # Purple
        'decision_text': '#ffffff',
        'start_end': '#059669',      # Emerald
        'arrow': '#64748b',
        'hitl': '#f59e0b',           # Amber (HITL)
        'bg': '#0f172a',             # Slate dark
        'title': '#e2e8f0',
    }
    
    fig.patch.set_facecolor(colors['bg'])
    ax.set_facecolor(colors['bg'])
    
    def draw_node(x, y, text, node_type='node', width=2.8, height=0.8):
        """Draw a rounded rectangle node."""
        color = colors.get(node_type, colors['node'])
        text_color = colors.get(f'{node_type}_text', colors['node_text'])
        
        if node_type == 'decision':
            # Diamond shape for decisions
            diamond = patches.RegularPolygon(
                (x, y), numVertices=4, radius=0.7,
                orientation=np.pi/4,
                facecolor=color, edgecolor='white', linewidth=2
            )
            ax.add_patch(diamond)
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=8, color=text_color, fontweight='bold')
        else:
            rect = FancyBboxPatch(
                (x - width/2, y - height/2), width, height,
                boxstyle="round,rounding_size=0.2",
                facecolor=color, edgecolor='white', linewidth=2
            )
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=10 if node_type != 'start_end' else 12,
                   color=text_color, fontweight='bold')
    
    def draw_arrow(start, end, color=None, style='->', curved=False, label=''):
        """Draw an arrow between two points."""
        arrow_color = color or colors['arrow']
        
        if curved:
            style_str = f"arc3,rad={curved}"
        else:
            style_str = "arc3,rad=0"
        
        arrow = FancyArrowPatch(
            start, end,
            arrowstyle='-|>',
            mutation_scale=15,
            color=arrow_color,
            linewidth=2,
            connectionstyle=style_str
        )
        ax.add_patch(arrow)
        
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
                   color=arrow_color, fontstyle='italic')
    
    # Title
    ax.text(6, 13.3, '🤖 OnsetLab Meta-Agent (Registry-Based)', 
           ha='center', va='center', fontsize=18, 
           color=colors['title'], fontweight='bold')
    ax.text(6, 12.8, 'Simplified Workflow with Human-in-the-Loop', 
           ha='center', va='center', fontsize=11, 
           color='#94a3b8', fontstyle='italic')
    
    # ===== NODES =====
    
    # START
    draw_node(6, 12, 'START', 'start_end', width=2, height=0.6)
    
    # parse_problem
    draw_node(6, 10.8, 'parse_problem', 'node')
    ax.text(8.5, 10.8, 'Extract services', fontsize=8, color='#94a3b8', style='italic')
    draw_arrow((6, 11.7), (6, 11.2))
    
    # load_registry
    draw_node(6, 9.6, 'load_registry', 'node')
    ax.text(8.5, 9.6, 'Load from JSON', fontsize=8, color='#94a3b8', style='italic')
    draw_arrow((6, 10.4), (6, 10.0))
    
    # filter_tools
    draw_node(6, 8.4, 'filter_tools', 'node')
    ax.text(8.5, 8.4, 'LLM selects 15-20', fontsize=8, color='#94a3b8', style='italic')
    draw_arrow((6, 9.2), (6, 8.8))
    
    # process_feedback (HITL)
    draw_node(6, 7.2, 'process_feedback', 'hitl')
    ax.text(8.5, 7.2, '👤 User reviews', fontsize=8, color='#fbbf24', style='italic', fontweight='bold')
    draw_arrow((6, 8.0), (6, 7.6))
    
    # Decision: feedback action
    draw_node(6, 6, '?', 'decision')
    ax.text(6, 5.2, 'feedback?', ha='center', fontsize=9, color='#94a3b8')
    draw_arrow((6, 6.8), (6, 6.5))
    
    # Routes from decision
    # approved -> generate_token_guides
    draw_node(6, 4.2, 'generate_token_guides', 'node')
    draw_arrow((6, 5.5), (6, 4.6), label='approved')
    
    # add_tools -> loop back to load_registry
    draw_node(3, 6, 'load_registry', 'node', width=2.2, height=0.6)
    draw_arrow((5.5, 6), (4.1, 6), label='add', curved=-0.1)
    draw_arrow((3, 6.3), (3, 9.3), color=colors['hitl'], curved=-0.3)
    draw_arrow((3, 9.3), (4.5, 9.6), color=colors['hitl'], curved=-0.1)
    
    # remove_tools -> loop back to filter_tools
    draw_node(9, 6, 'filter_tools', 'node', width=2.2, height=0.6)
    draw_arrow((6.5, 6), (7.9, 6), label='remove', curved=0.1)
    draw_arrow((9, 6.3), (9, 8.1), color=colors['hitl'], curved=0.3)
    draw_arrow((9, 8.1), (7.5, 8.4), color=colors['hitl'], curved=0.1)
    
    # generate_notebook
    draw_node(6, 3, 'generate_notebook', 'node')
    draw_arrow((6, 3.8), (6, 3.4))
    
    # END
    draw_node(6, 1.8, 'END', 'start_end', width=2, height=0.6)
    draw_arrow((6, 2.6), (6, 2.1))
    
    # Legend
    legend_x = 1
    legend_y = 3.5
    ax.text(legend_x, legend_y + 0.8, 'Legend:', fontsize=11, color=colors['title'], fontweight='bold')
    
    # Legend items
    draw_node(legend_x, legend_y, '', 'node', width=0.8, height=0.4)
    ax.text(legend_x + 0.8, legend_y, 'Node', fontsize=9, color='#94a3b8', va='center')
    
    draw_node(legend_x, legend_y - 0.7, '', 'decision')
    ax.text(legend_x + 0.8, legend_y - 0.7, 'Decision', fontsize=9, color='#94a3b8', va='center')
    
    draw_node(legend_x, legend_y - 1.4, '', 'hitl', width=0.8, height=0.4)
    ax.text(legend_x + 0.8, legend_y - 1.4, 'HITL', fontsize=9, color='#94a3b8', va='center')
    
    draw_node(legend_x, legend_y - 2.1, '', 'start_end', width=0.8, height=0.4)
    ax.text(legend_x + 0.8, legend_y - 2.1, 'Start/End', fontsize=9, color='#94a3b8', va='center')
    
    ax.plot([legend_x - 0.3, legend_x + 0.3], [legend_y - 2.8, legend_y - 2.8], 
            color=colors['hitl'], linewidth=3)
    ax.text(legend_x + 0.8, legend_y - 2.8, 'Feedback Loop', fontsize=9, color='#94a3b8', va='center')
    
    # Stats
    stats_y = 0.5
    ax.text(6, stats_y, '6 Nodes | 1 Decision | 3 LLM Calls', 
           ha='center', fontsize=9, color='#64748b', style='italic')
    
    # Save
    plt.tight_layout()
    plt.savefig('meta_agent/meta_agent_graph.png', 
                facecolor=colors['bg'], 
                dpi=150, 
                bbox_inches='tight',
                pad_inches=0.5)
    plt.savefig('meta_agent/meta_agent_graph.svg', 
                facecolor=colors['bg'],
                bbox_inches='tight',
                pad_inches=0.5)
    
    print("✅ Saved: meta_agent/meta_agent_graph.png")
    print("✅ Saved: meta_agent/meta_agent_graph.svg")
    
    return fig


if __name__ == "__main__":
    fig = create_graph_visualization()
    print("✅ Graph visualization created successfully!")
    # plt.show()  # Uncomment for interactive viewing
