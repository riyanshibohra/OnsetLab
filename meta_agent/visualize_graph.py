"""
Meta-Agent Graph Visualization
==============================
Creates a visual flowchart of the LangGraph workflow.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_graph_visualization():
    """Create a beautiful visualization of the meta-agent graph."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 18))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 18)
    ax.axis('off')
    
    # Colors - Dark theme inspired
    colors = {
        'node': '#1e3a5f',           # Deep blue
        'node_text': '#ffffff',
        'decision': '#7c3aed',       # Purple
        'decision_text': '#ffffff',
        'start_end': '#059669',      # Emerald
        'arrow': '#64748b',
        'loop': '#f59e0b',           # Amber
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
    ax.text(7, 17.3, '🤖 OnsetLab Meta-Agent Workflow', 
           ha='center', va='center', fontsize=20, 
           color=colors['title'], fontweight='bold')
    ax.text(7, 16.7, 'MCP Discovery & Agent Generation Pipeline', 
           ha='center', va='center', fontsize=12, 
           color='#94a3b8', fontstyle='italic')
    
    # ===== NODES =====
    
    # START
    draw_node(7, 15.5, 'START', 'start_end', width=2, height=0.6)
    
    # parse_problem
    draw_node(7, 14.3, 'parse_problem', 'node')
    draw_arrow((7, 15.2), (7, 14.7))
    
    # Decision: has services?
    draw_node(7, 13, '?', 'decision')
    ax.text(7, 12.2, 'has services?', ha='center', fontsize=9, color='#94a3b8')
    draw_arrow((7, 13.9), (7, 13.5))
    
    # No services -> compile_results (shortcut)
    draw_node(10.5, 13, 'compile_results\n(skip to end)', 'node', width=3)
    draw_arrow((7.5, 13), (9, 13), label='no')
    
    # Yes -> search_mcp_servers
    draw_node(7, 11.2, 'search_mcp_servers', 'node')
    draw_arrow((7, 12.5), (7, 11.6), label='yes')
    
    # evaluate_mcp_results
    draw_node(7, 9.8, 'evaluate_mcp_results', 'node')
    draw_arrow((7, 10.8), (7, 10.2))
    
    # Decision: good_mcp?
    draw_node(7, 8.5, '?', 'decision')
    ax.text(7, 7.7, 'good_mcp?', ha='center', fontsize=9, color='#94a3b8')
    draw_arrow((7, 9.4), (7, 9.0))
    
    # good_mcp -> extract_schemas
    draw_node(4.5, 7, 'extract_schemas', 'node')
    draw_arrow((6.5, 8.3), (5.3, 7.4), label='yes', curved=-0.2)
    
    # no_mcp -> mark_as_api
    draw_node(9.5, 7, 'mark_as_api', 'node')
    draw_arrow((7.5, 8.3), (8.7, 7.4), label='no', curved=0.2)
    
    # Decision: more services?
    draw_node(7, 5.5, '?', 'decision')
    ax.text(7, 4.7, 'more services?', ha='center', fontsize=9, color='#94a3b8')
    draw_arrow((4.5, 6.6), (6.5, 5.8), curved=-0.1)
    draw_arrow((9.5, 6.6), (7.5, 5.8), curved=0.1)
    
    # Loop back to search
    draw_arrow((6.5, 5.7), (5, 10.5), color=colors['loop'], curved=-0.5, label='yes')
    draw_arrow((5, 10.5), (6.2, 11.2), color=colors['loop'], curved=-0.2)
    
    # compile_results (main path)
    draw_node(7, 4, 'compile_results', 'node')
    draw_arrow((7, 5.0), (7, 4.4), label='no')
    
    # Connect shortcut to main compile
    draw_arrow((10.5, 12.4), (10.5, 4), curved=0)
    draw_arrow((10.5, 4), (8.4, 4), curved=0)
    
    # filter_tools
    draw_node(7, 2.8, 'filter_tools', 'node')
    ax.text(9.2, 2.8, '📋 Max 15-20 tools', fontsize=8, color='#94a3b8')
    draw_arrow((7, 3.6), (7, 3.2))
    
    # generate_token_guides
    draw_node(7, 1.6, 'generate_token_guides', 'node')
    draw_arrow((7, 2.4), (7, 2.0))
    
    # generate_notebook
    draw_node(7, 0.4, 'generate_notebook', 'node')
    draw_arrow((7, 1.2), (7, 0.8))
    
    # END
    draw_node(7, -0.8, 'END', 'start_end', width=2, height=0.6)
    draw_arrow((7, 0), (7, -0.5))
    
    # Legend
    legend_x = 1.5
    legend_y = 3
    ax.text(legend_x, legend_y + 1, 'Legend:', fontsize=11, color=colors['title'], fontweight='bold')
    
    # Legend items
    draw_node(legend_x, legend_y, '', 'node', width=0.8, height=0.4)
    ax.text(legend_x + 0.8, legend_y, 'Node', fontsize=9, color='#94a3b8', va='center')
    
    draw_node(legend_x, legend_y - 0.7, '', 'decision')
    ax.text(legend_x + 0.8, legend_y - 0.7, 'Decision', fontsize=9, color='#94a3b8', va='center')
    
    draw_node(legend_x, legend_y - 1.4, '', 'start_end', width=0.8, height=0.4)
    ax.text(legend_x + 0.8, legend_y - 1.4, 'Start/End', fontsize=9, color='#94a3b8', va='center')
    
    ax.plot([legend_x - 0.3, legend_x + 0.3], [legend_y - 2.1, legend_y - 2.1], 
            color=colors['loop'], linewidth=3)
    ax.text(legend_x + 0.8, legend_y - 2.1, 'Loop Back', fontsize=9, color='#94a3b8', va='center')
    
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
