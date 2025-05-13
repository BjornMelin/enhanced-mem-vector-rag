"""
Knowledge graph visualization component for EMVR UI.

This module provides functionality for visualizing the knowledge graph.
"""

import json
import logging
from typing import Any

import chainlit as cl

from emvr.memory.memory_manager import memory_manager

# Configure logging
logger = logging.getLogger(__name__)


# ----- Graph Visualization Functions -----


async def prepare_graph_data(
    center_entity: str | None = None,
    max_nodes: int = 50,
    max_depth: int = 2,
) -> dict[str, Any]:
    """
    Prepare graph data for visualization.

    Args:
    ----
        center_entity: Optional center entity name
        max_nodes: Maximum number of nodes to include
        max_depth: Maximum relationship depth

    Returns:
    -------
        Graph data structure for visualization

    """
    try:
        # Initialize memory manager if needed
        await memory_manager.initialize()

        # Get the graph data
        if center_entity:
            # If a center entity is specified, get the local graph around it
            graph_result = await memory_manager._graphiti.execute_cypher(
                f"""
                MATCH path = (center {{name: $name}})-[*1..{max_depth}]-(related)
                WITH nodes(path) AS nodes, relationships(path) AS rels
                UNWIND nodes AS node
                WITH COLLECT(DISTINCT node) AS allNodes, rels
                UNWIND rels AS rel
                WITH allNodes, COLLECT(DISTINCT rel) AS allRels
                RETURN allNodes, allRels
                LIMIT {max_nodes}
                """,
                {"name": center_entity},
            )
        else:
            # Otherwise, get a sample of the full graph
            graph_result = await memory_manager.read_graph()

        # Process nodes and relationships for visualization
        nodes = []
        edges = []

        # Process nodes
        for node in graph_result.get("nodes", []):
            if len(nodes) >= max_nodes:
                break

            nodes.append(
                {
                    "id": node.get("name", ""),
                    "label": node.get("name", ""),
                    "title": node.get("entityType", "Entity"),
                    "group": node.get("entityType", "Entity"),
                    "properties": {
                        "type": node.get("entityType", "Entity"),
                        "observations": node.get("observations", []),
                    },
                },
            )

        # Process relationships
        for rel in graph_result.get("relationships", []):
            edges.append(
                {
                    "from": rel.get("from", ""),
                    "to": rel.get("to", ""),
                    "label": rel.get("relationType", "related"),
                    "title": rel.get("relationType", "related"),
                    "properties": {
                        "type": rel.get("relationType", "related"),
                    },
                },
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "status": "success",
        }

    except Exception as e:
        logger.exception(f"Error preparing graph data: {e}")
        return {
            "nodes": [],
            "edges": [],
            "error": str(e),
            "status": "error",
        }


# ----- UI Components -----


async def show_graph_visualization(
    center_entity: str | None = None,
    max_nodes: int = 50,
    max_depth: int = 2,
) -> None:
    """
    Show the knowledge graph visualization.

    Args:
    ----
        center_entity: Optional center entity name
        max_nodes: Maximum number of nodes to include
        max_depth: Maximum relationship depth

    """
    try:
        # Prepare the graph data
        graph_data = await prepare_graph_data(
            center_entity=center_entity,
            max_nodes=max_nodes,
            max_depth=max_depth,
        )

        if graph_data.get("status") == "error":
            await cl.Message(
                content=f"❌ Error preparing graph data: {graph_data.get('error')}",
                author="System",
            ).send()
            return

        # Convert to JSON for the iframe
        graph_json = json.dumps(graph_data)

        # Create an HTML visualization using Vis.js
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>EMVR Knowledge Graph</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style type="text/css">
                #graph {{
                    width: 100%;
                    height: 600px;
                    border: 1px solid lightgray;
                }}

                .controls {{
                    margin-bottom: 10px;
                }}

                .legend {{
                    margin-top: 10px;
                    font-size: 12px;
                }}

                .legend-item {{
                    display: inline-block;
                    margin-right: 15px;
                }}

                .color-box {{
                    display: inline-block;
                    width: 12px;
                    height: 12px;
                    margin-right: 5px;
                    border: 1px solid #888;
                }}
            </style>
        </head>
        <body>
            <div class="controls">
                <button id="zoom-in">Zoom In</button>
                <button id="zoom-out">Zoom Out</button>
                <button id="fit">Fit View</button>
                <select id="layout">
                    <option value="standard">Standard Layout</option>
                    <option value="hierarchical">Hierarchical Layout</option>
                    <option value="circular">Circular Layout</option>
                </select>
            </div>

            <div id="graph"></div>

            <div class="legend">
                <div class="legend-item"><div class="color-box" style="background-color: #97C2FC;"></div>Entity</div>
                <div class="legend-item"><div class="color-box" style="background-color: #FFCCCC;"></div>Component</div>
                <div class="legend-item"><div class="color-box" style="background-color: #C2F0C2;"></div>Document</div>
                <div class="legend-item"><div class="color-box" style="background-color: #FFE0B2;"></div>Decision</div>
            </div>

            <script type="text/javascript">
                // Parse the data
                const graphData = {graph_json};

                // Prepare the data for the visualization
                const nodes = new vis.DataSet(graphData.nodes);
                const edges = new vis.DataSet(graphData.edges);

                // Create the network
                const container = document.getElementById('graph');
                const data = {{
                    nodes: nodes,
                    edges: edges
                }};

                // Configure the visualization
                const options = {{
                    nodes: {{
                        shape: 'dot',
                        size: 16,
                        font: {{
                            size: 14
                        }},
                        borderWidth: 2,
                        shadow: true
                    }},
                    edges: {{
                        width: 2,
                        shadow: true,
                        smooth: {{
                            type: 'continuous'
                        }},
                        arrows: {{
                            to: {{enabled: true, scaleFactor: 0.5}}
                        }},
                        font: {{
                            size: 12,
                            align: 'middle'
                        }}
                    }},
                    physics: {{
                        stabilization: true,
                        barnesHut: {{
                            gravitationalConstant: -2000,
                            centralGravity: 0.3,
                            springLength: 150,
                            springConstant: 0.04,
                            damping: 0.09
                        }}
                    }},
                    groups: {{
                        Entity: {{color: "#97C2FC"}},
                        Component: {{color: "#FFCCCC"}},
                        Document: {{color: "#C2F0C2"}},
                        Decision: {{color: "#FFE0B2"}}
                    }},
                    interaction: {{
                        tooltipDelay: 200,
                        hideEdgesOnDrag: true,
                        hover: true,
                        navigationButtons: true,
                        keyboard: true
                    }}
                }};

                // Create the network
                const network = new vis.Network(container, data, options);

                // Add event listeners for controls
                document.getElementById('zoom-in').addEventListener('click', function() {{
                    network.zoomIn();
                }});

                document.getElementById('zoom-out').addEventListener('click', function() {{
                    network.zoomOut();
                }});

                document.getElementById('fit').addEventListener('click', function() {{
                    network.fit();
                }});

                document.getElementById('layout').addEventListener('change', function(e) {{
                    const layout = e.target.value;

                    if (layout === 'hierarchical') {{
                        network.setOptions({{
                            layout: {{
                                hierarchical: {{
                                    direction: 'UD',
                                    sortMethod: 'directed',
                                    levelSeparation: 150,
                                    nodeSpacing: 150
                                }}
                            }}
                        }});
                    }} else if (layout === 'circular') {{
                        network.setOptions({{
                            layout: {{
                                hierarchical: false
                            }},
                            physics: {{
                                stabilization: false,
                                barnesHut: {{
                                    gravitationalConstant: -10000,
                                    centralGravity: 0.3,
                                    springLength: 150,
                                    springConstant: 0.04
                                }}
                            }}
                        }});

                        // Position nodes in a circle
                        const nodeCount = nodes.length;
                        const radius = 300;
                        const angle = 2 * Math.PI / nodeCount;

                        nodes.forEach((node, i) => {{
                            const x = radius * Math.cos(angle * i);
                            const y = radius * Math.sin(angle * i);
                            node.x = x;
                            node.y = y;
                        }});

                        network.redraw();
                    }} else {{
                        network.setOptions({{
                            layout: {{
                                hierarchical: false
                            }},
                            physics: {{
                                stabilization: true,
                                barnesHut: {{
                                    gravitationalConstant: -2000,
                                    centralGravity: 0.3,
                                    springLength: 150,
                                    springConstant: 0.04,
                                    damping: 0.09
                                }}
                            }}
                        }});
                    }}
                }});

                // Add click event for nodes
                network.on("click", function (params) {{
                    if (params.nodes.length > 0) {{
                        const nodeId = params.nodes[0];
                        const node = nodes.get(nodeId);

                        alert(`Node: ${node.label}\\nType: ${node.properties.type}\\nObservations: ${node.properties.observations.join("\\n- ")}`);
                    }}
                }});
            </script>
        </body>
        </html>
        """

        # Create a temporary HTML file for the visualization
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".html",
            delete=False,
        ) as f:
            f.write(html_content)
            html_path = f.name

        # Show the visualization
        await cl.Message(
            content="📊 Knowledge Graph Visualization:",
            elements=[
                cl.Iframe(
                    path=html_path,
                    display="inline",
                    height="700px",
                ),
            ],
            author="EMVR",
        ).send()

    except Exception as e:
        logger.exception(f"Error showing graph visualization: {e}")
        await cl.Message(
            content=f"❌ Error showing graph visualization: {e!s}",
            author="System",
        ).send()


async def show_graph_explorer_ui() -> None:
    """Show the graph explorer UI."""
    # Create graph explorer elements
    elements = [
        cl.Text(
            name="center_entity",
            label="Center Entity (Optional)",
            placeholder="Enter entity name to focus on...",
        ),
        cl.Slider(
            name="max_nodes",
            label="Maximum Nodes",
            min=10,
            max=200,
            step=10,
            initial=50,
        ),
        cl.Slider(
            name="max_depth",
            label="Maximum Depth",
            min=1,
            max=5,
            step=1,
            initial=2,
        ),
    ]

    await cl.Message(
        content="🔍 Explore the knowledge graph:",
        elements=elements,
        author="EMVR",
    ).send()
