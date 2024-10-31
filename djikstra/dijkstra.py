import math

def dijkstra(graph, start, end):
    inf = math.inf
    neginf = -math.inf
    # Initialize distances and visited nodes
    distances = {node: inf for node in graph}
    distances[start] = 0
    visited = set()
    previous = {node: None for node in graph}

    while len(visited) < len(graph):
        # Find unvisited node with minimum distance
        min_dist = inf
        min_node = None
        for node in graph:
            if node not in visited and distances[node] < min_dist:
                min_dist = distances[node]
                min_node = node

        if min_node is None:
            break

        visited.add(min_node)

        # Update distances to neighbors
        for neighbor, weight in graph[min_node].items():
            if neighbor not in visited:
                new_dist = distances[min_node] + weight
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = min_node

    # Build path from start to end
    if distances[end] == inf:
        return None  # No path exists

    path = []
    current = end
    while current is not None:
        path.append(current)
        current = previous[current]
    path.reverse()

    return path, distances[end]

    
def visualize_graph(graph, path=None):
    import matplotlib.pyplot as plt
    import networkx as nx

    # Create a new directed graph
    G = nx.Graph()

    # Add edges with weights
    for node in graph:
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)

    # Get position layout for nodes
    pos = nx.spring_layout(G)

    plt.figure(figsize=(10,8))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    
    # If path is provided, highlight the path edges
    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=2)
    
    # Draw nodes
    if path:
        # Color nodes in the path differently
        node_colors = ['red' if node in path else 'lightblue' for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500)
    else:
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    
    nx.draw_networkx_labels(G, pos)
    
    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("Graph Visualization")
    plt.axis('off')
    plt.show()
        

if __name__ == "__main__":
    graph = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    
    # First show the graph without path
    visualize_graph(graph)
    
    # Then show the shortest path
    path, distance = dijkstra(graph, 'A', 'D')
    print(f"Shortest path: {path}")
    print(f"Total distance: {distance}")
    visualize_graph(graph, path)
