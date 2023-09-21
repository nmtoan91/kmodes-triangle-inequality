import matplotlib.pyplot as plt
import networkx as nx

#nx.draw(G, with_labels=True, font_weight='bold')
#plt.show()
def DrawGra(nearclusters):
    G = nx.DiGraph(directed=True)
    for i in range(len(nearclusters)):
        G.add_node(i)
    for i in range(len(nearclusters)):
        for j in range(len(nearclusters[i])):
            G.add_edge(i,nearclusters[i][j] )

    options = {
        'edge_color':'#999999',
        #'node_color': '#dddddd', 
        'node_size': 200,
        'width': 0.5,
        'arrowstyle': '-|>',
        'arrowsize': 5,
    }
    nx.draw_networkx(G, arrows=True, **options)
    filename = "0000"
    plt.savefig("D:/Dropbox/PHD/FIGURES/" + filename+ ".pdf",bbox_inches='tight')
    plt.show()