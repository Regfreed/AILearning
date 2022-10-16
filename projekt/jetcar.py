#%%
MAP_GRAPH = [

   # o1,o2,o3,o4,o5,o6,o7,r1,r2,r3,r4,c1,c2,s, k
    [0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 3], # o1
    [5, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # o2
    [0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5], # o3
    [0, 0, 3, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # o4
    [0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 0], # o5
    [0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0], # o6
    [0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0], # o7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2, 0], # r1
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2], # r2
    [0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0], # r3
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], # r4
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0], # c1
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 3, 0, 0, 2], # c2
    [5, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0], # s
    [3, 0, 5, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0], # k
]

NODE_DICT = {
    "o1":0,
    "o2":1,
    "o3":2,
    "o4":3,
    "o5":4,
    "o6":5,
    "o7":6,
    "r1":7,
    "r2":8,
    "r3":9,
    "r4":10,
    "c1":11,
    "c2":12,
    "s":13,
    "k":14,
    }



def numFromName(node, node_dict=NODE_DICT):

    node = node.lower()

    if node in node_dict.keys():
        return node_dict[node]
    else:
        return -1

def nameFromNum(node, node_dict=NODE_DICT):

    keys = list(node_dict.keys())

    if node < len(keys):
        return keys[node]
    else:
        return -1




def dijkstraShortestPath(beginning_node, end_node, graph=MAP_GRAPH):

    def addAdjacentWeights():
        current_distance = distances[current_node]
        for node,dist in enumerate(graph[current_node]):
            if dist > 0:
                if node in not_visited:
                    if (dist + current_distance) < distances[node]:
                        distances[node] = dist + current_distance
                        parents[node] = current_node

    def nextNearestNode():
        minimum = float('inf')

        for node in not_visited:
            if distances[node] < minimum:
                minimum = distances[node]
                minimum_node = node

        return minimum_node

    def getPath(child_node):
        
        if child_node == numFromName(beginning_node):
            return beginning_node
        
        return getPath(parents[child_node]) + " ---> " + nameFromNum(child_node)


    current_node = numFromName(beginning_node)
    not_visited = [i for i in range(len(graph))]
    not_visited.remove(current_node)

    distances = {i:float('inf') for i in range(len(graph))}
    distances[current_node] = 0

    parents = {i:None for i in range(len(graph))}

    while not_visited:

        addAdjacentWeights()
        current_node = nextNearestNode()
        not_visited.remove(current_node)


    path = getPath(numFromName(end_node))
    print("distance:",distances[numFromName(end_node)])
    print("path:",path)
# %%
