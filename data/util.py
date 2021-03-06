import graphviz
from graphviz import Digraph
import csv
import numpy as np

import pdb


def createGraph(word_list, file_id, no_edge_factor):
    createEdgeProbability = .65
    diGraph = Digraph(format='png')
    id_list = []
    #create nodes for words
    for i in range(len(word_list)):
        diGraph.node(str(i), word_list[i])
        id_list.append(i)
    edge_hashList = []
    pair_wordList = []

    # #create edges between nodes with some probability
    # if len(id_list) > 1:
    #     for i in range(len(id_list)):
    #         for j in range(len(id_list)):
    #             reverse_edgeID = str(j) + str(i)
    #             current_edgeID = str(i) + str(j)
    #             words = word_list[i] + " to " + word_list[j]
    #             if i != j and np.random.uniform() > createEdgeProbability and reverse_edgeID not in edge_hashList and \
    #             current_edgeID not in edge_hashList:
    #                 edge_hashList.append(current_edgeID)
    #                 pair_wordList.append(words)
    # diGraph.edges(edge_hashList)


    totalNodeCount = int(np.ceil(len(word_list) * (len(word_list)-1) * no_edge_factor))
    # print(totalNodeCount)
    nodeCount = 0
        #create edges between nodes with some probability

    coordinateList = []
    if len(id_list) > 1:
        for i in range(len(id_list)):
            for j in range(len(id_list)):
                element = (i,j)
                
                if i != j and element not in coordinateList:
                    coordinateList.append(element)

    randIndex = np.random.choice(len(coordinateList), totalNodeCount, replace=False)
    # print(randIndex)
    connectEdges = np.asarray(coordinateList)[randIndex]

    for i,j in connectEdges:
        current_edgeID = str(i) + str(j)
        edge_hashList.append(current_edgeID)
        words = word_list[i] + " to " + word_list[j]
        pair_wordList.append(words)


    diGraph.edges(edge_hashList)

    #create plainText
    plain_graph = Digraph(format='plain')
    for i in range(len(word_list)):
        plain_graph.node(str(i), word_list[i])
        id_list.append(i)

    plain_graph.edges(edge_hashList)


    f = open('descriptions/' + str(file_id) + ".txt", "w")
    for i in pair_wordList:
        f.write(i)
        f.write("\n")
    f.close()

    
    #create graph image
    diGraph.render('data/' + str(file_id), view=False)  

    #create plain graph
    plain_graph.render('plain_data/' + str(file_id), view=False)  

def createPetri(word_list, file_id):
    createEdgeProbability = .4
    diGraph = Digraph(format='png')
    id_list_node = []
    id_list_trans = []
    #create nodes for words
    for i in range(len(word_list)//2):
        diGraph.node(str(i), word_list[i])
        id_list_node.append(i)
    for i in range(len(word_list)-len(word_list)//2):
        diGraph.node(str(i+len(word_list)//2),word_list[i+len(word_list)//2],shape='box')
        id_list_trans.append(i+len(word_list)//2)
    edge_hashList = []
    pair_wordList = []

    #create edges between nodes with some probability
    if len(id_list_node)+ len(id_list_trans) > 1:
        for i in id_list_node:
            for j in id_list_trans:
                reverse_edgeID = str(j) + str(i)
                current_edgeID = str(i) + str(j)
                words = word_list[i] + " to " + word_list[j]
                words_rev = word_list[j] + " to " + word_list[i]
                if np.random.uniform() > createEdgeProbability:
                    edge_hashList.append(current_edgeID)
                    pair_wordList.append(words)
                if np.random.uniform() > createEdgeProbability:
                    edge_hashList.append(reverse_edgeID)
                    pair_wordList.append(words_rev)
    diGraph.edges(edge_hashList)

    #create plainText for parsing
    plain_graph = Digraph(format='plain')
    for i in range(len(word_list)//2):
        plain_graph.node(str(i), word_list[i])
    for i in range(len(word_list)-len(word_list)//2):
        plain_graph.node(str(i+len(word_list)//2),word_list[i+len(word_list)//2],shape='box')

    plain_graph.edges(edge_hashList)


    f = open('descriptions/' + str(file_id) + ".txt", "w")
    for i in pair_wordList:
        f.write(i)
        f.write("\n")
    f.close()

    
    #create graph image
    diGraph.render('data/' + str(file_id), view=False)  

    #create plain graph
    plain_graph.render('plain_data/' + str(file_id), view=False) 

def offsetEquation(graphY):
    return int(graphY * 96 + 5.5)


def parsePlainFile(path):
	node_x_list = []
	node_y_list = []
	node_rx_list = []
	node_ry_list = []
	node_type_list = []


	with open(path) as fp:
		line = fp.readline()
		offset_y = (float)(line.split(" ")[3])
		while line:
			line = fp.readline()

			if "node" in line:

				node_x_list.append((float)(line.split(" ")[2]))
				node_y_list.append((float)(line.split(" ")[3]))
				node_rx_list.append((int)((float)(line.split(" ")[4]) * 100)/2)
				node_ry_list.append((int)((float)(line.split(" ")[5]) * 100)/2)
				node_type_list.append(line.split(" ")[8])
                
	return offset_y, node_x_list, node_y_list, node_rx_list, node_ry_list, node_type_list


def remove_nodes(cx, cy, rx, ry, temp, node_type='ellipse'):
    image_array = temp.copy()
    start_x = max(int(cx- rx) -1,0)
    end_x = min(int(cx + rx) + 1, temp.shape[1]-1)
    start_y = max(int(cy - ry)-1,0)
    end_y = min(int(cy + ry)+1, temp.shape[0]-1)
    for i in range(start_y,end_y):
        for j in range(start_x,end_x):
            if node_type == "ellipse":
                if (i - cy)**2/ry**2 + (j - cx)**2/rx**2 <= 1.0: 
                    image_array[i][j] = True
            elif node_type == "box":
                if max(abs(i-cy)/ry, abs(j-cx)/rx) <=1.0:
                    image_array[i][j] = True
    # image_array[start_y:end_y, start_x:end_x] = True
    return image_array

def add_nodes(cx, cy, rx, ry, temp):
    image_array = temp.copy()
    start_x = max(int(cx- rx) -1,0)
    end_x = min(int(cx + rx) + 1, temp.shape[1]-1)
    start_y = max(int(cy - ry)-1,0)
    end_y = min(int(cy + ry)+1, temp.shape[0]-1)
    for i in range(start_y,end_y):
        for j in range(start_x,end_x):
            if (i - cy)**2/ry**2 + (j - cx)**2/rx**2 <= 1.0: 
                image_array[i][j] = False
    # image_array[start_y:end_y, start_x:end_x] = True
    return image_array

def add_boxes(cx, cy, rx, ry, temp):
    image_array = temp.copy()
    start_x = max(int(cx- rx) -1,0)
    end_x = min(int(cx + rx) + 1, temp.shape[1]-1)
    start_y = max(int(cy - ry)-1,0)
    end_y = min(int(cy + ry)+1, temp.shape[0]-1)
    image_array[start_y:end_y, start_x:end_x] = False
    return image_array

def offsetEquation(graphY):
    return int(graphY * 96 + 5.5)




