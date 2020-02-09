import matplotlib.pyplot as plt
import pdb
import numpy as np
import cv2
from utils.dataset import BasicDataset
import sys
import os
from math import sqrt

def isSafe(mask, i, j, visited):
        # row number is in range, column number
        # is in range and value is 1
        # and not yet visited
        return (i >= 0 and i < mask.shape[0] and
                j >= 0 and j < mask.shape[1] and
                not visited[i][j] and mask[i][j]==255)

def DFS(mask,i,j,visited, threshold):            # prints all vertices in DFS manner from a given source.
                                # Initially mark all verices as not visited
        # Create a stack for DFS
        stack = []

        # Push the current source node.
        stack.append((i,j))
        rowNbr = [-1, -1, -1,  0, 0,  1, 1, 1]
        colNbr = [-1,  0,  1, -1, 1, -1, 0, 1]
        left_coord = sys.maxsize
        right_coord = -sys.maxsize -1
        bot_coord = -sys.maxsize -1
        top_coord = sys.maxsize
        count = 0
        while (len(stack)):
            # Pop a vertex from stack and print it
            s = stack[-1]
            stack.pop()

            # Stack may contain same vertex twice. So
            # we need to print the popped item only
            # if it is not visited.
            if (not visited[s[0]][s[1]]):
                if s[1]< left_coord:
                    left_coord = s[1]
                if s[1]> right_coord:
                    right_coord = s[1]
                if s[0]< top_coord:
                    top_coord = s[0]
                if s[0] > bot_coord:
                    bot_coord = s[0]
                visited[s[0]][s[1]] = True
                count+=1

            # Get all adjacent vertices of the popped vertex s
            # If a adjacent has not been visited, then push it
            # to the stack.

            for k in range(8):
                if isSafe(mask, s[0] + rowNbr[k], s[1] + colNbr[k], visited):
                    stack.append((s[0] + rowNbr[k], s[1] + colNbr[k]))
        # print(count)
        if count<threshold:
            return None
        return (left_coord,right_coord, top_coord, bot_coord)

def check_element(x,y, islands, radius=1.0, in_type="ellipse"):
    # Check if coordinate lies in one of the nodes
    elem_in_islands=False
    island_found=None
    for island in islands:
        (left_coord,right_coord, top_coord, bot_coord) = island
        y_centre = left_coord/2 + right_coord/2
        x_centre = top_coord/2 + bot_coord/2
        y_radius = (right_coord - left_coord)/2
        x_radius = (bot_coord - top_coord)/2
        if in_type=='box':
            if (((x - x_centre)/x_radius)**2 + ((y - y_centre)/y_radius)**2 <= radius**2):
                elem_in_islands = True
                island_found = island
                break
        else:
            if (max(abs(x - x_centre)/x_radius, abs(y - y_centre)/y_radius) <= radius):
                elem_in_islands = True
                island_found = island
                break
    return elem_in_islands, island_found

def convert_number(islands_connected, islands_copy):
    # Convert from tuple coordinate based representation to node numbers
    numbers = []
    for island in islands_connected:
        numbers.append(islands_copy.index(island))
    return numbers

def removeDuplicates(lst):
    return list(set([i for i in lst]))

def sort_closest_dir(rowNbr,colNbr, dir):
    dist = []
    for k,(row,col) in enumerate(zip(rowNbr,colNbr)):
        v1 = np.array([row,col])
        v0 = np.array(dir)
        # pdb.set_trace()
        angle = np.abs(np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)))
        dist.append(angle)
    # pdb.set_trace()

    return np.array(rowNbr)[np.argsort(dist)], np.array(colNbr)[np.argsort(dist)]

def dfs_island_check(mask, img, island, islands, in_type):
    #Finds all nodes connected to the current node in the image
    stack = []
    dir = None
    # Push the current source node.
    (left_coord,right_coord, top_coord, bot_coord) = island
    i = (top_coord+bot_coord)//2
    j = (left_coord+right_coord)//2
    stack.append(((i,j),None))
    rowNbr = [-1, -1, -1,  0, 0,  1, 1, 1]
    colNbr = [-1,  0,  1, -1, 1, -1, 0, 1]
    visited = [[False for j in range(mask.shape[1])]for i in range(mask.shape[0])]
    islands_connected = []
    num_pixels_connection = []
    box_connection = []
    points = []

    pixel_num_out = 0
    pixel_out = False
    while (len(stack)):
        # Pop a vertex from stack and print it
        s, dir_new = stack[-1]
        if dir is None:
            if dir_new is not None:
                dir = np.array(dir_new)
        else:
            dir = np.array(dir)*0.8+np.array(dir_new)*0.2
        if dir is not None:
            rowNbr,colNbr = sort_closest_dir(rowNbr,colNbr,dir)
        stack.pop()

        # Stack may contain same vertex twice. So
        # we need to print the popped item only
        # if it is not visited.
        if (not visited[s[0]][s[1]]):
            if s[1]< left_coord:
                left_coord = s[1]
            if s[1]> right_coord:
                right_coord = s[1]
            if s[0]< top_coord:
                top_coord = s[0]
            if s[0] > bot_coord:
                bot_coord = s[0]
            visited[s[0]][s[1]] = True

        # Get all adjacent vertices of the popped vertex s
        # If a adjacent has not been visited, then push it
        # to the stack.
        islands_copy = islands.copy()
        islands_copy.remove(island)
        box_radius = 7

        if in_type=='ellipse':
            elem_in_islands, __ = check_element(s[0],s[1], [island], radius = 1.1, in_type='box')
        if in_type=='box':
            elem_in_islands, __ = check_element(s[0],s[1], [island], radius = 1.1, in_type='ellipse')
        if elem_in_islands:
            for k in range(8):
                if isSafe(mask, s[0] + rowNbr[k], s[1] + colNbr[k], visited):
                    elem_in_islands, island_found = check_element(s[0] + rowNbr[k],s[1] + colNbr[k], islands_copy, radius=1.1, in_type=in_type)
                    # Add to stack if not reached another node
                    if not elem_in_islands:
                        stack.append(((s[0] + rowNbr[k], s[1] + colNbr[k]),[rowNbr[k],colNbr[k]]))
                        points.append((s[0] + rowNbr[k], s[1] + colNbr[k]))
                    #Check if island found has already been reached though dfs
                    elif island_found not in islands_connected:
                        print("Nothing is impossible")
                        islands_connected.append(island_found)
                        # Get number of pixels in a box around the connection point
                        num_pixels_connection.append(np.count_nonzero(img[s[0]+ rowNbr[k]-box_radius:s[0]+rowNbr[k]+box_radius
                        ,s[1]+ colNbr[k]-box_radius:s[1]+ colNbr[k]+box_radius]))
                        box_connection.append((s[0]+ rowNbr[k]-box_radius,s[0]+rowNbr[k]+box_radius
                        ,s[1]+ colNbr[k]-box_radius,s[1]+ colNbr[k]+box_radius))
                if isSafe(mask, s[0] + 2*rowNbr[k], s[1] + 2*colNbr[k], visited):
                    elem_in_islands, island_found = check_element(s[0] + 2*rowNbr[k],s[1] + 2*colNbr[k], islands_copy, radius=1.1, in_type=in_type)
                    # Add to stack if not reached another node
                    if not elem_in_islands:
                        stack.append(((s[0] + 2*rowNbr[k], s[1] + 2*colNbr[k]),[2*rowNbr[k],2*colNbr[k]]))
                        points.append((s[0] + 2*rowNbr[k], s[1] + 2*colNbr[k]))
                    #Check if island found has already been reached though dfs
                    elif island_found not in islands_connected:
                        print("Nothing is impossible")
                        islands_connected.append(island_found)
                        # Get number of pixels in a box around the connection point
                        num_pixels_connection.append(np.count_nonzero(img[s[0]+ 2*rowNbr[k]-box_radius:s[0]+2*rowNbr[k]+box_radius
                        ,s[1]+ 2*colNbr[k]-box_radius:s[1]+ 2*colNbr[k]+box_radius]))
                        box_connection.append((s[0]+ 2*rowNbr[k]-box_radius,s[0]+2*rowNbr[k]+box_radius
                        ,s[1]+ 2*colNbr[k]-box_radius,s[1]+ 2*colNbr[k]+box_radius))
        else:
            if not pixel_out:
                pixel_out = True
                pixel_num_out = np.count_nonzero(img[s[0]-box_radius:s[0]+box_radius\
                ,s[1]-box_radius:s[1]+box_radius])
                box_connection.append((s[0]-box_radius,s[0]+box_radius
                ,s[1]-box_radius,s[1]+box_radius))
            for k in range(8):
                if isSafe(mask, s[0] + rowNbr[k], s[1] + colNbr[k], visited):
                    elem_in_islands, island_found = check_element(s[0] + rowNbr[k],s[1] + colNbr[k], islands_copy, radius =1.1, in_type=in_type)
                    # Add to stack if not reached another node
                    if not elem_in_islands:
                        stack.append(((s[0] + rowNbr[k], s[1] + colNbr[k]),[rowNbr[k],colNbr[k]]))
                        points.append((s[0] + rowNbr[k], s[1] + colNbr[k]))
                        break
                    #Check if island found has already been reached though dfs
                    elif island_found not in islands_connected:
                        pixel_out = False
                        # Get number of pixels in a box around the connection point
                        if pixel_num_out<np.count_nonzero(img[s[0]+ rowNbr[k]-box_radius:s[0]+rowNbr[k]+box_radius
                        ,s[1]+ colNbr[k]-box_radius:s[1]+ colNbr[k]+box_radius]):
                            num_pixels_connection.append(1)
                            box_connection.append((s[0]+ rowNbr[k]-box_radius,s[0]+rowNbr[k]+box_radius
                            ,s[1]+ colNbr[k]-box_radius,s[1]+ colNbr[k]+box_radius))
                            islands_connected.append(island_found)
                        break
                if isSafe(mask, s[0] + 2*rowNbr[k], s[1] + 2*colNbr[k], visited):
                    elem_in_islands, island_found = check_element(s[0] + 2*rowNbr[k],s[1] + 2*colNbr[k], islands_copy, radius = 1.1, in_type=in_type)
                    # Add to stack if not reached another node
                    if not elem_in_islands:
                        stack.append(((s[0] + 2*rowNbr[k], s[1] + 2*colNbr[k]),[2*rowNbr[k],2*colNbr[k]]))
                        points.append((s[0] + 2*rowNbr[k], s[1] + 2*colNbr[k]))
                        break
                    #Check if island found has already been reached though dfs
                    elif island_found not in islands_connected:
                        pixel_out = False
                        if pixel_num_out<np.count_nonzero(img[s[0]+ 2*rowNbr[k]-box_radius:s[0]+2*rowNbr[k]+box_radius
                        ,s[1]+ 2*colNbr[k]-box_radius:s[1]+ 2*colNbr[k]+box_radius]):
                            islands_connected.append(island_found)
                            # Get number of pixels in a box around the connection point
                            num_pixels_connection.append(1)
                            box_connection.append((s[0]+ 2*rowNbr[k]-box_radius,s[0]+2*rowNbr[k]+box_radius
                            ,s[1]+ 2*colNbr[k]-box_radius,s[1]+ 2*colNbr[k]+box_radius))
                        break

    return islands_connected, num_pixels_connection, box_connection, points

def countIslands(mask, threshold):
    # Make a bool array to mark visited cells.
    # Initially all cells are unvisited
    visited = [[False for j in range(mask.shape[1])]for i in range(mask.shape[0])]

    # Initialize count as 0 and travese
    # through the all cells of
    # given matrix
    count = 0
    islands = []
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # If a cell with value 1 is not visited yet,
            # then new island found
            if visited[i][j] == False and mask[i][j] == 255:
                coord = DFS(mask, i, j, visited, threshold)
                if coord is not None:
                    islands.append(coord)

    return islands

def plot_img_and_mask(img, mask, fn):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img,cmap='gray')
    kernel_dilation = np.ones((5,5), np.uint8)
    kernel_erosion = np.ones((5,5),np.uint8)
    threshold = 400
    islands = []
    if classes > 1:
        for i in range(classes):
            if i==2:
                break
            ax[i+1].set_title(f'Output mask (class {i+1})')
            # pdb.set_trace()
            mask_draw = np.uint8(mask[i,:,:])
            islands += countIslands(mask_draw, threshold)
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            # fontScale
            fontScale = 0.5
            color = 127
            # print (islands)
            #Erode to get rid of the ellipse outline
            mask_draw_erode = cv2.erode(mask_draw, kernel_erosion, iterations=2)
            # if i<=1:
                # for iter, island in enumerate(islands):
                #     (left_coord,right_coord, top_coord, bot_coord) = island
                #     k = (top_coord+bot_coord)//2
                #     j = (left_coord+right_coord)//2
                #     mask_draw_erode = cv2.putText(mask_draw_erode, str(iter), (j,k), font,
                #        fontScale, color)
            # mask_draw = cv2.dilate(mask_draw, kernel_dilation, iterations=2)
            # mask_draw = cv2.erode(mask_draw, kernel_erosion, iterations=2)
            # ax[i+1].imshow(mask_draw, cmap='gray')

            if i==0:
                mask_box = mask_draw
            else:
                mask_ellipse = mask_draw
            if i==0:
                mask_box_erode = mask_draw_erode
            else:
                mask_ellipse_erode = mask_draw_erode
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    # mask_ellipse_erode = cv2.resize(mask_ellipse_erode,(img.shape[1],img.shape[0]))

    mask_box = cv2.resize(mask_box,(img.shape[1],img.shape[0]))*255
    mask_ellipse = cv2.resize(mask_ellipse,(img.shape[1],img.shape[0]))*255
    mask_box_erode = cv2.resize(mask_box_erode,(img.shape[1],img.shape[0]))*255
    mask_ellipse_erode = cv2.resize(mask_ellipse_erode,(img.shape[1],img.shape[0]))*255
    # Get all words
    letters = np.uint8((mask_ellipse_erode+mask_box_erode)*((255-img)/255))
    # letters = cv2.dilate(letters,kernel)
    # letters = cv2.erode(letters,kernel)

    mask_box_dilate = cv2.dilate(mask_box, kernel_erosion, iterations=2)
    mask_box_dilate = cv2.resize(mask_box_dilate,(img.shape[1],img.shape[0]))
    islands_box_dilated = countIslands(mask_box_dilate, threshold)

    mask_ellipse_dilate = cv2.dilate(mask_ellipse, kernel_erosion, iterations=2)
    mask_ellipse_dilate = cv2.resize(mask_ellipse_dilate,(img.shape[1],img.shape[0]))
    islands_ellipse_dilated = countIslands(mask_ellipse_dilate, threshold)
    # islands_eroded = countIslands(mask_draw_erode, threshold)

    #Get nodes+edges
    edges = (255-mask_ellipse_dilate - mask_box_dilate)/255
    edges = (1 - np.uint8(edges*img + mask_box_dilate+mask_ellipse_dilate)//255)*255
    node_edge = np.uint8(edges + mask_box_dilate + mask_ellipse_dilate)

    #Binarizing the image
    node_edge[node_edge>127] = 255
    node_edge[node_edge<=127] = 0

    #Binarize edges
    edges[edges>127]=255
    edges[edges<127]=0
    # skel = skeletonize(255-edges)
    # out = skeleton_endpoints(skel)
    for iter, island in enumerate(islands_box_dilated+islands_ellipse_dilated):
        (left_coord,right_coord, top_coord, bot_coord) = island
        k = (top_coord+bot_coord)//2
        j = (left_coord+right_coord)//2
        node_edge = cv2.putText(node_edge, str(iter), (j,k), font,
           fontScale, color)
    # ax[1].imshow(node_edge,cmap='gray')
    # ax[2].imshow(mask_box_dilate,cmap='gray')
    # ax[3].imshow(mask_ellipse_dilate,cmap='gray')
    # ax[3].imshow(edges,cmap='gray')
    # Save images
    # plt.xticks([]), plt.yticks([])
    # plt.show()
    for iter, island in enumerate(islands_box_dilated+islands_ellipse_dilated):
        (left_coord,right_coord, top_coord, bot_coord) = island
        # print(island)
        box = letters[top_coord+15:bot_coord-15,left_coord+15:right_coord-15]
        box = cv2.resize(box,(500,255))
        plt.imsave("../ocr/demo_image/" + str(iter) + ".png",box,cmap='gray')

    os.system("python ../ocr/demo.py --Transformation TPS --FeatureExtraction ResNet --SequenceModeling \
                BiLSTM --Prediction Attn --image_folder ../ocr/demo_image/ --saved_model ../ocr/TPS-ResNet-BiLSTM-Attn.pth")
    log = open(f'./log_demo_result.txt', 'r')
    node_name_dict = {}
    count = 0
    for line in log:
        if count == len(islands_box_dilated + islands_ellipse_dilated):
            break
        img_name = int(line.split()[0].split("/")[-1].split(".")[0])
        node_name = line.split()[1]
        node_name_dict[img_name] = node_name
        count+=1

    visited = [[False for j in range(node_edge.shape[1])]for i in range(node_edge.shape[0])]
    # print(islands)
    islands_ellipse_copy = islands_ellipse_dilated.copy()
    islands_box_copy = islands_box_dilated.copy()
    # print(len(islands_copy))
    #Initialize the graph
    graph = np.zeros((len(islands_ellipse_copy)+len(islands_box_copy),len(islands_ellipse_copy)+len(islands_box_copy)))
    node_edge_copy = node_edge.copy()
    for iter, island in enumerate(islands_box_copy+islands_ellipse_copy):
        # islands.remove(island)
        if iter<len(islands_box_copy):
            in_type = "box"
        else:
            in_type = "ellipse"
        islands_connected, num_pixels_connection, box_connection, points = dfs_island_check(node_edge\
                    , edges, island, islands_box_copy+islands_ellipse_copy, in_type)
        islands_numbered = convert_number(islands_connected,islands_box_copy+islands_ellipse_copy)
        for vertex,numbered in zip(islands_numbered, num_pixels_connection):
            graph[iter][vertex] = numbered
            #Draw rectangle along the connection of nodes and edges
            # node_edge_copy = cv2.putText(node_edge_copy, str(iter)+"."+str(vertex), (y1,x1), font,
            #    fontScale, color)
        for box in box_connection:
            print(box)
            (x1,x2,y1,y2) = box
            node_edge_copy = cv2.rectangle(node_edge_copy,(y1,x1),(y2,x2),color)
        if iter==3:
            for point in points:
                node_edge_copy[point[0],point[1]] = 127
        print(iter)
        print(islands_numbered)
    ax[i].imshow(edges,cmap='gray')
    ax[i+1].imshow(node_edge_copy,cmap='gray')

    # plt.xticks([]), plt.yticks([])
    # plt.show()

    #Check the direction of the connection between 2 nodes


    # for i in range(len(islands_box_copy+islands_ellipse_copy)):
    #     for j in range(len(islands_box_copy+islands_ellipse_copy)):
    #         if graph[i][j]>0 and graph[i][j]>=graph[j][i]:
    #             graph[j][i]=0
    #             graph[i][j]=1


    for i in range(len(node_name_dict)):
        for j in range(len(node_name_dict)):
            if graph[i][j] == 1:
                print(node_name_dict[i]+","+node_name_dict[j])

    file = fn.strip().split("/")[-1].split(".")[0]
    descriptions = open(f'../data/descriptions/'+file +'.txt', 'r')
    graph_gt = np.zeros((len(islands_box_copy+islands_ellipse_copy),len(islands_box_copy+islands_ellipse_copy)))
    # pdb.set_trace()
    for line in descriptions:
        first = line.strip().split(" ")[0]
        second = line.strip().split(" ")[-1]
        try:
            first_idx = list(node_name_dict.keys())[list(node_name_dict.values()).index(first.lower())]
            second_idx = list(node_name_dict.keys())[list(node_name_dict.values()).index(second.lower())]
            graph_gt[first_idx][second_idx] = 1
        except:
            continue

    if (np.all(graph_gt == graph)):
        return 1
    else:
        return 0
