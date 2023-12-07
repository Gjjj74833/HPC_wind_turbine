clear;
clc;

% MATH 3134 Research Project
% Probabilistic Roadmap Path Planning with A* Graph Search Algorithm
% Yihan Liu, Zelin Shen
% (12-01-23)
n = 1000; % Number of nodes to put in the roadmap
k = 5; % the number of closest nodes around each node to connect.
start_point = [-1 -3]; % set start point
end_point = [3 1]; % set end point
%start_point = [8 9];
%end_point = [-9 -8];

% Create a VideoWriter object for an MP4 file with specified frame rate
v = VideoWriter('path_planning_animation', 'MPEG-4');
v.FrameRate = 10; % Set frame rate
open(v);

% Create map with obstacles
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plot the square as the boundary of map
x = [-10 10 10 -10 -10];
y = [-10 -10 10 10 -10];

figure;



plot(x, y); 
hold on;

% Add some obstacle
obstacle_1_x = [5.5 -5 -5 5.5 5.5];
obstacle_1_y = [6 6 5.5 5.5 6];
obstacle_2_x = [6 -6 -6 6 6];
obstacle_2_y = [-6 -6 -5.5 -5.5 -6];
obstacle_3_x = [5 1 1 5 5];
obstacle_3_y = [0.25 0.25 -0.25 -0.25 0.25];
obstacle_4_x = [1.5 1 1 1.5 1.5];
obstacle_4_y = [-6 -6 6 6 -6];
obstacle_5_x = [-2.5 -3 -3 -2.5 -2.5];
obstacle_5_y = [-6 -6 1 1 -6];

fill(obstacle_1_x, obstacle_1_y, 'black');
fill(obstacle_2_x, obstacle_2_y, 'black');
fill(obstacle_3_x, obstacle_3_y, 'black');
fill(obstacle_4_x, obstacle_4_y, 'black');
fill(obstacle_5_x, obstacle_5_y, 'black');

axis equal;
axis([-12 12 -12 12]);
title('Path Planning Map');
grid on;
hold on;

frame = getframe(gcf); 
% pause 1 second
numFramesForPause = v.FrameRate * 1; 
for i = 1:numFramesForPause
    writeVideo(v, frame);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



% Start implementing PRM
% Generate Nodes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Random generate point uniformly in the graph
%n = 1000; % Number of nodes to put in the roadmap
%k = 5; % the number of closest nodes around each node to connect. 

% Uniformly generate random n nodes, including additional 2 nodes of 
% start and end location
node_x = rand(1, n) * 20 - 10;
node_y = rand(1, n) * 20 - 10;


% Define each obstacle as a struct with x and y fields
obstacles = struct('x', {}, 'y', {});

obstacles(1).x = obstacle_1_x;
obstacles(1).y = obstacle_1_y;
obstacles(2).x = obstacle_2_x;
obstacles(2).y = obstacle_2_y;
obstacles(3).x = obstacle_3_x;
obstacles(3).y = obstacle_3_y;
obstacles(4).x = obstacle_4_x;
obstacles(4).y = obstacle_4_y;
obstacles(5).x = obstacle_5_x;
obstacles(5).y = obstacle_5_y;

% Check if points are inside any obstacle
inside_obstacle = inpolygon(node_x, node_y, obstacles(1).x, obstacles(1).y) | ...
                  inpolygon(node_x, node_y, obstacles(2).x, obstacles(2).y) | ...
                  inpolygon(node_x, node_y, obstacles(3).x, obstacles(3).y) | ...
                  inpolygon(node_x, node_y, obstacles(4).x, obstacles(4).y) | ...
                  inpolygon(node_x, node_y, obstacles(5).x, obstacles(5).y);

% Exclude points that are inside obstacles
node_x = node_x(~inside_obstacle);
node_y = node_y(~inside_obstacle);

% Plot the Nodes
plot(node_x, node_y, '.', color='black'); 
hold on;

frame = getframe(gcf); 
numFramesForPause = v.FrameRate * 1; 
for i = 1:numFramesForPause
    writeVideo(v, frame);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Connect Edges
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% initialize edge list, edgeList[i, j] means connect ith and jth node
% call node_x[i], node_y[i] for the coordinate of the first node
% call node_x[j], node_y[j] for the coordinate of the second node
edgeList = []; % Maximum size, assuming each node connects to k neighbors
edgeCounter = 0;

% for each node
for i = 1:length(node_x)
    % connect nearest k nodes without interact with obstacle
    % Calculate the distance to all other nodes
    distances = sqrt((node_x - node_x(i)).^2 + (node_y - node_y(i)).^2);
    distances(i) = inf; % Set distance to itself as infinity to avoid self-connection
    
    % Sort distances and get indices
    [sortedDistances, sortedIndices] = sort(distances);

    % Initialize counter for connected neighbors
    connectedNeighbors = 0;

    % Iterate through sorted indices to find nearest neighbors
    for idx = 1:length(sortedIndices)
        if connectedNeighbors >= k
            break; % Stop if already connected to k neighbors
        end

        neighborIndex = sortedIndices(idx);

        % Check if the path between nodes i and neighborIndex intersects any obstacle
        % for each obstacle
        isFree = true;
        for o = 1:length(obstacles)
            if ~isCollisionFree(node_x(i), node_y(i), ...
                               node_x(neighborIndex), node_y(neighborIndex), ...
                               obstacles(o).x, obstacles(o).y)
                isFree = false;
                break;
            end
        end

        if isFree
            % No collision, so add this edge to the edge list and
            % update edgeCounter and add the edge
            edgeCounter = edgeCounter + 1;
            edgeList(edgeCounter, :) = [i, neighborIndex];
            % update connectedNeighbors
            connectedNeighbors = connectedNeighbors + 1;
            % Plot the edge
            node1 = edgeList(edgeCounter, 1);
            node2 = edgeList(edgeCounter, 2);
        
            % Get the coordinates of the nodes
            x1 = node_x(node1);
            y1 = node_y(node1);
            x2 = node_x(node2);
            y2 = node_y(node2);
        
            % Plot the line connecting these nodes
            plot([x1, x2], [y1, y2], 'r-');
            
        end
    end
end

frame = getframe(gcf); % Capture current figure
writeVideo(v, frame); % Write the frame to the video


% Graph Search, find the nearest path
% Implementing A* graph search algorithm
% Parameter required: k, node_x, node_y, edge_list, start_point, end_point,
% the axis to draw
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set the start and end position
%start_point = [8 9];
%end_point = [-9 -8];

% add start and end point to the node list, and connect to nearest k nodes
node_x(end+1) = start_point(1);
node_y(end+1) = start_point(2);
node_x(end+1) = end_point(1);
node_y(end+1) = end_point(2);

start_index = length(node_x) - 1;
end_index = length(node_x);

% Plot the start and end positon

scatter(start_point(1), start_point(2), 70, 'red', 'filled');
hold on;
scatter(end_point(1), end_point(2), 70, 'k', 'filled');
hold on;

frame = getframe(gcf); % Capture current figure
numFramesForPause = v.FrameRate * 1; 
for i = 1:numFramesForPause
    writeVideo(v, frame);
end

% Connect the start and end point to nearest 5 nodes without collision
for i = start_index:end_index
    % connect nearest k nodes without interact with obstacle
    % Calculate the distance to all other nodes
    distances = sqrt((node_x - node_x(i)).^2 + (node_y - node_y(i)).^2);
    distances(i) = inf; % Set distance to itself as infinity to avoid self-connection
    
    % Sort distances and get indices
    [sortedDistances, sortedIndices] = sort(distances);

    % Initialize counter for connected neighbors
    connectedNeighbors = 0;

    % Iterate through sorted indices to find nearest neighbors
    for idx = 1:length(sortedIndices)
        if connectedNeighbors >= k
            break; % Stop if already connected to k neighbors
        end

        neighborIndex = sortedIndices(idx);

        % Check if the path between nodes i and neighborIndex intersects any obstacle
        % for each obstacle
        isFree = true;
        for o = 1:length(obstacles)
            if ~isCollisionFree(node_x(i), node_y(i), ...
                               node_x(neighborIndex), node_y(neighborIndex), ...
                               obstacles(o).x, obstacles(o).y)
                isFree = false;
                break;
            end
        end

        if isFree
            % No collision, so add this edge to the edge list and
            % update edgeCounter and add the edge
            edgeCounter = edgeCounter + 1;
            edgeList(edgeCounter, :) = [i, neighborIndex];
            % update connectedNeighbors
            connectedNeighbors = connectedNeighbors + 1;
            % Plot the edge
            node1 = edgeList(edgeCounter, 1);
            node2 = edgeList(edgeCounter, 2);
        
            % Get the coordinates of the nodes
            x1 = node_x(node1);
            y1 = node_y(node1);
            x2 = node_x(node2);
            y2 = node_y(node2);
        
            % Plot the line connecting these nodes
            plot([x1, x2], [y1, y2], 'r-');

        end
    end
end

frame = getframe(gcf); % Capture current figure
numFramesForPause = v.FrameRate * 1; 
for i = 1:numFramesForPause
    writeVideo(v, frame);
end


% Exchange the position of end nodes with its connected nodes so that in
% A*, the nodes connected to end will always able to find the end node
edgeList(end-k+1:end, :) = edgeList(end-k+1:end, [2 1]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Lists used for the algorithm

% Open list, the set of nodes to be evaluated. We evaluate k nodes around
% the current node
% Strucuture: |IS ON LIST 1/0 |Node index |Parent node index |h(n) |g(n) |f(n)|
% h(n): the distance between the start and current node
% g(n): the distance between the target node and n node
% f(n) = g(n) + h(n) 
OPEN = [];

% Close list, the set of nodes already evaluated
% Structure: |node index|
CLOSE = [];

% Set the starting node as the first node
% Add the starting node to OPEN
% Evaluate the starting node
open_count = 1;
close_count = 0;
h = 0; % as it is the start node
g = distance(start_point(1), start_point(2), end_point(1), end_point(2));
f = g + h;

% Since the start node is already evaluated, set "IS ON LIST" to 0 in OPEN
% list, and add the start node to CLOSE list
OPEN(open_count, :) = [0, start_index, start_index, h, g, f];
close_count = close_count + 1;
CLOSE(close_count) = start_index;

% Set the start point as the current node
current = start_index;

% No path indicator, 1 means there is a path, 0 means there is no path.
% Initialize with 1, there is a path. 
no_path = 1;
% while the current node is not the target and there is a path
while (current ~= end_index && no_path == 1)
    % find k connected nodes with the current node (successor)
    % store those nodes in a list of node index
    % initialize neighbor list to length k as each node is connected to k
    % nearest nodes
    % neighbor_list structure: node_index | h(n) | g(n) | f(n)
    neighbor_list = [];
    % count the number of connected neighbors
    neighbor_count = 0;
    % for each edge
    for i = 1:length(edgeList)
        % if the edge start node is the current node
        if edgeList(i, 1) == current
            % store the neighbor
            neighbor = edgeList(i, 2);
            % check if the neighbor is not in CLOSE
            % initialize the neighbor is not in CLOSE
            in_closed = 0;
            % for each node in CLOSE:
            for j = 1:size(CLOSE,1)
                % if i neighbor is in the CLOSE list
                if neighbor == CLOSE(j)
                    % set in_closed to true
                    in_closed = 1;
                    % break the CLOSE list for loop
                    break;
                end % end neighbor is in CLOSE if
            end % end for j
            
            % if neighbor is not in closed
            if in_closed == 0
                % update neighbor list size
                neighbor_count = neighbor_count + 1;
                % add the current neighbor to neighbor_list
                neighbor_list(neighbor_count, 1) = neighbor;
                % h(n), the cost from start node to neighbor node,
                % h(current) + distance(current, neighbor)
                neighbor_list(neighbor_count, 2) = h + distance(node_x(current), node_y(current), ...
                                                                node_x(neighbor), node_y(neighbor));
                % g(n), the distance between the target node and neighbor
                neighbor_list(neighbor_count, 3) = distance(node_x(neighbor), node_y(neighbor), ...
                                                            node_x(end_index), node_y(end_index));
                % f(n), total cost h(n) + g(n)
                neighbor_list(neighbor_count, 4) = neighbor_list(neighbor_count, 2) + neighbor_list(neighbor_count, 3);
            end % end if not in closed
        elseif edgeList(i, 2) == current 
            % store the neighbor
            neighbor = edgeList(i, 1);
            % check if the neighbor is not in CLOSE
            % initialize the neighbor is not in CLOSE
            in_closed = 0;
            % for each node in CLOSE:
            for j = 1:size(CLOSE,1)
                % if i neighbor is in the CLOSE list
                if neighbor == CLOSE(j)
                    % set in_closed to true
                    in_closed = 1;
                    % break the CLOSE list for loop
                    break;
                end % end neighbor is in CLOSE if
            end % end for j
            
            % if neighbor is not in closed
            if in_closed == 0
                % update neighbor list size
                neighbor_count = neighbor_count + 1;
                % add the current neighbor to neighbor_list
                neighbor_list(neighbor_count, 1) = neighbor;
                % h(n), the cost from start node to neighbor node,
                % h(current) + distance(current, neighbor)
                neighbor_list(neighbor_count, 2) = h + distance(node_x(current), node_y(current), ...
                                                                node_x(neighbor), node_y(neighbor));
                % g(n), the distance between the target node and neighbor
                neighbor_list(neighbor_count, 3) = distance(node_x(neighbor), node_y(neighbor), ...
                                                            node_x(end_index), node_y(end_index));
                % f(n), total cost h(n) + g(n)
                neighbor_list(neighbor_count, 4) = neighbor_list(neighbor_count, 2) + neighbor_list(neighbor_count, 3);
            end % end if not in closed
        end
    end % end for i
    
    % show all the neighbors connect to current on plot with yellow

    % add neighbors to OPEN list
    % for each neighbor (successor)
    for i = 1:neighbor_count
        
        % show the neighbor connected
        x1 = node_x(current);
        y1 = node_y(current);
        x2 = node_x(neighbor_list(i, 1));
        y2 = node_y(neighbor_list(i, 1));

        plot([x1, x2], [y1, y2], 'y-', 'LineWidth', 2);  
        

        % check if the neighbor is already in OPEN list
        % by default, set neighbor i is not in OPEN
        in_OPEN = 0;
        % for each node in open
        for j = 1:open_count
            
            % plot trajectories in open, connect to its parents
            x1 = node_x(OPEN(j, 2));
            y1 = node_y(OPEN(j, 2));
            x2 = node_x(OPEN(j, 3));
            y2 = node_y(OPEN(j, 3));
            plot([x1, x2], [y1, y2], 'r-', 'LineWidth', 2);  
            

            % check i neighbor is j open node
            if neighbor_list(i, 1) == OPEN(j, 2)
                % set in_OPEN to true
                in_OPEN = 1;
                % check if this node in OPEN need to update
                % compare f(n), update if the new f(n) value of
                % neighbor is smaller then original stored in OPEN   
                if neighbor_list(i, 4) < OPEN(j, 6)
                    % the neighbor f is lower, update this node in OPEN
                    % update the parent node of neighbor to current node
                    OPEN(j, 3) = current;
                    % update h(n) to the value in neighbor_list
                    OPEN(j, 4) = neighbor_list(i, 2);
                    % update g(n) to the value in neighbor_list
                    OPEN(j, 5) = neighbor_list(i, 3);
                    % update f(n) to the value in neighbor_list
                    OPEN(j, 6) = neighbor_list(i, 4);
                    % Since each node can only appear once in OPEN, once
                    % found in OPEN, no need to check the rest in OPEN,
                    % thus break the j in open_count loop
                    break;
                end % end if neighbor cost is smaller
            end % end neighbor check with open
        end % end for j
        % if neighbor i is not in OPEN, add it to OPEN
        if in_OPEN == 0
            open_count = open_count + 1;
            OPEN(open_count, :) = [1, neighbor_list(i, 1), current, ...
                                   neighbor_list(i, 2), neighbor_list(i, 3), neighbor_list(i, 4)];
        end % end if in_OPEN == 0
    end % end for i

    frame = getframe(gcf); % Capture current figure
    writeVideo(v, frame); % Write the frame to the video
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Now all the neighbors are in OPEN
    % Find out the node with the smallest f(n) that is still under consideration
    % in OPEN for the next iteration
    min_neighbor_index = min_fn(OPEN, open_count, end_index);
    
    % check if the node index is valid. 
    if min_neighbor_index == -1
        % if not valid, set no path to 0 and break the main while loop
        no_path = 0;
        break;
    end

    % if the node index is valid
    % set the next current node to the given min index
    current = OPEN(min_neighbor_index, 2);
    % update the path cost h(n)
    h = OPEN(min_neighbor_index, 4);
    % remove the node from consideration in OPEN
    OPEN(min_neighbor_index, 1) = 0;
    % add this node to CLOSE
    close_count = close_count + 1;
    CLOSE(close_count) = current;

end % end the main while loop

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% first check if the path exist
if no_path == 0
    h=msgbox('No path exists to the Target!','warn');
    uiwait(h,5);
else
    % to get the optimal path, use OPEN list, start at the end node, traverse
    % though its parent node, until reaches the start node
    % Initialize the current node as the end node
    current = end_index;
    

    % Backtrack from the end node to the start node
    while current ~= start_index
        % Find the current node in the OPEN list
        x1 = node_x(current);
        y1 = node_y(current);

        for i = 1:size(OPEN, 1)
            if OPEN(i, 2) == current
                % Update current to be its parent node
                current = OPEN(i, 3);
                break; % Exit the loop once the parent is found
            end
        end

        x2 = node_x(current);
        y2 = node_y(current);
        plot([x1, x2], [y1, y2], 'b-', 'LineWidth', 3); 
        numFramesForPause = v.FrameRate * 0.05; 
        for i = 1:numFramesForPause
            writeVideo(v, frame);
        end
        frame = getframe(gcf); % Capture current figure
        writeVideo(v, frame); % Write the frame to the video
    end

end

numFramesForPause = v.FrameRate * 5; 
for i = 1:numFramesForPause
    writeVideo(v, frame);
end

close(v);

% Function declearation and implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function isFree = isCollisionFree(x1, y1, x2, y2, obstacle_x, obstacle_y) 
    % This function divide the line to points, 0.4 unit length apart,
    % check if any point is in the obstacle.
    % Return true if the edge is collision free

    % compute the norm of the edge
    edge_norm = sqrt((x2 - x1)^2 + (y2 - y1)^2);
    
    % compute the number of point to generate on the edge
    numPoints = floor(edge_norm / 0.2) + 1;

    % Calculate the points on the line
    xPoints = linspace(x1, x2, numPoints);
    yPoints = linspace(y1, y2, numPoints);

    % Check if each generated point is inside the obstacle
    % obstacle_points is a logical array
    obstacle_points = inpolygon(xPoints, yPoints, obstacle_x, obstacle_y);

    % Initialize isFree to true
    isFree = true;
    
    % Check if any point is inside the obstacle, is so, change isFree to
    % false
    if any(obstacle_points)
        isFree = false;
    end
end

function distance = distance(x1, y1, x2, y2)
    % This function return the distance between to nodes
    distance = sqrt((x2 - x1)^2 + (y2 - y1)^2);
end

function min_index = min_fn(OPEN, open_count, end_index)
    % This function return the node indax in OPEN with the minimum f(n).
    % The returned node index must still under the consideration 
    % IS ON LIST = 1

    % Initializes a temporary array to store nodes from the OPEN list that 
    % are still under consideration (not yet moved to the CLOSED list,
    % IS ON LIST = 1).
    in_OPEN_list = [];
    in_OPEN_count = 1; % index counter for in_OPEN_list
    end_OPEN_index = 0; % end index in OPEN list
    is_end = 0; % by default, the end point is not in OPEN
    % fill in_OPEN_list
    for i = 1:open_count
        % if this node is still under consideration
        if OPEN(i, 1) == 1
            % add this node to in_OPEN_list
            in_OPEN_list(in_OPEN_count, :) = [OPEN(i, :) i];
            in_OPEN_count = in_OPEN_count + 1;
            % Check if Node is the Target
            if OPEN(i, 2) == end_index
                is_end = 1; % the end point is in OPEN
                end_OPEN_index = i; % record the end index
                break;
            end
        end
    end
    
    % one of the successors is the goal node so send this node
    if is_end == 1
        min_index = end_OPEN_index;
    elseif size(in_OPEN_list, 1) == 0
        min_index = -1;
    else
        % Index of the smallest f(n) node in temp array
        [~,temp_min] = min(in_OPEN_list(:,6)); 
        % Index of the smallest node in the OPEN array
        min_index = in_OPEN_list(temp_min,7);
    end
end






