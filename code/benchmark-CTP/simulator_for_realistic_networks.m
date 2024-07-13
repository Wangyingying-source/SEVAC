function [LET,mean,std] = simulator_for_realistic_networks(policyFunc,network,originNode,destinationNode,maxTimes,edgeCostDistributionModel)
%UNTITLED2 ???????§Û?????????
%   ????????????
if originNode == destinationNode
    return;
end

[oriMuGraph,oriSigmaGraph,oriProbGraph,nodeNum] = loadGraphFromFile(network);
% if oriMuGraph(originNode,destinationNode) == 0
%     oriMuGraph(originNode,destinationNode) = 500;
%     oriSigmaGraph(originNode,destinationNode) = 10;
%     oriProbGraph(originNode,destinationNode) = 1;
% end
costList = [];

for item = 1 : maxTimes    %   simulation times
    %   parameters initialization
    traversalGraph = oriProbGraph;
    muGraph = oriMuGraph;
    sigmaGraph = oriSigmaGraph;
    currentState = originNode;
    synchronous = 0;
    travelCost = 0;
    path = [currentState];
    [muGraph,sigmaGraph,traversalGraph] = observationAndGraphUpdate(currentState,destinationNode,muGraph,sigmaGraph,traversalGraph);
    % offline
%     tic;
    nextAction = policyFunc(muGraph,sigmaGraph,traversalGraph,currentState,destinationNode,synchronous);
%     toc;

    %   begin a travel
    while currentState~= destinationNode  
        % arriving at a new state(node)
        [muGraph,sigmaGraph,traversalGraph] = observationAndGraphUpdate(currentState,destinationNode,muGraph,sigmaGraph,traversalGraph);    %   get observation and update graph
        synchronous = 1;  %   synchronous updating of observations and policy
%         tic
        nextAction = policyFunc(muGraph,sigmaGraph,traversalGraph,currentState,destinationNode,synchronous);
%         toc
        % traveling on the edge
        nextState = nextAction;             %   travel the edge from current node to next node
        if (traversalGraph(currentState,nextState) == 0)
            edgeCost = 10000;
            travelCost = travelCost + edgeCost;
            break;
        else
            edgeCost = sampleFromDistribution(muGraph(currentState,nextState),sigmaGraph(currentState,nextState),edgeCostDistributionModel);
        end
        travelCost = travelCost + edgeCost;
        if travelCost > 5000
            break;
        end
        synchronous = 2;   %   asynchronous updating of observations and policy
%         tic
        nextAction = policyFunc(muGraph,sigmaGraph,traversalGraph,currentState,destinationNode,synchronous);    %   2-3s
%         toc
        path = [path,nextState];
        currentState = nextState; 
    end
    costList = [costList,travelCost];
    if (currentState==destinationNode)
%         path
    end
    travelCost;
    item;
end
    [mean,std]=calculateMeanStd(costList,maxTimes);
    c = sparse(oriMuGraph);
    [LET,~,~] = graphshortestpath(c,originNode,destinationNode);
    LET;
%     graphshortestpath
%     mean = 0;
%     std = 0;
%     LET = 0;
    
    
    
    function edgeCost = sampleFromDistribution(mu,sigma,edgeCostDistributionModel)
        switch edgeCostDistributionModel
            case {'Gaussian','gaussian'}
                edgeCost = normrnd(mu,sigma);
                if edgeCost < 0.1
                    edgeCost = 0.1;
                end
            otherwise
                error('no available distribution model');
        end
    end
    
    function [muGraph,sigmaGraph,traversalGraph] = observationAndGraphUpdate(currentState,destinationNode,muGraph,sigmaGraph,traversalGraph)
        [nodeNum,~] = size(traversalGraph);
        for i = 1 : nodeNum
            currentState; i;
            if traversalGraph(currentState,i) ~= 0
                random1 = rand();
                if random1 <= traversalGraph(currentState,i)
                    traversalGraph(currentState,i) = 1;
%                     traversalGraph(i,currentState) = 1;
%                     if muGraph(i,currentState) == 0
%                         muGraph(i,currentState) = muGraph(currentState,i);
%                         sigmaGraph(i,currentState) = sigmaGraph(currentState,i);
%                     end
                else
                    traversalGraph(currentState,i) = 0;
                    muGraph(currentState,i) = 0;
                    sigmaGraph(currentState,i) = 0;
%                     traversalGraph(i,currentState) = 0;
%                     muGraph(i,currentState) = 0;
%                     sigmaGraph(i,currentState) = 0;
                end
                random2 = rand();
                if random2 <= traversalGraph(i,currentState)
%                     traversalGraph(currentState,i) = 1;
                    traversalGraph(i,currentState) = 1;
                else
%                     traversalGraph(currentState,i) = 0;
%                     muGraph(currentState,i) = 0;
%                     sigmaGraph(currentState,i) = 0;
                    traversalGraph(i,currentState) = 0;
                    muGraph(i,currentState) = 0;
                    sigmaGraph(i,currentState) = 0;
                end
            end 
        end
%         if currentState ~= destinationNode
%         muGraph(currentState,destinationNode) = 500;
%         sigmaGraph(currentState,destinationNode) = 60;
%         traversalGraph(currentState,destinationNode) = 1;
%         end
    end
    
    function [E,std] = calculateMeanStd(costList,maxTimes)
        M = sum(sum(costList.^2)) / maxTimes;
        E = sum(abs(costList)) / maxTimes;
        std = sqrt(M - E^2);
    end

    function [muGraph,sigmaGraph,probGraph,nodeNum] = loadGraphFromFile(network)
        network;
        [xlx,~,~]=xlsread(network);
        [n,~] = size(xlx);      %
        nodeNum = xlx(n,1);
        muGraph = zeros(nodeNum,nodeNum);
        probGraph = zeros(nodeNum,nodeNum);
        
        % probList = zeros(n,1);
        for i = 1:n
            startState = xlx(i,1);
            endState = xlx(i,2);
            muGraph(startState,endState) = xlx(i,4);
            sigmaGraph(startState,endState) = xlx(i,5);
            probGraph(startState,endState) = xlx(i,6);
        %     probList(i) = rand();   %?????????§Ú???
        %     probGraph(Num(i,1),Num(i,2)) = probList(i);
        end
    end
    
end
