function nextAction = DDP_DL(muGraph,sigmaGraph,traversalGraph,currentState,destinationNode,synchronous)
%UNTITLED2 ???????§Û?????????
%   ????????????
persistent VfuncList;
persistent stateValueListJ;
persistent fullyDecisionListMeanStd;
persistent stateActionsCell;
%persistent fileName;
persistent ZinPE;

% params
%fileName = 'plot_data.xlsx';
% ZinPE = zeros(1,100);
ZinPE = [];
zeta = 10;
randomDecisionListProb = 0;     %0.001
initDLProb = 0.006;
deltaZ = 0.01;
deltaPi = 1;
maxCountZ0 = 500;
maxCountPI0 = 100;
maxCountZ2 = 500;
maxCountPI2 = 100;
[nodeNum,~] = size(muGraph);


noActionPenaltyItemQ = 100;
noActionPenaltyItemJ = 100;

decisionListBuf = zeros(nodeNum,nodeNum);
bufZ = zeros(nodeNum,nodeNum);
bufVfunc = zeros(1,nodeNum);
countPolicy = 1;    %0
% count = 0;
if synchronous == 0 
    VfuncList = zeros(1,nodeNum);
    stateValueListJ = zeros(1,nodeNum);
    fullyDecisionListMeanStd = zeros(nodeNum,nodeNum);   %????????????decision list
    stateActionsCell = cell(1,nodeNum);
%     Qvar = zeros(nodeNum,nodeNum);
%     Qtable = zeros(nodeNum,nodeNum);
    Z = zeros(nodeNum,nodeNum);
    [fullyDecisionListMeanStd,stateActionsCell] = initStateActionsCell(muGraph,traversalGraph,destinationNode);
    VfuncList = updateVfuncList(VfuncList,stateActionsCell,noActionPenaltyItemQ,destinationNode,nodeNum);
    while countPolicy < maxCountPI0
        countVfunc = 0;
         while countVfunc < maxCountZ0
%             Qtable = updateQtable(muGraph,VfuncList,destinationNode,nodeNum);
            
            stateActionsCell = updateQInStateActionsCell(muGraph,VfuncList,stateActionsCell,destinationNode,nodeNum);
            VfuncList = updateVfuncList(VfuncList,stateActionsCell,noActionPenaltyItemQ,destinationNode,nodeNum);
            [stateActionsCell,delta] = updateJsaAndQvarInStateActionsCell(muGraph,sigmaGraph,stateValueListJ,VfuncList,stateActionsCell,zeta,destinationNode);
            stateValueListJ = updateStateValueListJ(stateValueListJ,stateActionsCell,VfuncList,noActionPenaltyItemJ,destinationNode,nodeNum);
            delta;
            if delta < deltaZ
                break;
            end
            if rand() < randomDecisionListProb
                [fullyDecisionListMeanStd,stateActionsCell] = updateDecisionLists(stateActionsCell,randomDecisionListProb,destinationNode,nodeNum);
            end
            countVfunc = countVfunc + 1;
%             count = count + 1;
%             Z = getZ(stateActionsCell,Z,destinationNode,zeta,nodeNum);

%             ZinPE = [ZinPE,stateActionsCell{currentState}(1).Z];

        end
%           PI
        [fullyDecisionListMeanStd,stateActionsCell] = updateDecisionLists(stateActionsCell,randomDecisionListProb,destinationNode,nodeNum);
%         if decisionListBuf == fullyDecisionListMeanStd
        errorPI = sum(sum(abs(decisionListBuf - fullyDecisionListMeanStd)));
        if errorPI < deltaPi
            break;
        end
        decisionListBuf = fullyDecisionListMeanStd;
        countPolicy = countPolicy + 1;
    end
     nextAction = fullyDecisionListMeanStd(1,currentState);
    
    %     a = nonzeros(fullyDecisionListMeanStd(:,currentState))
%     [~,len] = size(ZinPE);
%     x = 1 : 1 : len;
%     plot(x, ZinPE);
%   xlswrite(fileName, ZinPE, 'Sheet1', 'A1');
    
elseif synchronous == 1 
    for i = 1 : nodeNum
        if fullyDecisionListMeanStd(i,currentState) ~= 0
            if traversalGraph(currentState,fullyDecisionListMeanStd(i,currentState)) ~= 0
                nextAction = fullyDecisionListMeanStd(i,currentState);
                break;
            end
        else
            nextAction = currentState;
            break;
        end
    end
%     
    
elseif synchronous == 2
    stateActionsCell = updateTraversal(traversalGraph,stateActionsCell,destinationNode,nodeNum);
    while countPolicy < maxCountPI2
        countVfunc = 0;
%         VI
         while countVfunc < maxCountZ0
%             Qtable = updateQtable(muGraph,VfuncList,destinationNode,nodeNum);
            
            stateActionsCell = updateQInStateActionsCell(muGraph,VfuncList,stateActionsCell,destinationNode,nodeNum);
            VfuncList = updateVfuncList(VfuncList,stateActionsCell,noActionPenaltyItemQ,destinationNode,nodeNum);
            [stateActionsCell,delta] = updateJsaAndQvarInStateActionsCell(muGraph,sigmaGraph,stateValueListJ,VfuncList,stateActionsCell,zeta,destinationNode);
            stateValueListJ = updateStateValueListJ(stateValueListJ,stateActionsCell,VfuncList,noActionPenaltyItemJ,destinationNode,nodeNum);
%             delta
            if delta < deltaZ
                break;
            end
            if rand() < randomDecisionListProb
                [fullyDecisionListMeanStd,stateActionsCell] = updateDecisionLists(stateActionsCell,randomDecisionListProb,destinationNode,nodeNum);
            end
            countVfunc = countVfunc + 1;
        end
%           PI
        [fullyDecisionListMeanStd,stateActionsCell] = updateDecisionLists(stateActionsCell,randomDecisionListProb,destinationNode,nodeNum);
%         if decisionListBuf == fullyDecisionListMeanStd
        if sum(sum(abs(decisionListBuf - fullyDecisionListMeanStd))) < deltaPi
            break;
        end
        decisionListBuf = fullyDecisionListMeanStd;
        countPolicy = countPolicy + 1;
    end
%     a = nonzeros(fullyDecisionListMeanStd(:,currentState))
    nextAction = fullyDecisionListMeanStd(1,currentState);
end


    function delta = calculateDelta(Z,bufZ)
        [~,nodeNum] = size(Z);
        delta = 0;
        for i = 1 : nodeNum
            for j = 1 : nodeNum
                error = abs(Z(i,j) - bufZ(i,j));
                if error > delta
                    from = i;
                    to = j;
                    delta = error;
                end
            end
        end

    end


function Qtable = updateQtable(muGraph,VfuncList,destinationNode,nodeNum)
    for i = 1 : nodeNum
        for j = 1 : nodeNum
            if i ~= destinationNode && muGraph(i,j) ~= 0
                Qtable(i,j) = muGraph(i,j) + VfuncList(j);
            else
                Qtable(i,j) = 0;
            end
        end
    end
end

function [fullyDecisionListMeanStd,outputCell] = initStateActionsCell(muGraph,traversalGraph,destinationNode)
    [~,nodeNum] = size(muGraph);
    outputCell = cell(1,nodeNum);
    fullyDecisionListMeanStd = zeros(nodeNum,nodeNum);
    shortestPathList = zeros(1,nodeNum);
    c=sparse(muGraph);
    for i = 1 : nodeNum
        [dist,~,~] = graphshortestpath(c,i,destinationNode);
        shortestPathList(i) = dist;
    end
    
    for i = 1 : nodeNum
%         VfuncList(i) = 0;
        if i ~=destinationNode && shortestPathList(i) ~= inf
            
            bufQPJlist = [];        %Q(s,a)??J(s,a)??
            for j = 1 : nodeNum
               if  muGraph(i,j)~=0 && shortestPathList(j) ~= inf
%                    bufQPJ.Q = 0;
                   bufQPJ.Q = shortestPathList(j) + muGraph(i,j);
                   bufQPJ.P = traversalGraph(i,j);
                   bufQPJ.Jsa = 0;
                   bufQPJ.nextState = j;
                   bufQPJ.Qvar = 0;
                   bufQPJ.Z = 0;
                   bufQPJ.preZ = 0;
                   bufQPJlist = [bufQPJlist,bufQPJ];
               end
            end
            [~,sizeBufList] = size(bufQPJlist);
            if sizeBufList > 0
                for ei = 1 : sizeBufList - 1
                    for ej = ei + 1 : sizeBufList
                        if bufQPJlist(ei).Q > bufQPJlist(ej).Q
                            buffer = bufQPJlist(ei);
                            bufQPJlist(ei) = bufQPJlist(ej);
                            bufQPJlist(ej) = buffer;
                        end
                    end
                    fullyDecisionListMeanStd(ei,i) = bufQPJlist(ei).nextState;
                end
                 fullyDecisionListMeanStd(sizeBufList,i) = bufQPJlist(sizeBufList).nextState;
            end
            outputCell{i} = bufQPJlist;
        else
            bufQPJ.Q = 0;
            bufQPJ.P = 0;
            bufQPJ.Jsa = 0;
            bufQPJ.nextState = i;
            bufQPJ.Qvar = 0;
            bufQPJ.Z = 0;
            outputCell{i} = bufQPJ;
        end
    end
end

function stateActionsCell = updateQInStateActionsCell(muGraph,VfuncList,stateActionsCell,destinationNode,nodeNum)
    for i = 1 : nodeNum
        if i ~=destinationNode
            [~,actionsNum] = size(stateActionsCell{i});
            for j = 1 : actionsNum
                nextState = stateActionsCell{i}(j).nextState;
                stateActionsCell{i}(j).Q = muGraph(i,nextState) + VfuncList(nextState);
            end
        end
    end
end

function VfuncList = updateVfuncList(VfuncList,stateActionsCell,noActionPenaltyItemV,destinationNode,nodeNum)
    for i = 1 : nodeNum
        if i ~= destinationNode
            bufQPJlist = stateActionsCell{i};
            [~,sizeBufList] = size(bufQPJlist);            
%             update V(i)
            value = 0;
            for bi = 1 : sizeBufList
                multiplicative = 1;
                for ci = 1 : bi - 1
                   multiplicative  = multiplicative  * (1 - bufQPJlist(ci).P);
                end
                value = value + multiplicative * bufQPJlist(bi).P * bufQPJlist(bi).Q;
            end
            VfuncList(i) = value;
            penaltyV = noActionPenaltyItemV;
            for di = 1 : sizeBufList
                penaltyV = penaltyV * (1 - bufQPJlist(di).P);
            end

            VfuncList(i) = VfuncList(i) + penaltyV;
        else
            VfuncList(i) = 0;
        end
    end
end

function [stateActionsCell,delta] = updateJsaAndQvarInStateActionsCell(muGraph,sigmaGraph,stateValueListJ,VfuncList,stateActionsCell,zeta,destinationNode)
    [~,nodeNum] = size(muGraph);
    delta = 0;
    for i = 1 : nodeNum
        if i ~=destinationNode
            [~,actionsNum] = size(stateActionsCell{i});
            for j = 1 : actionsNum
%                 update Jsa
                nextState = stateActionsCell{i}(j).nextState;
                stateActionsCell{i}(j).preZ = stateActionsCell{i}(j).Z;
                stateActionsCell{i}(j).Jsa = sigmaGraph(i,nextState)^2 + muGraph(i,nextState)^2 + stateValueListJ(nextState) + 2 * muGraph(i,nextState) * VfuncList(nextState);
                stateActionsCell{i}(j).Qvar = abs(stateActionsCell{i}(j).Jsa - stateActionsCell{i}(j).Q^2);
                stateActionsCell{i}(j).Z = stateActionsCell{i}(j).Q + zeta * sqrt(stateActionsCell{i}(j).Qvar);
                error = abs(stateActionsCell{i}(j).Z - stateActionsCell{i}(j).preZ);
                if error > delta
                    delta = error;
                end

            end
        end
    end
end

function stateValueListJ = updateStateValueListJ(stateValueListJ,stateActionsCell,VfuncList,noActionPenaltyItemJ,destinationNode,nodeNum)
    for i = 1 : nodeNum
        if i ~=destinationNode
            bufQPJlist = stateActionsCell{i};
            [~,sizeBufList] = size(stateActionsCell{i});
            value = 0;
            for bi = 1 : sizeBufList
                multiplicative = 1;
                for ci = 1 : bi - 1
                   multiplicative  = multiplicative  * (1 - bufQPJlist(ci).P);
                end
                value = value + multiplicative * bufQPJlist(bi).P * bufQPJlist(bi).Jsa;
            end
            stateValueListJ(i) = value;
%             penaltyJ = noActionPenaltyItemJ;
            noActionProb = 1;
            for di = 1 : sizeBufList
                noActionProb = noActionProb * (1 - bufQPJlist(di).P);
            end
            penaltyJ = noActionProb * [noActionPenaltyItemJ^2 + stateValueListJ(i) + 2 * noActionPenaltyItemJ * VfuncList(i)];
            stateValueListJ(i) = stateValueListJ(i) + penaltyJ;
        end
    end
end


function [fullyDecisionListMeanStd,stateActionsCell] = updateDecisionLists(stateActionsCell,randomDecisionListProb,destinationNode,nodeNum)
    fullyDecisionListMeanStd = zeros(nodeNum,nodeNum);
    randomParam= rand();
    if randomParam > randomDecisionListProb
    for i = 1 : nodeNum
        if i ~= destinationNode
            bufQPJlist = stateActionsCell{i};
            [~,sizeBufList] = size(bufQPJlist);
            if sizeBufList == 0
                continue;
            end
        %           ????????
            for ei = 1 : sizeBufList - 1
                for ej = ei + 1 : sizeBufList
                    if bufQPJlist(ei).Z > bufQPJlist(ej).Z
                        buffer = bufQPJlist(ei);
                        bufQPJlist(ei) = bufQPJlist(ej);
                        bufQPJlist(ej) = buffer;
                    end
                end
                fullyDecisionListMeanStd(ei,i) = bufQPJlist(ei).nextState;
            end
            fullyDecisionListMeanStd(sizeBufList,i) = bufQPJlist(sizeBufList).nextState;
        end
        stateActionsCell{i} = bufQPJlist;
    end
    else
        for i = 1 : nodeNum
            if i ~= destinationNode
                bufQPJlist = stateActionsCell{i};
                [~,sizeBufList] = size(bufQPJlist);
                if sizeBufList == 0
                    continue;
                end
%                 ???policy
                randomOrder = randperm(sizeBufList);
                buffer = bufQPJlist;
                for ei = 1 : sizeBufList
                    bufQPJlist(ei) = buffer(randomOrder(ei));
                    fullyDecisionListMeanStd(ei,i) = bufQPJlist(ei).nextState;
                end
                fullyDecisionListMeanStd(sizeBufList,i) = bufQPJlist(sizeBufList).nextState;
            end
            stateActionsCell{i} = bufQPJlist;
        end

    end   
        
end

function [fullyDecisionListMeanStd,stateActionsCell] = randomPolicy(stateActionsCell,randomDecisionListProb,destinationNode,nodeNum)
    fullyDecisionListMeanStd = zeros(nodeNum,nodeNum);
    for i = 1 : nodeNum
        if i ~= destinationNode
            bufQPJlist = stateActionsCell{i};
            [~,sizeBufList] = size(bufQPJlist);
            if sizeBufList == 0
                continue;
            end
            randomParam= rand();
    %                 ???policy
            randomOrder = randperm(sizeBufList);
            buffer = bufQPJlist;
            for ei = 1 : sizeBufList
                bufQPJlist(ei) = buffer(randomOrder(ei));
                fullyDecisionListMeanStd(ei,i) = bufQPJlist(ei).nextState;
            end
        end
        stateActionsCell{i} = bufQPJlist;      
    end            
end

function [fullyDecisionListMeanStd,stateActionsCell] = initializeDecisionListPolicy(muGraph,stateActionsCell,destinationNode)
    [~,nodeNum] = size(muGraph);
    fullyDecisionListMeanStd = zeros(nodeNum,nodeNum);
    shortestPathList = zeros(1,nodeNum);
    c=sparse(muGraph);
%         currentState
    for i = 1 : nodeNum
        [dist,~,~] = graphshortestpath(c,i,destinationNode);
        shortestPathList(i) = dist;
    end
    for i = 1 : nodeNum
        if i ~= destinationNode
            bufList = stateActionsCell{i};
            [sizeBufList,~] = size(bufList);
            for ei = 1 : sizeBufList - 1
                for ej = ei + 1 : sizeBufList
                    if shortestPathList(bufList(ei).nextState) > shortestPathList(bufList(ej).nextState)
                        buffer = bufList(ei);
                        bufList(ei) = bufList(ej);
                        bufList(ej) = buffer;
                    end
                end
                fullyDecisionListMeanStd(ei,i) = bufList(ei).nextState;
            end
        end
    end
                
end

function delta = calculateDeltaFromStateActionsCell(stateActionsCell,destinationNode,zeta,nodeNum)
    delta = 0;
    for i = 1 : nodeNum
        if i ~=destinationNode
            [~,sizeBufList] = size(stateActionsCell{i});
            for j = 1 : sizeBufList
                stateActionsCell{i}(j).preZ = stateActionsCell{i}(j).Z;
                stateActionsCell{i}(j).Z = stateActionsCell{i}(j).Q + zeta * sqrt(stateActionsCell{i}(j).Qvar);
%                 Z(i,stateActionsCell{i}(j).nextState) = stateActionsCell{i}(j).Q + zeta * sqrt(stateActionsCell{i}(j).Qvar);
                error = abs(stateActionsCell{i}(j).Z - stateActionsCell{i}(j).preZ);
                if error > delta
%                     from = i;
%                     to = j;
                    delta = error;
                end
            end
        end
    end
end

function Z = getZ(stateActionsCell,Z,destinationNode,zeta,nodeNum)
    for i = 1 : nodeNum
        if i ~=destinationNode
            [~,sizeBufList] = size(stateActionsCell{i});
            for j = 1 : sizeBufList
                Z(i,stateActionsCell{i}(j).nextState) = stateActionsCell{i}(j).Q + zeta * sqrt(stateActionsCell{i}(j).Qvar);
            end
        end
    end
end



function Qtable = getQtable(stateActionsCell,Qtable,destinationNode,nodeNum)
    for i = 1 : nodeNum
        if i ~=destinationNode
            [~,sizeBufList] = size(stateActionsCell{i});
            for j = 1 : sizeBufList
                Qtable(i,stateActionsCell{i}(j).nextState) = stateActionsCell{i}(j).Q;
            end
        end
    end
end

function Qvar = getQvar(stateActionsCell,Qvar,destinationNode,nodeNum)
    for i = 1 : nodeNum
        if i ~=destinationNode
            [~,sizeBufList] = size(stateActionsCell{i});
            for j = 1 : sizeBufList
                Qvar(i,stateActionsCell{i}(j).nextState) = stateActionsCell{i}(j).Qvar;
            end
        end
    end
end

function stateActionsCell = updateTraversal(traversalGraph,stateActionsCell,destinationNode,nodeNum)
    for i = 1 : nodeNum
        if i ~=destinationNode
            [~,sizeBufList] = size(stateActionsCell{i});
%             nextStateToDelete = [];
            for j = sizeBufList : -1 : 1
                if traversalGraph(i,stateActionsCell{i}(j).nextState) == 0
                      stateActionsCell{i}(j) = [];
                else
                    stateActionsCell{i}(j).P = traversalGraph(i,stateActionsCell{i}(j).nextState);
                end
            end
            [~,sizeBufList] = size(stateActionsCell{i});
            for ti = 1 : nodeNum
                if traversalGraph(i,ti) ~= 0
                    isNewLink = true;
                    for tj = 1 : sizeBufList 
                        if stateActionsCell{i}(tj).nextState == ti
                            isNewLink = false;
                            break;
                        end
                    end
                    if isNewLink 
                        bufQPJ.Q = stateActionsCell{i}(sizeBufList).Q;
                        bufQPJ.P = traversalGraph(i,ti);
                        bufQPJ.Jsa =stateActionsCell{i}(sizeBufList).Jsa;
                        bufQPJ.nextState = ti;
                        bufQPJ.Qvar = stateActionsCell{i}(sizeBufList).Qvar;
                        bufQPJ.Z = stateActionsCell{i}(sizeBufList).Z;
                        bufQPJ.preZ = stateActionsCell{i}(sizeBufList).preZ;
                        stateActionsCell{i} = [stateActionsCell{i},bufQPJ];
                    end
                end
            end
        end
    end
end


end
