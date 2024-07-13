clc
clear all;
zeta = 1;
% network = '.\experiment\Sioux-medium-case-P=0.5.xlsx';
% network = '.\experiment\Anaheim-kappa=0.1-P=0.5.xlsx';
% networkod = '.\experiment\Anaheim_OD_pair.xlsx';
% network = '.\experiment\Friedrichshain-kapppa=0.1-P=0.5.xlsx';
% networkod = '.\experiment\Friedrichshain_OD_pair.xlsx';
% network = '.\experiment\Barcelona-kappa=0.1-P=0.9.xlsx';
% networkod = '.\experiment\Barcelona_OD_pair.xlsx';
% network = '.\experiment\Chengdu-weekend-offpeak.xlsx';
% networkod = '.\experiment\Chengdu_OD_pair.xlsx';
network = '.\Anaheim\Anaheim_network1.xlsx';
networkod = '.\Anaheim\Anaheim_OD.xlsx';
[xlxOD,~,~]=xlsread(networkod);
origins = xlxOD(:,1);
destinations= xlxOD(:,2);


maxTimes = 2;   
edgeCostDistributionModel = 'Gaussian';
CR = zeros(50,6);

for i = 1 : 10     %   run time barcenola: 23
    originNode = origins(i);
    destinationNode = destinations(i);
%     originNode = 1;
%     destinationNode = 15;
    countAlgrithm = 1;
    while(countAlgrithm < 5)
        switch countAlgrithm
            case 1
                policyFunc=@DDP_DL;
%             case 2
%                 policyFunc=@Pi_tau;
%             case 3
%                 policyFunc=@RAO_star; 
%             case 4
%                 policyFunc=@AO_star;
%             case 3
%                 policyFunc=@Kahni_RSP;
%             case 4
%                 policyFunc=@Zyf_RSP;  
        end
        [LET,mean,std] = simulator_for_realistic_networks(policyFunc,network,originNode,destinationNode,maxTimes,edgeCostDistributionModel);
        competitiveRatio = (mean + zeta * std) / LET;
        clear Function;
        CR(i,countAlgrithm) = competitiveRatio;
%         xlswrite('data.xlsx',CR); 写还是有问题
        countAlgrithm = countAlgrithm + 1;
    end
end
for i = 1 : 2
    data(i) = sum(CR(:,i)) / 10;
end

 CR;

    


 