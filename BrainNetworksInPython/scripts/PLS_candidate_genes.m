%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function performs permutation test for significance of up/down
% weighting of candidate gene lists in a PLS component

% INPUTS:
%fid1 is the PLS gene weights input file
%fid2 is the candidate genes input file
%fid3 is the name of the output file
%ABSOLUTE is an option defining whether we use absolute values of z-scores
%in permutation test.
%Use ABSOLUTE=true for schizophrenia gene set and ABSOLUTE=false for oligo gene set

%example:
%PLS_candidate_genes('PLS2_geneWeights.csv','Candidate_genes_schizophrenia.csv','schizophrenia_pls2_stats.csv',true);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Created by Petra Vertes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function y = PLS_candidate_genes(fid1,fid2,fid3,ABSOLUTE)

disp('  Running candidate gene analysis')

importdata(fid1);
PLSind=ans.data(:,1);
PLSweight=ans.data(:,2);
PLSgenes=ans.textdata;
clear ans

importdata(fid2);
cand=ans.data(:,1);
CANDgenes=ans.textdata;
clear ans

PLS_Z=zscore(PLSweight);
for i=1:length(cand)
    CANDind(i)=find(PLSind==cand(i));
end
myZ=PLS_Z(CANDind);

if (ABSOLUTE)
    R1=mean(abs(myZ));
else
    R1=mean(myZ);
end

Rperm=[];
count=0;
for r=1:10000
    permo=randperm(length(PLSgenes));
    myZr=PLS_Z(permo(1:length(cand)));
    if (ABSOLUTE)
        R2=mean(abs(myZr));
    else
        R2=mean(myZr);
    end

    if R2>=R1
    count=count+1;
    end

    Rperm=[Rperm;R2];
end

R=R1;
p=count/r;

myoutput=[R;Rperm];
csvwrite(fid3,myoutput);
