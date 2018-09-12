function PLS_calculate_stats(response_var_file, predictor_var_file, output_dir)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the PLS calculate stats function with the following arguments
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% response_var_file ------ full path to the PLS_MRI_response_vars.csv file
%%%                           that is created by the NSPN_CorticalMyelination
%%%                           wrapper script
%%% predictor_var_file ----- full path to the PLS_gene_predictor_vars.csv file
%%%                           that is provided as raw data
%%% output_dir ------------- where to save the PLS_stats file (for PLS1 and PLS2 together)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Created by Petra Vertes
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Re-run PLS to get explained variance and associated stats')

%import response variables
importdata(response_var_file);

%unwrap and tidy MRI response variable names
ROIname=ans.textdata(:,1);
ResponseVarNames=ans.textdata(1,:);
ResponseVarNames=ans.textdata(1,2:4);
ResponseVarNames=ans.textdata(1,:);
ResponseVarNames(1)=[];
ROIname(1)=[];
%and store the response variables in matrix Y
MRIdata=ans.data;
clear ans

%import predictor variables
indata=importdata(predictor_var_file);
GENEdata=indata.data;
GENEdata(1,:)=[];
genes=indata.textdata;
genes=genes(2:length(genes));
clear indata

%DO PLS in 2 dimensions (with 2 components)
X=GENEdata';
Y=zscore(MRIdata);
dim=2;
[XL,YL,XS,YS,BETA,PCTVAR,MSE,stats]=plsregress(X,Y,dim);
temp=cumsum(100*PCTVAR(2,1:dim));
Rsquared = temp(dim);

%align PLS components with desired direction%
[R1,p1]=corr([XS(:,1),XS(:,2)],MRIdata);
if R1(1,2)<0
    XS(:,1)=-1*XS(:,1);
end
if R1(2,4)<0
    XS(:,2)=-1*XS(:,2);
end

%calculate correlations of PLS components with MRI variables
[R1,p1]=corr(XS(:,1),MRIdata);
[R2,p2]=corr(XS(:,2),MRIdata);
a=[R1',p1',R2',p2'];


%assess significance of PLS result
for j=1:1000
    order=randperm(size(Y,1));
    Yp=Y(order,:);
    [XLr,YLr,XSr,YSr,BETAr,PCTVARr,MSEr,statsr]=plsregress(X,Yp,dim);
    temp=cumsum(100*PCTVARr(2,1:dim));
    Rsq(j) = temp(dim);
end
p=length(find(Rsq>=Rsquared))/j;

% plot histogram
% hist(Rsq,30)
% hold on
% plot(Rsquared,20,'.r','MarkerSize',15)
% set(gca,'Fontsize',14)
% xlabel('R squared','FontSize',14);
% ylabel('Permuted runs','FontSize',14);
% title('p<0.0001')

%save stats
myStats=[PCTVAR; p, j];
csvwrite(fullfile(output_dir,'PLS_stats.csv'),myStats);
