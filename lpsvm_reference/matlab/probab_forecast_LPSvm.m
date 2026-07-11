clear all;
close all;
% Datasets are bundled inside final_github/shared/datasets
% This script lives at final_github/lpsvm_reference/matlab/
dataset_path = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'datasets', 'beer.csv');
y = readmatrix(dataset_path);
plot(y,'b-');
%%

win =1;
kernel=2;               % window size (number of days)
s1= 2^25;
s2=2^25;                       
C1=2^25;
C2= 2^25;
q_lower = 0.025;
q_upper = 0.975;
target_cov = 0.95;
c3=1;
%%
  win_size= 1:25;
  s1val= -25:1:25;
  c1val = -25:1:25;


for ii=1:length(win_size)
    disp(ii);
    clear X_train;
    clear y_train;
  win= win_size(ii);
for jj=1:length(s1val)
  s1 = 2^s1val(jj);
  s2=2^s1val(jj);
 kerfPara1.type = 'rbf';
   kerfPara1.pars = s1;
    kerfPara2.type = 'rbf';
   kerfPara2.pars = s2;
 for kk =1:length(c1val)
    C1 = 2^c1val(kk); 
    C2 = 2^c1val(kk); 

for i = 1:(length(y) - win)
                    idx_window = i:(i+win-1);
                    X_train(i,:) = y(idx_window, :);
                    y_train(i,:) =y(idx_window(end)+1,:);
   end

              Xtest= X_train(floor(length(y_train)*0.7)+1:end,:);      
              ytest= y_train(floor(length(y_train)*0.7)+1:end,:);
              X_train= X_train(1:floor(length(y_train)*0.7),:); 
              y_train= y_train(1:floor(length(y_train)*0.7),:);
              Xval=  X_train(floor(length(y_train)*0.9)+1:end,:);
              yval= y_train(floor(length(y_train)*0.9)+1:end,:);
              X_train= X_train(1:floor(length(y_train)*0.9),:); 
              y_train= y_train(1:floor(length(y_train)*0.9),:);
%%
          
                      % Lower quantile prediction.
                     % [Pred_val_lower, f_lower, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xval, kerfPara1, C1, q_lower, 0);
                        % Upper quantile prediction.
                    %  [Pred_val_upper, f_upper, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xval, kerfPara2, C2, q_upper,0);
                       %  [Pred_val_lower, f_lower, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xtest, kerfPara1, C1, q_lower, 0);
                        % Upper quantile prediction.
                      %[Pred_val_upper, f_upper, ~, ~] = epsilon_quantilesvr2(X_train, y_train, Xtest, kerfPara2, C2, q_upper,0);

[Low_Q,Pred_val_lower, sparsity_lower] = quantileLPONENORMTSVR12(X_train, y_train, Xval, s1, c3, C1, q_lower);
[Up_Q,Pred_val_upper,  sparsity_upper] = quantileLPONENORMTSVR12(X_train, y_train, Xval, s2, c3, C1, q_upper);
     %%                 
     target=yval;
    [PICP(ii,jj,kk), MPIW(ii,jj,kk)] = evaluate_PICP(target,Pred_val_lower, Pred_val_upper);
end
end
end
%%
[maxVal, linearIdx] = max(PICP(:));
[i, j, k] = ind2sub(size(PICP), linearIdx);
 threshold = 0.95;

% Get all linear indices where PICP > threshold
validIdx = find(PICP > threshold);

if isempty(validIdx)
    fprintf('No indices found where PICP > %f.\n', threshold);
else
    % Extract the corresponding MPIW values at these indices
    MPIW_valid = MPIW(validIdx);
    
    % Find the minimum MPIW value among these and its index in MPIW_valid
    [minMPIW, idxMin] = min(MPIW_valid);
    
    % Get the corresponding linear index in the full MPIW array
    bestLinearIdx = validIdx(idxMin);
    
    % Convert the linear index to subscript indices
    [i, j, k] = ind2sub(size(MPIW), bestLinearIdx);
    
    % Display the result
    fprintf('Minimum MPIW value among indices with PICP > %.2f is %f.\n', threshold, minMPIW);
    fprintf('This MPIW value is located at index (%d, %d, %d).\n', i, j, k);
end




function [PICP, MPIW] = evaluate_PICP(y, Low_Q, Up_Q)
    PICP = mean(y >= Low_Q & y <= Up_Q);
    MPIW = mean(Up_Q - Low_Q);
end
hold on;
plot(yval, ' b-');
hold on;
plot(Pred_val_lower, ' r-');
plot(Pred_val_upper, ' k-');
