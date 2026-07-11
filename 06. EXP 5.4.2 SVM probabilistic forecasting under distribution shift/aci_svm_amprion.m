clear; clc;

% Add shared MATLAB utilities to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'shared', 'matlab_utils'));

%% ============================================================
%  Amprion SVM + ACI (Adaptive Conformal Inference)
%  Reviewer Response: Distribution Shift Experiment
%
%  Train on first 5000 points, test on next 10000 points.
%  ACI with rolling calibration window = 500.
%  alpha = 0.1, gamma = 0.001.
%
%  Best hyperparams (from prior tuning, no re-tuning):
%    window = 24, s = 2^20, C = 2^16
%% ============================================================

%% 1. Load and Prepare the Dataset
fprintf('Loading Amprion dataset ...\n');

rawData = readmatrix(fullfile(fileparts(mfilename('fullpath')), '..', 'shared', 'datasets', 'Amprion.csv'));
rawData(:, 1) = [];   % Drop Date column (NaNs)

% Row-major flatten — chronological day-major ordering
y = reshape(rawData', [], 1);

% Remove first 96 values (header-row NaNs parsed by readmatrix)
y = y(97:end);

% Fill remaining NaNs
y = fillmissing(y, 'linear');
y(isnan(y)) = [];

fprintf('Total points before downsample: %d\n', length(y));
y = y(1:2:end);
fprintf('Total points after  downsample: %d\n', length(y));
fprintf('Final dataset size: %d samples.\n\n', length(y));

%% 2. Config
q_lower    = 0.05;
q_upper    = 0.95;
target_cov = 0.90;
win_size   = 24;

TRAIN_END  = 5000;    % indices 1–5000 for training
TEST_SIZE  = 10000;   % indices 5001–15000 for test
CAL_WINDOW = 500;     % ACI rolling calibration window
ALPHA      = 0.1;
GAMMA      = 0.001;

% Best hyperparams from prior tuning
best_s_exp = 20;      % s = 2^20
best_c_exp = 16;      % C = 2^16

TOTAL_NEEDED = TRAIN_END + TEST_SIZE;  % = 15000
assert(length(y) >= TOTAL_NEEDED, ...
    sprintf('Need at least %d samples; have %d.', TOTAL_NEEDED, length(y)));

fprintf('Split: training=%d | test=%d\n', TRAIN_END, TEST_SIZE);
fprintf('ACI: cal_window=%d, alpha=%.3f, gamma=%.4f\n\n', CAL_WINDOW, ALPHA, GAMMA);

%% 3. Output folder
out_folder = fullfile(fileparts(mfilename('fullpath')), 'aci_svm_results');
if ~exist(out_folder, 'dir')
    mkdir(out_folder);
end

%% 4. Build windows from training set
fprintf('Building windows (win=%d)...\n', win_size);

y_train_raw = y(1:TRAIN_END);
y_test_raw  = y(TRAIN_END+1 : TRAIN_END+TEST_SIZE);

win = win_size;

% Training windows
clear X_tr y_tr;
for i = 1:(length(y_train_raw) - win - 1)
    idx_window = i:(i+win-1);
    X_tr(i,:)  = y_train_raw(idx_window, :);
    y_tr(i,:)  = y_train_raw(idx_window(end)+1, :);
end

% Test windows
clear X_te y_te;
for i = 1:(length(y_test_raw) - win - 1)
    idx_window = i:(i+win-1);
    X_te(i,:)  = y_test_raw(idx_window, :);
    y_te(i,:)  = y_test_raw(idx_window(end)+1, :);
end

fprintf('  train=%d  test=%d\n', size(X_tr,1), size(X_te,1));

%% 5. Train SVM quantile models (no re-tuning, fixed hyperparams)
fprintf('\nTraining SVM quantile models (s=2^%d, C=2^%d)...\n', best_s_exp, best_c_exp);

s_val = 2^best_s_exp;
C_val = 2^best_c_exp;

kerfPara = struct('type', 'rbf', 'pars', s_val);

% Fit each quantile once — QP is solved a total of two times, ever.
tic;
model_lower = train_qsvr(X_tr, y_tr, kerfPara, C_val, q_lower, 0);
model_upper = train_qsvr(X_tr, y_tr, kerfPara, C_val, q_upper, 0);
t_train = toc;
fprintf('  Training time: %.2f s\n', t_train);

% Training predictions (for initial calibration scores) — pure inference
Pred_lower_tr = predict_qsvr(X_tr, model_lower);
Pred_upper_tr = predict_qsvr(X_tr, model_upper);

% Training residuals for initial calibration
resid_lower_tr = Pred_lower_tr - y_tr;
resid_upper_tr = y_tr - Pred_upper_tr;
scores_tr = max(resid_lower_tr, resid_upper_tr);

%% 6. Batch predict on test set (pure inference — no retraining)
fprintf('\nBatch predicting on %d test points...\n', size(X_te,1));

n_test = size(X_te, 1);

tic;
pred_lower_raw = predict_qsvr(X_te, model_lower);
pred_upper_raw = predict_qsvr(X_te, model_upper);
t_predict = toc;
fprintf('  Batch prediction time: %.2f s\n', t_predict);

%% 7. Run ACI on stored predictions (no retraining)
fprintf('Running ACI on %d test points...\n', n_test);

% Storage
pred_lower_aci = zeros(n_test, 1);
pred_upper_aci = zeros(n_test, 1);
alpha_trajectory = zeros(n_test, 1);
adapt_err_seq    = zeros(n_test, 1);
y_true_all       = y_te;

% Initial calibration scores from training set (last CAL_WINDOW)
cal_scores = scores_tr(max(1, end-CAL_WINDOW+1):end);

alphat = ALPHA;

tic;
for t = 1:n_test
    q_lo = pred_lower_raw(t);
    q_hi = pred_upper_raw(t);
    y_t  = y_te(t);

    % ACI: compute adaptive conformal quantile
    alphat = max(min(alphat, 1 - 1e-6), 1e-6);
    level_adapt = (1.0 - alphat) * (1.0 + 1.0 / length(cal_scores));
    level_adapt = max(min(level_adapt, 1.0), 0.0);
    confQuant = quantile(cal_scores, level_adapt);

    % Apply conformal correction
    pred_lower_aci(t) = q_lo - confQuant;
    pred_upper_aci(t) = q_hi + confQuant;

    % Compute conformity score for this point
    newScore = max(q_lo - y_t, y_t - q_hi);
    adaptErr = double(newScore > confQuant);

    % Update alpha_t
    alphat_new = alphat + GAMMA * (ALPHA - adaptErr);
    alphat = max(min(alphat_new, 1 - 1e-6), 1e-6);

    % Update rolling calibration window
    cal_scores = [cal_scores(2:end); newScore];

    alpha_trajectory(t) = alphat;
    adapt_err_seq(t)    = adaptErr;
end
t_aci = toc;
fprintf('  ACI loop time: %.2f s\n', t_aci);
fprintf('  Total time: %.2f s (train=%.2f + predict=%.2f + aci=%.2f)\n', t_train+t_predict+t_aci, t_train, t_predict, t_aci);

%% 7. Evaluate
[PICP_raw, MPIW_raw] = evaluate_PICP(y_true_all, pred_lower_raw, pred_upper_raw);
[PICP_aci, MPIW_aci] = evaluate_PICP(y_true_all, pred_lower_aci, pred_upper_aci);

% Local PICP (window=24)
local_picp_raw = find_local_picp(pred_lower_raw, pred_upper_raw, y_true_all, 24);
local_picp_aci = find_local_picp(pred_lower_aci, pred_upper_aci, y_true_all, 24);

mean_lpicp_raw = mean(local_picp_raw);
mean_lpicp_aci = mean(local_picp_aci);

% LPICE
lpice_raw = sqrt(mean(max(target_cov - local_picp_raw, 0).^2));
lpice_aci = sqrt(mean(max(target_cov - local_picp_aci, 0).^2));

% Accuracy (fraction of windows meeting target coverage)
acc_raw = mean(local_picp_raw >= target_cov);
acc_aci = mean(local_picp_aci >= target_cov);

fprintf('\n========== RESULTS ==========\n');
fprintf('%-12s  %-10s  %-10s  %-10s  %-10s\n', 'Method', 'PICP', 'MPIW', 'LPICE', 'Accuracy');
fprintf('%-12s  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n', 'SVM (raw)', PICP_raw, MPIW_raw, lpice_raw, acc_raw);
fprintf('%-12s  %-10.4f  %-10.4f  %-10.4f  %-10.4f\n', 'SVM+ACI', PICP_aci, MPIW_aci, lpice_aci, acc_aci);

%% 8. Plots

% 1. Test predictions: raw vs ACI
fig = figure('Visible','off');
hold on; grid on; box on;
plot(y_true_all,        'b-', 'LineWidth', 1.2);
plot(pred_lower_raw,    'r--', 'LineWidth', 1.0);
plot(pred_upper_raw,    'r--', 'LineWidth', 1.0);
plot(pred_lower_aci,    'g-', 'LineWidth', 1.2);
plot(pred_upper_aci,    'g-', 'LineWidth', 1.2);
legend('Actual', 'Lower (raw)', 'Upper (raw)', 'Lower (ACI)', 'Upper (ACI)', 'Location', 'best');
xlabel('Time Index'); ylabel('Value');
title(sprintf('SVM: Raw vs ACI | PICP_{raw}=%.3f PICP_{aci}=%.3f', PICP_raw, PICP_aci));
set(gca,'FontSize',12);
hold off;
saveas(fig, fullfile(out_folder, 'svm_raw_vs_aci.png'));

% 2. Alpha trajectory
fig = figure('Visible','off');
plot(alpha_trajectory, 'b-', 'LineWidth', 1.2);
hold on; grid on; box on;
yline(ALPHA, 'r--', sprintf('Target \\alpha=%.3f', ALPHA), 'LineWidth', 1.5);
xlabel('Test Step'); ylabel('\alpha_t');
title('SVM+ACI: Alpha Trajectory');
set(gca,'FontSize',12);
hold off;
saveas(fig, fullfile(out_folder, 'svm_alpha_trajectory.png'));

% 3. Local PICP comparison
fig = figure('Visible','off');
hold on; grid on; box on;
plot(local_picp_raw, 'r-', 'LineWidth', 1.0);
plot(local_picp_aci, 'b-', 'LineWidth', 1.2);
yline(target_cov, 'k--', sprintf('Target %.2f', target_cov), 'LineWidth', 1.5);
legend('SVM (raw)', 'SVM+ACI', 'Location', 'best');
xlabel('Time Index'); ylabel('Local PICP');
title(sprintf('Local PICP | LPICE_{raw}=%.4f LPICE_{aci}=%.4f', lpice_raw, lpice_aci));
set(gca,'FontSize',12);
hold off;
saveas(fig, fullfile(out_folder, 'svm_local_picp_comparison.png'));

% 4. PICP/MPIW bar chart
fig = figure('Visible','off');
subplot(1,2,1);
bar([PICP_raw, PICP_aci], 'FaceColor', 'flat');
xticklabels({'SVM (raw)','SVM+ACI'});
ylabel('PICP'); title('PICP Comparison');
yline(target_cov,'r--',sprintf('Target %.2f', target_cov),'LineWidth',1.5);
grid on;

subplot(1,2,2);
bar([MPIW_raw, MPIW_aci], 'FaceColor', 'flat');
xticklabels({'SVM (raw)','SVM+ACI'});
ylabel('MPIW'); title('MPIW Comparison');
grid on;
saveas(fig, fullfile(out_folder, 'svm_picp_mpiw_bars.png'));

fprintf('\nAll plots saved to: %s/\n', out_folder);

%% 9. Save results to text file
timestamp  = datestr(now,'yyyy_mm_dd_HH_MM_SS');
outputFile = fullfile(out_folder, sprintf('results_svm_aci_%s.txt', timestamp));
fid = fopen(outputFile, 'w');

fprintf(fid, '========== Amprion SVM + ACI Results ==========\n\n');
fprintf(fid, 'Experiment Protocol: Distribution Shift (ACI)\n');
fprintf(fid, '--------------------------------------------------\n');
fprintf(fid, 'Training set  : first %d samples (indices 1–%d)\n', TRAIN_END, TRAIN_END);
fprintf(fid, 'Test set      : next  %d samples (indices %d–%d)\n', TEST_SIZE, TRAIN_END+1, TRAIN_END+TEST_SIZE);
fprintf(fid, 'ACI cal window: %d\n', CAL_WINDOW);
fprintf(fid, 'Alpha         : %.3f\n', ALPHA);
fprintf(fid, 'Gamma         : %.4f\n\n', GAMMA);

fprintf(fid, 'Hyperparameters (from prior tuning, no re-tuning)\n');
fprintf(fid, '--------------------------------------------------\n');
fprintf(fid, 'Window Size : %d\n', win_size);
fprintf(fid, 's           : 2^%d = %e\n', best_s_exp, s_val);
fprintf(fid, 'C           : 2^%d = %e\n\n', best_c_exp, C_val);

fprintf(fid, 'Performance Summary\n');
fprintf(fid, '--------------------------------------------------------------------------------\n');
fprintf(fid, '%-12s %-10s %-10s %-10s %-10s\n', 'Method', 'PICP', 'MPIW', 'LPICE', 'Accuracy');
fprintf(fid, '--------------------------------------------------------------------------------\n');
fprintf(fid, '%-12s %-10.6f %-10.4f %-10.6f %-10.4f\n', 'SVM(raw)', PICP_raw, MPIW_raw, lpice_raw, acc_raw);
fprintf(fid, '%-12s %-10.6f %-10.4f %-10.6f %-10.4f\n', 'SVM+ACI', PICP_aci, MPIW_aci, lpice_aci, acc_aci);
fprintf(fid, '--------------------------------------------------------------------------------\n\n');

fprintf(fid, 'Timing\n');
fprintf(fid, '--------------------------------------------------\n');
fprintf(fid, '  Training (2 models) : %.2f s\n', t_train);
fprintf(fid,  '  Batch prediction    : %.2f s\n', t_predict);
fprintf(fid,  '  ACI loop            : %.2f s\n', t_aci);
fprintf(fid,  '  Total               : %.2f s\n\n', t_train + t_predict + t_aci);

fprintf(fid, 'Execution completed successfully.\n');
fprintf(fid, 'Timestamp : %s\n', datestr(now));
fclose(fid);

fprintf('Results saved to: %s\n', outputFile);

%% ============================================================
%  Helper functions
%% ============================================================
function [PICP, MPIW] = evaluate_PICP(y, Low_Q, Up_Q)
    PICP = mean(y >= Low_Q & y <= Up_Q);
    MPIW = mean(Up_Q - Low_Q);
end

function local_picp = find_local_picp(lower, upper, predictions, win)
    window = win;
    n = length(lower);
    local_picp = zeros(n, 1);
    half_left  = floor((window - 1) / 2);
    half_right = floor(window / 2);

    for i = 1:n
        start_idx = max(1, i - half_left);
        end_idx   = min(n, i + half_right);

        cur = 0;
        for j = start_idx:end_idx
            if lower(j) <= predictions(j) && predictions(j) <= upper(j)
                cur = cur + 1;
            end
        end
        local_picp(i) = cur / (end_idx - start_idx + 1);
    end
end
