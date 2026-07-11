clear; clc;

% Add shared MATLAB utilities to path
addpath(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'matlab_utils'));

%% 0. Define Multi-Dataset Sizes to Evaluate
% Add or change values in this array to test different dataset sizes


%% 1. Load and Prepare the Dataset
fprintf('Loading Amprion dataset ...\n');

rawData = readmatrix(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'datasets', 'Amprion.csv'));
rawData(:, 1) = [];   % Drop Date column (NaNs)

% Row-major flatten to match Python: chronological day-major ordering
% rawData is (N_days × 96) — each row = one full day of 96 power readings
y = reshape(rawData', [], 1);   % day1_int1, day1_int2, ..., day1_int96, day2_int1, ...

% Remove the first 96 values (the header-row time stamps that readmatrix parsed as NaNs)
y = y(97:end);

% Fill any remaining NaN values
y = fillmissing(y, 'linear');
y(isnan(y)) = [];

fprintf('Total points before downsample: %d\n', length(y));
y = y(1:2:end);
fprintf('Total points after downsample (step=2): %d\n', length(y));
fprintf('Final dataset size: %d samples.\n\n', length(y));
data_sizes = [4000, 6000, 8000, 10000, 12000,16000];
%% Grid search parameters
win = [24];
q_lower = 0.05;
q_upper = 0.95;
target_cov = 0.90;

win_size = win;

% Define the exhaustive grid bounds (Step = 1 checks every single member)
s1val_grid = -20:1:20;
c1val_grid = -20:1:20;


% Start parallel pool safely matching your machine's profile limits
pool = gcp('nocreate');
if isempty(pool)
    myCluster = parcluster('local');
    numWorkers = min(feature('numcores'), myCluster.NumWorkers);
    fprintf('Starting parallel pool with %d workers...\n', numWorkers);
    parpool('local', numWorkers);
end

% Slice a 1,000-point subset for fast hyperparameter tuning
y_tune = y(1:min(1000, length(y)));
all_candidates = struct('win', {}, 's_exp', {}, 'c_exp', {}, 'PICP', {}, 'MPIW', {});

% =====================================================================
% STAGE 1 — Grid Search Tuning on 1k Data Subset (80/20 train/val)
% =====================================================================
for ii = 1:length(win_size)
    disp(ii);
    win = win_size(ii);

    % Build dataset from windows
    clear X_all y_all;
    for i = 1:(length(y_tune) - win - 1)
        idx_window = i:(i+win-1);
        X_all(i,:) = y_tune(idx_window, :);
        y_all(i,:) = y_tune(idx_window(end)+1,:);
    end

    % 80/20 train/val split
    split1 = floor(length(y_all) * 0.80);
    X_tr  = X_all(1:split1, :);
    y_tr  = y_all(1:split1, :);
    Xval  = X_all(split1+1:end, :);
    yval  = y_all(split1+1:end, :);

    [JJ, KK] = ndgrid(1:length(s1val_grid), 1:length(c1val_grid));
    JJ = JJ(:);  KK = KK(:);
    nTotal = length(JJ);

    % Preallocate results vectors
    PICP_arr = zeros(nTotal, 1);
    MPIW_arr = zeros(nTotal, 1);

    fprintf('Tuning Phase | win=%d | Checking all grid members = %d ...\n', win, nTotal);

    parfor idx = 1:nTotal
        jj = JJ(idx);  kk = KK(idx);
        s1 = 2^s1val_grid(jj);  s2 = 2^s1val_grid(jj);
        C1 = 2^c1val_grid(kk);  C2 = 2^c1val_grid(kk);

        % Defined locally inside parfor to prevent worker classification errors
        kerfPara1 = struct('type', 'rbf', 'pars', s1);
        kerfPara2 = struct('type', 'rbf', 'pars', s2);

        [Pred_lower, ~, ~, ~] = epsilon_quantilesvr2(X_tr, y_tr, Xval, kerfPara1, C1, q_lower, 0);
        [Pred_upper, ~, ~, ~] = epsilon_quantilesvr2(X_tr, y_tr, Xval, kerfPara2, C2, q_upper, 0);

        [PICP_arr(idx), MPIW_arr(idx)] = evaluate_PICP(yval, Pred_lower, Pred_upper);
    end

    % Pre-extend the struct array size once to bypass loop reallocation lag
    start_pos = length(all_candidates);
    all_candidates(start_pos + nTotal).win = [];

    for idx = 1:nTotal
        all_candidates(start_pos + idx).win = win;
        all_candidates(start_pos + idx).s_exp = s1val_grid(JJ(idx));
        all_candidates(start_pos + idx).c_exp = c1val_grid(KK(idx));
        all_candidates(start_pos + idx).PICP = PICP_arr(idx);
        all_candidates(start_pos + idx).MPIW = MPIW_arr(idx);
    end
end

% =====================================================================
% SELECT THE SINGLE BEST HYPERPARAMETER CONFIGURATION
% =====================================================================
% Selection: if any config meets PICP >= threshold, pick min MPIW among them.
% Otherwise, fall back to highest PICP (tiebreak by lowest MPIW).
threshold = 0.90;
valid_tune_idx = find([all_candidates.PICP] >= threshold);

if isempty(valid_tune_idx)
    picp_vals = [all_candidates.PICP];
    max_picp  = max(picp_vals);
    top_idx   = find(picp_vals == max_picp);
    [~, tmp_min] = min([all_candidates(top_idx).MPIW]);
    best_idx  = top_idx(tmp_min);
else
    valid_candidates = all_candidates(valid_tune_idx);
    [~, tmp_min] = min([valid_candidates.MPIW]);
    best_idx = valid_tune_idx(tmp_min);
end

best_cand = all_candidates(best_idx);

fprintf('\n========== BEST PARAMETERS FOUND (ON 1K SUBSET) ==========\n');
fprintf('Window Size : %d\n', best_cand.win);
fprintf('s exponent  : %d (s = %e)\n', best_cand.s_exp, 2^best_cand.s_exp);
fprintf('C exponent  : %d (C = %e)\n', best_cand.c_exp, 2^best_cand.c_exp);
fprintf('Tune PICP   : %.4f\n', best_cand.PICP);
fprintf('Tune MPIW   : %.4f\n\n', best_cand.MPIW);

% =====================================================================
% STAGE 2 — Sliding Test Set Evaluation Across Multi Data Sizes
% =====================================================================
% For each size N: last 1k = test, rest = train+val pool (90/10 split).
% Example: 4k → 3k train+val + 1k test; 6k → 5k train+val + 1k test.
TEST_SIZE = 1000;
VAL_FRAC = 0.10;

% Create output folder for all plots and results
out_folder = fullfile(fileparts(mfilename('fullpath')), 'results');
if ~exist(out_folder, 'dir')
    mkdir(out_folder);
end

fprintf('Evaluating the single best configuration on requested dataset sizes...\n');

n_full = length(y);
assert(max(data_sizes) <= n_full, ...
    sprintf('Need at least %d samples; have %d.', max(data_sizes), n_full));

size_results = struct('size', {}, 'PICP_val', {}, 'MPIW_val', {}, 'PICP_test', {}, 'MPIW_test', {}, ...
                      'train_time_lower', {}, 'train_time_upper', {}, 'total_time', {});

for s_idx = 1:length(data_sizes)
    curr_size = data_sizes(s_idx);

    % Take first N samples
    y_chunk = y(1:curr_size);

    % Split: last TEST_SIZE = test, rest = train+val pool (at sample level)
    y_test_raw = y_chunk(curr_size - TEST_SIZE + 1 : end);
    y_pool     = y_chunk(1 : curr_size - TEST_SIZE);

    win = best_cand.win;

    % Build windows from test portion
    clear X_te y_te;
    for i = 1:(length(y_test_raw) - win - 1)
        idx_window = i:(i+win-1);
        X_te(i,:) = y_test_raw(idx_window, :);
        y_te(i,:) = y_test_raw(idx_window(end)+1,:);
    end

    % Build windows from train+val pool
    clear X_pool y_pool_t;
    for i = 1:(length(y_pool) - win - 1)
        idx_window = i:(i+win-1);
        X_pool(i,:) = y_pool(idx_window, :);
        y_pool_t(i,:) = y_pool(idx_window(end)+1,:);
    end

    % 90/10 chronological train/val split inside the pool
    n_pool = length(y_pool_t);
    i_val = floor(n_pool * (1.0 - VAL_FRAC));
    X_train_f = X_pool(1:i_val, :);
    y_train_f = y_pool_t(1:i_val, :);
    Xval_f    = X_pool(i_val+1:end, :);
    yval_f    = y_pool_t(i_val+1:end, :);

    fprintf('  Size %d: train=%d  val=%d  test=%d\n', ...
            curr_size, size(X_train_f,1), size(Xval_f,1), size(X_te,1));

    kerfPara1 = struct('type', 'rbf', 'pars', 2^best_cand.s_exp);
    kerfPara2 = struct('type', 'rbf', 'pars', 2^best_cand.s_exp);

    % --- Time the lower quantile SVR training ---
    % --- Stack val + test for single-train prediction ---
    X_pred_both = [Xval_f; X_te];
    nVal = size(Xval_f, 1);

    % --- Time the lower quantile SVR training ---
    tic;
    [Pred_lower_both, ~, ~, ~] = epsilon_quantilesvr2(X_train_f, y_train_f, X_pred_both, kerfPara1, 2^best_cand.c_exp, q_lower, 0);
    t_lower = toc;
    Pred_val_lower  = Pred_lower_both(1:nVal);
    Pred_test_lower = Pred_lower_both(nVal+1:end);

    % --- Time the upper quantile SVR training ---
    tic;
    [Pred_upper_both, ~, ~, ~] = epsilon_quantilesvr2(X_train_f, y_train_f, X_pred_both, kerfPara2, 2^best_cand.c_exp, q_upper, 0);
    t_upper = toc;
    Pred_val_upper  = Pred_upper_both(1:nVal);
    Pred_test_upper = Pred_upper_both(nVal+1:end);

    total_time = t_lower + t_upper;

    % --- Evaluate on val and sliding test ---
    [curr_PICP_val, curr_MPIW_val]   = evaluate_PICP(yval_f, Pred_val_lower, Pred_val_upper);
    [curr_PICP_test, curr_MPIW_test] = evaluate_PICP(y_te, Pred_test_lower, Pred_test_upper);

    % Store metrics
    size_results(s_idx).size = curr_size;
    size_results(s_idx).PICP_val = curr_PICP_val;
    size_results(s_idx).MPIW_val = curr_MPIW_val;
    size_results(s_idx).PICP_test = curr_PICP_test;
    size_results(s_idx).MPIW_test = curr_MPIW_test;
    size_results(s_idx).train_time_lower = t_lower;
    size_results(s_idx).train_time_upper = t_upper;
    size_results(s_idx).total_time = total_time;

    fprintf('  -> Complete: Size %d | Val PICP: %.4f MPIW: %.4f | Test PICP: %.4f MPIW: %.4f | Time: %.2f s\n', ...
            curr_size, curr_PICP_val, curr_MPIW_val, curr_PICP_test, curr_MPIW_test, total_time);

    % --- Plot: Validation Set ---
    fig = figure;
    hold on; grid on; box on;
    plot(yval_f, 'b-', 'LineWidth', 1.8);
    plot(Pred_val_lower, 'r-', 'LineWidth', 1.5);
    plot(Pred_val_upper, 'k-', 'LineWidth', 1.5);
    legend('Actual', 'Lower Quantile', 'Upper Quantile', 'Location', 'best');
    xlabel('Time Index'); ylabel('Value');
    title(sprintf('Validation | Size: %dk | Win=%d | s=2^{%d} | C=2^{%d}', ...
                  curr_size/1000, win, best_cand.s_exp, best_cand.c_exp));
    set(gca,'FontSize',12);
    hold off;
    saveas(fig, fullfile(out_folder, sprintf('%dk_val.png', curr_size/1000)));

    % --- Plot: Test Set (sliding) ---
    fig = figure;
    hold on; grid on; box on;
    plot(y_te, 'b-', 'LineWidth', 1.8);
    plot(Pred_test_lower, 'r-', 'LineWidth', 1.5);
    plot(Pred_test_upper, 'k-', 'LineWidth', 1.5);
    legend('Actual', 'Lower Quantile', 'Upper Quantile', 'Location', 'best');
    xlabel('Time Index'); ylabel('Value');
    title(sprintf('Test (last 1k) | Size: %dk | Win=%d | s=2^{%d} | C=2^{%d}', ...
                  curr_size/1000, win, best_cand.s_exp, best_cand.c_exp));
    set(gca,'FontSize',12);
    hold off;
    saveas(fig, fullfile(out_folder, sprintf('%dk_test.png', curr_size/1000)));
end

% =====================================================================
% SUMMARY PLOTS (saved to svm_amprion_results/)
% =====================================================================
sizes_arr  = [size_results.size];
picp_val   = [size_results.PICP_val];
mpiw_val   = [size_results.MPIW_val];
picp_test  = [size_results.PICP_test];
mpiw_test  = [size_results.MPIW_test];
time_arr   = [size_results.total_time];
ratio_val  = mpiw_val ./ picp_val;
ratio_test = mpiw_test ./ picp_test;

% 1. MPIW/PICP Ratio vs Data Size
fig = figure;
hold on; grid on; box on;
hb = bar(sizes_arr, [ratio_val(:), ratio_test(:)], 0.6, 'FaceColor', 'flat', 'EdgeColor', 'k');
hb(1).FaceColor = [0.2 0.6 0.85];
hb(2).FaceColor = [0.85 0.3 0.3];
xlabel('Dataset Size (samples)', 'FontSize', 13);
ylabel('MPIW / PICP Ratio', 'FontSize', 13);
title('MPIW/PICP vs. Data Size', 'FontSize', 14, 'FontWeight', 'bold');
legend('Validation', 'Test', 'Location', 'best');
set(gca, 'FontSize', 12);
saveas(fig, fullfile(out_folder, 'mpiw_picp_ratio_vs_data_size.png'));

% 2. PICP vs Data Size
fig = figure;
hold on; grid on; box on;
plot(sizes_arr, picp_val, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.6 0.85]);
plot(sizes_arr, picp_test, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.3 0.3]);
xlabel('Dataset Size (samples)', 'FontSize', 13);
ylabel('PICP (Coverage)', 'FontSize', 13);
title('PICP vs. Data Size', 'FontSize', 14, 'FontWeight', 'bold');
legend('Validation', 'Test', 'Location', 'best');
set(gca, 'FontSize', 12);
yline(target_cov, 'r--', sprintf('Target: %.2f', target_cov), 'LineWidth', 1.5);
saveas(fig, fullfile(out_folder, 'picp_vs_data_size.png'));

% 3. MPIW vs Data Size
fig = figure;
hold on; grid on; box on;
plot(sizes_arr, mpiw_val, '-s', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.6 0.85]);
plot(sizes_arr, mpiw_test, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0.85 0.3 0.3]);
xlabel('Dataset Size (samples)', 'FontSize', 13);
ylabel('MPIW (Width)', 'FontSize', 13);
title('MPIW vs. Data Size', 'FontSize', 14, 'FontWeight', 'bold');
legend('Validation', 'Test', 'Location', 'best');
set(gca, 'FontSize', 12);
saveas(fig, fullfile(out_folder, 'mpiw_vs_data_size.png'));

% 4. Training Time vs Data Size
fig = figure;
hold on; grid on; box on;
plot(sizes_arr, time_arr, '-o', 'LineWidth', 2, 'MarkerSize', 8, 'MarkerFaceColor', [0.2 0.6 0.85]);
xlabel('Dataset Size (samples)', 'FontSize', 13);
ylabel('Total Training Time (seconds)', 'FontSize', 13);
title('Training Time vs. Dataset Size', 'FontSize', 14, 'FontWeight', 'bold');
set(gca, 'FontSize', 12);
saveas(fig, fullfile(out_folder, 'training_time_vs_data_size.png'));

fprintf('\nAll plots saved to: %s/\n', out_folder);

function [PICP, MPIW] = evaluate_PICP(y, Low_Q, Up_Q)
    PICP = mean(y >= Low_Q & y <= Up_Q);
    MPIW = mean(Up_Q - Low_Q);
end

%% ============================================================
% Save ALL Multi-Size Results to One Single Text File
%% ============================================================
timestamp = datestr(now,'yyyy_mm_dd_HH_MM_SS');
outputFile = fullfile(out_folder, sprintf('results_amprion_%s.txt', timestamp));
fid = fopen(outputFile, 'w');

fprintf(fid, '========== Amprion Quantile SVR — Sliding Test Results ==========\n\n');

fprintf(fid, 'Tuning Workspace Parameters\n');
fprintf(fid, '--------------------------------------------------\n');
fprintf(fid, 'Tuning Sample Size                    : 1000 points\n');
fprintf(fid, 'Tuning Split                          : 80/20 train/val\n');
fprintf(fid, 'Target Confidence Quantiles           : [%.2f, %.2f]\n', q_lower, q_upper);
fprintf(fid, 'Minimum Acceptance Target Coverage    : %.2f\n', target_cov);
fprintf(fid, 'Selection Criterion                   : Min MPIW among PICP >= threshold.\n');
fprintf(fid, '                                         Fallback: max PICP (tiebreak = min MPIW)\n\n');

fprintf(fid, 'Selected Hyperparameters (From 1k Tuning)\n');
fprintf(fid, '--------------------------------------------------\n');
fprintf(fid, 'Chosen Window Size : %d\n', best_cand.win);
fprintf(fid, 's Kernel Exponent  : %d (s = %e)\n', best_cand.s_exp, 2^best_cand.s_exp);
fprintf(fid, 'C Cost Exponent    : %d (C = %e)\n', best_cand.c_exp, 2^best_cand.c_exp);
fprintf(fid, 'Tuning Phase PICP  : %.6f\n', best_cand.PICP);
fprintf(fid, 'Tuning Phase MPIW  : %.6f\n\n', best_cand.MPIW);

fprintf(fid, 'Stage 2 Protocol: SLIDING test set\n');
fprintf(fid, '  For each size N: last %d = test, first (N-%d) = train+val (90/10).\n', TEST_SIZE, TEST_SIZE);
fprintf(fid, '  Example: 4k -> 3k train+val + 1k test; 6k -> 5k train+val + 1k test.\n');
fprintf(fid, '  sigma2_model = 0 (SVM is deterministic).\n\n');

fprintf(fid, 'Performance Summary Across Different Data Sizes\n');
fprintf(fid, '--------------------------------------------------------------------------------\n');
fprintf(fid, '%-10s %-8s %-10s %-10s %-10s %-10s %-12s %-10s\n', ...
        'Data Size', 'Train', 'Val PICP', 'Val MPIW', 'Test PICP', 'Test MPIW', 'MPIW/PICP', 'Time (s)');
fprintf(fid, '--------------------------------------------------------------------------------\n');
for s_idx = 1:length(size_results)
    s = size_results(s_idx);
    train_size = s.size - TEST_SIZE;
    ratio = s.MPIW_test / s.PICP_test;
    fprintf(fid, '%-10d %-8d %-10.6f %-10.4f %-10.6f %-10.4f %-12.6f %-10.2f\n', ...
            s.size, train_size, s.PICP_val, s.MPIW_val, ...
            s.PICP_test, s.MPIW_test, ratio, s.total_time);
end
fprintf(fid, '--------------------------------------------------------------------------------\n\n');

fprintf(fid, 'Execution completed successfully.\n');
fprintf(fid, 'Timestamp : %s\n', datestr(now));
fclose(fid);

fprintf('Results saved to: %s\n', outputFile);