function model = train_qsvr(X, Y, kerfPara, C, tau, eps1)
% TRAIN_QSVR  Fit a quantile SVR (dual form) — training only.
%
%   model = train_qsvr(X, Y, kerfPara, C, tau, eps1)
%
%   Solves the same QP as epsilon_quantilesvr2 and returns the fitted
%   parameters in a struct. No prediction is done here.
%
%   Inputs
%     X        : n-by-d training features
%     Y        : n-by-1 training targets
%     kerfPara : struct('type', 'rbf'|'lin'|'poly', 'pars', ...)
%     C        : regularization constant
%     tau      : quantile level in (0, 1)
%     eps1     : epsilon-insensitive width (0 for standard quantile loss)
%
%   Output
%     model : struct with fields
%       .beta      n-by-1 dual coefficients (alpha - alpha*)
%       .bias      scalar bias term (0 for RBF/poly, computed for linear/sigmoid)
%       .X_train   the training features, needed for the test kernel
%       .kerfPara  the kernel spec, needed for the test kernel
%       .nsv       number of support vectors
%       .sparsity  fraction of zero beta entries

    epsilon = svtol(C);
    n = size(X, 1);

    % Train kernel matrix
    H = kernelfun(X, kerfPara);

    % QP setup — identical to epsilon_quantilesvr2
    Hb  = [H -H; -H H];
    c   = [((1 - tau) * eps1 * ones(n, 1) - Y); (tau * eps1 * ones(n, 1) + Y)];
    vlb = zeros(2 * n, 1);
    vub = [tau * C * ones(n, 1); ((1 - tau) * C) * ones(n, 1)];
    x0  = zeros(2 * n, 1);

    neqcstr = nobias(kerfPara.type);
    if neqcstr
        A = [ones(1, n) -ones(1, n)];
        b = 0;
    else
        A = [];
        b = [];
    end

    % Regularize for numerical PSD-ness
    Hb = Hb + 1e-10 * eye(size(Hb));

    options = optimoptions('quadprog', 'Display', 'off', 'MaxIterations', 1000);
    alpha = quadprog(Hb, c, [], [], A, b, vlb, vub, x0, options);

    alpha1 = alpha(1:n);
    beta1  = alpha(n+1:2*n);
    beta   = alpha1 - beta1;

    sparsity = 1 - (nnz(beta) / length(beta));

    % Bias (only for kernels flagged by nobias — linear/sigmoid)
    bias = 0;
    if neqcstr ~= 0
        if tau > 0.5
            svii = find(abs(beta) > epsilon & abs(beta) < (tau * C - epsilon));
        else
            svii = find(abs(beta) > epsilon & abs(beta) < ((1 - tau) * C - epsilon));
        end

        if ~isempty(svii)
            bias = mean(Y(svii) - H(svii, :) * beta);
        else
            fprintf('No SVs found for bias — using midpoint.\n');
            bias = (max(Y) + min(Y)) / 2;
        end
    end

    nsv = length(find(abs(beta) > epsilon));

    model = struct( ...
        'beta',     beta, ...
        'bias',     bias, ...
        'X_train',  X, ...
        'kerfPara', kerfPara, ...
        'nsv',      nsv, ...
        'sparsity', sparsity);
end
