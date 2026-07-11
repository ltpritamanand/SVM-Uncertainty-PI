function [trainf1, f1, sparsity] = quantileLPONENORMTSVR12(train, ytrain, test, s, c3, c1, tau1)
    n1 = length(train);
    e = ones(n1, 1);
    kerfPara.type = 'rbf';
    kerfPara.pars = s;

    % Calculate kernel matrix for training data
    A = kernelfun(train, kerfPara);
    H = [A, e];

    %% Optimization Problem
    % Objective function coefficients
    f = [c3 * ones(n1 + 1, 1); 
         c3 * ones(n1 + 1, 1); 
         c1 * tau1 * e; 
         c1 * (1 - tau1) * e];

    % Construct constraints
    A1 = [-H, H, -eye(n1, n1), zeros(n1, n1);
           H, -H, zeros(n1, n1), -eye(n1, n1)];
    B1 = [-ytrain; ytrain];

    % Lower and upper bounds
    lb1 = zeros(size(f));
    ub1 = inf(size(f));

    % Debugging output
    % disp('Objective function f:');
    % disp(f);
    % disp('Constraints A1:');
    % disp(A1);
    % disp('Constraints B1:');
    % disp(B1);
    %disp('Lower bounds lb1:');
    %disp(lb1);
    %disp('Upper bounds ub1:');
    %disp(ub1);

    % Suppress output for linprog
    %options = optimoptions('linprog', 'Display', 'off');

    % Solve the linear programming problem
    options = optimoptions('linprog','Algorithm','interior-point','Display', 'off');

    [x, ~, exitflag] = linprog(f, A1, B1, [], [], lb1, ub1, options);
    if exitflag <= 0
    warning('linprog did not converge. Exit flag: %d. Using arbitrary fallback value for x.', exitflag);
        x= lb1;
    end
   
    % Calculate predictions and sparsity
    Htest = [kernelfun(test, kerfPara, train), ones(size(test, 1), 1)];
    u1 = x(1:n1 + 1) - x(n1 + 2:2 * (n1 + 1));
    
    trainf1 = H * u1;   % Predictions for training data
    f1 = Htest * u1;    % Predictions for test data
    sparsity = sum(u1 ~= 0) / length(u1); % Calculate sparsity ratio
end

% Kernel function for RBF kernel
function K = kernelfun(X, kerfPara, Y)
    if nargin == 2
        Y = X;
    end
    if strcmp(kerfPara.type, 'rbf')
        gamma = kerfPara.pars;  % Extract the gamma (s) parameter
        sqdist = pdist2(X, Y, 'euclidean').^2;
        K = exp(-gamma * sqdist);  % Apply the kernel formula correctly
    else
        error('Unknown kernel function');
    end
end


