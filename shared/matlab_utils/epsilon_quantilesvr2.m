function [PredictY,f1,nsv,sparsity] = epsilon_quantilesvr2(X,Y,test,kerfPara,C,tau,eps1) 
    epsilon = svtol(C);
    n = size(X,1);
    fprintf('Constructing ...\n');
    H = kernelfun(X, kerfPara);

    Hb = [H -H; -H H];
    c  = [((1-tau)*eps1*ones(n,1) - Y); (tau*eps1*ones(n,1) + Y)];
    vlb = zeros(2*n,1);
    vub = [tau*C*ones(n,1); ((1-tau)*C)*ones(n,1)];
    x0  = zeros(2*n,1);
    neqcstr = nobias(kerfPara.type);

    if neqcstr
        A = [ones(1,n) -ones(1,n)]; b = 0;
    else
        A = []; b = [];
    end

    Hb = Hb + 1e-10*eye(size(Hb));

    options = optimoptions('quadprog','Display','off','MaxIterations',1000);
    [alpha] = quadprog(Hb, c, [], [], A, b, vlb, vub, x0, options);

    alpha1 = alpha(1:n);
    beta1  = alpha(n+1:2*n);
    beta   = alpha1 - beta1;

    sparsity = 1 - (nnz(beta)/length(beta));

    %% Compute bias
    bias = 0;
    if nobias(kerfPara.type) ~= 0
        if tau > 0.5
            svii = find(abs(beta) > epsilon & abs(beta) < (tau*C - epsilon));
        else
            svii = find(abs(beta) > epsilon & abs(beta) < ((1-tau)*C - epsilon));
        end

        if ~isempty(svii)
            % b = y_i - H_i * beta  averaged over support vectors
            bias = mean(Y(svii) - H(svii,:)*beta);
        else
            fprintf('No SVs found for bias — using midpoint.\n');
            bias = (max(Y) + min(Y)) / 2;
        end
    end

    Htest    = kernelfun(test, kerfPara, X);
    f1       = H*beta + bias;
    PredictY = Htest*beta + bias;
    nsv      = length(find(abs(beta) > epsilon));
end