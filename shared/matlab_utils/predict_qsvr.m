function PredictY = predict_qsvr(Xtest, model)
% PREDICT_QSVR  Score a fitted quantile SVR on new points.
%
%   PredictY = predict_qsvr(Xtest, model)
%
%   Runs pure inference: builds the m-by-n test kernel and applies
%   PredictY = Htest * beta + bias. No QP is solved.
%
%   Inputs
%     Xtest : m-by-d test features
%     model : struct from train_qsvr with fields
%             beta, bias, X_train, kerfPara
%
%   Output
%     PredictY : m-by-1 predictions

    % Test kernel — same call pattern as inside epsilon_quantilesvr2:
    %   Htest(i,j) = K(Xtest_i, X_train_j)   →   m-by-n
    Htest = kernelfun(Xtest, model.kerfPara, model.X_train);

    PredictY = Htest * model.beta + model.bias;
end
