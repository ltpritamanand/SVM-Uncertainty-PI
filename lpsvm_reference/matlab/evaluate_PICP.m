function [PICP, MPIW] = evaluate_PICP(y, Low_Q, Up_Q)
    PICP = mean(y >= Low_Q & y <= Up_Q);
    MPIW = mean(Up_Q - Low_Q);
end