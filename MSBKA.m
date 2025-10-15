%% ========================================================================
%  Multi-strategy Black-winged Kite Algorithm (MSBKA)
%
%  Developed in MATLAB R2022a
%
%  Author and programmer: zhuoxuanli
%  MSBKA: A multi-strategy black-winged kite algorithm for
%  global optimization and engineering problems
%
%  Cluster Computing
%  DOI:
% ========================================================================
%%
function [Best_Fitness_BKA, Best_Pos_BKA, Convergence_curve] = MSBKA(pop, Max_NFE, lb, ub, dim, fobj, fhd)

%% ----------------Initialize population------------------%
r = rand;
U1 = ones(1, dim);
U2 = rand(1, dim) < 0.5;
useArchive = false;


XPos = initialization(pop, dim, lb, ub);
XFit = zeros(pop,1);
parfor i = 1:pop
    XFit(i) = feval(fhd, XPos(i,:)', fobj);
end

NFE = pop;
Convergence_curve = [];

%% ------------------Initialize Archive--------------------%
ArchiveSize = 5;
Archive_Pos = zeros(ArchiveSize, dim);
Archive_Fit = inf(ArchiveSize, 1);
[~, sorted_indexes] = sort(XFit);
Archive_Pos(1,:) = XPos(sorted_indexes(1),:);
Archive_Fit(1) = XFit(sorted_indexes(1));
ArchiveCount = 1;

%% -------------------Main loop----------------------------%
t = 0;
while NFE < Max_NFE
    t = t + 1;

    [~, sorted_indexes] = sort(XFit);
    XLeader_Pos = XPos(sorted_indexes(1),:);
    XLeader_Fit = XFit(sorted_indexes(1));

    nfe_ratio = NFE / Max_NFE;
    p = exp(-tan((pi/4) * nfe_ratio)^2);

    %% ----------------Attacking behavior--------------------%
    for i = 1:pop
        if NFE >= Max_NFE
            break;
        end

        n = 0.05 * exp(-2 * nfe_ratio^2);
        RL = levy(pop, dim, 1.5);
        b = 1 - (1 ./ (1 + exp(-0.01 * (NFE - Max_NFE / 2))));

        if p > r
            if rand < rand
                XPosNew(i,:) = XPos(i,:) + n .* RL(i,:) .* XPos(i,:);
            else
                XPosNew(i,:) = XPos(i,:) + n .* (1 + sin(r)) .* XPos(i,:);
            end
        else
            if rand > rand
                Opp = lb + rand(1, dim) .* (ub - lb - XPos(i,:));
                alpha = 0.5 * (1 + cos(pi * nfe_ratio));
                XPosNew(i,:) = XPos(i,:) + n .* ((1-alpha) .* (Opp - XPos(i,:)) + alpha .* (XLeader_Pos - XPos(i,:)));
            else
                XPosNew(i,:) = XPos(i,:) .* (n*(2*rand(1,dim)-1) + 1);
            end
        end

        %% -------- Boundary handling --------%
        X = XPosNew(i,:);
        range = ub - lb;
        idx_low = X < lb;
        X(idx_low) = lb(idx_low) + rand(1, sum(idx_low)) .* range(idx_low);
        idx_high = X > ub;
        X(idx_high) = ub(idx_high) - rand(1, sum(idx_high)) .* range(idx_high);
        XPosNew(i,:) = X;

        %% -------- Evaluation and selection --------%
        XFit_New(i) = feval(fhd, XPosNew(i,:)', fobj);
        NFE = NFE + 1;

        if XFit_New(i) < XFit(i)
            XPos(i,:) = XPosNew(i,:);
            XFit(i) = XFit_New(i);
        end

        %% ---------------- Migration behavior ----------------%
        [~, sorted_indexes] = sort(XFit);
        n_temp = length(sorted_indexes);
        twentyPercentIndex = ceil(0.2 * n_temp);
        RandGroupNumber = randperm(twentyPercentIndex, 1);
        RandGroup = sorted_indexes(randperm(length(sorted_indexes), RandGroupNumber));
        if length(RandGroup) > 1
            MeanGroup = mean(XPos(RandGroup, :));
        else
            MeanGroup = XPos(RandGroup, :);
        end

        m = 2 * sin(r + pi/2);
        Avg_XFitness = feval(fhd, MeanGroup', fobj);
        NFE = NFE + 1;

        ori_value = rand(1,dim);
        cauchy_value = tan((ori_value - 0.5) * pi);
        if XFit(i) < Avg_XFitness
            XPosNew(i,:) = XPos(i,:) + cauchy_value(:,dim).* (XPos(i,:) - p*MeanGroup - (1-p)*XLeader_Pos);
        else
            XPosNew(i,:) = XPos(i,:) + cauchy_value(:,dim) .* (U2 .* MeanGroup + (U1-U2) .* XLeader_Pos - m .* XPos(i,:));
        end

        %% -------- Archive-guided adjustment --------%
        if useArchive && ArchiveCount > 1
            idx = randi(ArchiveCount);
            arch_ref = Archive_Pos(idx,:);
            gamma = 0.5;
            guidance = gamma * XLeader_Pos + (1 - gamma) * arch_ref;
            XPos(i,:) = XPos(i,:) + b .* U2 .* (guidance - XPos(i,:));
        end


        %% -------- Boundary handling --------%
        X = XPosNew(i,:);
        range = ub - lb;
        idx_low = X < lb;
        X(idx_low) = lb(idx_low) + rand(1, sum(idx_low)) .* range(idx_low);
        idx_high = X > ub;
        X(idx_high) = ub(idx_high) - rand(1, sum(idx_high)) .* range(idx_high);
        XPosNew(i,:) = X;

        XFit_New(i) = feval(fhd, XPosNew(i,:)', fobj);
        NFE = NFE + 1;

        if XFit_New(i) < XFit(i)
            XPos(i,:) = XPosNew(i,:);
            XFit(i) = XFit_New(i);
        end

        %% ----------------- Archive Update -----------------------%
        dist = sqrt(sum((XPosNew(i,:) - Archive_Pos(1,:)).^2));
        alpha = 1 - (1 ./ (1 + exp(-0.01 * (NFE - Max_NFE / 2))));
        D = rand*alpha*(ub(1,5)-lb(1,5));

        if XFit_New(i) < Archive_Fit(1) && dist > D
            if ArchiveCount < ArchiveSize
                ArchiveCount = ArchiveCount + 1;
                Archive_Pos(ArchiveCount,:) = XPosNew(i,:);
                Archive_Fit(ArchiveCount) = XFit_New(i);
            else
                [max_archive_fit, idx_max] = max(Archive_Fit);
                if XFit_New(i) < max_archive_fit
                    Archive_Pos(idx_max,:) = XPosNew(i,:);
                    Archive_Fit(idx_max) = XFit_New(i);
                end
            end
        end

        if NFE >= Max_NFE
            break;
        end
    end

    %% ---------- Record best fitness -----------------%
    Convergence_curve(t) = min(XFit);
end

[Best_Fitness_BKA, Best_idx] = min(XFit);
Best_Pos_BKA = XPos(Best_idx, :);
end
