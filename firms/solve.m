%% File Info.

%{

    solve.m
    -------
    This code solves the model.

%}

%% Solve class.

classdef solve
    methods(Static)
        %% Solve the model using VFI. 
        
        function sol = firm_problem(par)            
            %% Structure array for model solution.
            
            sol = struct();
            
            %% Model parameters, grids and functions.
            
            beta = par.beta; % Discount factor.
            delta = par.delta; % Depreciation rate.
            gamma = par.gamma;
            klen = par.klen;  % Grid size for k.
            kgrid = par.kgrid; % Grid for k (state and choice).

            Alen = par.Alen; % Grid size for A.
            Agrid = par.Agrid; % Grid for A.
            pmat_A = par.pmat_A; % Transition matrix for A.
            plen = par.plen;       % Grid size for p
            pgrid = par.pgrid;     % Grid for p
            pmat_p = par.pmat_p;   % Transition matrix for p

            %% Value Function iteration.

  
            v0 = zeros(klen, Alen, plen);  % Initial guess of value function
     
            v1 = zeros(klen, Alen, plen);  % Container for value function
            k1 = zeros(klen, Alen, plen);  % Container for next period capital
            i1 = zeros(klen, Alen, plen);  % Container for investment
            r1 = zeros(klen, Alen, plen);  % Container for revenue
            l1 = zeros(klen, Alen, plen);  % Container for labor
            e1 = zeros(klen, Alen, plen);  % Container for expenditure
            p1 = zeros(klen, Alen, plen);  % Container for profit
            
            crit = 1e-6;
            maxiter = 1000;
            diff = 1;
            iter = 0;
            
            fprintf('------------Beginning Value Function Iteration.------------\n\n')
            
            while diff > crit && iter < maxiter % Iterate on the Bellman Equation until convergence.
                for k_idx = 1:klen      % Loop over current capital
                    for a_idx = 1:Alen  % Loop over productivity states
                        for p_idx = 1:plen % Loop over price states
                            
                            k = kgrid(k_idx);             % Current capital
                            A = Agrid(a_idx);             % Current productivity
                            p = pgrid(p_idx);             % Current price
                            
                            [rev, lab] = model.production(A, k, par); % Revenue and optimal variable input
                            
                            max_val = -1e10;
                            max_k_next = k;
                            max_inv = 0;
                            max_exp = 0;
                            max_prof = -1e10;
                            max_lab = lab;
                        
                            
                            % Solve the maximization problem
                            for k_next_idx = 1:klen
                                k_next = kgrid(k_next_idx);  % Next period capital
                                
                                [exp, inv] = model.total_cost(k, k_next, p, par); % Investment and expenditure
                                 
                                profit = rev - par.w * lab - exp; % Current period profit
                                
                                
                                % Expected value next period
                                ev = 0;
                                for a_next_idx = 1:Alen
                                    for p_next_idx = 1:plen
                                        ev = ev + pmat_A(a_idx, a_next_idx) * pmat_p(p_idx, p_next_idx) * ...
                                             v0(k_next_idx, a_next_idx, p_next_idx);
                                    end
                                end
                                
                                % Current value
                                value = profit + beta * ev;
                                
                                if value > max_val
                                    max_val = value;
                                    max_k_next = k_next;
                                    max_inv = inv;
                                    max_exp = exp;
                                    max_prof = profit;
                                  
                                end
                            end
                            
                            % Store optimal values
                            v1(k_idx, a_idx, p_idx) = max_val;      % Value function
                            k1(k_idx, a_idx, p_idx) = max_k_next;   % Capital policy
                            i1(k_idx, a_idx, p_idx) = max_inv;      % Investment
                            r1(k_idx, a_idx, p_idx) = rev;          % Revenue
                            l1(k_idx, a_idx, p_idx) = lab;          % Labor
                            e1(k_idx, a_idx, p_idx) = max_exp;      % Expenditure
                            p1(k_idx, a_idx, p_idx) = max_prof;     % Profit
                        end
                    end
                end
                
                diff = max(abs(v1(:) - v0(:)));  % Check for convergence
                v0 = v1;                         % Update guess
                
                iter = iter + 1;                 % Update counter
                
                % Print progress
                if mod(iter, 10) == 0
                    fprintf('Iteration: %d, Diff: %f\n', iter, diff)
                end
            end
            
            fprintf('\nConverged in %d iterations.\n\n', iter)
            fprintf('------------End of Value Function Iteration.------------\n')
            
            %% Save solution
            sol.v = v1;  % Value function
            sol.k = k1;  % Capital policy function
            sol.i = i1;  % Investment policy function
            sol.r = r1;  % Revenue function
            sol.l = l1;  % Labor functions, materials for large)
            sol.e = e1;  % Expenditure function
            sol.p = p1;  % Profit function
        
        end
    end
end