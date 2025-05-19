%% File Info.

%{

    simulate.m
    ----------
    This code simulates the model.

%}

%% Simulate class.

classdef simulate
    methods(Static)
        %% Simulate the model.
          function sim = firm_dynamics(par, sol) 
            kgrid = par.kgrid; % Capital today (state variable).
            Agrid = par.Agrid; % Productivity (state variable).
            pgrid = par.pgrid;    % Price grid
            v = sol.v;            % Value function
            k_pol = sol.k;        % Capital policy
            i_pol = sol.i;        % Investment policy
            r_pol = sol.r;        % Revenue 
            l_pol = sol.l;        % Labor
            e_pol = sol.e;        % Expenditure
            p_pol = sol.p;        % Profit
            
            T = par.T;            % Simulation periods
            burn_in = par.burn_in;  % Burn-in periods
            
            % Initialize simulation arrays
            A_sim = zeros(T, 1);  % Productivity
            p_sim = zeros(T, 1);  % Price
            k_sim = zeros(T, 1);  % Capital
            i_sim = zeros(T, 1);  % Investment
            r_sim = zeros(T, 1);  % Revenue
            l_sim = zeros(T, 1);  % Labor
            e_sim = zeros(T, 1);  % Expenditure
            pi_sim = zeros(T, 1); % Profit
            v_sim = zeros(T, 1);  % Value
            
            %% Begin simulation
            
            rng(par.seed);  % Set random seed
            
            % Initial conditions
            k_idx = round(par.klen/2);    % Start with middle capital
            A_idx = round(par.Alen/2);    % Start with middle productivity
            p_idx = round(par.plen/2);    % Start with middle price
            
            % First period values
            k_sim(1) = kgrid(k_idx);
            A_sim(1) = Agrid(A_idx);
            p_sim(1) = pgrid(p_idx);
            i_sim(1) = i_pol(k_idx, A_idx, p_idx);
            r_sim(1) = r_pol(k_idx, A_idx, p_idx);
            l_sim(1) = l_pol(k_idx, A_idx, p_idx);
            e_sim(1) = e_pol(k_idx, A_idx, p_idx);
            pi_sim(1) = p_pol(k_idx, A_idx, p_idx);
            v_sim(1) = v(k_idx, A_idx, p_idx);
            
            %% Main simulation loop
            for t = 2:T
                % Find next period capital (from previous period decision)
                k_next = k_pol(k_idx, A_idx, p_idx);
                k_sim(t) = k_next;
                
                % Find index of current capital 
                [~, k_idx] = min(abs(kgrid - k_next));
                
                % Draw shocks for current period
                rand_A = rand();
                rand_p = rand();
                
                % Get new productivity and price state
                cum_prob_A = cumsum(par.pmat_A(A_idx, :));
                cum_prob_p = cumsum(par.pmat_p(p_idx, :));
                
                A_idx = find(cum_prob_A >= rand_A, 1);
                p_idx = find(cum_prob_p >= rand_p, 1);
                
                if isempty(A_idx), A_idx = par.Alen; end
                if isempty(p_idx), p_idx = par.plen; end
                
                % Record current state
                A_sim(t) = Agrid(A_idx);
                p_sim(t) = pgrid(p_idx);
                
                % Record optimal decisions
                i_sim(t) = i_pol(k_idx, A_idx, p_idx);
                r_sim(t) = r_pol(k_idx, A_idx, p_idx);
                l_sim(t) = l_pol(k_idx, A_idx, p_idx);
                e_sim(t) = e_pol(k_idx, A_idx, p_idx);
                pi_sim(t) = p_pol(k_idx, A_idx, p_idx);
                v_sim(t) = v(k_idx, A_idx, p_idx);
            end
            
            %% Prepare output (skip burn-in periods)
            sim = struct();
            sim.A = A_sim(burn_in+1:end);
            sim.p = p_sim(burn_in+1:end);
            sim.k = k_sim(burn_in+1:end);
            sim.i = i_sim(burn_in+1:end);
            sim.r = r_sim(burn_in+1:end);
            sim.l = l_sim(burn_in+1:end);
            sim.e = e_sim(burn_in+1:end);
            sim.pi = pi_sim(burn_in+1:end);
            sim.v = v_sim(burn_in+1:end);
        end
        
        %% Perform parameter analysis
        function results = parameter_analysis(firm_type)
            % Different delta and gamma values
            delta_values = [0.05, 0.06, 0.07, 0.08];
            gamma_values = [0.10, 0.15, 0.20, 0.25];
            
            % Storage for results
            avg_k = zeros(length(gamma_values), length(delta_values));
            avg_i = zeros(length(gamma_values), length(delta_values));
            
            fprintf('\nPerforming parameter analysis for %s firms...\n', firm_type);
            
            % Loop over parameter combinations
            for g = 1:length(gamma_values)
                for d = 1:length(delta_values)
                    fprintf('  Testing gamma = %.2f, delta = %.2f\n', gamma_values(g), delta_values(d));
                    
                    % Setup model with current parameters
                    par = model.setup(firm_type);
                    par.delta = delta_values(d);
                    par.gamma = gamma_values(g);
                    par = model.gen_grids(par);
                    
                    % Solve model
                    sol = solve.firm_problem(par);
                    
                    % Simulate model
                    sim = simulate.firm_dynamics(par, sol);
                    
                    % Compute averages
                    avg_k(g, d) = mean(sim.k);
                    avg_i(g, d) = mean(sim.i);
                end
            end
            
            % Store results
            results = struct();
            results.avg_k = avg_k;
            results.avg_i = avg_i;
            results.delta_values = delta_values;
            results.gamma_values = gamma_values;
            
            fprintf('Parameter analysis complete for %s firms.\n', firm_type);
        end
    end
end