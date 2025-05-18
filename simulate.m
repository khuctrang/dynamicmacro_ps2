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
        
        function sim = lc(par,sol)            
            %% Set up.
            
            agrid = par.agrid; % Assets today (state variable).
            Gt = par.Gt;         % Age-specific average income

            apol = sol.a; % Policy function for capital.
            cpol = sol.c; % Policy function for consumption.

            TT = par.TT; % Time periods.
            NN = par.NN; % People.
            T = par.T; % Life span.
            tr = par.tr; % Retirement.
            r = par.r; % Interest rate.

            kappa = par.kappa; % Share of income as pension.
            ygrid = par.ygrid; % Exogenous income.
            pmat = par.pmat; % Transition matrix.            
            rho = par.rho;       % Income persistence
            sigma_eps = par.sigma_eps; % Standard deviation of income shocks
            
            % Initialize simulation arrays correctly
            ysim = zeros(NN, TT); % Income
            asim = zeros(NN, TT); % Assets
            csim = zeros(NN, TT); % Consumption
          
            %% Begin simulation.
            
            rng(par.seed);

          % All households start at age 0 with no assets
            asim(:, 1) = 0; % Initial assets (a0 = 0)
            
            % Generate income shocks for all periods
            eps = sigma_eps * randn(NN, TT);
            
            % Initial income with shock (at age 0)
            ysim(:, 1) = Gt(1) * exp(eps(:, 1));
            
            % Find closest income grid point for each household
            y_idx = zeros(NN, 1);
            for i = 1:NN
                [~, y_idx(i)] = min(abs(par.ygrid - ysim(i, 1)/Gt(1)));
            end
            
            % Initial consumption based on policy function
            % All households start with a0 = 0, which corresponds to the first grid point
            a_idx = ones(NN, 1);
            
            for i = 1:NN
                % Age 0 corresponds to index 1 in the policy function
                csim(i, 1) = cpol(a_idx(i), 1, y_idx(i));
            end
            
            %% Simulate forward
            for t = 2:TT
                % Current age (0-indexed in problem, 1-indexed in code)
                age = t - 1;
                age_idx = t;
                
                for i = 1:NN
                    % Previous period assets and income
                    a_prev = asim(i, t-1);
                    y_prev = ysim(i, t-1);
                    
                    % Find closest asset grid point
                    [~, a_idx(i)] = min(abs(agrid - a_prev));
                    
                    % Calculate next period assets using policy function
                    if age < T
                        asim(i, t) = apol(a_idx(i), age_idx, y_idx(i));
                    else
                        asim(i, t) = 0; % No savings in final period
                    end
                    
                    % Calculate current period income
                    if age < tr
                        % Working age: income follows AR(1) process with age-specific profile
                        if age < length(Gt)
                            % Log income follows AR(1) around age-specific mean
                            log_dev_prev = log(y_prev / Gt(age));
                            log_dev = rho * log_dev_prev + eps(i, t);
                            ysim(i, t) = Gt(age+1) * exp(log_dev);
                        else
                            % If age exceeds Gt data, use last available value
                            log_dev_prev = log(y_prev / Gt(end));
                            log_dev = rho * log_dev_prev + eps(i, t);
                            ysim(i, t) = Gt(end) * exp(log_dev);
                        end
                    else
                        % Retired: pension is a fraction of last working period income
                        if age == tr
                            % Just retired: calculate pension
                            ysim(i, t) = kappa * y_prev;
                        else
                            % Already retired: maintain pension
                            ysim(i, t) = ysim(i, t-1);
                        end
                    end
                    
                    % Update income grid index
                    % During retirement, income doesn't vary stochastically
                    if age < tr
                        [~, y_idx(i)] = min(abs(par.ygrid - ysim(i, t)/Gt(min(age+1, length(Gt)))));
                    end
                    
                    % Calculate consumption
                    if age < T
                        csim(i, t) = cpol(a_idx(i), age_idx, y_idx(i));
                    else
                        % In final period, consume everything
                        csim(i, t) = a_prev + ysim(i, t);
                    end
                end
            end
            
            %% Calculate life-cycle profiles
            
            % Age indicators (0 to T-1)
            ages = 0:(T-1);
            
            % Initialize arrays for age profiles
            c_profile = zeros(T, 1);
            a_profile = zeros(T, 1);
            y_profile = zeros(T, 1);
            
            % Calculate average by age
            for age = 0:(T-1)
                c_profile(age+1) = mean(csim(:, age+1));
                a_profile(age+1) = mean(asim(:, age+1));
                y_profile(age+1) = mean(ysim(:, age+1));
            end
            
            %% Store results
            sim = struct();
            sim.ysim = ysim;
            sim.asim = asim;
            sim.csim = csim;
            sim.c_profile = c_profile;
            sim.a_profile = a_profile;
            sim.y_profile = y_profile;
            sim.ages = ages;
        end
        
        %% Simulate with parameter variations
        function results = param_variations(par, sol_collection)
            % This function simulates the model with different beta and gamma values
            
            %% Define parameter values
            beta_values = [0.90, 0.92, 0.94, 0.96];
            gamma_values = [2.00, 3.00, 4.00, 5.00];
            
            n_beta = length(beta_values);
            n_gamma = length(gamma_values);
            
            %% 1. Varying beta with fixed gamma=2.00
            
            % Storage for profiles
            c_profiles_beta = zeros(par.T, n_beta);
            a_profiles_beta = zeros(par.T, n_beta);
            
            % Fixed gamma
            fixed_gamma = 2.00;
            
            fprintf('Simulating for different beta values with gamma = %.2f...\n', fixed_gamma);
            
            for i = 1:n_beta
                beta_i = beta_values(i);
                
                % Find the solution with this beta and fixed gamma
                idx = find([sol_collection.beta] == beta_i & [sol_collection.gamma] == fixed_gamma);
                
                if ~isempty(idx)
                    fprintf('  Simulating for beta = %.2f\n', beta_i);
                    
                    % Set parameters
                    par_i = par;
                    par_i.beta = beta_i;
                    
                    % Simulate
                    sim_i = simulate.lc(par_i, sol_collection(idx));
                    
                    % Store profiles
                    c_profiles_beta(:, i) = sim_i.c_profile;
                    a_profiles_beta(:, i) = sim_i.a_profile;
                else
                    fprintf('  Warning: No solution found for beta = %.2f and gamma = %.2f\n', beta_i, fixed_gamma);
                end
            end
            
            %% 2. Varying gamma with fixed beta=0.96
            
            % Storage for profiles
            c_profiles_gamma = zeros(par.T, n_gamma);
            a_profiles_gamma = zeros(par.T, n_gamma);
            
            % Fixed beta
            fixed_beta = 0.96;
            
            fprintf('Simulating for different gamma values with beta = %.2f...\n', fixed_beta);
            
            for i = 1:n_gamma
                gamma_i = gamma_values(i);
                
                % Find the solution with fixed beta and this gamma
                idx = find([sol_collection.beta] == fixed_beta & [sol_collection.gamma] == gamma_i);
                
                if ~isempty(idx)
                    fprintf('  Simulating for gamma = %.2f\n', gamma_i);
                    
                    % Set parameters
                    par_i = par;
                    par_i.sigma = gamma_i; % Note: gamma is called sigma in the model
                    
                    % Simulate
                    sim_i = simulate.lc(par_i, sol_collection(idx));
                    
                    % Store profiles
                    c_profiles_gamma(:, i) = sim_i.c_profile;
                    a_profiles_gamma(:, i) = sim_i.a_profile;
                else
                    fprintf('  Warning: No solution found for beta = %.2f and gamma = %.2f\n', fixed_beta, gamma_i);
                end
            end
            
            %% 3. Create data for heat map of average wealth
            
            % Initialize heat map data
            avg_wealth = zeros(n_beta, n_gamma);
            
            fprintf('Creating data for average wealth heat map...\n');
            
            for i = 1:n_beta
                for j = 1:n_gamma
                    beta_ij = beta_values(i);
                    gamma_ij = gamma_values(j);
                    
                    % Find the solution with this beta and gamma
                    idx = find([sol_collection.beta] == beta_ij & [sol_collection.gamma] == gamma_ij);
                    
                    if ~isempty(idx)
                        % Set parameters
                        par_ij = par;
                        par_ij.beta = beta_ij;
                        par_ij.sigma = gamma_ij;
                        
                        % Simulate
                        sim_ij = simulate.lc(par_ij, sol_collection(idx));
                        
                        % Store average wealth
                        avg_wealth(i, j) = mean(sim_ij.a_profile);
                    else
                        fprintf('  Warning: No solution found for beta = %.2f and gamma = %.2f\n', beta_ij, gamma_ij);
                    end
                end
            end
            
            %% Return results
            results = struct();
            results.c_profiles_beta = c_profiles_beta;
            results.a_profiles_beta = a_profiles_beta;
            results.c_profiles_gamma = c_profiles_gamma;
            results.a_profiles_gamma = a_profiles_gamma;
            results.avg_wealth = avg_wealth;
            results.beta_values = beta_values;
            results.gamma_values = gamma_values;
        end
    end
end