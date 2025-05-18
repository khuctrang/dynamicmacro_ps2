%% File Info.

%{

    simulate_remittance.m
    ---------------------
    This code simulates the model with remittances.

%}

%% Simulate Remittance class.

classdef simulate_remittance
    methods(Static)
        %% Simulate the model with remittances. 
        
        function sim = lc(par,sol)            
            %% Set up.
            
            agrid = par.agrid; % Assets today (state variable).
            Gt = par.Gt;     % Age-specific average income
            p_remit = par.p_remit; % Age-specific probability of receiving remittances
            mu_remit = par.mu_remit; % Age-specific average remittance factor

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
            rho = par.rho;     % Income persistence
            sigma_eps = par.sigma_eps; % Standard deviation of income shocks
            
            rgrid = par.rgrid; % Remittance grid
            rmat = par.rmat; % Remittance transition matrix
            eta = par.eta; % Remittance persistence
            sigma_nu = par.sigma_nu; % Std. dev of remittance shocks
            sigma_eps_nu = par.sigma_eps_nu; % Covariance of income and remittance shocks
            
            % Initialize simulation arrays
            ysim = zeros(NN, TT); % Income
            rsim = zeros(NN, TT); % Remittances
            receive_remit = false(NN, TT); % Indicator for receiving remittances
            asim = zeros(NN, TT); % Assets
            csim = zeros(NN, TT); % Consumption
          
            %% Begin simulation.
            
            rng(par.seed);

            % All households start at age 0 with no assets
            asim(:, 1) = 0; % Initial assets (a0 = 0)
            
            % Generate income and remittance shocks accounting for correlation
            % Draw correlated normal random variables
            z1 = randn(NN, TT); % Income shock
            z2 = randn(NN, TT); % Independent component
            
            % Create correlated remittance shock
            corr = sigma_eps_nu / (sigma_eps * sigma_nu);
            z_remit = corr * z1 + sqrt(1 - corr^2) * z2;
            
            % Convert to actual shocks
            eps = sigma_eps * z1;
            nu = sigma_nu * z_remit;
            
            % Draw uniform random variables for remittance receipt
            u_remit = rand(NN, TT);
            
            % Initial income with shock (at age 0)
            ysim(:, 1) = Gt(1) * exp(eps(:, 1));
            
            % Initial remittance status and amount
            for i = 1:NN
                receive_remit(i, 1) = u_remit(i, 1) < p_remit(1);
                if receive_remit(i, 1)
                    rsim(i, 1) = mu_remit(1) * exp(nu(i, 1));
                else
                    rsim(i, 1) = 0;
                end
            end
            
            % Find closest grid points for initial states
            y_idx = zeros(NN, 1);
            r_idx = zeros(NN, 1);
            
            for i = 1:NN
                % Find closest income grid point
                [~, y_idx(i)] = min(abs(ygrid - ysim(i, 1)/Gt(1)));
                
                % Find closest remittance grid point (if receiving)
                if receive_remit(i, 1)
                    normalized_remit = rsim(i, 1) / mu_remit(1);
                    [~, r_idx(i)] = min(abs(rgrid - normalized_remit));
                else
                    r_idx(i) = 1; % Use smallest grid point when not receiving
                end
            end
            
            % Initial consumption based on policy function
            % All households start with a0 = 0, which corresponds to the first grid point
            a_idx = ones(NN, 1);
            
            for i = 1:NN
                % Age 0 corresponds to index 1 in the policy function
                csim(i, 1) = cpol(a_idx(i), 1, y_idx(i), r_idx(i));
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
                    r_prev = rsim(i, t-1);
                    
                    % Find closest asset grid point
                    [~, a_idx(i)] = min(abs(agrid - a_prev));
                    
                    % Calculate next period assets using policy function
                    if age < T
                        asim(i, t) = apol(a_idx(i), age_idx, y_idx(i), r_idx(i));
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
                    
                    % Calculate current period remittances
                    current_p_remit = min(max(p_remit(min(age+1, length(p_remit))), 0), 1);
                    receive_remit(i, t) = u_remit(i, t) < current_p_remit;
                    
                    if receive_remit(i, t)
                        % If received remittances in previous period, apply AR(1) process
                        if receive_remit(i, t-1) && r_prev > 0
                            current_mu = mu_remit(min(age+1, length(mu_remit)));
                            log_r_prev = log(r_prev / mu_remit(min(age, length(mu_remit))));
                            log_r = eta * log_r_prev + nu(i, t);
                            rsim(i, t) = current_mu * exp(log_r);
                        else
                            % Start new remittance stream
                            current_mu = mu_remit(min(age+1, length(mu_remit)));
                            rsim(i, t) = current_mu * exp(nu(i, t));
                        end
                    else
                        rsim(i, t) = 0;
                    end
                    
                    % Update income and remittance grid indices
                    % During retirement, income doesn't vary stochastically
                    if age < tr
                        [~, y_idx(i)] = min(abs(ygrid - ysim(i, t)/Gt(min(age+1, length(Gt)))));
                    end
                    
                    if receive_remit(i, t)
                        normalized_remit = rsim(i, t) / mu_remit(min(age+1, length(mu_remit)));
                        [~, r_idx(i)] = min(abs(rgrid - normalized_remit));
                    else
                        r_idx(i) = 1; % Use smallest grid point when not receiving
                    end
                    
                    % Calculate consumption
                    if age < T
                        csim(i, t) = cpol(a_idx(i), age_idx, y_idx(i), r_idx(i));
                    else
                        % In final period, consume everything
                        csim(i, t) = a_prev + ysim(i, t) + rsim(i, t);
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
            r_profile = zeros(T, 1);
            remit_freq = zeros(T, 1);
            
            % Calculate average by age
            for age = 0:(T-1)
                age_idx = age + 1;
                c_profile(age_idx) = mean(csim(:, age_idx));
                a_profile(age_idx) = mean(asim(:, age_idx));
                y_profile(age_idx) = mean(ysim(:, age_idx));
                r_profile(age_idx) = mean(rsim(:, age_idx));
                remit_freq(age_idx) = mean(receive_remit(:, age_idx));
            end
            
            %% Store results
            sim = struct();
            sim.ysim = ysim;
            sim.rsim = rsim;
            sim.asim = asim;
            sim.csim = csim;
            sim.receive_remit = receive_remit;
            sim.c_profile = c_profile;
            sim.a_profile = a_profile;
            sim.y_profile = y_profile;
            sim.r_profile = r_profile;
            sim.remit_freq = remit_freq;
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
            r_profiles_beta = zeros(par.T, n_beta);
            
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
                    sim_i = simulate_remittance.lc(par_i, sol_collection(idx));
                    
                    % Store profiles
                    c_profiles_beta(:, i) = sim_i.c_profile;
                    a_profiles_beta(:, i) = sim_i.a_profile;
                    r_profiles_beta(:, i) = sim_i.r_profile;
                else
                    fprintf('  Warning: No solution found for beta = %.2f and gamma = %.2f\n', beta_i, fixed_gamma);
                end
            end
            
            %% 2. Varying gamma with fixed beta=0.96
            
            % Storage for profiles
            c_profiles_gamma = zeros(par.T, n_gamma);
            a_profiles_gamma = zeros(par.T, n_gamma);
            r_profiles_gamma = zeros(par.T, n_gamma);
            
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
                    sim_i = simulate_remittance.lc(par_i, sol_collection(idx));
                    
                    % Store profiles
                    c_profiles_gamma(:, i) = sim_i.c_profile;
                    a_profiles_gamma(:, i) = sim_i.a_profile;
                    r_profiles_gamma(:, i) = sim_i.r_profile;
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
                        sim_ij = simulate_remittance.lc(par_ij, sol_collection(idx));
                        
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
            results.r_profiles_beta = r_profiles_beta;
            results.c_profiles_gamma = c_profiles_gamma;
            results.a_profiles_gamma = a_profiles_gamma;
            results.r_profiles_gamma = r_profiles_gamma;
            results.avg_wealth = avg_wealth;
            results.beta_values = beta_values;
            results.gamma_values = gamma_values;
        end
        
        %% Compare baseline model with remittance model
        function comparison = compare_with_baseline(par_base, sol_base, par_remit, sol_remit)
            % This function compares the baseline model with the remittance model
            
            fprintf('Comparing baseline model with remittance model...\n');
            
            % Simulate both models
            sim_base = simulate.lc(par_base, sol_base);
            sim_remit = simulate_remittance.lc(par_remit, sol_remit);
            
            % Store results for comparison
            comparison = struct();
            comparison.ages = sim_base.ages;
            
            % Consumption
            comparison.c_base = sim_base.c_profile;
            comparison.c_remit = sim_remit.c_profile;
            comparison.c_diff = sim_remit.c_profile - sim_base.c_profile;
            comparison.c_pct_diff = 100 * (sim_remit.c_profile ./ sim_base.c_profile - 1);
            
            % Assets
            comparison.a_base = sim_base.a_profile;
            comparison.a_remit = sim_remit.a_profile;
            comparison.a_diff = sim_remit.a_profile - sim_base.a_profile;
            comparison.a_pct_diff = 100 * (sim_remit.a_profile ./ max(sim_base.a_profile, 1e-10) - 1);
            
            % Income
            comparison.y_base = sim_base.y_profile;
            comparison.y_remit = sim_remit.y_profile;
            comparison.y_total_remit = sim_remit.y_profile + sim_remit.r_profile;
            
            % Remittances
            comparison.r_profile = sim_remit.r_profile;
            comparison.remit_freq = sim_remit.remit_freq;
            
            % Summary statistics
            comparison.avg_c_base = mean(sim_base.c_profile);
            comparison.avg_c_remit = mean(sim_remit.c_profile);
            comparison.avg_a_base = mean(sim_base.a_profile);
            comparison.avg_a_remit = mean(sim_remit.a_profile);
            
            % Consumption smoothing measure (standard deviation of consumption over life cycle)
            comparison.c_sd_base = std(sim_base.c_profile);
            comparison.c_sd_remit = std(sim_remit.c_profile);
            
            % Consumption-to-income ratio
            comparison.c_y_ratio_base = sim_base.c_profile ./ max(sim_base.y_profile, 1e-10);
            comparison.c_y_ratio_remit = sim_remit.c_profile ./ max(sim_remit.y_profile + sim_remit.r_profile, 1e-10);
            
            fprintf('Comparison completed.\n');
        end
    end
end