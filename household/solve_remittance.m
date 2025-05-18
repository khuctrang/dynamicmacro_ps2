%% File Info.

%{

    solve_remittance.m
    ------------------
    This code solves the extended model with remittances.

%}

%% Solve Remittance class.

classdef solve_remittance
    methods(Static)
        %% Solve the model using BI. 
        
        function sol = lc(par)            
            %% Structure array for model solution.
            
            sol = struct();
            
            %% Model parameters, grids and functions.
            
            T = par.T; % Last period of life.
            tr = par.tr; % First year of retirement.
            Gt = par.Gt;     % Age-specific average income profile
            p_remit = par.p_remit;  % Age-specific probability of receiving remittances
            mu_remit = par.mu_remit; % Age-specific average remittance amount

            beta = par.beta; % Discount factor.
            gamma = par.sigma; % Risk aversion parameter

            alen = par.alen; % Grid size for a.
            agrid = par.agrid; % Grid for a (state and choice).

            ylen = par.ylen; % Grid size for y.
            ygrid = par.ygrid; % Grid for y.
            pmat = par.pmat; % Transition matrix for y.
            
            rlen = par.rlen; % Grid size for remittance.
            rgrid = par.rgrid; % Grid for remittance.
            rmat = par.rmat; % Transition matrix for remittance.
            
            joint_trans = par.joint_trans; % Joint transition matrix for y and r

            r = par.r; % Real interest rate.
            kappa = par.kappa; % Share of income as pension.

            %% Backward induction.
            
            v1 = nan(alen, T, ylen, rlen); % Container for V.
            a1 = nan(alen, T, ylen, rlen); % Container for a'.
            c1 = nan(alen, T, ylen, rlen); % Container for c'.

   
            fprintf('------------Solving the Vietnamese Household Life Cycle Model with Remittances------------\n\n')
            
            for age = T:-1:1 % Start in the last period (T) and iterate backward
                
                if age == T % Last period of life (age T-1 in problem notation)
                    % In the last period, consume everything (aT = 0)
                    for y_idx = 1:ylen
                        for r_idx = 1:rlen
                            % Determine income based on whether retired
                            if age-1 >= tr % Already retired
                                y_income = kappa * Gt(tr) * ygrid(y_idx); % Pension
                            else
                                y_income = Gt(age) * ygrid(y_idx); % Still working
                            end
                            
                            % Determine remittance income
                            prob_remit = p_remit(age);
                            remit_income = prob_remit * mu_remit(age) * rgrid(r_idx);
                            
                            total_income = y_income + remit_income;
                            
                            c1(:, T, y_idx, r_idx) = agrid + total_income; % Consume all assets plus income
                            a1(:, T, y_idx, r_idx) = 0.0; % No savings in terminal period
                            v1(:, T, y_idx, r_idx) = model_remittance.utility(c1(:, T, y_idx, r_idx), par); % Terminal value function
                        end
                    end
                else % All other periods
                    % Current age for the problem definition (0-indexed)
                    current_age = age - 1;
                    
                    for y_idx = 1:ylen % Loop over income states
                        for r_idx = 1:rlen % Loop over remittance states
                            % Current income based on retirement status
                            if current_age >= tr
                                % Retired
                                y_income = kappa * Gt(tr) * ygrid(y_idx);
                            else
                                % Working
                                y_income = Gt(age) * ygrid(y_idx);
                            end
                            
                            % Current remittance income
                            prob_remit = p_remit(age);
                            remit_income = prob_remit * mu_remit(age) * rgrid(r_idx);
                            
                            total_income = y_income + remit_income;
                            
                            % Calculate expected future value
                            ev = zeros(alen, 1);
                            for a_idx = 1:alen
                                ev_sum = 0;
                                
                                % If retired, no income uncertainty but still remittance uncertainty
                                if current_age >= tr
                                    for next_r_idx = 1:rlen
                                        % Fixed income state but stochastic remittances
                                        next_y_idx = y_idx; % Keep same income state
                                        ev_sum = ev_sum + rmat(r_idx, next_r_idx) * v1(a_idx, age+1, next_y_idx, next_r_idx);
                                    end
                                else
                                    % Working: both income and remittance are stochastic
                                    for next_y_idx = 1:ylen
                                        for next_r_idx = 1:rlen
                                            % Joint transition probability
                                            joint_prob = joint_trans(y_idx, r_idx, next_y_idx, next_r_idx);
                                            ev_sum = ev_sum + joint_prob * v1(a_idx, age+1, next_y_idx, next_r_idx);
                                        end
                                    end
                                end
                                
                                ev(a_idx) = ev_sum;
                            end
                            
                            for a_idx = 1:alen % Loop over asset states
                                % Cash-on-hand
                                coh = agrid(a_idx) + total_income;
                                
                                % For each possible next-period asset choice
                                vals = zeros(alen, 1);
                                for ap_idx = 1:alen
                                    % Consumption implied by asset choice
                                    c = coh - agrid(ap_idx)/(1+r);
                                    
                                    if c > 0
                                        % Utility plus discounted expected future value
                                        vals(ap_idx) = model_remittance.utility(c, par) + beta * ev(ap_idx);
                                    else
                                        vals(ap_idx) = -1e10; % Large negative value for infeasible choices
                                    end
                                end
                                
                                % Find the optimal choice
                                [vmax, amax_idx] = max(vals);
                                
                                % Store the results
                                v1(a_idx, age, y_idx, r_idx) = vmax;
                                a1(a_idx, age, y_idx, r_idx) = agrid(amax_idx);
                                c1(a_idx, age, y_idx, r_idx) = coh - a1(a_idx, age, y_idx, r_idx)/(1+r);
                            end
                        end
                    end
                end

                % Print progress
                if mod(age, 10) == 0 || age == T || age == 1
                    fprintf('Solving for age: %d\n', age-1) % Convert to 0-indexed age for consistency
                end
            end
            
            fprintf('------------Life Cycle Problem with Remittances Solved------------\n')
            
            %% Store solution
            sol.c = c1; % Consumption policy function
            sol.a = a1; % Saving policy function
            sol.v = v1; % Value function
            sol.beta = par.beta; % Store beta used for this solution
            sol.gamma = par.sigma; % Store gamma used for this solution
        end
        
        %% Solve the model for multiple parameter combinations
        function sol_collection = lc_param_variations(par)
            % This function solves the model for different combinations of beta and gamma
            
            % Beta and gamma values to test
            beta_values = [0.90, 0.92, 0.94, 0.96];
            gamma_values = [2.00, 3.00, 4.00, 5.00];
            
            % Number of combinations
            n_beta = length(beta_values);
            n_gamma = length(gamma_values);
            total_combinations = n_beta * n_gamma;
            
            % Initialize storage for solutions
            sol_collection = struct('beta', {}, 'gamma', {}, 'c', {}, 'a', {}, 'v', {});
            
            % Track progress
            fprintf('Solving model with remittances for %d parameter combinations...\n', total_combinations);
            
            % Loop over all combinations
            counter = 0;
            for i_beta = 1:n_beta
                for i_gamma = 1:n_gamma
                    counter = counter + 1;
                    
                    % Set parameters for this run
                    par_i = par;
                    par_i.beta = beta_values(i_beta);
                    par_i.sigma = gamma_values(i_gamma); % Note: gamma is called sigma in the model
                    
                    % Print progress
                    fprintf('Solving combination %d/%d: beta = %.2f, gamma = %.2f\n', ...
                        counter, total_combinations, par_i.beta, par_i.sigma);
                    
                    % Solve the model with remittances
                    sol_i = solve_remittance.lc(par_i);
                    
                    % Store the solution
                    sol_collection(counter).beta = par_i.beta;
                    sol_collection(counter).gamma = par_i.sigma;
                    sol_collection(counter).c = sol_i.c;
                    sol_collection(counter).a = sol_i.a;
                    sol_collection(counter).v = sol_i.v;
                end
            end
            
            fprintf('All parameter combinations solved for remittance model.\n');
        end
        
        %% Sensitivity analysis for remittance parameters
        function results = remittance_sensitivity(par, base_sol)
            % This function analyzes sensitivity to remittance parameters
            
            % Parameters to vary
            eta_values = [0.5, 0.7, 0.9];            % Persistence of remittance process
            corr_values = [-0.5, -0.3, -0.1, 0.1];   % Correlation between income and remittance shocks
            prob_scale = [0.5, 1.0, 1.5];            % Scaling factor for remittance probability
            amount_scale = [0.5, 1.0, 1.5];          % Scaling factor for remittance amount
            
            % Initialize storage for results
            results = struct();
            results.eta_values = eta_values;
            results.corr_values = corr_values;
            results.prob_scale = prob_scale;
            results.amount_scale = amount_scale;
            
            % Base parameter values
            base_eta = par.eta;
            base_corr = par.corr_eps_nu;
            base_p_remit = par.p_remit;
            base_mu_remit = par.mu_remit;
            
            % 1. Sensitivity to eta (persistence of remittance process)
            fprintf('Analyzing sensitivity to remittance persistence (eta)...\n');
            
            eta_c_profiles = zeros(par.T, length(eta_values));
            eta_a_profiles = zeros(par.T, length(eta_values));
            
            for i = 1:length(eta_values)
                eta_i = eta_values(i);
                fprintf('  Simulating for eta = %.2f\n', eta_i);
                
                % Create new parameter structure with modified eta
                par_i = par;
                par_i.eta = eta_i;
                
                % Regenerate grids with new eta
                par_i = model_remittance.gen_grids(par_i);
                
                % Solve and simulate
                sol_i = solve_remittance.lc(par_i);
                sim_i = simulate_remittance.lc(par_i, sol_i);
                
                % Store profiles
                eta_c_profiles(:, i) = sim_i.c_profile;
                eta_a_profiles(:, i) = sim_i.a_profile;
            end
            
            results.eta_c_profiles = eta_c_profiles;
            results.eta_a_profiles = eta_a_profiles;
            
            % 2. Sensitivity to correlation between income and remittance shocks
            fprintf('Analyzing sensitivity to income-remittance correlation...\n');
            
            corr_c_profiles = zeros(par.T, length(corr_values));
            corr_a_profiles = zeros(par.T, length(corr_values));
            
            for i = 1:length(corr_values)
                corr_i = corr_values(i);
                fprintf('  Simulating for correlation = %.2f\n', corr_i);
                
                % Create new parameter structure with modified correlation
                par_i = par;
                par_i.corr_eps_nu = corr_i;
                par_i.sigma_eps_nu = corr_i * par.sigma_eps * par.sigma_nu;
                
                % Regenerate grids with new correlation
                par_i = model_remittance.gen_grids(par_i);
                
                % Solve and simulate
                sol_i = solve_remittance.lc(par_i);
                sim_i = simulate_remittance.lc(par_i, sol_i);
                
                % Store profiles
                corr_c_profiles(:, i) = sim_i.c_profile;
                corr_a_profiles(:, i) = sim_i.a_profile;
            end
            
            results.corr_c_profiles = corr_c_profiles;
            results.corr_a_profiles = corr_a_profiles;
            
            % 3. Sensitivity to probability of receiving remittances
            fprintf('Analyzing sensitivity to remittance probability...\n');
            
            prob_c_profiles = zeros(par.T, length(prob_scale));
            prob_a_profiles = zeros(par.T, length(prob_scale));
            
            for i = 1:length(prob_scale)
                scale_i = prob_scale(i);
                fprintf('  Simulating for probability scale = %.2f\n', scale_i);
                
                % Create new parameter structure with scaled probabilities
                par_i = par;
                par_i.p_remit = min(base_p_remit * scale_i, 1); % Cap at 1
                
                % Solve and simulate
                sol_i = solve_remittance.lc(par_i);
                sim_i = simulate_remittance.lc(par_i, sol_i);
                
                % Store profiles
                prob_c_profiles(:, i) = sim_i.c_profile;
                prob_a_profiles(:, i) = sim_i.a_profile;
            end
            
            results.prob_c_profiles = prob_c_profiles;
            results.prob_a_profiles = prob_a_profiles;
            
            % 4. Sensitivity to remittance amount
            fprintf('Analyzing sensitivity to remittance amount...\n');
            
            amount_c_profiles = zeros(par.T, length(amount_scale));
            amount_a_profiles = zeros(par.T, length(amount_scale));
            
            for i = 1:length(amount_scale)
                scale_i = amount_scale(i);
                fprintf('  Simulating for amount scale = %.2f\n', scale_i);
                
                % Create new parameter structure with scaled amounts
                par_i = par;
                par_i.mu_remit = base_mu_remit * scale_i;
                
                % Solve and simulate
                sol_i = solve_remittance.lc(par_i);
                sim_i = simulate_remittance.lc(par_i, sol_i);
                
                % Store profiles
                amount_c_profiles(:, i) = sim_i.c_profile;
                amount_a_profiles(:, i) = sim_i.a_profile;
            end
            
            results.amount_c_profiles = amount_c_profiles;
            results.amount_a_profiles = amount_a_profiles;
            
            fprintf('Sensitivity analysis completed.\n');
        end
    end
end