%% File Info.

%{

    solve.m
    -------
    This code solves the model.

%}

%% Solve class.

classdef solve
    methods(Static)
        %% Solve the model using BI. 
        
        function sol = lc(par)            
            %% Structure array for model solution.
            
            sol = struct();
            
            %% Model parameters, grids and functions.
            
            T = par.T; % Last period of life.
            tr = par.tr; % First year of retirement.
            Gt = par.Gt;     % Age-specific average income profile

            beta = par.beta; % Discount factor.
            gamma = par.sigma; % Risk aversion parameter

            alen = par.alen; % Grid size for a.
            agrid = par.agrid; % Grid for a (state and choice).

            ylen = par.ylen; % Grid size for y.
            ygrid = par.ygrid; % Grid for y.
            pmat = par.pmat; % Transition matrix for y.

            r = par.r; % Real interest rate.
            kappa = par.kappa; % Share of income as pension.

            %% Backward induction.
            
            v1 = nan(alen,T,ylen); % Container for V.
            a1 = nan(alen,T,ylen); % Container for a'.
            c1 = nan(alen,T,ylen); % Container for c'.

   
            fprintf('------------Solving the Vietnamese Household Life Cycle Model------------\n\n')
            
            for age = T:-1:1 % Start in the last period (T) and iterate backward
                
                if age == T % Last period of life (age T-1 in problem notation)
                    % In the last period, consume everything (aT = 0)
                    for i = 1:ylen
                        % Determine income based on whether retired
                        if age-1 >= tr % Already retired
                            yt = kappa *Gt(tr) *ygrid(i); % Pension
                        else
                            yt = Gt(age)*grid(i); % Still working
                        end
                        
                        c1(:, T, i) = agrid + yt; % Consume all assets plus income
                        a1(:, T, i) = 0.0; % No savings in terminal period
                        v1(:, T, i) = model.utility(c1(:, T, i), par); % Terminal value function
                    end

                else % All other periods
                    % Current age for the problem definition (0-indexed)
                    current_age = age - 1;
                    
                    for i = 1:ylen % Loop over income states
                        % Current income based on retirement status
                        if current_age >= tr
                            % Retired
                            yt = kappa * Gt(tr) * ygrid(i);
                            
                            % In retirement, no income uncertainty
                            ev = v1(:, age+1, i);
                        else
                            % Working
                            yt = Gt(age) * ygrid(i);
                            
                            % Expected value with income transitions
                            ev = v1(:, age+1, :);
                            ev = reshape(ev, [alen, ylen]);
                            ev = ev * pmat(i, :)';
                        end
                        
                        for p = 1:alen % Loop over asset states
                            % Cash-on-hand
                            coh = agrid(p) + yt;
                            
                            % For each possible next-period asset choice
                            vals = zeros(alen, 1);
                            for ap = 1:alen
                                % Consumption implied by asset choice
                                c = coh - agrid(ap)/(1+r);
                                
                                if c > 0
                                    % Utility plus discounted expected future value
                                    vals(ap) = model.utility(c, par) + beta * ev(ap);
                                else
                                    vals(ap) = -1e10; % Large negative value for infeasible choices
                                end
                            end
                            
                            % Find the optimal choice
                            [vmax, amax_idx] = max(vals);
                            
                            % Store the results
                            v1(p, age, i) = vmax;
                            a1(p, age, i) = agrid(amax_idx);
                            c1(p, age, i) = coh - a1(p, age, i)/(1+r);
                        end
                    end
                end

                % Print progress
                if mod(age, 10) == 0 || age == T || age == 1
                    fprintf('Solving for age: %d\n', age-1) % Convert to 0-indexed age for consistency
                end
            end
            
            fprintf('------------Life Cycle Problem Solved------------\n')
            
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
            fprintf('Solving model for %d parameter combinations...\n', total_combinations);
            
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
                    
                    % Solve the model
                    sol_i = solve.lc(par_i);
                    
                    % Store the solution
                    sol_collection(counter).beta = par_i.beta;
                    sol_collection(counter).gamma = par_i.sigma;
                    sol_collection(counter).c = sol_i.c;
                    sol_collection(counter).a = sol_i.a;
                    sol_collection(counter).v = sol_i.v;
                end
            end
            
            fprintf('All parameter combinations solved.\n');
        end
    end
end
