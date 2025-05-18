%% File Info.

%{

    model_remittance.m
    ------------------
    This code extends the baseline model to include remittances as an 
    additional income source for Vietnamese households.

%}

%% Model Remittance class.

classdef model_remittance
    methods(Static)
        %% Set up structure array for model parameters and set the simulation parameters.
        
        function par = setup()            
            %% Structure array for model parameters.
            
            par = struct();
            
            %% Preferences.

            par.T = 75; % Last period of life.
            par.tr = 60; % First period of retirement.
            
            par.beta = 0.94; % Discount factor: Lower values of this mean that consumers are impatient and consume more today.
            par.sigma = 2.0; % CRRA: Higher values of this mean that consumers are risk averse and do not want to consume too much today.
            
            assert(par.T > par.tr,'Cannot retire after dying.\n')
            assert(par.beta > 0.0 && par.beta < 1.0,'Discount factor should be between 0 and 1.\n')
            assert(par.sigma > 0.0,'CRRA should be at least 0.\n')

            %% Prices and Income.

            par.r = 0.03; % Interest rate.
            par.kappa = 0.7; % Share of income as pension.

            par.sigma_eps = 0.1; % Std. dev of productivity shocks
            par.rho = 0.95; % Persistence of AR(1) process as specified in problem
            par.mu = 0.0; % Intercept of AR(1) process
            

            %% Remittance Parameters
            par.eta = 0.7; % Persistence of remittance process
            par.sigma_nu = 0.2; % Std. dev of remittance shocks
            par.corr_eps_nu = -0.3; % Correlation between income and remittance shocks (negative: counter-cyclical)
            par.sigma_eps_nu = par.corr_eps_nu * par.sigma_eps * par.sigma_nu; % Covariance between income and remittance shocks
            
            % Age-specific probability of receiving remittances
            % Higher probability for working age and elderly
            par.p_remit = zeros(par.T, 1);
            for t = 1:par.T
                if t <= 20 % Young (0-19)
                    par.p_remit(t) = 0.1;
                elseif t <= 40 % Early working age (20-39)
                    par.p_remit(t) = 0.3;
                elseif t < par.tr % Late working age (40-59)
                    par.p_remit(t) = 0.5;
                else % Retirement (60+)
                    par.p_remit(t) = 0.7;
                end
            
            end
            
            % Age-specific average remittance factor (relative to income)
            par.mu_remit = zeros(par.T, 1);
            for t = 1:par.T
                if t <= 20 % Young
                    par.mu_remit(t) = 0.05;
                elseif t <= 40 % Early working age
                    par.mu_remit(t) = 0.1;
                elseif t < par.tr % Late working age
                    par.mu_remit(t) = 0.2;
                else % Retirement
                    par.mu_remit(t) = 0.3;
                end
            end

            assert(par.kappa >= 0.0 && par.kappa <= 1.0,'The share of income received as pension should be from 0 to 1.\n')
            assert(par.sigma_eps > 0,'The standard deviation of the shock must be positive.\n')
            assert(abs(par.rho) < 1,'The persistence must be less than 1 in absolute value so that the series is stationary.\n')
            assert(abs(par.eta) < 1,'The remittance persistence must be less than 1 in absolute value.\n')
            assert(par.sigma_nu > 0,'The standard deviation of remittance shocks must be positive.\n')

            %% Simulation parameters.

            par.seed = 2025; % Seed for simulation.
            par.TT = par.T; % Number of time periods.
            par.NN = 3000; % Number of people.
            par.Gt = model_remittance.create_synthetic_income_profile(par.T, par.tr, par.kappa);

            %% Load and process VHLSS data for Gt
            %par = model_remittance.load_vhlss_data(par);
            
            %% Load and process VHLSS data for remittances
            %par = model_remittance.process_remittance_data(par);
        end 
        
        %% Load and process VHLSS data
        function par = load_vhlss_data(par)
            try
                % Step 1: Load data files
                fprintf('Loading VHLSS data...\n');
                
                % Step 1: Load the household member file (muc123a)
                fprintf('  Loading household member data (muc123a)...\n');
                member_file = 'muc123a.csv'; % Adjust the path as needed
                members_data = readtable(member_file);
                fprintf('  Loading income data (muc4a)...\n');
                income_file = 'muc4a.csv'; % Adjust the path as needed
                income_data = readtable(income_file);
             
                
                % Step 2: Find male heads
                male_head_mask = (members_data.m1ac2 == 1) & (members_data.m1ac3 == 1);
                male_heads = members_data(male_head_mask, :);
                
                % Step 3: Create household identifier function
                createHHKey = @(data) string(data.tinh) + "_" + string(data.huyen) + "_" + ...
                                      string(data.xa) + "_" + string(data.diaban) + "_" + string(data.hoso);
                
                % Step 4: Calculate total income for each household
                household_income = containers.Map('KeyType', 'char', 'ValueType', 'double');
                
                % Process each income record
                for i = 1:height(income_data)
                    hh_key = char(createHHKey(income_data(i,:)));
                    
                    % Calculate income from components
                    income_sum = 0;
                    income_cols = {'m4ac11', 'm4ac12f', 'm4ac21', 'm4ac22f', 'm4ac25'};
                    
                    for j = 1:length(income_cols)
                        col = income_cols{j};
                        if ismember(col, income_data.Properties.VariableNames)
                            val = income_data.(col)(i);
                            if ~isnan(val)
                                income_sum = income_sum + val;
                            end
                        end
                    end
                    
                    % Add to household total
                    if ~isKey(household_income, hh_key)
                        household_income(hh_key) = income_sum;
                    else
                        household_income(hh_key) = household_income(hh_key) + income_sum;
                    end
                end
                
                % Step 5: Associate male household heads with their household income
                ages = [];
                incomes = [];
                
                for i = 1:height(male_heads)
                    head = male_heads(i, :);
                    hh_key = char(createHHKey(head));
                    
                    if isKey(household_income, hh_key)
                        age = head.m1ac5;
                        total_income = household_income(hh_key);
                        
                        if age >= 0 && total_income > 0
                            ages = [ages; age];
                            incomes = [incomes; total_income];
                        end
                    end
                end
                
                % Step 6: Calculate age-specific income profile
                max_age = min(max(ages), par.T-1);
                par.Gt = zeros(par.T, 1);
                
                for age = 0:max_age
                    age_incomes = incomes(ages == age);
                    
                    if ~isempty(age_incomes)
                        % Use geometric mean (exponentiate the mean of logs)
                        par.Gt(age+1) = exp(mean(log(age_incomes)));
                    end
                end
                
                % Step 7: Apply smoothing
                window_size = 5;
                smoothed_Gt = par.Gt;
                
                for t = 1:length(par.Gt)
                    window_start = max(1, t - floor(window_size/2));
                    window_end = min(length(par.Gt), t + floor(window_size/2));
                    
                    window_values = par.Gt(window_start:window_end);
                    window_values = window_values(window_values > 0);
                    
                    if ~isempty(window_values)
                        smoothed_Gt(t) = mean(window_values);
                    end
                end
                
                par.Gt = smoothed_Gt;
                
                % Step 8: Interpolate missing values
                for t = 1:length(par.Gt)
                    if par.Gt(t) == 0
                        % Find nearest non-zero values
                        prev_t = t-1;
                        while prev_t >= 1 && par.Gt(prev_t) == 0
                            prev_t = prev_t - 1;
                        end
                        
                        next_t = t+1;
                        while next_t <= length(par.Gt) && par.Gt(next_t) == 0
                            next_t = next_t + 1;
                        end
                        
                        if prev_t >= 1 && next_t <= length(par.Gt)
                            % Linear interpolation
                            par.Gt(t) = par.Gt(prev_t) + (par.Gt(next_t) - par.Gt(prev_t)) * (t - prev_t) / (next_t - prev_t);
                        elseif prev_t >= 1
                            par.Gt(t) = par.Gt(prev_t);
                        elseif next_t <= length(par.Gt)
                            par.Gt(t) = par.Gt(next_t);
                        else
                            par.Gt(t) = 1.0;
                        end
                    end
                end
                
                % Step 9: Extend to cover all ages
                if length(par.Gt) < par.T
                    for t = length(par.Gt)+1:par.T
                        if t-1 >= par.tr
                            par.Gt(t) = par.Gt(par.tr) * par.kappa;
                        else
                            par.Gt(t) = par.Gt(length(par.Gt));
                        end
                    end
                end
                
                % Step 10: Normalize to value 1.0 at age 20
                if par.Gt(21) > 0
                    par.Gt = par.Gt / par.Gt(21);
                end
                
            catch e
                par.Gt = model_remittance.create_synthetic_income_profile(par.T, par.tr, par.kappa);
            end
        end
        
        %% Process remittance data from VHLSS
        function par = process_remittance_data(par)
            try
                % Load remittance data file (e.g., muc4a or dedicated remittance file)
                fprintf('Loading and processing remittance data...\n');
                remit_file = 'muc1c.csv'; % Adjust to the actual file containing remittance data
                remit_data = readtable(remit_file);
                
                % Load household member file for age data
                member_file = 'muc123a.csv';
                members_data = readtable(member_file);
                
                % Create household identifier function
                createHHKey = @(data) string(data.tinh) + "_" + string(data.huyen) + "_" + ...
                                     string(data.xa) + "_" + string(data.diaban) + "_" + string(data.hoso);
                
                % Find male heads for household age mapping
                male_head_mask = (members_data.m1ac2 == 1) & (members_data.m1ac3 == 1);
                male_heads = members_data(male_head_mask, :);
                
                % Create mapping from household ID to age of head
                head_ages = containers.Map('KeyType', 'char', 'ValueType', 'double');
                for i = 1:height(male_heads)
                    head = male_heads(i, :);
                    hh_key = char(createHHKey(head));
                    head_ages(hh_key) = head.m1ac5;
                end
                
                % Calculate remittances by household
                % Adjust column names based on actual VHLSS data structure
                remittance_columns = {'m1cc10'}; 
                
                household_remittances = containers.Map('KeyType', 'char', 'ValueType', 'double');
                household_incomes = containers.Map('KeyType', 'char', 'ValueType', 'double');
                
                % Process each record
                for i = 1:height(remit_data)
                    hh_key = char(createHHKey(remit_data(i,:)));
                    
                    % Calculate total remittances
                    remit_sum = 0;
                    for j = 1:length(remittance_columns)
                        col = remittance_columns{j};
                        if ismember(col, remit_data.Properties.VariableNames)
                            val = remit_data.(col)(i);
                            if ~isnan(val)
                                remit_sum = remit_sum + val;
                            end
                        end
                    end
                    
                    % Add to household total
                    if ~isKey(household_remittances, hh_key)
                        household_remittances(hh_key) = remit_sum;
                    else
                        household_remittances(hh_key) = household_remittances(hh_key) + remit_sum;
                    end
                    
                    % Also calculate total income for ratio calculations
                    income_sum = 0;
                    income_cols = {'m4ac11', 'm4ac12f', 'm4ac21', 'm4ac22f', 'm4ac25'};
                    
                    for j = 1:length(income_cols)
                        col = income_cols{j};
                        if ismember(col, remit_data.Properties.VariableNames)
                            val = remit_data.(col)(i);
                            if ~isnan(val)
                                income_sum = income_sum + val;
                            end
                        end
                    end
                    
                    if ~isKey(household_incomes, hh_key)
                        household_incomes(hh_key) = income_sum;
                    else
                        household_incomes(hh_key) = household_incomes(hh_key) + income_sum;
                    end
                end
                
                % Collect data for each age group
                age_remit_prob = zeros(par.T, 1);
                age_remit_ratio = zeros(par.T, 1);
                age_counts = zeros(par.T, 1);
                
                % Process each household with a male head
                household_keys = keys(household_remittances);
                for i = 1:length(household_keys)
                    hh_key = household_keys{i};
                    
                    % Check if we have age data for this household
                    if isKey(head_ages, hh_key) && isKey(household_incomes, hh_key)
                        age = head_ages(hh_key);
                        
                        if age >= 0 && age < par.T
                            age_idx = age + 1;
                            
                            % Increment count for this age
                            age_counts(age_idx) = age_counts(age_idx) + 1;
                            
                            % Check if household received remittances
                            if household_remittances(hh_key) > 0
                                age_remit_prob(age_idx) = age_remit_prob(age_idx) + 1;
                                
                                % Calculate remittance ratio relative to income
                                income = max(household_incomes(hh_key), 1); % Avoid division by zero
                                remit_ratio = household_remittances(hh_key) / income;
                                
                                age_remit_ratio(age_idx) = age_remit_ratio(age_idx) + remit_ratio;
                            end
                        end
                    end
                end
                
                % Calculate probabilities and mean ratios
                for age = 0:par.T-1
                    age_idx = age + 1;
                    if age_counts(age_idx) > 0
                        age_remit_prob(age_idx) = age_remit_prob(age_idx) / age_counts(age_idx);
                        
                        if age_remit_prob(age_idx) > 0
                            age_remit_ratio(age_idx) = age_remit_ratio(age_idx) / (age_remit_prob(age_idx) * age_counts(age_idx));
                        end
                    else
                        % Default values if no data
                        if age < 20
                            age_remit_prob(age_idx) = 0.1;
                            age_remit_ratio(age_idx) = 0.05;
                        elseif age < 40
                            age_remit_prob(age_idx) = 0.3;
                            age_remit_ratio(age_idx) = 0.1;
                        elseif age < par.tr
                            age_remit_prob(age_idx) = 0.5;
                            age_remit_ratio(age_idx) = 0.2;
                        else
                            age_remit_prob(age_idx) = 0.7;
                            age_remit_ratio(age_idx) = 0.3;
                        end
                    end
                end
                
                % Smooth the values
                window_size = 5;
                smoothed_prob = age_remit_prob;
                smoothed_ratio = age_remit_ratio;
                
                for t = 1:par.T
                    window_start = max(1, t - floor(window_size/2));
                    window_end = min(par.T, t + floor(window_size/2));
                    
                    prob_values = age_remit_prob(window_start:window_end);
                    prob_values = prob_values(prob_values > 0);
                    
                    ratio_values = age_remit_ratio(window_start:window_end);
                    ratio_values = ratio_values(ratio_values > 0);
                    
                    if ~isempty(prob_values)
                        smoothed_prob(t) = mean(prob_values);
                    end
                    
                    if ~isempty(ratio_values)
                        smoothed_ratio(t) = mean(ratio_values);
                    end
                end
                
                % Store in parameter structure
                par.p_remit = smoothed_prob;
                par.mu_remit = smoothed_ratio;
                
                fprintf('Remittance data processed successfully.\n');
                
            catch e
                fprintf('Error processing remittance data: %s\n', e.message);
                fprintf('Using default remittance parameters.\n');
                
                % Keep the default parameters already set in setup()
            end
        end

        %% Create synthetic income profile
        function Gt = create_synthetic_income_profile(T, tr, kappa)
            % Create a synthetic age-income profile when real data is not available
            Gt = zeros(T, 1);
            
            % Parameters for the hump shape
            peak_age = 45;
            growth_rate = 0.04;
            decline_rate = 0.02;
            
            % Create hump-shaped profile
            for age = 0:T-1
                if age < peak_age
                    % Growth phase
                    Gt(age+1) = (1 + growth_rate)^age;
                else
                    % Decline phase (before retirement)
                    if age < tr
                        Gt(age+1) = (1 + growth_rate)^peak_age * (1 - decline_rate)^(age - peak_age);
                    else
                        % Retirement phase - use kappa * last working income
                        if age == tr
                            Gt(age+1) = Gt(age) * kappa;
                        else
                            Gt(age+1) = Gt(tr+1); % Keep pension constant
                        end
                    end
                end
            end
            
            % Normalize to have value of 1 at age 20
            normalization = Gt(21);
            Gt = Gt / normalization;
        end        
        
        %% Generate state grids.
        function par = gen_grids(par)
            %% Capital grid.

            par.alen = 300; % Grid size for a.
            par.amax = 30.0; % Upper bound for a.
            par.amin = 0.0; % Minimum a.
            
            assert(par.alen > 5,'Grid size for a should be positive and greater than 5.\n')
            assert(par.amax > par.amin,'Minimum a should be less than maximum value.\n')
            
            par.agrid = linspace(par.amin,par.amax,par.alen)'; % Equally spaced, linear grid for a and a'.
                
            %% Discretized income process.
                  
            par.ylen = 7; % Grid size for y.
            par.m = 3; % Scaling parameter for Tauchen.
            
            assert(par.ylen > 3,'Grid size for A should be positive and greater than 3.\n')
            assert(par.m > 0,'Scaling parameter for Tauchen should be positive.\n')
            
            [ygrid,pmat] = model_remittance.tauchen(par.mu,par.rho,par.sigma_eps,par.ylen,par.m); % Tauchen's Method to discretize the AR(1) process for log productivity.
            par.ygrid = exp(ygrid); % The AR(1) is in logs so exponentiate it to get A.
            par.pmat = pmat; % Transition matrix.
            
            %% Discretized remittance process
            par.rlen = 5; % Grid size for remittances
            
            [rgrid,rmat] = model_remittance.tauchen(0, par.eta, par.sigma_nu, par.rlen, par.m);
            par.rgrid = exp(rgrid); % Remittance amounts (will be multiplied by age-specific factor)
            par.rmat = rmat; % Transition matrix for remittances
            
            %% Joint transition matrix (if correlation between income and remittance shocks)
            if abs(par.corr_eps_nu) > 0
                % Create joint transition matrix that accounts for correlation
                par.joint_trans = model_remittance.create_joint_transition(par.ylen, par.rlen, par.pmat, par.rmat, par.corr_eps_nu);
            else
                % If no correlation, just use the product of independent transitions
                par.joint_trans = zeros(par.ylen, par.rlen, par.ylen, par.rlen);
                for y1 = 1:par.ylen
                    for r1 = 1:par.rlen
                        for y2 = 1:par.ylen
                            for r2 = 1:par.rlen
                                par.joint_trans(y1, r1, y2, r2) = par.pmat(y1, y2) * par.rmat(r1, r2);
                            end
                        end
                    end
                end
            end
        end
        
        %% Tauchen's Method
        function [y,pi] = tauchen(mu,rho,sigma,N,m)
            %% Construct equally spaced grid.
        
            ar_mean = mu/(1-rho); % The mean of a stationary AR(1) process is mu/(1-rho).
            ar_sd = sigma/((1-rho^2)^(1/2)); % The std. dev of a stationary AR(1) process is sigma/sqrt(1-rho^2)
            
            y1 = ar_mean-(m*ar_sd); % Smallest grid point is the mean of the AR(1) process minus m*std.dev of AR(1) process.
            yn = ar_mean+(m*ar_sd); % Largest grid point is the mean of the AR(1) process plus m*std.dev of AR(1) process.
            
            y = linspace(y1,yn,N); % Equally spaced grid.
            d = y(2)-y(1); % Step size.
            
            %% Compute transition probability matrix from state j (row) to k (column).
        
            ymatk = repmat(y,N,1); % States next period.
            ymatj = mu+rho*ymatk'; % States this period.
        
            pi = normcdf(ymatk,ymatj-(d/2),sigma) - normcdf(ymatk,ymatj+(d/2),sigma); % Transition probabilities to state 2, ..., N-1.
            pi(:,1) = normcdf(y(1),mu+rho*y-(d/2),sigma); % Transition probabilities to state 1.
            pi(:,N) = 1 - normcdf(y(N),mu+rho*y+(d/2),sigma); % Transition probabilities to state N.
        end
        
        %% Create joint transition matrix with correlation
        function joint_trans = create_joint_transition(ylen, rlen, pmat, rmat, corr)
            % This function creates a joint transition matrix for income and remittances
            % accounting for the correlation between their innovations
            
            joint_trans = zeros(ylen, rlen, ylen, rlen);
            
            % Parameter for the mixing of transition probabilities
            mix_param = abs(corr);
            
            % Direction of correlation
            corr_sign = sign(corr);
            
            for y1 = 1:ylen
                for r1 = 1:rlen
                    for y2 = 1:ylen
                        for r2 = 1:rlen
                            % Independent probabilities
                            indep_prob = pmat(y1, y2) * rmat(r1, r2);
                            
                            % Correlated adjustment
                            % When income increases (y2 > y1):
                            % - For negative correlation: decrease r (favor lower r2)
                            % - For positive correlation: increase r (favor higher r2)
                            if y2 > y1
                                if corr_sign < 0
                                    corr_adjust = (1 - (r2-1)/(rlen-1)); % Lower r2 favored
                                else
                                    corr_adjust = (r2-1)/(rlen-1); % Higher r2 favored
                                end
                            % When income decreases (y2 < y1):
                            % - For negative correlation: increase r (favor higher r2)
                            % - For positive correlation: decrease r (favor lower r2)
                            elseif y2 < y1
                                if corr_sign < 0
                                    corr_adjust = (r2-1)/(rlen-1); % Higher r2 favored
                                else
                                    corr_adjust = (1 - (r2-1)/(rlen-1)); % Lower r2 favored
                                end
                            % No change in income
                            else
                                corr_adjust = 0.5; % Neutral
                            end
                            
                            % Mix independent and correlated components
                            joint_trans(y1, r1, y2, r2) = (1 - mix_param) * indep_prob + mix_param * corr_adjust * pmat(y1, y2);
                        end
                    end
                    
                    % Normalize to ensure rows sum to 1
                    row_sum = 0;
                    for y2 = 1:ylen
                        for r2 = 1:rlen
                            row_sum = row_sum + joint_trans(y1, r1, y2, r2);
                        end
                    end
                    
                    for y2 = 1:ylen
                        for r2 = 1:rlen
                            joint_trans(y1, r1, y2, r2) = joint_trans(y1, r1, y2, r2) / row_sum;
                        end
                    end
                end
            end
        end
        
        %% Utility function.
        function u = utility(c,par)
            %% CRRA utility.
            
            if par.sigma == 1
                u = log(max(c, 1e-10)); % Log utility.
            else
                u = (max(c, 1e-10).^(1-par.sigma))./(1-par.sigma); % CRRA utility.
            end
        end
        
    end
end