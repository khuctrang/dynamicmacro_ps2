%% File Info.

%{

    model.m
    -------
    This code sets up the model.

%}

%% Model class.

classdef model
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

            assert(par.kappa >= 0.0 && par.kappa <= 1.0,'The share of income received as pension should be from 0 to 1.\n')
            assert(par.sigma_eps > 0,'The standard deviation of the shock must be positive.\n')
            assert(abs(par.rho) < 1,'The persistence must be less than 1 in absolute value so that the series is stationary.\n')

            %% Simulation parameters.

            par.seed = 2025; % Seed for simulation.
            par.TT = par.T; % Number of time periods.
            par.NN = 3000; % Number of people.

        
        	%% Load and process VHLSS data for Gt
            par = model.load_vhlss_data(par); %(to calculate using simulate Gt, diable this line and insert this here:  par.Gt = model.create_synthetic_income_profile(par.T, par.tr, par.kappa);)
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
        par.Gt = model.create_synthetic_income_profile(par.T, par.tr, par.kappa);
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
            
            [ygrid,pmat] = model.tauchen(par.mu,par.rho,par.sigma_eps,par.ylen,par.m); % Tauchen's Method to discretize the AR(1) process for log productivity.
            par.ygrid = exp(ygrid); % The AR(1) is in logs so exponentiate it to get A.
            par.pmat = pmat; % Transition matrix.
        
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
