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
      function par = setup(firm_type)
            %% Structure array for model parameters.
            
            par = struct();
            
            %% Technology parameters - different for large and small firms
            if strcmp(firm_type, 'large')
                par.beta = 0.95;           % Discount factor
                par.alpha_K = 0.40;        % Capital's share of income
                par.alpha_L = 0.50;        % Labor's share of income
                par.delta = 0.08;          % Depreciation rate (baseline)
                par.w = 1.5;               % Wage rate for large firms
                par.rho_A = 0.85;          % Persistence of productivity for large firms
                par.sigma_eps = 0.10;      % Std. dev of productivity shocks for large firms
                par.kmax = 50.0;           % Upper bound for capital
                par.kmin = 1.0;            % Minimum capital
            else % small firms
                par.beta = 0.95;           % Discount factor
                par.alpha_K = 0.30;        % Capital's share of income
                par.alpha_L = 0.60;        % Labor's share of income
                par.delta = 0.08;          % Depreciation rate (baseline)
                par.w = 1.0;               % Wage rate for small firms
                par.rho_A = 0.70;          % Persistence of productivity for small firms
                par.sigma_eps = 0.20;      % Std. dev of productivity shocks for small firms
                par.kmax = 20.0;           % Upper bound for capital
                par.kmin = 0.5;            % Minimum capital
            end
            
            % Price process parameters - same for both firm types
            par.rho_p = 0.80;          % Persistence of price
            par.sigma_p = 0.05;        % Standard deviation of price shocks
            par.p_mean = 1.0;          % Mean price of investment
            
            %% Adjustment costs
            par.gamma = 0.10;          % Adjustment cost parameter (baseline)
            
            %% Simulation parameters
            par.seed = 2025;           % Seed for simulation
            par.T = 1000;              % Number of time periods
            par.burn_in = 200;         % Burn-in periods for simulation
            
            % Parameter validation
            assert(par.delta >= 0.0 && par.delta <= 1.0, 'The depreciation rate should be from 0 to 1.')
            assert(par.beta > 0.0 && par.beta < 1.0, 'Discount factor should be between 0 and 1.')
            assert(par.alpha_K > 0.0 && par.alpha_K < 1.0, 'Capital share of income should be between 0 and 1.')
            assert(par.alpha_L > 0.0 && par.alpha_L < 1.0, 'Labor share of income should be between 0 and 1.')
            assert(par.alpha_K + par.alpha_L < 1.0, 'Sum of income shares should be less than 1.')
            assert(par.gamma >= 0.0, 'The cost function coefficient should be non-negative.')
            assert(par.sigma_eps > 0, 'The standard deviation of the shock must be positive.')
            assert(abs(par.rho_A) < 1, 'The productivity persistence must be less than 1 in absolute value.')
            assert(abs(par.rho_p) < 1, 'The price persistence must be less than 1 in absolute value.')
      end
        
        %% Generate state grids.
        
        function par = gen_grids(par)
            %% Capital grid
            par.klen = 100;                        % Grid size for capital
            par.kgrid = linspace(par.kmin, par.kmax, par.klen)'; % Equally spaced linear grid
            
            %% Productivity grid
            par.Alen = 15;                         % Grid size for productivity
            par.m = 3;                             % Scaling parameter for Tauchen
            
            [Agrid, pmat_A] = model.tauchen(0, par.rho_A, par.sigma_eps, par.Alen, par.m);
            par.Agrid = exp(Agrid);                % Exponentiate to get productivity levels
            par.pmat_A = pmat_A;                   % Transition matrix for productivity
            
            %% Investment price grid
            par.plen = 7;                          % Grid size for price
            [pgrid, pmat_p] = model.tauchen(log(par.p_mean), par.rho_p, par.sigma_p, par.plen, par.m);
            par.pgrid = exp(pgrid);                % Exponentiate to get price levels
            par.pmat_p = pmat_p;                   % Transition matrix for price
        end
        %% Tauchen's Method
        
        function [y, pi] = tauchen(mu, rho, sigma, N, m)
            %% Construct equally spaced grid.
            ar_mean = mu/(1-rho);                  % Mean of stationary AR(1) process
            ar_sd = sigma/((1-rho^2)^(1/2));       % Std dev of stationary AR(1) process
            
            y1 = ar_mean-(m*ar_sd);                % Smallest grid point
            yn = ar_mean+(m*ar_sd);                % Largest grid point
            
            y = linspace(y1, yn, N)';              % Equally spaced grid
            d = y(2)-y(1);                         % Step size
            
            %% Compute transition probability matrix
            pi = zeros(N, N);
            
            for i = 1:N
                for j = 1:N
                    if j == 1
                        pi(i, j) = normcdf((y(j) - mu - rho*y(i) + d/2)/sigma);
                    elseif j == N
                        pi(i, j) = 1 - normcdf((y(j) - mu - rho*y(i) - d/2)/sigma);
                    else
                        pi(i, j) = normcdf((y(j) - mu - rho*y(i) + d/2)/sigma) - ...
                                   normcdf((y(j) - mu - rho*y(i) - d/2)/sigma);
                    end
                end
            end
        end
        
        %% Revenue function with static labor choice
        
        function [output, labor] = production(A, k, par)
            % Optimal labor choice given productivity and capital
            labor = (par.alpha_L * A * k.^par.alpha_K / par.w).^(1/(1-par.alpha_L));
            
            % Revenue function with optimal labor
            output = A .* k.^par.alpha_K .* labor.^par.alpha_L;
        end
        
        %% Cost function
        
        function [cost, invest] = total_cost(k, k_next, p, par)
            % Investment in new capital
            invest = k_next - (1-par.delta).*k;
            
            % Convex adjustment cost
            adj_cost = (par.gamma/2) .* (k_next - k).^2;
            
            % Total investment cost
            cost = adj_cost + p.*invest;
        end
    end
end