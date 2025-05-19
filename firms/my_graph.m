%% File Info.

%{

    my_graph.m
    ----------
    This code plots the value and policy functions and the time path of the variables.

%}

%% Graph class.

classdef my_graph
    methods(Static)
        %% Plot policy functions
        function [] = plot_policy(par, sol, firm_type)
            % Get mid-point indices for productivity and price
            a_mid = round(par.Alen/2);
            p_mid = round(par.plen/2);
            
            % Plot capital policy function
            figure('Name', [firm_type ' Firm - Capital Policy Function']);
            plot(par.kgrid, squeeze(sol.k(:, a_mid, p_mid)), 'LineWidth', 2);
            hold on;
            plot(par.kgrid, par.kgrid, 'k--');
            xlabel('Current Capital (K)');
            ylabel('Next Period Capital (K'')');
            title([firm_type ' Firm - Capital Policy Function']);
            legend('Policy Function', '45-degree Line');
            grid on;
            
            % Plot investment policy function
            figure('Name', [firm_type ' Firm - Investment Policy Function']);
            plot(par.kgrid, squeeze(sol.i(:, a_mid, p_mid)), 'LineWidth', 2);
            xlabel('Current Capital (K)');
            ylabel('Investment (I)');
            title([firm_type ' Firm - Investment Policy Function']);
            grid on;
            
            % Plot labor policy function
            figure('Name', [firm_type ' Firm - Labor Policy Function']);
            plot(par.kgrid, squeeze(sol.l(:, a_mid, p_mid)), 'LineWidth', 2);
            xlabel('Current Capital (K)');
            ylabel('Labor Demand (L)');
            title([firm_type ' Firm - Labor Policy Function']);
            grid on;
            
            % Plot value function
            figure('Name', [firm_type ' Firm - Value Function']);
            plot(par.kgrid, squeeze(sol.v(:, a_mid, p_mid)), 'LineWidth', 2);
            xlabel('Current Capital (K)');
            ylabel('Firm Value (V)');
            title([firm_type ' Firm - Value Function']);
            grid on;
        end
        
        %% Plot simulation results
        function [] = plot_simulation(sim, firm_type)
            T = length(sim.k);
            time = 1:T;
            
            % Plot simulated capital
            figure('Name', [firm_type ' Firm - Simulated Capital']);
            plot(time, sim.k, 'LineWidth', 1.5);
            xlabel('Time Period');
            ylabel('Capital Stock');
            title([firm_type ' Firm - Simulated Capital']);
            grid on;
            
            % Plot simulated investment
            figure('Name', [firm_type ' Firm - Simulated Investment']);
            plot(time, sim.i, 'LineWidth', 1.5);
            xlabel('Time Period');
            ylabel('Investment');
            title([firm_type ' Firm - Simulated Investment']);
            grid on;
        end
        
        %% Plot parameter analysis for delta variations
        function [] = plot_delta_analysis(par_large, sol_large, par_small, sol_small)
            % Parameter values
            delta_values = [0.05, 0.06, 0.07, 0.08];
            gamma_fixed = 0.10;
            
            % Create separate figures for capital and investment
            % Capital plots
            figure('Name', 'Capital with Different Depreciation Rates', 'Visible', 'on');
            % Add this line to ensure the first plot is visible
            set(0, 'DefaultFigureVisible', 'on');
            % Large firms - Capital
            subplot(2, 1, 1);
            hold on;
            for d = 1:length(delta_values)
                % Setup model with current delta
                par = par_large;
                par.delta = delta_values(d);
                par.gamma = gamma_fixed;
                par = model.gen_grids(par);
                
                % Simulate model
                sol = solve.firm_problem(par);
                sim = simulate.firm_dynamics(par, sol);
                
                % Plot average capital for the first 100 periods (for visibility)
                plot(1:100, sim.k(1:100), 'LineWidth', 1.5);
            end
            title('Capital - Large Firms');
            xlabel('Time Period');
            ylabel('Capital Stock');
            legend('\delta = 0.05', '\delta = 0.06', '\delta = 0.07', '\delta = 0.08');
            grid on;
            
            % Small firms - Capital
            subplot(2, 1, 2);
            hold on;
            for d = 1:length(delta_values)
                % Setup model with current delta
                par = par_small;
                par.delta = delta_values(d);
                par.gamma = gamma_fixed;
                par = model.gen_grids(par);
                
                % Simulate model
                sol = solve.firm_problem(par);
                sim = simulate.firm_dynamics(par, sol);
                
                % Plot average capital for the first 100 periods (for visibility)
                plot(1:100, sim.k(1:100), 'LineWidth', 1.5);
            end
            title('Capital - Small Firms');
            xlabel('Time Period');
            ylabel('Capital Stock');
            legend('\delta = 0.05', '\delta = 0.06', '\delta = 0.07', '\delta = 0.08');
            grid on;
            
            % Investment plots
            figure('Name', 'Investment with Different Depreciation Rates');
            
            % Large firms - Investment
            subplot(2, 1, 1);
            hold on;
            for d = 1:length(delta_values)
                % Setup model with current delta
                par = par_large;
                par.delta = delta_values(d);
                par.gamma = gamma_fixed;
                par = model.gen_grids(par);
                
                % Simulate model
                sol = solve.firm_problem(par);
                sim = simulate.firm_dynamics(par, sol);
                
                % Plot average investment for the first 100 periods (for visibility)
                plot(1:100, sim.i(1:100), 'LineWidth', 1.5);
            end
            title('Investment - Large Firms');
            xlabel('Time Period');
            ylabel('Investment');
            legend('\delta = 0.05', '\delta = 0.06', '\delta = 0.07', '\delta = 0.08');
            grid on;
            
            % Small firms - Investment
            subplot(2, 1, 2);
            hold on;
            for d = 1:length(delta_values)
                % Setup model with current delta
                par = par_small;
                par.delta = delta_values(d);
                par.gamma = gamma_fixed;
                par = model.gen_grids(par);
                
                % Simulate model
                sol = solve.firm_problem(par);
                sim = simulate.firm_dynamics(par, sol);
                
                % Plot average investment for the first 100 periods (for visibility)
                plot(1:100, sim.i(1:100), 'LineWidth', 1.5);
            end
            title('Investment - Small Firms');
            xlabel('Time Period');
            ylabel('Investment');
            legend('\delta = 0.05', '\delta = 0.06', '\delta = 0.07', '\delta = 0.08');
            grid on;
            drawnow;
            pause(1);
        end
       
        %% Plot parameter analysis for gamma variations
        function [] = plot_gamma_analysis(par_large, sol_large, par_small, sol_small)
            % Parameter values
            delta_fixed = 0.08;
            gamma_values = [0.10, 0.15, 0.20, 0.25];
            
            % Create separate figures for capital and investment
            % Capital plots
            figure('Name', 'Capital with Different Adjustment Costs');
            set(0, 'DefaultFigureVisible', 'on');
            % Large firms - Capital
            subplot(2, 1, 1);
            hold on;
            for g = 1:length(gamma_values)
                % Setup model with current gamma
                par = par_large;
                par.delta = delta_fixed;
                par.gamma = gamma_values(g);
                par = model.gen_grids(par);
                
                % Simulate model
                sol = solve.firm_problem(par);
                sim = simulate.firm_dynamics(par, sol);
                
                % Plot average capital for the first 100 periods (for visibility)
                plot(1:100, sim.k(1:100), 'LineWidth', 1.5);
            end
            title('Capital - Large Firms');
            xlabel('Time Period');
            ylabel('Capital Stock');
            legend('\gamma = 0.10', '\gamma = 0.15', '\gamma = 0.20', '\gamma = 0.25');
            grid on;
            
            % Small firms - Capital
            subplot(2, 1, 2);
            hold on;
            for g = 1:length(gamma_values)
                % Setup model with current gamma
                par = par_small;
                par.delta = delta_fixed;
                par.gamma = gamma_values(g);
                par = model.gen_grids(par);
                
                % Simulate model
                sol = solve.firm_problem(par);
                sim = simulate.firm_dynamics(par, sol);
                
                % Plot average capital for the first 100 periods (for visibility)
                plot(1:100, sim.k(1:100), 'LineWidth', 1.5);
            end
            title('Capital - Small Firms');
            xlabel('Time Period');
            ylabel('Capital Stock');
            legend('\gamma = 0.10', '\gamma = 0.15', '\gamma = 0.20', '\gamma = 0.25');
            grid on;
            
            % Investment plots
            figure('Name', 'Investment with Different Adjustment Costs');
            
            % Large firms - Investment
            subplot(2, 1, 1);
            hold on;
            for g = 1:length(gamma_values)
                % Setup model with current gamma
                par = par_large;
                par.delta = delta_fixed;
                par.gamma = gamma_values(g);
                par = model.gen_grids(par);
                
                % Simulate model
                sol = solve.firm_problem(par);
                sim = simulate.firm_dynamics(par, sol);
                
                % Plot average investment for the first 100 periods (for visibility)
                plot(1:100, sim.i(1:100), 'LineWidth', 1.5);
            end
            title('Investment - Large Firms');
            xlabel('Time Period');
            ylabel('Investment');
            legend('\gamma = 0.10', '\gamma = 0.15', '\gamma = 0.20', '\gamma = 0.25');
            grid on;
            
            % Small firms - Investment
            subplot(2, 1, 2);
            hold on;
            for g = 1:length(gamma_values)
                % Setup model with current gamma
                par = par_small;
                par.delta = delta_fixed;
                par.gamma = gamma_values(g);
                par = model.gen_grids(par);
                
                % Simulate model
                sol = solve.firm_problem(par);
                sim = simulate.firm_dynamics(par, sol);
                
                % Plot average investment for the first 100 periods (for visibility)
                plot(1:100, sim.i(1:100), 'LineWidth', 1.5);
            end
            title('Investment - Small Firms');
            xlabel('Time Period');
            ylabel('Investment');
            legend('\gamma = 0.10', '\gamma = 0.15', '\gamma = 0.20', '\gamma = 0.25');
            grid on;
            drawnow;
            pause(1);
        end
        
        %% Plot heat maps for parameter analysis
        function [] = plot_parameter_heatmaps(results_large, results_small)
            % Parameter values
            delta_values = results_large.delta_values;
            gamma_values = results_large.gamma_values;
            
            % Create figure for capital
            figure('Name', 'Parameter Heat Maps - Average Capital', 'Position', [100, 100, 1000, 450]);
            set(0, 'DefaultFigureVisible', 'on');
            % Large firms heat map
            subplot(1, 2, 1);
            imagesc(results_large.avg_k);
            colorbar;
            title('Average Capital - Large Firms');
            set(gca, 'XTick', 1:length(delta_values), 'XTickLabel', delta_values);
            set(gca, 'YTick', 1:length(gamma_values), 'YTickLabel', gamma_values);
            xlabel('\delta (Depreciation Rate)');
            ylabel('\gamma (Adjustment Cost)');
            
            % Small firms heat map
            subplot(1, 2, 2);
            imagesc(results_small.avg_k);
            colorbar;
            title('Average Capital - Small Firms');
            set(gca, 'XTick', 1:length(delta_values), 'XTickLabel', delta_values);
            set(gca, 'YTick', 1:length(gamma_values), 'YTickLabel', gamma_values);
            xlabel('\delta (Depreciation Rate)');
            ylabel('\gamma (Adjustment Cost)');
            
            % Create figure for investment
            figure('Name', 'Parameter Heat Maps - Average Investment', 'Position', [100, 100, 1000, 450]);
            
            % Large firms heat map
            subplot(1, 2, 1);
            imagesc(results_large.avg_i);
            colorbar;
            title('Average Investment - Large Firms');
            set(gca, 'XTick', 1:length(delta_values), 'XTickLabel', delta_values);
            set(gca, 'YTick', 1:length(gamma_values), 'YTickLabel', gamma_values);
            xlabel('\delta (Depreciation Rate)');
            ylabel('\gamma (Adjustment Cost)');
            
            % Small firms heat map
            subplot(1, 2, 2);
            imagesc(results_small.avg_i);
            colorbar;
            title('Average Investment - Small Firms');
            set(gca, 'XTick', 1:length(delta_values), 'XTickLabel', delta_values);
            set(gca, 'YTick', 1:length(gamma_values), 'YTickLabel', gamma_values);
            xlabel('\delta (Depreciation Rate)');
            ylabel('\gamma (Adjustment Cost)');
            drawnow;
            pause(1);
        end
    end
end