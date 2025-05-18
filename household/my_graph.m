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
        function plot_policy_functions(par, sol)
            % This function plots the consumption and asset policy functions
            
            % Select ages to plot (e.g., early, middle, and late working life)
            ages_to_plot = [5, 20, 40, 60];
            
            % Select a middle income state
            y_idx = ceil(par.ylen / 2);
            
            % Create figure for consumption policy
            figure('Name', 'Consumption Policy Functions');
            hold on;
            
            % Plot consumption policy for different ages
            for i = 1:length(ages_to_plot)
                age = ages_to_plot(i);
                age_idx = age + 1; % Convert to 1-indexed
                if age_idx <= par.T
                    plot(par.agrid, sol.c(:, age_idx, y_idx), 'LineWidth', 2);
                end
            end
            
            title('Consumption Policy Function');
            xlabel('Current Assets (a_t)');
            ylabel('Consumption (c_t)');
            legend('Age 5', 'Age 20', 'Age 40', 'Age 60', 'Location', 'best');
            grid on;
            
            % Create figure for asset policy
            figure('Name', 'Asset Policy Functions');
            hold on;
            
            % Plot asset policy for different ages
            for i = 1:length(ages_to_plot)
                age = ages_to_plot(i);
                age_idx = age + 1; % Convert to 1-indexed
                if age_idx <= par.T
                    plot(par.agrid, sol.a(:, age_idx, y_idx), 'LineWidth', 2);
                end
            end
            
            % Add 45-degree line
            plot(par.agrid, par.agrid, 'k--', 'LineWidth', 1);
            
            title('Asset Policy Function');
            xlabel('Current Assets (a_t)');
            ylabel('Next Period Assets (a_{t+1})');
            legend('Age 5', 'Age 20', 'Age 40', 'Age 60', '45-degree line', 'Location', 'best');
            grid on;
        end
        
        %% Plot life cycle profiles
        function plot_lifecycle_profiles(sim)
            % This function plots the life cycle profiles of consumption, income, and assets
            
            % Extract data
            ages = sim.ages;
            c_profile = sim.c_profile;
            a_profile = sim.a_profile;
            y_profile = sim.y_profile;
            
            % Figure for consumption and income
            figure('Name', 'Life Cycle Profiles');
            
            % Plot consumption and income
            subplot(2, 1, 1);
            plot(ages, c_profile, 'b-', 'LineWidth', 2);
            hold on;
            plot(ages, y_profile, 'g--', 'LineWidth', 2);
            title('Life Cycle Profile of Consumption and Income');
            xlabel('Age');
            ylabel('Value');
            legend('Consumption', 'Income', 'Location', 'best');
            grid on;
            
            % Plot assets
            subplot(2, 1, 2);
            plot(ages, a_profile, 'r-', 'LineWidth', 2);
            title('Life Cycle Profile of Assets');
            xlabel('Age');
            ylabel('Assets');
            grid on;
            
            % Adjust figure size
            set(gcf, 'Position', [100, 100, 800, 600]);
        end
        
        %% Plot parameter variations - Beta
        function plot_beta_variations(ages, c_profiles, a_profiles, beta_values)
            % This function plots the life cycle profiles for different beta values
            
            figure('Name', 'Beta Variations');
            
            % Consumption profiles
            subplot(2, 1, 1);
            hold on;
            for i = 1:length(beta_values)
                plot(ages, c_profiles(:, i), 'LineWidth', 2);
            end
            title('Life Cycle Profile of Consumption - Varying \beta');
            xlabel('Age');
            ylabel('Consumption');
            
            % Create legend strings with beta values
            legend_strs = cell(length(beta_values), 1);
            for i = 1:length(beta_values)
                legend_strs{i} = sprintf('\\beta = %.2f', beta_values(i));
            end
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Asset profiles
            subplot(2, 1, 2);
            hold on;
            for i = 1:length(beta_values)
                plot(ages, a_profiles(:, i), 'LineWidth', 2);
            end
            title('Life Cycle Profile of Assets - Varying \beta');
            xlabel('Age');
            ylabel('Assets');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Adjust figure size
            set(gcf, 'Position', [100, 100, 800, 600]);
        end
        
        %% Plot parameter variations - Gamma
        function plot_gamma_variations(ages, c_profiles, a_profiles, gamma_values)
            % This function plots the life cycle profiles for different gamma values
            
            figure('Name', 'Gamma Variations');
            
            % Consumption profiles
            subplot(2, 1, 1);
            hold on;
            for i = 1:length(gamma_values)
                plot(ages, c_profiles(:, i), 'LineWidth', 2);
            end
            title('Life Cycle Profile of Consumption - Varying \gamma');
            xlabel('Age');
            ylabel('Consumption');
            
            % Create legend strings with gamma values
            legend_strs = cell(length(gamma_values), 1);
            for i = 1:length(gamma_values)
                legend_strs{i} = sprintf('\\gamma = %.2f', gamma_values(i));
            end
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Asset profiles
            subplot(2, 1, 2);
            hold on;
            for i = 1:length(gamma_values)
                plot(ages, a_profiles(:, i), 'LineWidth', 2);
            end
            title('Life Cycle Profile of Assets - Varying \gamma');
            xlabel('Age');
            ylabel('Assets');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Adjust figure size
            set(gcf, 'Position', [100, 100, 800, 600]);
        end
        
        %% Plot heat map for average wealth
        function plot_wealth_heatmap(beta_values, gamma_values, avg_wealth)
            % This function plots a heat map of average wealth for different beta and gamma combinations
            
            figure('Name', 'Average Wealth Heat Map');
            
            % Create heat map
            imagesc(gamma_values, beta_values, avg_wealth);
            colorbar;
            title('Average Wealth for Different \beta and \gamma Combinations');
            xlabel('\gamma (Risk Aversion)');
            ylabel('\beta (Discount Factor)');
            set(gca, 'YDir', 'normal'); % Higher beta at top
            
            % Add value labels to the heat map
            textStrings = num2str(avg_wealth(:), '%.2f');
            textStrings = strtrim(cellstr(textStrings));
            [x, y] = meshgrid(1:length(gamma_values), 1:length(beta_values));
            hStrings = text(x(:), y(:), textStrings(:), ...
                'HorizontalAlignment', 'center', ...
                'Color', 'white', 'FontWeight', 'bold');
            
            % Adjust x and y tick labels
            xticks(1:length(gamma_values));
            yticks(1:length(beta_values));
            xticklabels(arrayfun(@(x) sprintf('%.2f', x), gamma_values, 'UniformOutput', false));
            yticklabels(arrayfun(@(x) sprintf('%.2f', x), beta_values, 'UniformOutput', false));
            
            % Use a colormap that shows the variations well
            colormap('jet');
            
            % Adjust figure size
            set(gcf, 'Position', [100, 100, 600, 500]);
        end
        
        %% Plot age-specific income profile (Gt)
        function plot_income_profile(par)
            % This function plots the age-specific income profile (Gt)
            
            figure('Name', 'Age-Specific Income Profile');
            
            % Plot Gt
            plot(0:(length(par.Gt)-1), par.Gt, 'b-', 'LineWidth', 2);
            title('Age-Specific Average Income Profile (G_t)');
            xlabel('Age');
            ylabel('Average Income');
            grid on;
            
            % Mark retirement age
            hold on;
            if par.tr <= length(par.Gt)
                plot([par.tr, par.tr], [0, max(par.Gt)*1.1], 'r--', 'LineWidth', 1.5);
                text(par.tr+1, max(par.Gt)*0.9, 'Retirement', 'Color', 'r');
            end
            
            % Adjust y-axis to start from 0
            ylim([0, max(par.Gt)*1.1]);
            
            % Adjust figure size
            set(gcf, 'Position', [100, 100, 800, 400]);
        end
        
        %% Comprehensive plotting function for all required plots
        function plot_all(par, sol, sim, param_results)
            % This function creates all required plots for Problem 1
            
            fprintf('Creating all plots for the analysis...\n');
            
            % 1. Plot age-specific income profile (Gt)
            my_graph.plot_income_profile(par);
            fprintf('- Age-specific income profile plot created\n');
            
            % 2. Plot policy functions
            my_graph.plot_policy_functions(par, sol);
            fprintf('- Policy function plots created\n');
            
            % 3. Plot life cycle profiles
            my_graph.plot_lifecycle_profiles(sim);
            fprintf('- Life cycle profile plots created\n');
            
            % 4. Plot results for different beta values
            my_graph.plot_beta_variations(sim.ages, param_results.c_profiles_beta, ...
                param_results.a_profiles_beta, param_results.beta_values);
            fprintf('- Beta variation plots created\n');
            
            % 5. Plot results for different gamma values
            my_graph.plot_gamma_variations(sim.ages, param_results.c_profiles_gamma, ...
                param_results.a_profiles_gamma, param_results.gamma_values);
            fprintf('- Gamma variation plots created\n');
            
            % 6. Plot heat map for average wealth
            my_graph.plot_wealth_heatmap(param_results.beta_values, param_results.gamma_values, ...
                param_results.avg_wealth);
            fprintf('- Heat map for average wealth created\n');
            
            fprintf('All plots created successfully.\n');
        end
    end
end
