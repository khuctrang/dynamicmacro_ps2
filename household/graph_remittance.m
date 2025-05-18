%% File Info.

%{

    graph_remittance.m
    ------------------
    This code plots the results for the extended model with remittances.

%}

%% Graph Remittance class.

classdef graph_remittance
    methods(Static)
        %% Plot policy functions
        function plot_policy_functions(par, sol)
            % This function plots the consumption and asset policy functions for different remittance levels
            
            % Select ages to plot (e.g., early, middle, and late working life)
            ages_to_plot = [5, 20, 40, 60];
            
            % Select a middle income state
            y_idx = ceil(par.ylen / 2);
            
            % Select remittance states: none, medium, high
            r_states = [1, ceil(par.rlen/2), par.rlen];
            r_labels = {'No Remittance', 'Medium Remittance', 'High Remittance'};
            
            % Create figure for consumption policy by remittance level
            figure('Name', 'Consumption Policy Functions by Remittance Level');
            
            % Plot for middle age (e.g., 40)
            age = 40;
            age_idx = age + 1; % Convert to 1-indexed
            
            for i = 1:length(r_states)
                r_idx = r_states(i);
                subplot(1, length(r_states), i);
                
                plot(par.agrid, sol.c(:, age_idx, y_idx, r_idx), 'LineWidth', 2);
                
                title(sprintf('Age %d, %s', age, r_labels{i}));
                xlabel('Current Assets (a_t)');
                ylabel('Consumption (c_t)');
                grid on;
            end
            
            % Create figure for consumption policy by age
            figure('Name', 'Consumption Policy Functions by Age');
            
            % Fixed remittance level (medium)
            r_idx = ceil(par.rlen/2);
            
            for i = 1:length(ages_to_plot)
                age = ages_to_plot(i);
                age_idx = age + 1; % Convert to 1-indexed
                
                if age_idx <= par.T
                    subplot(2, 2, i);
                    plot(par.agrid, sol.c(:, age_idx, y_idx, r_idx), 'LineWidth', 2);
                    
                    title(sprintf('Age %d', age));
                    xlabel('Current Assets (a_t)');
                    ylabel('Consumption (c_t)');
                    grid on;
                end
            end
            
            % Create figure for asset policy by remittance level
            figure('Name', 'Asset Policy Functions by Remittance Level');
            
            % Plot for middle age (e.g., 40)
            age = 40;
            age_idx = age + 1; % Convert to 1-indexed
            
            for i = 1:length(r_states)
                r_idx = r_states(i);
                subplot(1, length(r_states), i);
                
                plot(par.agrid, sol.a(:, age_idx, y_idx, r_idx), 'LineWidth', 2);
                hold on;
                plot(par.agrid, par.agrid, 'k--', 'LineWidth', 1); % 45-degree line
                
                title(sprintf('Age %d, %s', age, r_labels{i}));
                xlabel('Current Assets (a_t)');
                ylabel('Next Period Assets (a_{t+1})');
                legend('Policy Function', '45-degree line', 'Location', 'best');
                grid on;
            end
        end
        
        %% Plot life cycle profiles
        function plot_lifecycle_profiles(sim)
            % This function plots the life cycle profiles of consumption, income, assets, and remittances
            
            % Extract data
            ages = sim.ages;
            c_profile = sim.c_profile;
            a_profile = sim.a_profile;
            y_profile = sim.y_profile;
            r_profile = sim.r_profile;
            remit_freq = sim.remit_freq;
            
            % Figure for consumption, income, and remittances
            figure('Name', 'Life Cycle Profiles with Remittances');
            
            % Plot consumption, income, and total income (including remittances)
            subplot(3, 1, 1);
            plot(ages, c_profile, 'b-', 'LineWidth', 2);
            hold on;
            plot(ages, y_profile, 'g--', 'LineWidth', 2);
            plot(ages, y_profile + r_profile, 'r-.', 'LineWidth', 2);
            title('Life Cycle Profile of Consumption and Income');
            xlabel('Age');
            ylabel('Value');
            legend('Consumption', 'Income (excl. remittances)', 'Total Income (incl. remittances)', 'Location', 'best');
            grid on;
            
            % Plot assets
            subplot(3, 1, 2);
            plot(ages, a_profile, 'r-', 'LineWidth', 2);
            title('Life Cycle Profile of Assets');
            xlabel('Age');
            ylabel('Assets');
            grid on;
            
            % Plot remittances and frequency
            subplot(3, 1, 3);
            yyaxis left;
            bar(ages, r_profile, 0.5, 'FaceColor', [0.8, 0.8, 0.8]);
            ylabel('Remittance Amount');
            
            yyaxis right;
            plot(ages, remit_freq, 'r-', 'LineWidth', 2);
            ylabel('Probability of Receiving');
            
            title('Life Cycle Profile of Remittances');
            xlabel('Age');
            legend('Amount', 'Frequency', 'Location', 'best');
            grid on;
            
            % Adjust figure size
            set(gcf, 'Position', [100, 100, 800, 800]);
        end
        
        %% Plot parameter variations - Beta
        function plot_beta_variations(ages, c_profiles, a_profiles, r_profiles, beta_values)
            % This function plots the life cycle profiles for different beta values
            
            figure('Name', 'Beta Variations with Remittances');
            
            % Consumption profiles
            subplot(3, 1, 1);
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
            subplot(3, 1, 2);
            hold on;
            for i = 1:length(beta_values)
                plot(ages, a_profiles(:, i), 'LineWidth', 2);
            end
            title('Life Cycle Profile of Assets - Varying \beta');
            xlabel('Age');
            ylabel('Assets');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Remittance impact (showing how remittances interact with different betas)
            subplot(3, 1, 3);
            hold on;
            for i = 1:length(beta_values)
                plot(ages, r_profiles(:, i), 'LineWidth', 2);
            end
            title('Life Cycle Profile of Remittances - Varying \beta');
            xlabel('Age');
            ylabel('Remittance Amount');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Adjust figure size
            set(gcf, 'Position', [100, 100, 800, 800]);
        end
        
        %% Plot parameter variations - Gamma
        function plot_gamma_variations(ages, c_profiles, a_profiles, r_profiles, gamma_values)
            % This function plots the life cycle profiles for different gamma values
            
            figure('Name', 'Gamma Variations with Remittances');
            
            % Consumption profiles
            subplot(3, 1, 1);
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
            subplot(3, 1, 2);
            hold on;
            for i = 1:length(gamma_values)
                plot(ages, a_profiles(:, i), 'LineWidth', 2);
            end
            title('Life Cycle Profile of Assets - Varying \gamma');
            xlabel('Age');
            ylabel('Assets');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Remittance impact
            subplot(3, 1, 3);
            hold on;
            for i = 1:length(gamma_values)
                plot(ages, r_profiles(:, i), 'LineWidth', 2);
            end
            title('Life Cycle Profile of Remittances - Varying \gamma');
            xlabel('Age');
            ylabel('Remittance Amount');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Adjust figure size
            set(gcf, 'Position', [100, 100, 800, 800]);
        end
        
        %% Plot heat map for average wealth
        function plot_wealth_heatmap(beta_values, gamma_values, avg_wealth)
            % This function plots a heat map of average wealth for different beta and gamma combinations
            
            figure('Name', 'Average Wealth Heat Map with Remittances');
            
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
        
        %% Plot comparison between baseline and remittance model
        function plot_comparison(comparison)
            % This function plots a comparison between the baseline model and the remittance model
            
            % Extract data
            ages = comparison.ages;
            c_base = comparison.c_base;
            c_remit = comparison.c_remit;
            a_base = comparison.a_base;
            a_remit = comparison.a_remit;
            y_base = comparison.y_base;
            y_remit = comparison.y_remit;
            y_total_remit = comparison.y_total_remit;
            r_profile = comparison.r_profile;
            c_y_ratio_base = comparison.c_y_ratio_base;
            c_y_ratio_remit = comparison.c_y_ratio_remit;
            
            % Figure for consumption comparison
            figure('Name', 'Consumption Comparison');
            
            % Plot consumption levels
            subplot(2, 1, 1);
            plot(ages, c_base, 'b-', 'LineWidth', 2);
            hold on;
            plot(ages, c_remit, 'r--', 'LineWidth', 2);
            title('Consumption: Baseline vs. Remittance Model');
            xlabel('Age');
            ylabel('Consumption');
            legend('Baseline', 'With Remittances', 'Location', 'best');
            grid on;
            
            % Plot consumption difference
            subplot(2, 1, 2);
            bar(ages, comparison.c_pct_diff);
            title('Percentage Difference in Consumption');
            xlabel('Age');
            ylabel('% Difference');
            grid on;
            
            % Figure for asset comparison
            figure('Name', 'Asset Comparison');
            
            % Plot asset levels
            subplot(2, 1, 1);
            plot(ages, a_base, 'b-', 'LineWidth', 2);
            hold on;
            plot(ages, a_remit, 'r--', 'LineWidth', 2);
            title('Assets: Baseline vs. Remittance Model');
            xlabel('Age');
            ylabel('Assets');
            legend('Baseline', 'With Remittances', 'Location', 'best');
            grid on;
            
            % Plot asset difference
            subplot(2, 1, 2);
            bar(ages, comparison.a_pct_diff);
            title('Percentage Difference in Assets');
            xlabel('Age');
            ylabel('% Difference');
            grid on;
            
            % Figure for income comparison
            figure('Name', 'Income Comparison');
            
            % Plot income components
            plot(ages, y_base, 'b-', 'LineWidth', 2);
            hold on;
            plot(ages, y_remit, 'g--', 'LineWidth', 2);
            plot(ages, y_total_remit, 'r-.', 'LineWidth', 2);
            plot(ages, r_profile, 'k:', 'LineWidth', 2);
            title('Income Components');
            xlabel('Age');
            ylabel('Income');
            legend('Baseline Income', 'Income (excl. remittances)', 'Total Income (incl. remittances)', 'Remittances', 'Location', 'best');
            grid on;
            
            % Figure for consumption-to-income ratio
            figure('Name', 'Consumption-to-Income Ratio');
            
            plot(ages, c_y_ratio_base, 'b-', 'LineWidth', 2);
            hold on;
            plot(ages, c_y_ratio_remit, 'r--', 'LineWidth', 2);
            title('Consumption-to-Income Ratio');
            xlabel('Age');
            ylabel('C/Y Ratio');
            legend('Baseline', 'With Remittances', 'Location', 'best');
            grid on;
            
            % Adjust figure size for all figures
            figHandles = findall(0, 'Type', 'figure');
            for i = 1:length(figHandles)
                set(figHandles(i), 'Position', [100, 100, 800, 600]);
            end
        end
        
        %% Plot remittance parameter sensitivity
        function plot_remittance_sensitivity(ages, results)
            % This function plots sensitivity analysis for remittance parameters
            
            % 1. Sensitivity to eta (persistence of remittance process)
            figure('Name', 'Sensitivity to Remittance Persistence');
            
            % Consumption profiles
            subplot(2, 1, 1);
            hold on;
            for i = 1:length(results.eta_values)
                plot(ages, results.eta_c_profiles(:, i), 'LineWidth', 2);
            end
            title('Consumption - Varying Remittance Persistence (\eta)');
            xlabel('Age');
            ylabel('Consumption');
            
            % Create legend strings
            legend_strs = cell(length(results.eta_values), 1);
            for i = 1:length(results.eta_values)
                legend_strs{i} = sprintf('\\eta = %.2f', results.eta_values(i));
            end
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Asset profiles
            subplot(2, 1, 2);
            hold on;
            for i = 1:length(results.eta_values)
                plot(ages, results.eta_a_profiles(:, i), 'LineWidth', 2);
            end
            title('Assets - Varying Remittance Persistence (\eta)');
            xlabel('Age');
            ylabel('Assets');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % 2. Sensitivity to correlation between income and remittance shocks
            figure('Name', 'Sensitivity to Income-Remittance Correlation');
            
            % Consumption profiles
            subplot(2, 1, 1);
            hold on;
            for i = 1:length(results.corr_values)
                plot(ages, results.corr_c_profiles(:, i), 'LineWidth', 2);
            end
            title('Consumption - Varying Income-Remittance Correlation');
            xlabel('Age');
            ylabel('Consumption');
            
            % Create legend strings
            legend_strs = cell(length(results.corr_values), 1);
            for i = 1:length(results.corr_values)
                legend_strs{i} = sprintf('Corr = %.2f', results.corr_values(i));
            end
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Asset profiles
            subplot(2, 1, 2);
            hold on;
            for i = 1:length(results.corr_values)
                plot(ages, results.corr_a_profiles(:, i), 'LineWidth', 2);
            end
            title('Assets - Varying Income-Remittance Correlation');
            xlabel('Age');
            ylabel('Assets');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % 3. Sensitivity to probability of receiving remittances
            figure('Name', 'Sensitivity to Remittance Probability');
            
            % Consumption profiles
            subplot(2, 1, 1);
            hold on;
            for i = 1:length(results.prob_scale)
                plot(ages, results.prob_c_profiles(:, i), 'LineWidth', 2);
            end
            title('Consumption - Varying Remittance Probability');
            xlabel('Age');
            ylabel('Consumption');
            
            % Create legend strings
            legend_strs = cell(length(results.prob_scale), 1);
            for i = 1:length(results.prob_scale)
                legend_strs{i} = sprintf('Prob Scale = %.2f', results.prob_scale(i));
            end
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Asset profiles
            subplot(2, 1, 2);
            hold on;
            for i = 1:length(results.prob_scale)
                plot(ages, results.prob_a_profiles(:, i), 'LineWidth', 2);
            end
            title('Assets - Varying Remittance Probability');
            xlabel('Age');
            ylabel('Assets');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % 4. Sensitivity to remittance amount
            figure('Name', 'Sensitivity to Remittance Amount');
            
            % Consumption profiles
            subplot(2, 1, 1);
            hold on;
            for i = 1:length(results.amount_scale)
                plot(ages, results.amount_c_profiles(:, i), 'LineWidth', 2);
            end
            title('Consumption - Varying Remittance Amount');
            xlabel('Age');
            ylabel('Consumption');
            
            % Create legend strings
            legend_strs = cell(length(results.amount_scale), 1);
            for i = 1:length(results.amount_scale)
                legend_strs{i} = sprintf('Amount Scale = %.2f', results.amount_scale(i));
            end
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Asset profiles
            subplot(2, 1, 2);
            hold on;
            for i = 1:length(results.amount_scale)
                plot(ages, results.amount_a_profiles(:, i), 'LineWidth', 2);
            end
            title('Assets - Varying Remittance Amount');
            xlabel('Age');
            ylabel('Assets');
            legend(legend_strs, 'Location', 'best');
            grid on;
            
            % Adjust figure size for all figures
            figHandles = findall(0, 'Type', 'figure');
            for i = 1:length(figHandles)
                set(figHandles(i), 'Position', [100, 100, 800, 600]);
            end
        end
        
        %% Comprehensive plotting function for all required plots
        function plot_all(par, sol, sim, param_results, comparison, sensitivity_results)
            % This function creates all required plots for the remittance model analysis
            
            fprintf('Creating all plots for the remittance model analysis...\n');
            
            % 1. Plot policy functions
            graph_remittance.plot_policy_functions(par, sol);
            fprintf('- Policy function plots created\n');
            
            % 2. Plot life cycle profiles
            graph_remittance.plot_lifecycle_profiles(sim);
            fprintf('- Life cycle profile plots created\n');
            
            % 3. Plot results for different beta values
            graph_remittance.plot_beta_variations(sim.ages, param_results.c_profiles_beta, ...
                param_results.a_profiles_beta, param_results.r_profiles_beta, param_results.beta_values);
            fprintf('- Beta variation plots created\n');
            
            % 4. Plot results for different gamma values
            graph_remittance.plot_gamma_variations(sim.ages, param_results.c_profiles_gamma, ...
                param_results.a_profiles_gamma, param_results.r_profiles_gamma, param_results.gamma_values);
            fprintf('- Gamma variation plots created\n');
            
            % 5. Plot heat map for average wealth
            graph_remittance.plot_wealth_heatmap(param_results.beta_values, param_results.gamma_values, ...
                param_results.avg_wealth);
            fprintf('- Heat map for average wealth created\n');
            
            % 6. Plot comparison with baseline model
            if ~isempty(comparison)
                graph_remittance.plot_comparison(comparison);
                fprintf('- Comparison with baseline model plots created\n');
            end
            
            % 7. Plot sensitivity analysis for remittance parameters
            if ~isempty(sensitivity_results)
                graph_remittance.plot_remittance_sensitivity(sim.ages, sensitivity_results);
                fprintf('- Remittance parameter sensitivity plots created\n');
            end
            
            fprintf('All plots created successfully.\n');
        end
    end
end