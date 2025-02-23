import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def get_category_color(title):
    """
    Determine color based on category:
    - sa: warm red
    - sb: warm orange
    - ha: cool blue
    - hb: cool purple
    """
    if 's' in title.lower():
        if 'a' in title.lower():
            return '#FF4444'  # warm red
        elif 'b' in title.lower():
            return '#FFA500'  # warm orange
    elif 'h' in title.lower():
        if 'a' in title.lower():
            return '#4444FF'  # cool blue
        elif 'b' in title.lower():
            return '#8A2BE2'  # cool purple
    return None

def get_category_name(title):
    """Get full category name from title"""
    if 's' in title.lower():
        if 'a' in title.lower():
            return 'Spasticity (45°)'
        elif 'b' in title.lower():
            return 'Spasticity (90°)'
    elif 'h' in title.lower():
        if 'a' in title.lower():
            return 'Healthy (45°)'
        elif 'b' in title.lower():
            return 'Healthy (90°)'
    return None

def process_excel_comparison(file1_path, file2_path, output_path):
    try:
        # Read both Excel files and convert index to string
        df1 = pd.read_excel(file1_path, index_col=0)
        df2 = pd.read_excel(file2_path, index_col=0)
        
        df1.index = df1.index.astype(str)
        df2.index = df2.index.astype(str)
        
        common_titles = set(df1.columns).intersection(set(df2.columns))
        if not common_titles:
            raise ValueError("No matching titles found between the two files")
        
        # Initialize dictionaries to store data for each category
        categories = {
            'sa': {'titles': [], 'values1': [], 'values2': [], 'color': '#FF4444', 'label': 'Spasticity (45°)'},
            'sb': {'titles': [], 'values1': [], 'values2': [], 'color': '#FFA500', 'label': 'Spasticity (90°)'},
            'ha': {'titles': [], 'values1': [], 'values2': [], 'color': '#4444FF', 'label': 'Healthy (45°)'},
            'hb': {'titles': [], 'values1': [], 'values2': [], 'color': '#8A2BE2', 'label': 'Healthy (90°)'}
        }
        
        # Process each title and categorize data
        for title in common_titles:
            try:
                p_value1 = df1[title].loc['p1']
                p_value2 = df2[title].loc['p1']
                
                if pd.notnull(p_value1) and pd.notnull(p_value2):
                    # Determine category
                    title_lower = title.lower()
                    if 'c' in title_lower:  # Skip if contains 'c'
                        continue
                        
                    category = None
                    if 's' in title_lower:
                        if 'a' in title_lower:
                            category = 'sa'
                        elif 'b' in title_lower:
                            category = 'sb'
                    elif 'h' in title_lower:
                        if 'a' in title_lower:
                            category = 'ha'
                        elif 'b' in title_lower:
                            category = 'hb'
                    
                    if category:
                        categories[category]['titles'].append(title)
                        categories[category]['values1'].append(float(p_value1))
                        categories[category]['values2'].append(float(p_value2))
                        
            except (KeyError, ValueError) as e:
                print(f"Skipping {title} due to: {str(e)}")
                continue
        
        # Convert lists to numpy arrays
        for cat in categories.values():
            if cat['values1']:
                cat['values1'] = np.array(cat['values1'])
                cat['values2'] = np.array(cat['values2'])
        
        # Create figure
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))  # 更高的圖形

        
        # First subplot: Data Overlay
        current_index = 0
        for cat_key, cat_data in categories.items():
            if len(cat_data['values1']) > 0:
                indices = range(current_index, current_index + len(cat_data['values1']))
                ax1.scatter(indices, cat_data['values1'], 
                          label=f"elec - {cat_data['label']}", 
                          color=cat_data['color'])
                ax1.scatter(indices, cat_data['values2'], 
                          label=f"3D - {cat_data['label']}", 
                          color=cat_data['color'], marker='x')
                current_index += len(cat_data['values1'])
        
        ax1.set_title('P1 Values Comparison')
        ax1.set_xlabel('Measurement Index')
        ax1.set_ylabel('P1 Value')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Second subplot: Bland-Altman Plot
        all_means = []
        all_diffs = []
        
        for cat_key, cat_data in categories.items():
            if len(cat_data['values1']) > 0:
                means = (cat_data['values1'] + cat_data['values2']) / 2
                diffs = cat_data['values1'] - cat_data['values2']
                ax2.scatter(means, diffs, 
                          color=cat_data['color'], 
                          label=cat_data['label'], 
                          alpha=0.6)
                all_means.extend(means)
                all_diffs.extend(diffs)
        
        # Calculate overall statistics for Bland-Altman
        mean_diff = np.mean(all_diffs)
        std_diff = np.std(all_diffs)
        lower_limit = mean_diff - 1.96 * std_diff
        upper_limit = mean_diff + 1.96 * std_diff
        
        # Add lines for mean and limits
        ax2.axhline(y=mean_diff, color='k', linestyle='--')
        ax2.axhline(y=lower_limit, color='k', linestyle=':')
        ax2.axhline(y=upper_limit, color='k', linestyle=':')
        
        ax2.set_xlabel('Means')
        ax2.set_ylabel('Difference')
        ax2.set_title('Bland-Altman Plot')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add text annotations
        ax2.annotate(f'Mean Difference: {mean_diff:.4f}\n'
                    f'Std of Difference: {std_diff:.4f}\n'
                    f'Lower Limit: {lower_limit:.4f}\n'
                    f'Upper Limit: {upper_limit:.4f}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Adjust layout to prevent legend overlap
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)  # Make room for legend
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results to Excel
        results_data = []
        for cat_key, cat_data in categories.items():
            for title, val1, val2 in zip(cat_data['titles'], 
                                       cat_data['values1'], 
                                       cat_data['values2']):
                results_data.append({
                    'Title': title,
                    'Category': cat_data['label'],
                    'Elec_P1': val1,
                    '3D_P1': val2,
                    'Difference': val1 - val2
                })
        
        results_df = pd.DataFrame(results_data)
        results_path = output_path.rsplit('.', 1)[0] + '_results.xlsx'
        results_df.to_excel(results_path, index=False)
        
        # Print summary
        print("\nAnalysis completed.")
        for cat_key, cat_data in categories.items():
            if len(cat_data['values1']) > 0:
                mean_diff_cat = np.mean(cat_data['values1'] - cat_data['values2'])
                std_diff_cat = np.std(cat_data['values1'] - cat_data['values2'])
                print(f"\n{cat_data['label']}:")
                print(f"Number of cases: {len(cat_data['values1'])}")
                print(f"Mean difference: {mean_diff_cat:.4f}")
                print(f"Standard deviation: {std_diff_cat:.4f}")
        
        return {
            'overall': {
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'lower_limit': lower_limit,
                'upper_limit': upper_limit,
            },
            'categories': {key: {
                'n_matches': len(cat['values1']),
                'mean_difference': np.mean(cat['values1'] - cat['values2']) if len(cat['values1']) > 0 else None,
                'std_difference': np.std(cat['values1'] - cat['values2']) if len(cat['values1']) > 0 else None
            } for key, cat in categories.items()}
        }
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None

def main():
    # Set paths
    file1_path = "C:/Users/cherr/anaconda3/envs/STCformer/STCFormer-main/analysis_results.xlsx"
    file2_path = "C:/Users/cherr/anaconda3/envs/STCformer/STCFormer-main/analysis_results_hpe_v3.xlsx"
    # file1_path = "C:/Users/cherr/anaconda3/envs/STCformer/STCFormer-main/exponential_results_elec.xlsx"
    # file2_path = "C:/Users/cherr/anaconda3/envs/STCformer/STCFormer-main/exponential_results_3d.xlsx"
    output_path = "C:/Users/cherr/anaconda3/envs/STCformer/STCFormer-main/bland_altman_plot_p1.png"
    
    # Run the analysis
    results = process_excel_comparison(file1_path, file2_path, output_path)

if __name__ == "__main__":
    main()