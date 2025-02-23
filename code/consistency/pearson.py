import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

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
                    title_lower = title.lower()
                    if 'c' in title_lower:  
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

        # Calculate Pearson correlation and create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # First subplot: Data Overlay (same as before)
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
        
        # Second subplot: Pearson Correlation Analysis
        pearson_results = {}
        all_values1 = []
        all_values2 = []
        
        for cat_key, cat_data in categories.items():
            if len(cat_data['values1']) > 0:
                values1 = np.array(cat_data['values1'])
                values2 = np.array(cat_data['values2'])
                
                # Calculate Pearson correlation and p-value
                r, p = stats.pearsonr(values1, values2)
                pearson_results[cat_key] = {
                    'r': r,
                    'p_value': p,
                    'n_samples': len(values1)
                }
                
                # Store values for overall calculation
                all_values1.extend(values1)
                all_values2.extend(values2)
                
                # Plot correlation
                ax2.scatter(values1, values2, 
                          color=cat_data['color'],
                          label=f"{cat_data['label']} (r={r:.2f}, p={p:.3f})",
                          alpha=0.6)
        
        # Calculate overall correlation
        overall_r, overall_p = stats.pearsonr(all_values1, all_values2)
        
        ax2.set_xlabel('Elec Values')
        ax2.set_ylabel('3D Values')
        ax2.set_title('Pearson Correlation Analysis')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Add text annotations for correlation interpretation
        correlation_interpretation = """
        Correlation Interpretation:
        0.00-0.19: Very weak
        0.20-0.39: Weak
        0.40-0.59: Moderate
        0.60-0.79: Strong
        0.80-1.00: Very strong
        
        P-value Interpretation:
        p < 0.05: Statistically significant
        p ≥ 0.05: Not statistically significant
        """
        
        ax2.annotate(f'Overall Correlation:\nr = {overall_r:.2f}\np = {overall_p:.3f}\n\n{correlation_interpretation}', 
                    xy=(0.05, 0.95), xycoords='axes fraction',
                    verticalalignment='top', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Adjust layout and save
        plt.tight_layout()
        fig.subplots_adjust(right=0.85)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save detailed results
        results_data = []
        for cat_key, cat_data in categories.items():
            if len(cat_data['values1']) > 0:
                r = pearson_results[cat_key]['r']
                p = pearson_results[cat_key]['p_value']
                for title, val1, val2 in zip(cat_data['titles'], 
                                           cat_data['values1'], 
                                           cat_data['values2']):
                    results_data.append({
                        'Title': title,
                        'Category': cat_data['label'],
                        'Elec_P1': val1,
                        '3D_P1': val2,
                        'Category_R': r,
                        'Category_P_Value': p
                    })
        
        results_df = pd.DataFrame(results_data)
        results_path = output_path.rsplit('.', 1)[0] + '_correlation_results.xlsx'
        results_df.to_excel(results_path, index=False)
        
        # Print summary
        print("\nPearson Correlation Analysis completed.")
        print(f"\nOverall Correlation: r = {overall_r:.20f}, p = {overall_p:.20f}")
        for cat_key, results in pearson_results.items():
            print(f"\n{categories[cat_key]['label']}:")
            print(f"Number of cases: {results['n_samples']}")
            print(f"Correlation coefficient: {results['r']:.20f}")
            print(f"P-value: {results['p_value']:.20f}")
        
        return {
            'overall_correlation': {'r': overall_r, 'p': overall_p},
            'category_correlations': pearson_results
        }
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None

def main():
    # Set paths
    file1_path = "C:/Users/cherr/anaconda3/envs/STCformer/STCFormer-main/analysis_results.xlsx"
    file2_path = "C:/Users/cherr/anaconda3/envs/STCformer/STCFormer-main/analysis_results_hpe_v3.xlsx"
    output_path = "pearson_p1.png"
    
    # Run the analysis
    results = process_excel_comparison(file1_path, file2_path, output_path)

if __name__ == "__main__":
    main()