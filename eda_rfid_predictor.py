"""
WESAD EDA to RFID Read Range Prediction System
Body-mounted UHF RFID (865-928 MHz)

This system:
1. Loads and processes WESAD dataset
2. Extracts EDA features per emotional state
3. Predicts RFID read range using physics-based and ML models
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set visualization style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class WESADLoader:
    """Load and process WESAD dataset"""
    
    EMOTION_LABELS = {
        0: 'Transient',
        1: 'Baseline',
        2: 'Stress',
        3: 'Amusement',
        4: 'Meditation'
    }
    
    def __init__(self, data_path):
        """
        Initialize WESAD loader
        
        Args:
            data_path: Path to WESAD dataset directory containing subject folders
        """
        self.data_path = Path(data_path)
        self.subjects = [f'S{i}' for i in range(2, 18) if i not in [1, 12]]
        
    def load_subject(self, subject_id):
        """Load data for a single subject"""
        pkl_file = self.data_path / subject_id / f'{subject_id}.pkl'
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            return data
        except FileNotFoundError:
            print(f"File not found: {pkl_file}")
            return None
    
    def extract_eda_features(self, subject_data):
        """
        Extract EDA features per emotional state from E4 wrist data
        
        Returns:
            DataFrame with EDA features per emotional state
        """
        # Get E4 wrist EDA (sampled at 4 Hz, already in microsiemens)
        eda = subject_data['signal']['wrist']['EDA']
        
        # Get labels (sampled at 700 Hz, need to downsample to match EDA)
        labels = subject_data['label']
        
        # Downsample labels from 700 Hz to 4 Hz (E4 EDA sampling rate)
        downsample_factor = 700 // 4  # 175
        labels_downsampled = labels[::downsample_factor]
        
        # Ensure same length
        min_len = min(len(eda), len(labels_downsampled))
        eda = eda[:min_len]
        labels_downsampled = labels_downsampled[:min_len]
        
        features = []
        
        for label_id in [1, 2, 3, 4]:  # Baseline, Stress, Amusement, Meditation
            mask = labels_downsampled == label_id
            eda_segment = eda[mask]
            
            if len(eda_segment) > 0:
                # Calculate EDA features
                features.append({
                    'emotion': self.EMOTION_LABELS[label_id],
                    'label_id': label_id,
                    'eda_mean': np.mean(eda_segment),
                    'eda_std': np.std(eda_segment),
                    'eda_min': np.min(eda_segment),
                    'eda_max': np.max(eda_segment),
                    'eda_range': np.max(eda_segment) - np.min(eda_segment),
                    'eda_median': np.median(eda_segment),
                    'n_samples': len(eda_segment)
                })
        
        return pd.DataFrame(features)
    
    def load_all_subjects(self):
        """Load EDA features for all subjects"""
        all_features = []
        
        for subject_id in self.subjects:
            print(f"Loading {subject_id}...")
            data = self.load_subject(subject_id)
            
            if data is not None:
                features = self.extract_eda_features(data)
                features['subject'] = subject_id
                all_features.append(features)
        
        return pd.concat(all_features, ignore_index=True)


class RFIDRangePredictor:
    """Predict RFID read range based on EDA using physics and ML models"""
    
    def __init__(self, frequency_mhz=896.5, baseline_range_m=0.65):
        """
        Initialize RFID range predictor
        
        Args:
            frequency_mhz: Center frequency in MHz (default: 896.5 MHz, center of 865-928 MHz)
            baseline_range_m: Baseline read range in meters (default: 0.65m, middle of 0.3-1.0m)
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
        self.wavelength = 3e8 / self.frequency_hz  # Speed of light / frequency
        self.baseline_range = baseline_range_m
        
        # Typical UHF RFID parameters for body-mounted tags
        self.reader_power_w = 0.5  # 500 mW (typical UHF RFID reader)
        self.reader_gain_db = 6  # Reader antenna gain
        self.tag_gain_db = 2  # Tag antenna gain (body-mounted)
        self.tag_sensitivity_dbm = -18  # Tag activation threshold
        
    def friis_equation(self, eda_factor=1.0):
        """
        Calculate RFID read range using modified Friis transmission equation
        
        Args:
            eda_factor: Modifier based on EDA (1.0 = baseline, >1 = enhanced, <1 = reduced)
        
        Returns:
            Read range in meters
        """
        # Convert to linear scale
        reader_gain_linear = 10 ** (self.reader_gain_db / 10)
        tag_gain_linear = 10 ** (self.tag_gain_db / 10)
        tag_sensitivity_w = 10 ** ((self.tag_sensitivity_dbm - 30) / 10)
        
        # Friis equation for RFID: d = (λ/4π) * sqrt(P_t * G_r * G_t / P_th)
        # Modified by EDA factor (affects body tissue absorption and coupling)
        
        range_m = (self.wavelength / (4 * np.pi)) * np.sqrt(
            (self.reader_power_w * reader_gain_linear * tag_gain_linear * eda_factor) / 
            tag_sensitivity_w
        )
        
        return range_m
    
    def eda_to_body_factor(self, eda_microsiemens, baseline_eda=2.0):
        """
        Convert EDA to body tissue factor affecting RFID propagation
        
        Higher EDA → Higher skin conductivity → More moisture → Higher permittivity
        → Better coupling but potentially more absorption
        
        Args:
            eda_microsiemens: EDA value in μS
            baseline_eda: Baseline EDA value in μS
        
        Returns:
            Factor affecting RFID range (typically 0.7 to 1.3)
        """
        # Normalized EDA ratio
        eda_ratio = eda_microsiemens / baseline_eda
        
        # Model: Moderate EDA increase (stress) can improve coupling
        # But very high EDA might increase absorption
        # This is a simplified quadratic model - adjust based on experiments
        
        if eda_ratio < 1.0:
            # Lower than baseline: reduced coupling
            factor = 0.7 + 0.3 * eda_ratio
        elif eda_ratio < 1.5:
            # Moderate increase: improved coupling
            factor = 1.0 + 0.2 * (eda_ratio - 1.0)
        else:
            # High increase: increased absorption starts to dominate
            factor = 1.1 - 0.1 * (eda_ratio - 1.5)
        
        return np.clip(factor, 0.6, 1.4)
    
    def predict_range_physics(self, eda_value, baseline_eda=2.0):
        """Predict read range using physics-based model"""
        body_factor = self.eda_to_body_factor(eda_value, baseline_eda)
        return self.friis_equation(body_factor)
    
    def generate_training_data(self, eda_features_df):
        """
        Generate training data by calculating theoretical ranges
        
        Args:
            eda_features_df: DataFrame with EDA features per emotion
        
        Returns:
            X (features), y (predicted ranges)
        """
        baseline_eda = eda_features_df[
            eda_features_df['emotion'] == 'Baseline'
        ]['eda_mean'].mean()
        
        X = eda_features_df[[
            'eda_mean', 'eda_std', 'eda_min', 
            'eda_max', 'eda_range'
        ]].values
        
        y = np.array([
            self.predict_range_physics(row['eda_mean'], baseline_eda)
            for _, row in eda_features_df.iterrows()
        ])
        
        return X, y, baseline_eda
    
    def train_ml_models(self, X, y):
        """Train machine learning models"""
        # Split into 70% train, 30% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"\nDataset split:")
        print(f"  Training set: {len(X_train)} samples (70%)")
        print(f"  Test set: {len(X_test)} samples (30%)")
        
        # Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # Random Forest
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # Evaluate on TEST set only
        print("\n=== Model Performance (on Test Set - 30%) ===")
        print("\nLinear Regression:")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, lr_pred)):.4f} m")
        print(f"  MAE:  {mean_absolute_error(y_test, lr_pred):.4f} m")
        print(f"  R²:   {r2_score(y_test, lr_pred):.4f}")
        
        print("\nRandom Forest:")
        print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, rf_pred)):.4f} m")
        print(f"  MAE:  {mean_absolute_error(y_test, rf_pred):.4f} m")
        print(f"  R²:   {r2_score(y_test, rf_pred):.4f}")
        
        return lr_model, rf_model, (X_test, y_test, lr_pred, rf_pred)


def visualize_results(eda_features_df, predictor, baseline_eda, test_results=None):
    """Create comprehensive visualizations"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. EDA Distribution by Emotion - Using KDE curves
    ax1 = axes[0, 0]
    emotion_order = ['Baseline', 'Meditation', 'Amusement', 'Stress']
    colors = sns.color_palette('Set2', len(emotion_order))
    
    for emotion, color in zip(emotion_order, colors):
        emotion_data = eda_features_df[eda_features_df['emotion'] == emotion]['eda_mean']
        sns.kdeplot(data=emotion_data, ax=ax1, label=emotion, linewidth=2.5, 
                   color=color, fill=True, alpha=0.3)
    
    ax1.set_title('EDA Distribution by Emotional State', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Mean EDA (μS)', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Predicted Range by Emotion - Using KDE curves
    ax2 = axes[0, 1]
    
    for emotion, color in zip(emotion_order, colors):
        emotion_data = eda_features_df[eda_features_df['emotion'] == emotion]
        emotion_ranges = [
            predictor.predict_range_physics(eda, baseline_eda) * 100  # Convert to cm
            for eda in emotion_data['eda_mean']
        ]
        sns.kdeplot(data=emotion_ranges, ax=ax2, label=emotion, linewidth=2.5,
                   color=color, fill=True, alpha=0.3)
    
    ax2.set_title('Friis RFID Read Range by Emotional State', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Friis Read Range (cm)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.axvline(x=65, color='red', linestyle='--', linewidth=2, label='Baseline (65 cm)')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. EDA vs Range Relationship - Using curves for each emotion
    # 3. EDA vs Range Scatter
    ax3 = axes[0, 2]
    emotions = eda_features_df['emotion'].unique()
    colors = sns.color_palette('Set2', len(emotions))
    
    for emotion, color in zip(emotion_order, colors):
        emotion_data = eda_features_df[eda_features_df['emotion'] == emotion]
        eda_vals = emotion_data['eda_mean'].values
        range_vals = [predictor.predict_range_physics(eda, baseline_eda) * 100 
                      for eda in eda_vals]
        ax3.scatter(eda_vals, range_vals, label=emotion, alpha=0.6, 
                   s=100, color=color, edgecolors='black', linewidth=1)
    
    ax3.set_title('EDA to Friis Read Range Relationship', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Mean EDA (μS)', fontsize=12)
    ax3.set_ylabel('Friis Read Range (cm)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. EDA to ML Predicted Read Range Relationship
    if test_results is not None:
        X_test, y_test, lr_pred, rf_pred = test_results
        
        # Extract EDA mean (first feature in X_test)
        eda_test_vals = X_test[:, 0]
        
        ax4 = axes[1, 1]
        
        # Sort by EDA values for line plot
        sort_idx_lr = np.argsort(eda_test_vals)
        eda_sorted_lr = eda_test_vals[sort_idx_lr]
        lr_pred_sorted = (lr_pred * 100)[sort_idx_lr]
        
        sort_idx_rf = np.argsort(eda_test_vals)
        eda_sorted_rf = eda_test_vals[sort_idx_rf]
        rf_pred_sorted = (rf_pred * 100)[sort_idx_rf]
        
        # Plot Linear Regression - line and scatter
        ax4.plot(eda_sorted_lr, lr_pred_sorted, color='steelblue', linewidth=2.5, 
                alpha=0.7, label='Linear Regression')
        ax4.scatter(eda_test_vals, lr_pred * 100, alpha=0.6, s=80, 
                   color='steelblue', edgecolors='black', linewidth=0.5)
        
        # Plot Random Forest - line and scatter
        ax4.plot(eda_sorted_rf, rf_pred_sorted, color='forestgreen', linewidth=2.5, 
                alpha=0.7, label='Random Forest')
        ax4.scatter(eda_test_vals, rf_pred * 100, alpha=0.6, s=80, 
                   color='forestgreen', edgecolors='black', linewidth=0.5)
        
        ax4.set_title('EDA to ML Predicted Read Range Relationship', 
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Mean EDA (μS)', fontsize=11)
        ax4.set_ylabel('Predicted Read Range (cm)', fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax5 = axes[1, 2]
        
        # Sort by actual values for line plot
        sort_idx = np.argsort(y_test)
        y_test_sorted = (y_test * 100)[sort_idx]
        rf_pred_sorted = (rf_pred * 100)[sort_idx]
        
        # Plot Random Forest - line and scatter
        ax5.plot(y_test_sorted, rf_pred_sorted, color='forestgreen', linewidth=2.5, 
                alpha=0.7, label='RF Trend')
        ax5.scatter(y_test * 100, rf_pred * 100, alpha=0.6, s=80, 
                   color='forestgreen', edgecolors='black', linewidth=0.5, label='Random Forest')
        
        # Perfect prediction line
        min_val = min(y_test.min(), rf_pred.min()) * 100
        max_val = max(y_test.max(), rf_pred.max()) * 100
        
        ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax5.set_title('Random Forest: Predicted vs Actual Range', 
                     fontsize=12, fontweight='bold')
        ax5.set_xlabel('Actual Read Range (cm)', fontsize=11)
        ax5.set_ylabel('Predicted Read Range (cm)', fontsize=11)
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3)
    
    # 6. Summary Statistics Table
    ax6 = axes[1, 0]
    ax6.axis('tight')
    ax6.axis('off')
    
    summary_data = []
    for emotion in emotion_order:
        emotion_data = eda_features_df[eda_features_df['emotion'] == emotion]
        eda_mean = emotion_data['eda_mean'].mean()
        range_mean = predictor.predict_range_physics(eda_mean, baseline_eda) * 100
        
        summary_data.append([
            emotion,
            f"{eda_mean:.2f}",
            f"{range_mean:.1f}",
            f"{len(emotion_data)}"
        ])
    
    table = ax6.table(cellText=summary_data,
                     colLabels=['Emotion', 'Avg EDA (μS)', 'Avg Friis Range (cm)', 'Samples'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.3, 0.25, 0.25, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('eda_rfid_analysis.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'eda_rfid_analysis.png'")
    plt.show()


# ============================================ MAIN EXECUTION =================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("WESAD EDA to RFID Read Range Prediction System")
    print("Body-mounted UHF RFID (866 MHz)")
    print("=" * 70)
    
    # STEP 1: Load WESAD data
    print("\n[STEP 1] Loading WESAD Dataset...")
    print("NOTE: Update 'data_path' to point to your WESAD dataset directory")
    print("Expected structure: data_path/S2/S2.pkl, data_path/S3/S3.pkl, etc.")
    
    # UPDATE THIS PATH to your WESAD dataset location
    data_path = "./WESAD"  # Change this to your WESAD directory
    
    loader = WESADLoader(data_path)
    
    # Try to load data (this will fail if path is wrong - that's expected)
    try:
        eda_features_df = loader.load_all_subjects()
        print(f"\nLoaded data from {len(eda_features_df['subject'].unique())} subjects")
        print(f"Total samples: {len(eda_features_df)}")
        
        # STEP 2: Initialize RFID predictor
        print("\n[STEP 2] Initializing RFID Range Predictor...")
        predictor = RFIDRangePredictor(
            frequency_mhz=866,  # Center of 865-928 MHz
            baseline_range_m=0.65  # 65 cm baseline
        )
        
        print(f"  Frequency: {predictor.frequency_mhz} MHz")
        print(f"  Wavelength: {predictor.wavelength*100:.2f} cm")
        print(f"  Baseline Range: {predictor.baseline_range*100:.1f} cm")
        
        # STEP 3: Generate training data using Friis equation
        print("\n[STEP 3] Generating Training Data (X = EDA, Y = Friis Read Range)...")
        X, y, baseline_eda = predictor.generate_training_data(eda_features_df)
        print(f"  X (EDA features): {X.shape[0]} samples × {X.shape[1]} features")
        print(f"  Y (Friis Read Range): {y.shape[0]} samples")
        print(f"  Baseline EDA: {baseline_eda:.2f} μS")
        
        # STEP 4: Train ML models on training set, evaluate on test set
        print("\n[STEP 4] Training Models (70% train, 30% test split)...")
        lr_model, rf_model, test_results = predictor.train_ml_models(X, y)
        
        # STEP 5: Visualize
        print("\n[STEP 5] Creating Visualizations...")
        visualize_results(eda_features_df, predictor, baseline_eda, test_results)
        
        # STEP 6: Example predictions
        print("\n[STEP 6] Example Predictions")
        print("=" * 70)
        
        print("\nPhysics-Based Model (Friis Equation):")
        print("-" * 70)
        example_edas = {
            'Low EDA (Meditation)': 1.5,
            'Baseline EDA': baseline_eda,
            'Moderate EDA (Amusement)': 3.5,
            'High EDA (Stress)': 5.0
        }
        
        for label, eda in example_edas.items():
            predicted_range = predictor.predict_range_physics(eda, baseline_eda)
            print(f"{label:25s}: EDA = {eda:.2f} μS → Friis Range = {predicted_range*100:.1f} cm")
        
        # ML Model predictions on test set
        print("\n\nMachine Learning Models (Evaluated on Test Set):")
        print("-" * 70)
        print("\nLinear Regression Predictions (Test Set):")
        print(f"  Mean Prediction: {np.mean(lr_pred)*100:.1f} cm")
        print(f"  Std Deviation: {np.std(lr_pred)*100:.1f} cm")
        print(f"  R² Score: {r2_score(y_test, lr_pred):.4f}")
        
        print("\nRandom Forest Predictions (Test Set):")
        print(f"  Mean Prediction: {np.mean(rf_pred)*100:.1f} cm")
        print(f"  Std Deviation: {np.std(rf_pred)*100:.1f} cm")
        print(f"  R² Score: {r2_score(y_test, rf_pred):.4f}")
        
        print("\n" + "=" * 70)
        print("Analysis Complete!")
        print("=" * 70)
        print("\nNote: ML model visualizations are shown in plots:")
        print("  - Linear Regression: Bottom-middle plot (Predicted vs Actual)")
        print("  - Random Forest: Bottom-right plot (Predicted vs Actual)")
        print("  - Both models trained on 70% of data, evaluated on 30% test set")
        
    except FileNotFoundError:
        print("\n⚠️  WESAD dataset not found at specified path.")
        print("\nTo use this system:")
        print("1. Download WESAD dataset from: https://ubicomp.eti.uni-siegen.de/home/datasets/icmi18/")
        print("2. Extract the dataset")
        print("3. Update 'data_path' variable in this script to point to the WESAD folder")
        print("4. Run this script again")
        print("\nThe script is ready - just update the path!")
