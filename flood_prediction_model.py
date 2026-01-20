"""
PINN-Point Climate - Stage 1: Binary Flood Classification Model
Binary classification model using Random Forest to predict if a station will flood next week.
Output probability score triggers Stage 2 PINN simulation if >0.7.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class FloodPredictionModel:
    """Binary classification model for flood prediction with PINN integration."""
    
    def __init__(self, data_path='DATA/'):
        """Initialize model with data path."""
        self.data_path = data_path
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.probability_threshold = 0.7  # Threshold for PINN Stage 2 trigger
        
    def load_data(self):
        """Load all CSV files and merge into master dataframe."""
        print("Loading data files...")
        
        # Load all CSV files
        self.catchment_df = pd.read_csv(f'{self.data_path}catchment_characteristics_indofloods.csv')
        self.flood_events_df = pd.read_csv(f'{self.data_path}floodevents_indofloods.csv')
        self.metadata_df = pd.read_csv(f'{self.data_path}metadata_indofloods.csv')
        self.precip_df = pd.read_csv(f'{self.data_path}precipitation_variables_indofloods.csv')
        
        print(f"✓ Loaded {len(self.catchment_df)} catchment records")
        print(f"✓ Loaded {len(self.flood_events_df)} flood event records")
        print(f"✓ Loaded {len(self.metadata_df)} metadata records")
        print(f"✓ Loaded {len(self.precip_df)} precipitation records")
        
        return self
    
    def prepare_flood_events(self):
        """Prepare flood events with positive labels (Label=1)."""
        print("\nPreparing flood events...")
        
        # Extract GaugeID from EventID (e.g., INDOFLOODS-gauge-1010-1 -> INDOFLOODS-gauge-1010)
        self.flood_events_df['GaugeID'] = self.flood_events_df['EventID'].str.rsplit('-', n=1).str[0]
        
        # Convert dates to datetime
        self.flood_events_df['Start Date'] = pd.to_datetime(self.flood_events_df['Start Date'])
        self.flood_events_df['End Date'] = pd.to_datetime(self.flood_events_df['End Date'])
        self.flood_events_df['Peak FL Date'] = pd.to_datetime(self.flood_events_df['Peak FL Date'])
        
        # Create target date (we'll use Peak FL Date as the flood occurrence date)
        self.flood_events_df['Target_Date'] = self.flood_events_df['Peak FL Date']
        
        # Add label for flood events
        self.flood_events_df['Flood_Label'] = 1
        
        print(f"✓ Prepared {len(self.flood_events_df)} positive samples (floods)")
        
        return self
    
    def generate_negative_samples(self, samples_per_station=10):
        """Generate negative samples (non-flood dates) for each station."""
        print(f"\nGenerating negative samples ({samples_per_station} per station)...")
        
        negative_samples = []
        
        # Get unique gauge IDs from flood events
        unique_gauges = self.flood_events_df['GaugeID'].unique()
        
        for gauge_id in unique_gauges:
            # Get flood dates for this station
            station_floods = self.flood_events_df[self.flood_events_df['GaugeID'] == gauge_id]
            flood_dates = set(station_floods['Peak FL Date'].dt.date)
            
            # Get date range for this station from metadata
            if gauge_id in self.metadata_df['GaugeID'].values:
                station_meta = self.metadata_df[self.metadata_df['GaugeID'] == gauge_id].iloc[0]
                start_date = pd.to_datetime(station_meta['Start_date'])
                end_date = pd.to_datetime(station_meta['End_date'])
                
                # Generate random dates
                date_range_days = (end_date - start_date).days
                
                generated = 0
                attempts = 0
                max_attempts = samples_per_station * 10  # Limit attempts
                
                while generated < samples_per_station and attempts < max_attempts:
                    # Random date within the station's operational period
                    random_days = np.random.randint(0, date_range_days)
                    random_date = start_date + timedelta(days=int(random_days))
                    
                    # Check if this date is not a flood date (within 7-day window)
                    is_flood = False
                    for flood_date in flood_dates:
                        flood_datetime = pd.to_datetime(flood_date)
                        if abs((random_date - flood_datetime).days) <= 7:
                            is_flood = True
                            break
                    
                    if not is_flood:
                        # Create event ID for negative sample
                        event_id = f"{gauge_id}-NEG-{generated+1}"
                        
                        # Generate precipitation values for negative samples
                        # 30% get zero precipitation (dry season)
                        # 30% get very low precipitation (0-5mm) 
                        # 40% get low-moderate precipitation (5-20mm) - light rain, no flood
                        rand = np.random.random()
                        
                        if rand < 0.3:
                            # Completely dry (0mm all days)
                            low_precip = {
                                'T1d': 0, 'T2d': 0, 'T3d': 0, 'T4d': 0, 'T5d': 0,
                                'T6d': 0, 'T7d': 0, 'T8d': 0, 'T9d': 0, 'T10d': 0
                            }
                        elif rand < 0.6:
                            # Very low / dry season (0-5mm per day)
                            low_precip = {
                                'T1d': np.random.uniform(0, 5),
                                'T2d': np.random.uniform(0, 5),
                                'T3d': np.random.uniform(0, 5),
                                'T4d': np.random.uniform(0, 5),
                                'T5d': np.random.uniform(0, 5),
                                'T6d': np.random.uniform(0, 5),
                                'T7d': np.random.uniform(0, 5),
                                'T8d': np.random.uniform(0, 5),
                                'T9d': np.random.uniform(0, 5),
                                'T10d': np.random.uniform(0, 5)
                            }
                        else:
                            # Light rain, no flood (5-20mm per day)
                            low_precip = {
                                'T1d': np.random.uniform(5, 20),
                                'T2d': np.random.uniform(5, 20),
                                'T3d': np.random.uniform(5, 20),
                                'T4d': np.random.uniform(5, 20),
                                'T5d': np.random.uniform(5, 20),
                                'T6d': np.random.uniform(5, 20),
                                'T7d': np.random.uniform(5, 20),
                                'T8d': np.random.uniform(5, 20),
                                'T9d': np.random.uniform(5, 20),
                                'T10d': np.random.uniform(5, 20)
                            }
                        
                        negative_samples.append({
                            'EventID': event_id,
                            'GaugeID': gauge_id,
                            'Target_Date': random_date,
                            'Flood_Label': 0,
                            **low_precip  # Add precipitation values
                        })
                        generated += 1
                    
                    attempts += 1
        
        # Create dataframe from negative samples
        self.negative_df = pd.DataFrame(negative_samples)
        
        print(f"✓ Generated {len(self.negative_df)} negative samples (non-floods)")
        
        return self
    
    def create_features(self):
        """Create features including API, geomorphological, and categorical variables."""
        print("\nCreating features...")
        
        # Combine positive and negative samples
        # For positive samples (floods), only include basic columns
        flood_base = self.flood_events_df[['EventID', 'GaugeID', 'Target_Date', 'Flood_Label']]
        
        # For negative samples, include precipitation if available
        neg_cols = ['EventID', 'GaugeID', 'Target_Date', 'Flood_Label']
        precip_cols = ['T1d', 'T2d', 'T3d', 'T4d', 'T5d', 'T6d', 'T7d', 'T8d', 'T9d', 'T10d']
        
        # Check if negative samples have precipitation data
        if all(col in self.negative_df.columns for col in precip_cols):
            neg_base = self.negative_df[neg_cols + precip_cols]
        else:
            neg_base = self.negative_df[neg_cols]
        
        combined_df = pd.concat([flood_base, neg_base], ignore_index=True)
        
        print(f"✓ Combined dataset: {len(combined_df)} samples ({combined_df['Flood_Label'].sum()} floods, {(combined_df['Flood_Label']==0).sum()} non-floods)")
        
        # Merge with precipitation data for flood events (if not already present)
        if 'T1d' not in combined_df.columns:
            combined_df = combined_df.merge(
                self.precip_df[['EventID'] + precip_cols],
                on='EventID',
                how='left'
            )
        
        # For any remaining missing precipitation data, use station averages
        for col in ['T1d', 'T2d', 'T3d', 'T4d', 'T5d', 'T6d', 'T7d', 'T8d', 'T9d', 'T10d']:
            # Calculate mean precipitation for each gauge
            gauge_means = self.precip_df.merge(
                self.flood_events_df[['EventID', 'GaugeID']], 
                on='EventID'
            ).groupby('GaugeID')[col].mean()
            
            # Fill missing values with gauge average
            for gauge_id in combined_df['GaugeID'].unique():
                mask = (combined_df['GaugeID'] == gauge_id) & (combined_df[col].isna())
                if gauge_id in gauge_means.index:
                    combined_df.loc[mask, col] = gauge_means[gauge_id]
        
        # Fill any remaining NaNs with overall mean
        for col in ['T1d', 'T2d', 'T3d', 'T4d', 'T5d', 'T6d', 'T7d', 'T8d', 'T9d', 'T10d']:
            combined_df[col].fillna(combined_df[col].mean(), inplace=True)
        
        # Create Antecedent Precipitation Index (API) features
        combined_df['API_3day'] = combined_df['T1d'] + combined_df['T2d'] + combined_df['T3d']
        combined_df['API_7day'] = combined_df[['T1d', 'T2d', 'T3d', 'T4d', 'T5d', 'T6d', 'T7d']].sum(axis=1)
        combined_df['API_10day'] = combined_df[['T1d', 'T2d', 'T3d', 'T4d', 'T5d', 'T6d', 'T7d', 'T8d', 'T9d', 'T10d']].sum(axis=1)
        
        print(f"✓ Created API features (3-day, 7-day, 10-day)")
        
        # Merge with catchment characteristics (static features)
        combined_df = combined_df.merge(self.catchment_df, on='GaugeID', how='left')
        
        # Merge with metadata
        combined_df = combined_df.merge(
            self.metadata_df[['GaugeID', 'Latitude', 'Longitude', 'Catchment Area', 'Warning Level', 'Danger Level']],
            on='GaugeID',
            how='left'
        )
        
        print(f"✓ Merged catchment characteristics and metadata")
        
        # Select key geomorphological features (excluding Latitude/Longitude to prevent location overfitting)
        geomorph_features = [
            'Drainage Area', 'Catchment Relief', 'Catchment Length',
            'Relief Ratio', 'Drainage Density', 'Stream Order',
            'Catchment Area'
        ]
        
        # Select categorical features
        categorical_features = ['Land cover', 'Soil type', 'lithology type', 'KoppenGeiger Climate Type']
        
        # Encode categorical variables
        for col in categorical_features:
            if col in combined_df.columns:
                le = LabelEncoder()
                combined_df[col + '_encoded'] = le.fit_transform(combined_df[col].astype(str))
                self.label_encoders[col] = le
        
        print(f"✓ Encoded categorical features")
        
        # Define final feature set
        self.feature_columns = (
            ['API_3day', 'API_7day', 'API_10day'] +
            ['T1d', 'T2d', 'T3d', 'T4d', 'T5d', 'T6d', 'T7d', 'T8d', 'T9d', 'T10d'] +
            [f for f in geomorph_features if f in combined_df.columns] +
            [col + '_encoded' for col in categorical_features if col in combined_df.columns]
        )
        
        # Handle missing values in feature columns
        for col in self.feature_columns:
            if col in combined_df.columns:
                combined_df[col].fillna(combined_df[col].median(), inplace=True)
        
        self.master_df = combined_df
        
        print(f"✓ Final dataset shape: {self.master_df.shape}")
        print(f"✓ Number of features: {len(self.feature_columns)}")
        print(f"✓ Features: {', '.join(self.feature_columns[:10])}...")
        
        return self
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train Random Forest classifier."""
        print("\n" + "="*60)
        print("TRAINING RANDOM FOREST MODEL")
        print("="*60)
        
        # Prepare features and target
        X = self.master_df[self.feature_columns]
        y = self.master_df['Flood_Label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Class distribution - Flood: {y_train.sum()}, No Flood: {(y_train==0).sum()}")
        
        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        print("\nTraining Random Forest...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Store test sets for later use
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        print("✓ Model training completed")
        
        return self
    
    def evaluate_model(self):
        """Evaluate model performance and display metrics."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, self.y_pred, 
                                    target_names=['No Flood', 'Flood']))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # ROC-AUC score
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        print(f"\nROC-AUC Score: {roc_auc:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(feature_importance.head(15).to_string(index=False))
        
        # PINN integration check
        high_risk_samples = (self.y_pred_proba > self.probability_threshold).sum()
        print(f"\n" + "="*60)
        print(f"PINN STAGE 2 TRIGGER ANALYSIS")
        print("="*60)
        print(f"Samples with probability > {self.probability_threshold}: {high_risk_samples} ({high_risk_samples/len(self.y_test)*100:.2f}%)")
        print(f"These samples would trigger Stage 2 PINN simulation")
        
        return self
    
    def predict_flood_probability(self, features_dict):
        """
        Predict flood probability for new data.
        
        Args:
            features_dict: Dictionary with feature values
            
        Returns:
            Dictionary with prediction results and PINN trigger status
        """
        # Create feature vector
        feature_vector = pd.DataFrame([features_dict])[self.feature_columns]
        
        # Predict probability
        probability = self.model.predict_proba(feature_vector)[0, 1]
        prediction = int(probability > 0.5)
        
        # Check PINN trigger
        trigger_pinn = probability > self.probability_threshold
        
        result = {
            'flood_probability': float(probability),
            'prediction': 'Flood' if prediction == 1 else 'No Flood',
            'trigger_pinn_stage2': trigger_pinn,
            'confidence': 'HIGH' if probability > 0.7 or probability < 0.3 else 'MEDIUM'
        }
        
        return result
    
    def save_model(self, filepath='models/flood_prediction_rf.pkl'):
        """Save trained model to file."""
        import pickle
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'probability_threshold': self.probability_threshold
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved to {filepath}")
        
        return self
    
    def visualize_results(self):
        """Create visualizations for model results."""
        print("\nGenerating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Confusion Matrix
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_ylabel('True Label')
        axes[0, 0].set_xlabel('Predicted Label')
        axes[0, 0].set_xticklabels(['No Flood', 'Flood'])
        axes[0, 0].set_yticklabels(['No Flood', 'Flood'])
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = roc_auc_score(self.y_test, self.y_pred_proba)
        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True)
        
        # 3. Feature Importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)
        
        axes[1, 0].barh(range(len(feature_importance)), feature_importance['importance'])
        axes[1, 0].set_yticks(range(len(feature_importance)))
        axes[1, 0].set_yticklabels(feature_importance['feature'])
        axes[1, 0].set_xlabel('Importance')
        axes[1, 0].set_title('Top 15 Feature Importances')
        axes[1, 0].invert_yaxis()
        
        # 4. Probability Distribution
        axes[1, 1].hist(self.y_pred_proba[self.y_test == 0], bins=30, alpha=0.5, label='No Flood', color='blue')
        axes[1, 1].hist(self.y_pred_proba[self.y_test == 1], bins=30, alpha=0.5, label='Flood', color='red')
        axes[1, 1].axvline(x=self.probability_threshold, color='green', linestyle='--', 
                          label=f'PINN Trigger Threshold ({self.probability_threshold})')
        axes[1, 1].set_xlabel('Predicted Probability')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Probability Distribution by True Label')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('flood_model_evaluation.png', dpi=300, bbox_inches='tight')
        print("✓ Saved visualization to 'flood_model_evaluation.png'")
        
        return self


def main():
    """Main execution function."""
    print("\n" + "="*60)
    print("PINN-POINT CLIMATE - STAGE 1")
    print("Binary Flood Classification Model")
    print("="*60)
    
    # Initialize model
    model = FloodPredictionModel(data_path='DATA/')
    
    # Execute pipeline
    (model
        .load_data()
        .prepare_flood_events()
        .generate_negative_samples(samples_per_station=10)
        .create_features()
        .train_model(test_size=0.2, random_state=42)
        .evaluate_model()
        .visualize_results()
        .save_model('models/flood_prediction_rf.pkl')
    )
    
    print("\n" + "="*60)
    print("STAGE 1 COMPLETE - Model Ready for PINN Integration")
    print("="*60)
    print(f"\nNext Steps:")
    print(f"1. Use model.predict_flood_probability() for new predictions")
    print(f"2. Samples with probability > {model.probability_threshold} trigger Stage 2 PINN simulation")
    print(f"3. Load saved model from 'models/flood_prediction_rf.pkl' for deployment")
    
    return model


if __name__ == "__main__":
    model = main()
