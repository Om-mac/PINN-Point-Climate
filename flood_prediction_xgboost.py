"""
Alternative implementation using XGBoost instead of Random Forest.
XGBoost often provides better performance for structured/tabular data.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import the data processing pipeline from main model
from flood_prediction_model import FloodPredictionModel


class XGBoostFloodModel(FloodPredictionModel):
    """XGBoost variant of the flood prediction model."""
    
    def train_model(self, test_size=0.2, random_state=42):
        """Train XGBoost classifier."""
        print("\n" + "="*60)
        print("TRAINING XGBOOST MODEL")
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
        
        # Calculate scale_pos_weight for imbalanced dataset
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        print("\nTraining XGBoost...")
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Store test sets
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
        
        print("✓ Model training completed")
        
        return self
    
    def save_model(self, filepath='models/flood_prediction_xgb.pkl'):
        """Save trained XGBoost model."""
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'label_encoders': self.label_encoders,
            'probability_threshold': self.probability_threshold,
            'model_type': 'XGBoost'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ XGBoost model saved to {filepath}")
        
        return self


def main():
    """Main execution for XGBoost model."""
    print("\n" + "="*60)
    print("PINN-POINT CLIMATE - STAGE 1 (XGBoost Variant)")
    print("Binary Flood Classification Model")
    print("="*60)
    
    # Initialize XGBoost model
    model = XGBoostFloodModel(data_path='DATA/')
    
    # Execute pipeline
    (model
        .load_data()
        .prepare_flood_events()
        .generate_negative_samples(samples_per_station=10)
        .create_features()
        .train_model(test_size=0.2, random_state=42)
        .evaluate_model()
        .save_model('models/flood_prediction_xgb.pkl')
    )
    
    print("\n" + "="*60)
    print("XGBoost MODEL COMPLETE - Ready for PINN Integration")
    print("="*60)
    
    return model


if __name__ == "__main__":
    model = main()
