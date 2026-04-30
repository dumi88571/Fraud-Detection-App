import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
from functools import partial
import time

class OptimizedBatchProcessor:
    def __init__(self, model_package, max_workers=None):
        self.model_package = model_package
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.label_encoders = model_package['label_encoders']
        self.feature_columns = model_package['feature_columns']
        self.max_workers = max_workers or min(4, multiprocessing.cpu_count())
    
    def create_features_batch(self, df):
        df_featured = df.copy()
        
        df_featured['amount_log'] = np.log1p(df_featured['amount'])
        df_featured['hour_sin'] = np.sin(2 * np.pi * df_featured['hour'] / 24)
        df_featured['hour_cos'] = np.cos(2 * np.pi * df_featured['hour'] / 24)
        df_featured['is_weekend'] = ((df_featured['hour'] >= 0) & (df_featured['hour'] <= 6)) | (df_featured['hour'] >= 22)
        df_featured['is_night'] = (df_featured['hour'] >= 22) | (df_featured['hour'] <= 6)
        df_featured['combined_risk_score'] = (df_featured['device_risk_score'] + df_featured['ip_risk_score']) / 2
        df_featured['risk_diff'] = abs(df_featured['device_risk_score'] - df_featured['ip_risk_score'])
        df_featured['is_high_amount'] = df_featured['amount'] > df_featured['amount'].quantile(0.9)
        df_featured['is_high_risk'] = df_featured['combined_risk_score'] > 0.7
        
        for col, le in self.label_encoders.items():
            df_featured[f'{col}_encoded'] = le.transform(df_featured[col])
        
        return df_featured[self.feature_columns]
    
    def process_chunk(self, chunk_data, chunk_index):
        try:
            X_chunk = self.create_features_batch(chunk_data)
            X_chunk_scaled = self.scaler.transform(X_chunk)
            
            predictions = self.model.predict(X_chunk_scaled)
            probabilities = self.model.predict_proba(X_chunk_scaled)[:, 1]
            
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                risk_level = 'HIGH' if prob > 0.7 else 'MEDIUM' if prob > 0.3 else 'LOW'
                results.append({
                    'index': chunk_index + i,
                    'prediction': int(pred),
                    'fraud_probability': float(prob),
                    'risk_level': risk_level
                })
            
            return results
        except Exception as e:
            print(f"Error processing chunk {chunk_index}: {str(e)}")
            return []
    
    def predict_batch_optimized(self, df_transactions, batch_size=1000):
        start_time = time.time()
        
        if len(df_transactions) <= batch_size:
            results = self.process_chunk(df_transactions, 0)
            processing_time = time.time() - start_time
            return {
                'results': results,
                'processing_time': processing_time,
                'transactions_per_second': len(df_transactions) / processing_time,
                'total_transactions': len(df_transactions)
            }
        
        chunks = [df_transactions.iloc[i:i+batch_size] 
                 for i in range(0, len(df_transactions), batch_size)]
        
        all_results = []
        chunk_indices = [i * batch_size for i in range(len(chunks))]
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_chunk = {
                executor.submit(self.process_chunk, chunk, idx): chunk 
                for chunk, idx in zip(chunks, chunk_indices)
            }
            
            for future in as_completed(future_to_chunk):
                chunk_results = future.result()
                all_results.extend(chunk_results)
                print(f"Completed chunk {len(all_results)//batch_size}/{len(chunks)}")
        
        processing_time = time.time() - start_time
        all_results.sort(key=lambda x: x['index'])
        
        return {
            'results': all_results,
            'processing_time': processing_time,
            'transactions_per_second': len(df_transactions) / processing_time,
            'total_transactions': len(df_transactions)
        }
    
    def predict_batch_streaming(self, df_transactions, chunk_size=500):
        start_time = time.time()
        results = []
        
        for i in range(0, len(df_transactions), chunk_size):
            chunk = df_transactions.iloc[i:i+chunk_size]
            chunk_results = self.process_chunk(chunk, i)
            results.extend(chunk_results)
            
            yield {
                'chunk_index': i // chunk_size,
                'chunk_results': chunk_results,
                'cumulative_results': results,
                'processed_so_far': i + len(chunk),
                'total_transactions': len(df_transactions)
            }
        
        processing_time = time.time() - start_time
        
        yield {
            'final_results': results,
            'processing_time': processing_time,
            'transactions_per_second': len(df_transactions) / processing_time,
            'total_transactions': len(df_transactions)
        }
