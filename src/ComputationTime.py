import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
from . import flops

class PerformanceAnalysis:
    def __init__(self, model, batch_sizes=[32, 64, 128]):
        self.model = model
        self.batch_sizes = batch_sizes
        self.training_metrics = {}
        self.inference_metrics = {}
        
    def _rebuild_dataset(self, dataset, batch_size):
        """Rebuild dataset with new batch size"""
        return dataset.unbatch().batch(batch_size)
    
    def measure_training_time(self, dataset, epochs=1, max_batches_per_epoch=50, warmup_batches=5):
        """
        Measure training time metrics for different batch sizes with faster sampling
        
        Args:
            epochs: Number of epochs to measure
            max_batches_per_epoch: Maximum number of batches to measure per epoch
            warmup_batches: Number of batches to run before starting measurements
        """
        training_metrics = {}
        
        for batch_size in self.batch_sizes:
            # Rebuild dataset with current batch size
            train_ds = self._rebuild_dataset(dataset, batch_size)
            
            # Initialize metrics
            epoch_times = []
            batch_times = []
            memory_usage = []
            
            for epoch in range(epochs):
                epoch_start = time.time()
                batch_count = 0
                
                # Warmup phase
                for _ in range(warmup_batches):
                    next(iter(train_ds))
                
                # Measurement phase
                for batch in train_ds.take(max_batches_per_epoch):
                    batch_start = time.time()
                    self.model.train_on_batch(batch[0], batch[1])
                    batch_time = time.time() - batch_start
                    batch_times.append(batch_time)
                    batch_count += 1
                    
                    # Track memory usage
                    try:
                        memory = tf.config.experimental.get_memory_info('GPU:0')
                        memory_usage.append(memory['peak'] / (1024 * 1024 * 1024))
                    except:
                        memory_usage.append(0)
                
                epoch_time = time.time() - epoch_start
                epoch_times.append(epoch_time)
            
            # Calculate metrics
            metrics = {
                'avg_epoch_time': np.mean(epoch_times),
                'std_epoch_time': np.std(epoch_times),
                'avg_batch_time': np.mean(batch_times),
                'std_batch_time': np.std(batch_times),
                'avg_sample_time': np.mean(batch_times) / batch_size,
                'avg_memory_usage': np.mean(memory_usage),
                'peak_memory_usage': np.max(memory_usage),
                'measured_batches': max_batches_per_epoch,
                'warmup_batches': warmup_batches
            }
            
            training_metrics[batch_size] = metrics
        
        self.training_metrics = training_metrics
        return training_metrics
    
    def measure_inference_time(self, dataset, num_samples=None):
        """Measure inference time metrics for different batch sizes"""
        inference_metrics = {}
        
        for batch_size in self.batch_sizes:
            # Rebuild dataset with current batch size
            test_ds = self._rebuild_dataset(dataset, batch_size)
            # Limit number of samples
            if num_samples:
                test_ds = test_ds.take(num_samples // batch_size)
            
            # Warmup run
            for batch in test_ds.take(1):
                self.model.predict(batch[0])
            
            # Initialize metrics
            batch_times = []
            latencies = []
            memory_usage = []
            
            # Measure inference time
            for batch in test_ds:
                batch_start = time.time()
                _ = self.model.predict(batch[0])
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                # Calculate per-sample latency
                latencies.extend([batch_time / batch_size] * batch_size)
                
                # Track memory usage
                try:
                    memory = tf.config.experimental.get_memory_info('GPU:0')
                    memory_usage.append(memory['peak'] / (1024 * 1024 * 1024))  # Convert to GB
                except:
                    # If GPU memory tracking is not available
                    memory_usage.append(0)
            
            # Calculate metrics
            metrics = {
                'avg_batch_time': np.mean(batch_times),
                'std_batch_time': np.std(batch_times),
                'avg_latency': np.mean(latencies),
                'p95_latency': np.percentile(latencies, 95),
                'p99_latency': np.percentile(latencies, 99),
                'throughput': batch_size / np.mean(batch_times),
                'avg_memory_usage': np.mean(memory_usage),
                'peak_memory_usage': np.max(memory_usage)
            }
            
            inference_metrics[batch_size] = metrics
        
        self.inference_metrics = inference_metrics
        return inference_metrics
    
    def plot_training_metrics(self, figsize = (10,10), save_dir=None,show=True):
        """Generate individual training performance visualizations"""
        if not self.training_metrics:
            raise ValueError("No training metrics available. Run measure_training_time first.")
        
        batch_sizes = list(self.training_metrics.keys())
        
        # Epoch Time vs Batch Size
        plt.figure(figsize=figsize)
        epoch_times = [m['avg_epoch_time'] for m in self.training_metrics.values()]
        epoch_stds = [m['std_epoch_time'] for m in self.training_metrics.values()]
        plt.errorbar(batch_sizes, epoch_times, yerr=epoch_stds, marker='o')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Epoch Time (s)')
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'training_epoch_time.png'),
                        dpi=300,
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # Batch Time vs Batch Size
        plt.figure(figsize=figsize)
        batch_times = [m['avg_batch_time'] for m in self.training_metrics.values()]
        batch_stds = [m['std_batch_time'] for m in self.training_metrics.values()]
        plt.errorbar(batch_sizes, batch_times, yerr=batch_stds, marker='o')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Batch Time (s)')
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'training_batch_time.png'),
                        dpi=300,
                        bbox_inches='tight')

        if show:
            plt.show()
        plt.close()
        
        # Sample Time vs Batch Size
        plt.figure(figsize=figsize)
        sample_times = [m['avg_sample_time'] for m in self.training_metrics.values()]
        plt.plot(batch_sizes, sample_times, marker='o')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Sample Time (s)')
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'training_sample_time.png'),
                        dpi=300,
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # Memory Usage
        plt.figure(figsize=figsize)
        memory_usage = [m['avg_memory_usage'] for m in self.training_metrics.values()]
        peak_memory = [m['peak_memory_usage'] for m in self.training_metrics.values()]
        plt.plot(batch_sizes, memory_usage, marker='o', label='Average')
        plt.plot(batch_sizes, peak_memory, marker='o', label='Peak')
        plt.xlabel('Batch Size')
        plt.ylabel('Memory Usage (GB)')
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'training_memory_usage.png'),
                        dpi=300,
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
    def plot_inference_metrics(self, figsize = (10,10), save_dir=None, show=True):
        """Generate individual inference performance visualizations"""
        if not self.inference_metrics:
            raise ValueError("No inference metrics available. Run measure_inference_time first.")
        
        batch_sizes = list(self.inference_metrics.keys())
        
        # Latency vs Batch Size
        plt.figure(figsize=figsize)
        latencies = [m['avg_latency'] for m in self.inference_metrics.values()]
        p95_latencies = [m['p95_latency'] for m in self.inference_metrics.values()]
        p99_latencies = [m['p99_latency'] for m in self.inference_metrics.values()]
        plt.plot(batch_sizes, latencies, marker='o', label='Average')
        plt.plot(batch_sizes, p95_latencies, marker='o', label='P95')
        plt.plot(batch_sizes, p99_latencies, marker='o', label='P99')
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (s)')
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'inference_latency.png'),
                        dpi=300,
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # Throughput vs Batch Size
        plt.figure(figsize=figsize)
        throughput = [m['throughput'] for m in self.inference_metrics.values()]
        plt.plot(batch_sizes, throughput, marker='o')
        plt.xlabel('Batch Size')
        plt.ylabel('Samples/Second')
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'inference_throughput.png'),
                        dpi=300,
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # Batch Time vs Batch Size
        plt.figure(figsize=figsize)
        batch_times = [m['avg_batch_time'] for m in self.inference_metrics.values()]
        batch_stds = [m['std_batch_time'] for m in self.inference_metrics.values()]
        plt.errorbar(batch_sizes, batch_times, yerr=batch_stds, marker='o')
        plt.xlabel('Batch Size')
        plt.ylabel('Average Batch Time (s)')
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'inference_batch_time.png'),
                        dpi=300,
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
        # Memory Usage
        plt.figure(figsize=figsize)
        memory_usage = [m['avg_memory_usage'] for m in self.inference_metrics.values()]
        peak_memory = [m['peak_memory_usage'] for m in self.inference_metrics.values()]
        plt.plot(batch_sizes, memory_usage, marker='o', label='Average')
        plt.plot(batch_sizes, peak_memory, marker='o', label='Peak')
        plt.xlabel('Batch Size')
        plt.ylabel('Memory Usage (GB)')
        plt.legend()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'inference_memory_usage.png'),
                        dpi=300,
                        bbox_inches='tight')
        if show:
            plt.show()
        plt.close()
        
    def generate_report(self, save_path=None):
        """Generate a comprehensive performance report"""
        # Create initial report dictionary
        # flops_analysis = flops.ModelAnalyzer(self.model)
        # flops_n = flops_analysis.get_flops_keras()
        report = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': self.model.name,
            'training_metrics': pd.DataFrame(self.training_metrics).T,
            'inference_metrics': pd.DataFrame(self.inference_metrics).T
        }
        
        # Create combined DataFrame
        training_df = report['training_metrics'].add_prefix('training_')
        inference_df = report['inference_metrics'].add_prefix('inference_')
        combined_df = pd.concat([training_df, inference_df], axis=1)
        combined_df.insert(0, 'timestamp', report['timestamp'])
        combined_df.insert(1, 'model_name', report['model_name'])
        # combined_df.insert(2, 'FLOPs in M', flops_n['flops in M'])
        # combined_df.insert(3, 'FLOPs in G', flops_n['flops in G'])
        
        if save_path:
            # Save separate metrics if needed
            # report['training_metrics'].to_csv(os.path.join(save_path, 'TrainingMetrics.csv'))
            # report['inference_metrics'].to_csv(os.path.join(save_path, 'InferenceMetrics.csv'))
            
            # Save combined metrics
            combined_df.to_csv(os.path.join(save_path, 'ComputationTime.csv'))
            
            # Save plots
            self.plot_training_metrics(save_dir=save_path, show=False)
            self.plot_inference_metrics(save_dir=save_path, show=False)
        
        return combined_df