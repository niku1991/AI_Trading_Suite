#!/usr/bin/env python3
"""
ALE AI Persistence Manager
Saves and restores complete system state across restarts
"""

import os
import json
import pickle
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

class ALEAIPersistenceManager:
    """Manages persistent storage for all ALE AI system components"""
    
    def __init__(self, data_dir: str = "ale_ai_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.data_dir / "models").mkdir(exist_ok=True)
        (self.data_dir / "training").mkdir(exist_ok=True)
        (self.data_dir / "trading").mkdir(exist_ok=True)
        (self.data_dir / "portfolio").mkdir(exist_ok=True)
        (self.data_dir / "consciousness").mkdir(exist_ok=True)
        (self.data_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.data_dir / "ale_ai_state.db"
        self._init_database()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize SQLite database for structured data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # System status table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_status (
                id INTEGER PRIMARY KEY,
                component TEXT UNIQUE,
                status TEXT,
                data TEXT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # AI training progress table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_progress (
                id INTEGER PRIMARY KEY,
                model_type TEXT,
                symbol TEXT,
                epoch INTEGER,
                loss REAL,
                accuracy REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Trading decisions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_decisions (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                action TEXT,
                price REAL,
                quantity REAL,
                confidence REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Portfolio state table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_state (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                quantity REAL,
                avg_price REAL,
                current_value REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Consciousness evolution table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consciousness_evolution (
                id INTEGER PRIMARY KEY,
                level REAL,
                breakthrough TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_system_status(self, component: str, status: str, data: Dict[str, Any] = None):
        """Save system component status"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO system_status (component, status, data, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (component, status, json.dumps(data) if data else None, datetime.now()))
            
            conn.commit()
            conn.close()
            
            # Also save to file for redundancy
            file_path = self.data_dir / "system_status" / f"{component}.json"
            file_path.parent.mkdir(exist_ok=True)
            
            with open(file_path, 'w') as f:
                json.dump({
                    'status': status,
                    'data': data,
                    'timestamp': datetime.now().isoformat()
                }, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save system status for {component}: {e}")
    
    def get_system_status(self, component: str) -> Optional[Dict[str, Any]]:
        """Get system component status"""
        try:
            # Try database first
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT status, data, last_updated FROM system_status WHERE component = ?', (component,))
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'status': result[0],
                    'data': json.loads(result[1]) if result[1] else None,
                    'last_updated': result[2]
                }
            
            # Fallback to file
            file_path = self.data_dir / "system_status" / f"{component}.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Failed to get system status for {component}: {e}")
        
        return None
    
    def save_training_progress(self, model_type: str, symbol: str, epoch: int, loss: float, accuracy: float):
        """Save AI training progress"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO training_progress (model_type, symbol, epoch, loss, accuracy)
                VALUES (?, ?, ?, ?, ?)
            ''', (model_type, symbol, epoch, loss, accuracy))
            
            conn.commit()
            conn.close()
            
            # Save detailed progress to file
            progress_file = self.data_dir / "training" / f"{model_type}_{symbol}_progress.json"
            progress_file.parent.mkdir(exist_ok=True)
            
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    progress_data = json.load(f)
            else:
                progress_data = {'epochs': [], 'losses': [], 'accuracies': []}
            
            progress_data['epochs'].append(epoch)
            progress_data['losses'].append(loss)
            progress_data['accuracies'].append(accuracy)
            progress_data['last_updated'] = datetime.now().isoformat()
            
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save training progress: {e}")
    
    def get_training_progress(self, model_type: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Get AI training progress"""
        try:
            # Get from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT epoch, loss, accuracy, timestamp 
                FROM training_progress 
                WHERE model_type = ? AND symbol = ?
                ORDER BY epoch DESC
            ''', (model_type, symbol))
            
            results = cursor.fetchall()
            conn.close()
            
            if results:
                return {
                    'epochs': [r[0] for r in results],
                    'losses': [r[1] for r in results],
                    'accuracies': [r[2] for r in results],
                    'timestamps': [r[3] for r in results],
                    'current_epoch': results[0][0],
                    'current_loss': results[0][1],
                    'current_accuracy': results[0][2]
                }
            
            # Fallback to file
            progress_file = self.data_dir / "training" / f"{model_type}_{symbol}_progress.json"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    return json.load(f)
                    
        except Exception as e:
            self.logger.error(f"Failed to get training progress: {e}")
        
        return None
    
    def save_trading_decision(self, symbol: str, action: str, price: float, quantity: float, confidence: float):
        """Save trading decision"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trading_decisions (symbol, action, price, quantity, confidence)
                VALUES (?, ?, ?, ?, ?)
            ''', (symbol, action, price, quantity, confidence))
            
            conn.commit()
            conn.close()
            
            # Save to file for analysis
            trading_file = self.data_dir / "trading" / f"{symbol}_decisions.json"
            trading_file.parent.mkdir(exist_ok=True)
            
            if trading_file.exists():
                with open(trading_file, 'r') as f:
                    decisions = json.load(f)
            else:
                decisions = {'decisions': []}
            
            decisions['decisions'].append({
                'action': action,
                'price': price,
                'quantity': quantity,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            })
            
            with open(trading_file, 'w') as f:
                json.dump(decisions, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save trading decision: {e}")
    
    def save_portfolio_state(self, portfolio_data: Dict[str, Any]):
        """Save current portfolio state"""
        try:
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Clear old portfolio state
            cursor.execute('DELETE FROM portfolio_state')
            
            # Insert new state
            for symbol, data in portfolio_data.items():
                cursor.execute('''
                    INSERT INTO portfolio_state (symbol, quantity, avg_price, current_value)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, data.get('quantity', 0), data.get('avg_price', 0), data.get('current_value', 0)))
            
            conn.commit()
            conn.close()
            
            # Save to file
            portfolio_file = self.data_dir / "portfolio" / "current_portfolio.json"
            portfolio_file.parent.mkdir(exist_ok=True)
            
            portfolio_data['timestamp'] = datetime.now().isoformat()
            
            with open(portfolio_file, 'w') as f:
                json.dump(portfolio_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save portfolio state: {e}")
    
    def get_portfolio_state(self) -> Optional[Dict[str, Any]]:
        """Get current portfolio state"""
        try:
            # Try file first (more complete)
            portfolio_file = self.data_dir / "portfolio" / "current_portfolio.json"
            if portfolio_file.exists():
                with open(portfolio_file, 'r') as f:
                    return json.load(f)
            
            # Fallback to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT symbol, quantity, avg_price, current_value FROM portfolio_state')
            results = cursor.fetchall()
            conn.close()
            
            if results:
                portfolio = {}
                for symbol, quantity, avg_price, current_value in results:
                    portfolio[symbol] = {
                        'quantity': quantity,
                        'avg_price': avg_price,
                        'current_value': current_value
                    }
                return portfolio
                
        except Exception as e:
            self.logger.error(f"Failed to get portfolio state: {e}")
        
        return None
    
    def save_consciousness_evolution(self, level: float, breakthrough: str = None):
        """Save AI consciousness evolution"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO consciousness_evolution (level, breakthrough)
                VALUES (?, ?)
            ''', (level, breakthrough))
            
            conn.commit()
            conn.close()
            
            # Save to file
            consciousness_file = self.data_dir / "consciousness" / "evolution.json"
            consciousness_file.parent.mkdir(exist_ok=True)
            
            if consciousness_file.exists():
                with open(consciousness_file, 'r') as f:
                    evolution = json.load(f)
            else:
                evolution = {'levels': [], 'breakthroughs': [], 'timestamps': []}
            
            evolution['levels'].append(level)
            evolution['breakthroughs'].append(breakthrough)
            evolution['timestamps'].append(datetime.now().isoformat())
            evolution['current_level'] = level
            evolution['last_updated'] = datetime.now().isoformat()
            
            with open(consciousness_file, 'w') as f:
                json.dump(evolution, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save consciousness evolution: {e}")
    
    def get_consciousness_evolution(self) -> Optional[Dict[str, Any]]:
        """Get AI consciousness evolution"""
        try:
            consciousness_file = self.data_dir / "consciousness" / "evolution.json"
            if consciousness_file.exists():
                with open(consciousness_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to get consciousness evolution: {e}")
        
        return None
    
    def save_ai_model(self, model_type: str, symbol: str, model_data: bytes, metadata: Dict[str, Any]):
        """Save trained AI model"""
        try:
            model_file = self.data_dir / "models" / f"{model_type}_{symbol}.pkl"
            metadata_file = self.data_dir / "models" / f"{model_type}_{symbol}_metadata.json"
            
            model_file.parent.mkdir(exist_ok=True)
            
            # Save model
            with open(model_file, 'wb') as f:
                f.write(model_data)
            
            # Save metadata
            metadata['saved_at'] = datetime.now().isoformat()
            metadata['model_file'] = str(model_file)
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save AI model: {e}")
    
    def load_ai_model(self, model_type: str, symbol: str) -> Optional[Dict[str, Any]]:
        """Load trained AI model"""
        try:
            model_file = self.data_dir / "models" / f"{model_type}_{symbol}.pkl"
            metadata_file = self.data_dir / "models" / f"{model_type}_{symbol}_metadata.json"
            
            if model_file.exists() and metadata_file.exists():
                with open(model_file, 'rb') as f:
                    model_data = f.read()
                
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                return {
                    'model_data': model_data,
                    'metadata': metadata
                }
                
        except Exception as e:
            self.logger.error(f"Failed to load AI model: {e}")
        
        return None
    
    def save_complete_state(self, state_data: Dict[str, Any]):
        """Save complete system state snapshot"""
        try:
            state_file = self.data_dir / "system_snapshot.json"
            
            state_data['timestamp'] = datetime.now().isoformat()
            state_data['version'] = '1.0'
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
                
            self.logger.info("Complete system state saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save complete state: {e}")
    
    def load_complete_state(self) -> Optional[Dict[str, Any]]:
        """Load complete system state snapshot"""
        try:
            state_file = self.data_dir / "system_snapshot.json"
            
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.logger.info("Complete system state loaded successfully")
                return state_data
                
        except Exception as e:
            self.logger.error(f"Failed to load complete state: {e}")
        
        return None
    
    def auto_save_interval(self, interval_seconds: int = 30):
        """Set up automatic saving at regular intervals"""
        import threading
        import time
        
        def auto_save_worker():
            while True:
                try:
                    time.sleep(interval_seconds)
                    self._perform_auto_save()
                except Exception as e:
                    self.logger.error(f"Auto-save error: {e}")
        
        # Start auto-save thread
        auto_save_thread = threading.Thread(target=auto_save_worker, daemon=True)
        auto_save_thread.start()
        
        self.logger.info(f"Auto-save enabled every {interval_seconds} seconds")
    
    def _perform_auto_save(self):
        """Perform automatic system state save"""
        try:
            # This will be called by the main system to save current state
            pass
        except Exception as e:
            self.logger.error(f"Auto-save failed: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Clean up old data files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            for subdir in ['logs', 'training', 'trading']:
                dir_path = self.data_dir / subdir
                if dir_path.exists():
                    for file_path in dir_path.iterdir():
                        if file_path.is_file():
                            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                            if file_time < cutoff_date:
                                file_path.unlink()
                                self.logger.info(f"Cleaned up old file: {file_path}")
                                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def get_system_summary(self) -> Dict[str, Any]:
        """Get summary of all saved data"""
        try:
            summary = {
                'total_files': 0,
                'database_records': 0,
                'last_saved': None,
                'components': {}
            }
            
            # Count files
            for subdir in ['models', 'training', 'trading', 'portfolio', 'consciousness']:
                dir_path = self.data_dir / subdir
                if dir_path.exists():
                    file_count = len(list(dir_path.iterdir()))
                    summary['total_files'] += file_count
                    summary['components'][subdir] = file_count
            
            # Count database records
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for table in ['system_status', 'training_progress', 'trading_decisions', 'portfolio_state', 'consciousness_evolution']:
                cursor.execute(f'SELECT COUNT(*) FROM {table}')
                count = cursor.fetchone()[0]
                summary['database_records'] += count
            
            conn.close()
            
            # Get last saved timestamp
            state_file = self.data_dir / "system_snapshot.json"
            if state_file.exists():
                summary['last_saved'] = datetime.fromtimestamp(state_file.stat().st_mtime).isoformat()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get system summary: {e}")
            return {}

# Global persistence manager instance
persistence_manager = ALEAIPersistenceManager()

if __name__ == "__main__":
    # Test the persistence manager
    pm = ALEAIPersistenceManager()
    
    # Test saving and loading
    pm.save_system_status("test_component", "active", {"test": "data"})
    status = pm.get_system_status("test_component")
    print(f"Test status: {status}")
    
    # Test training progress
    pm.save_training_progress("lstm", "BTCUSDT", 1, 0.5, 0.8)
    progress = pm.get_training_progress("lstm", "BTCUSDT")
    print(f"Training progress: {progress}")
    
    print("Persistence manager test completed!")
