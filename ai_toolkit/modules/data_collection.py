#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Collection Module - Տվյալների հավաքում և մշակում
Comprehensive data collection, preprocessing and management
"""

import os
import json
import csv
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import hashlib


class DataCollector:
    """Հզոր տվյալների հավաքման գործիք"""
    
    def __init__(self, storage_path: str = "./data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.collected_data = []
        self.metadata = {}
        
    def collect_from_api(self, url: str, headers: Optional[Dict] = None, 
                        params: Optional[Dict] = None) -> Dict:
        """Տվյալների հավաքում API-ից"""
        try:
            import requests
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            record = {
                'source': 'api',
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'hash': hashlib.md5(json.dumps(data).encode()).hexdigest()
            }
            self.collected_data.append(record)
            return record
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def collect_from_file(self, file_path: str, file_type: str = 'auto') -> Dict:
        """Տվյալների հավաքում ֆայլից (CSV, JSON, TXT)"""
        path = Path(file_path)
        if not path.exists():
            return {'error': f'File not found: {file_path}'}
        
        if file_type == 'auto':
            file_type = path.suffix.lower().replace('.', '')
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                if file_type == 'json':
                    data = json.load(f)
                elif file_type == 'csv':
                    data = list(csv.DictReader(f))
                elif file_type == 'txt':
                    data = f.read().splitlines()
                else:
                    data = f.read()
            
            record = {
                'source': 'file',
                'file_path': str(path),
                'file_type': file_type,
                'timestamp': datetime.now().isoformat(),
                'data': data,
                'size': path.stat().st_size
            }
            self.collected_data.append(record)
            return record
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def collect_from_web_scrape(self, url: str, selectors: Optional[Dict] = None) -> Dict:
        """Տվյալների հավաքում վեբ էջերից"""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            scraped_data = {}
            if selectors:
                for key, selector in selectors.items():
                    elements = soup.select(selector)
                    scraped_data[key] = [el.get_text(strip=True) for el in elements]
            else:
                scraped_data['title'] = soup.title.string if soup.title else ''
                scraped_data['links'] = [a.get('href') for a in soup.find_all('a', href=True)]
                scraped_data['text'] = soup.get_text(strip=True)[:5000]
            
            record = {
                'source': 'web_scrape',
                'url': url,
                'timestamp': datetime.now().isoformat(),
                'data': scraped_data
            }
            self.collected_data.append(record)
            return record
        except Exception as e:
            return {'error': str(e), 'timestamp': datetime.now().isoformat()}
    
    def save_to_database(self, db_name: str = "collected_data.db", 
                        table_name: str = "data_records") -> bool:
        """Պահպանել տվյալները SQLite բազայում"""
        try:
            db_path = self.storage_path / db_name
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source TEXT,
                    timestamp TEXT,
                    data_hash TEXT,
                    data_json TEXT,
                    metadata TEXT
                )
            ''')
            
            for record in self.collected_data:
                cursor.execute(f'''
                    INSERT INTO {table_name} (source, timestamp, data_hash, data_json, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    record.get('source', 'unknown'),
                    record.get('timestamp', ''),
                    record.get('hash', ''),
                    json.dumps(record.get('data', {})),
                    json.dumps({k: v for k, v in record.items() 
                               if k not in ['data', 'source', 'timestamp', 'hash']})
                ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Database error: {e}")
            return False
    
    def export_to_json(self, filename: str = "export_data.json") -> str:
        """Արտահանել տվյալները JSON ֆայլ"""
        output_path = self.storage_path / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'total_records': len(self.collected_data),
                'data': self.collected_data
            }, f, ensure_ascii=False, indent=2)
        return str(output_path)
    
    def get_statistics(self) -> Dict:
        """Ստանալ հավաքված տվյալների վիճակագրություն"""
        sources = {}
        total_size = 0
        
        for record in self.collected_data:
            source = record.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
            total_size += record.get('size', 0)
        
        return {
            'total_records': len(self.collected_data),
            'sources': sources,
            'total_size_bytes': total_size,
            'collection_start': self.collected_data[0]['timestamp'] if self.collected_data else None,
            'collection_end': self.collected_data[-1]['timestamp'] if self.collected_data else None
        }


class DataPreprocessor:
    """Տվյալների նախնական մշակում և մաքրում"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Մաքրել տեքստը ավելորդ նշաններից"""
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
        return text.strip()
    
    @staticmethod
    def normalize_data(data: List[Dict], fields: List[str]) -> List[Dict]:
        """Նորմալիզացնել տվյալները"""
        normalized = []
        for item in data:
            norm_item = {}
            for field in fields:
                value = item.get(field, '')
                if isinstance(value, str):
                    norm_item[field] = value.lower().strip()
                else:
                    norm_item[field] = value
            normalized.append(norm_item)
        return normalized
    
    @staticmethod
    def remove_duplicates(data: List[Dict], key_field: str) -> List[Dict]:
        """Հեռացնել կրկնվող գրառումները"""
        seen = set()
        unique_data = []
        for item in data:
            key = item.get(key_field)
            if key not in seen:
                seen.add(key)
                unique_data.append(item)
        return unique_data
    
    @staticmethod
    def split_dataset(data: List[Any], train_ratio: float = 0.8, 
                     val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict:
        """Բաժանել տվյալները train/validation/test"""
        import random
        random.shuffle(data)
        
        n = len(data)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        return {
            'train': data[:train_end],
            'validation': data[train_end:val_end],
            'test': data[val_end:]
        }


__all__ = ['DataCollector', 'DataPreprocessor']
