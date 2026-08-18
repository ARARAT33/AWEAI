#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Device Control Module - Սարքերի կառավարում
IoT devices, hardware control and automation
"""

import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class DeviceController:
    """Սարքերի կառավարման հզոր գործիք"""
    
    def __init__(self):
        self.connected_devices = {}
        self.device_history = []
        self.automation_rules = []
        
    def scan_devices(self, connection_type: str = "usb") -> List[Dict]:
        """Սկանավորել միացված սարքերը"""
        detected_devices = []
        
        if connection_type == "usb":
            try:
                import usb.core
                import usb.util
                
                devices = usb.core.find(find_all=True)
                for device in devices:
                    device_info = {
                        'id': f"usb_{device.idVendor}_{device.idProduct}",
                        'vendor_id': hex(device.idVendor),
                        'product_id': hex(device.idProduct),
                        'manufacturer': usb.util.get_string(device, device.iManufacturer) if device.iManufacturer else "Unknown",
                        'product': usb.util.get_string(device, device.iProduct) if device.iProduct else "Unknown",
                        'connection_type': 'usb',
                        'detected_at': datetime.now().isoformat()
                    }
                    detected_devices.append(device_info)
            except ImportError:
                # Fallback: check /dev directory on Linux
                import subprocess
                try:
                    result = subprocess.run(['lsusb'], capture_output=True, text=True)
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        detected_devices.append({
                            'raw_info': line,
                            'connection_type': 'usb',
                            'detected_at': datetime.now().isoformat()
                        })
                except:
                    pass
                    
        elif connection_type == "serial":
            try:
                import serial.tools.list_ports
                ports = serial.tools.list_ports.comports()
                for port in ports:
                    device_info = {
                        'id': port.device,
                        'name': port.description,
                        'device': port.device,
                        'hwid': port.hwid,
                        'connection_type': 'serial',
                        'detected_at': datetime.now().isoformat()
                    }
                    detected_devices.append(device_info)
            except ImportError:
                pass
        
        self.connected_devices.update({d['id']: d for d in detected_devices})
        return detected_devices
    
    def connect_device(self, device_id: str, connection_params: Optional[Dict] = None) -> bool:
        """Միանալ սարքին"""
        if device_id not in self.connected_devices:
            return False
        
        device = self.connected_devices[device_id]
        
        try:
            if device['connection_type'] == 'serial':
                import serial
                baudrate = connection_params.get('baudrate', 9600) if connection_params else 9600
                ser = serial.Serial(device['device'], baudrate, timeout=1)
                device['connection'] = ser
                device['connected'] = True
                
            elif device['connection_type'] == 'usb':
                # USB connection logic
                device['connected'] = True
            
            self.device_history.append({
                'action': 'connect',
                'device_id': device_id,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            return True
        except Exception as e:
            self.device_history.append({
                'action': 'connect',
                'device_id': device_id,
                'timestamp': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            })
            return False
    
    def send_command(self, device_id: str, command: Union[str, bytes]) -> Dict:
        """Ուղարկել հրաման սարքին"""
        if device_id not in self.connected_devices:
            return {'error': 'Device not found'}
        
        device = self.connected_devices[device_id]
        
        if not device.get('connected', False):
            return {'error': 'Device not connected'}
        
        try:
            if device['connection_type'] == 'serial':
                if isinstance(command, str):
                    command = command.encode()
                
                device['connection'].write(command)
                time.sleep(0.1)
                
                response = device['connection'].readline().decode().strip()
                
                result = {
                    'command_sent': command.decode() if isinstance(command, bytes) else command,
                    'response': response,
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
                
            elif device['connection_type'] == 'usb':
                # USB command logic
                result = {
                    'command_sent': command,
                    'response': 'USB command sent',
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            
            self.device_history.append({
                'action': 'send_command',
                'device_id': device_id,
                'command': command if isinstance(command, str) else command.decode(),
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            return result
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def read_sensor_data(self, device_id: str, timeout: float = 1.0) -> Dict:
        """Կարդալ սենսորի տվյալները"""
        if device_id not in self.connected_devices:
            return {'error': 'Device not found'}
        
        device = self.connected_devices[device_id]
        
        if not device.get('connected', False):
            return {'error': 'Device not connected'}
        
        try:
            if device['connection_type'] == 'serial':
                device['connection'].flushInput()
                start_time = time.time()
                data_lines = []
                
                while time.time() - start_time < timeout:
                    if device['connection'].in_waiting > 0:
                        line = device['connection'].readline().decode().strip()
                        if line:
                            data_lines.append(line)
                    
                    time.sleep(0.01)
                
                return {
                    'data': data_lines,
                    'lines_count': len(data_lines),
                    'timestamp': datetime.now().isoformat(),
                    'success': True
                }
            
            return {'error': 'Unsupported connection type', 'success': False}
            
        except Exception as e:
            return {'error': str(e), 'success': False}
    
    def disconnect_device(self, device_id: str) -> bool:
        """Անջատել սարքը"""
        if device_id not in self.connected_devices:
            return False
        
        device = self.connected_devices[device_id]
        
        try:
            if device.get('connection'):
                if device['connection_type'] == 'serial':
                    device['connection'].close()
                device['connected'] = False
                del device['connection']
            
            self.device_history.append({
                'action': 'disconnect',
                'device_id': device_id,
                'timestamp': datetime.now().isoformat(),
                'success': True
            })
            
            return True
        except Exception as e:
            return False
    
    def add_automation_rule(self, rule: Dict) -> bool:
        """Ավելացնել ավտոմատացման կանոն"""
        required_fields = ['trigger', 'condition', 'action']
        if not all(field in rule for field in required_fields):
            return False
        
        rule['id'] = f"rule_{len(self.automation_rules) + 1}"
        rule['created_at'] = datetime.now().isoformat()
        rule['active'] = True
        
        self.automation_rules.append(rule)
        return True
    
    def execute_automation(self) -> List[Dict]:
        """Կատարել ավտոմատացման կանոնները"""
        results = []
        
        for rule in self.automation_rules:
            if not rule.get('active', False):
                continue
            
            try:
                trigger = rule['trigger']
                condition = rule['condition']
                action = rule['action']
                
                # Check trigger
                if trigger['type'] == 'sensor_value':
                    device_id = trigger['device_id']
                    threshold = trigger['threshold']
                    
                    sensor_data = self.read_sensor_data(device_id)
                    if 'data' in sensor_data and sensor_data['data']:
                        value = float(sensor_data['data'][0])
                        
                        if condition['operator'] == '>' and value > threshold:
                            result = self.send_command(action['device_id'], action['command'])
                            results.append({
                                'rule_id': rule['id'],
                                'triggered': True,
                                'value': value,
                                'action_result': result
                            })
                        
            except Exception as e:
                results.append({
                    'rule_id': rule['id'],
                    'error': str(e)
                })
        
        return results
    
    def get_device_history(self, device_id: Optional[str] = None) -> List[Dict]:
        """Ստանալ սարքի պատմությունը"""
        if device_id:
            return [h for h in self.device_history if h.get('device_id') == device_id]
        return self.device_history
    
    def export_device_log(self, filename: str = "device_log.json") -> str:
        """Արտահանել սարքերի լոգը"""
        log_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_events': len(self.device_history),
            'connected_devices': list(self.connected_devices.keys()),
            'automation_rules': len(self.automation_rules),
            'history': self.device_history
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        return filename


class GPIOPinController:
    """GPIO pins-ի կառավարում (Raspberry Pi և այլն)"""
    
    def __init__(self):
        self.pin_states = {}
        self.gpio_available = False
        
    def setup(self, pin_number: int, mode: str = 'out') -> bool:
        """Նախապատրաստել GPIO pin"""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            
            if mode == 'out':
                GPIO.setup(pin_number, GPIO.OUT)
            elif mode == 'in':
                GPIO.setup(pin_number, GPIO.IN)
            
            self.pin_states[pin_number] = {
                'mode': mode,
                'state': False,
                'configured': True
            }
            self.gpio_available = True
            return True
        except ImportError:
            # Simulation mode
            self.pin_states[pin_number] = {
                'mode': mode,
                'state': False,
                'configured': True,
                'simulated': True
            }
            return True
        except Exception as e:
            return False
    
    def set_pin_high(self, pin_number: int) -> bool:
        """Սահմանել pin-ը HIGH"""
        if pin_number not in self.pin_states:
            return False
        
        try:
            if not self.pin_states[pin_number].get('simulated', False):
                import RPi.GPIO as GPIO
                GPIO.output(pin_number, GPIO.HIGH)
            
            self.pin_states[pin_number]['state'] = True
            return True
        except:
            self.pin_states[pin_number]['state'] = True
            return True
    
    def set_pin_low(self, pin_number: int) -> bool:
        """Սահմանել pin-ը LOW"""
        if pin_number not in self.pin_states:
            return False
        
        try:
            if not self.pin_states[pin_number].get('simulated', False):
                import RPi.GPIO as GPIO
                GPIO.output(pin_number, GPIO.LOW)
            
            self.pin_states[pin_number]['state'] = False
            return True
        except:
            self.pin_states[pin_number]['state'] = False
            return True
    
    def read_pin(self, pin_number: int) -> Optional[bool]:
        """Կարդալ pin-ի վիճակը"""
        if pin_number not in self.pin_states:
            return None
        
        if self.pin_states[pin_number].get('mode') != 'in':
            return None
        
        try:
            if not self.pin_states[pin_number].get('simulated', False):
                import RPi.GPIO as GPIO
                return GPIO.input(pin_number) == GPIO.HIGH
            return self.pin_states[pin_number].get('state', False)
        except:
            return self.pin_states[pin_number].get('state', False)
    
    def cleanup(self):
        """Մաքրել GPIO ռեսուրսները"""
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except:
            pass
        self.pin_states.clear()
        self.gpio_available = False


__all__ = ['DeviceController', 'GPIOPinController']
