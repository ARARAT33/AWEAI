#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules Module - Մոդուլների կառավարում և ընդլայնում
Plugin system, module loading and extension management
"""

import os
import sys
import json
import importlib
import importlib.util
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable, Type
from pathlib import Path


class ModuleManager:
    """Մոդուլների կառավարման համակարգ"""
    
    def __init__(self, modules_path: str = "./modules"):
        self.modules_path = Path(modules_path)
        self.modules_path.mkdir(parents=True, exist_ok=True)
        self.loaded_modules = {}
        self.module_registry = {}
        self.module_dependencies = {}
        
    def discover_modules(self) -> List[Dict]:
        """Գտնել բոլոր առկա մոդուլները"""
        discovered = []
        
        for file_path in self.modules_path.glob("*.py"):
            if file_path.name.startswith('_'):
                continue
            
            module_name = file_path.stem
            
            try:
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    
                    module_info = {
                        'name': module_name,
                        'path': str(file_path),
                        'size_bytes': file_path.stat().st_size,
                        'modified_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        'discoverable': True
                    }
                    
                    discovered.append(module_info)
                    
            except Exception as e:
                discovered.append({
                    'name': module_name,
                    'path': str(file_path),
                    'error': str(e),
                    'discoverable': False
                })
        
        return discovered
    
    def load_module(self, module_name: str, force_reload: bool = False) -> Optional[Any]:
        """Բեռնել մոդուլ"""
        if module_name in self.loaded_modules and not force_reload:
            return self.loaded_modules[module_name]
        
        module_file = self.modules_path / f"{module_name}.py"
        if not module_file.exists():
            return None
        
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                
                self.loaded_modules[module_name] = module
                
                module_info = {
                    'name': module_name,
                    'loaded_at': datetime.now().isoformat(),
                    'path': str(module_file),
                    'status': 'loaded'
                }
                self.module_registry[module_name] = module_info
                
                return module
        except Exception as e:
            error_info = {
                'name': module_name,
                'error': str(e),
                'loaded_at': datetime.now().isoformat(),
                'status': 'failed'
            }
            self.module_registry[module_name] = error_info
            return None
    
    def unload_module(self, module_name: str) -> bool:
        """Ապաբեռնել մոդուլ"""
        if module_name not in self.loaded_modules:
            return False
        
        try:
            del self.loaded_modules[module_name]
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            if module_name in self.module_registry:
                self.module_registry[module_name]['status'] = 'unloaded'
                self.module_registry[module_name]['unloaded_at'] = datetime.now().isoformat()
            
            return True
        except Exception as e:
            return False
    
    def reload_module(self, module_name: str) -> Optional[Any]:
        """Վերաբեռնել մոդուլ"""
        self.unload_module(module_name)
        return self.load_module(module_name, force_reload=True)
    
    def register_plugin(self, plugin_name: str, plugin_class: Type, 
                       dependencies: Optional[List[str]] = None) -> bool:
        """Գրանցել պլագին"""
        if dependencies:
            missing_deps = [dep for dep in dependencies if dep not in self.loaded_modules]
            if missing_deps:
                return False
        
        self.module_registry[plugin_name] = {
            'type': 'plugin',
            'class': plugin_class,
            'dependencies': dependencies or [],
            'registered_at': datetime.now().isoformat(),
            'status': 'registered'
        }
        
        return True
    
    def get_module_classes(self, module_name: str, base_class: Optional[Type] = None) -> List[Type]:
        """Ստանալ մոդուլի դասերը"""
        if module_name not in self.loaded_modules:
            return []
        
        module = self.loaded_modules[module_name]
        classes = []
        
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type):
                if base_class is None or issubclass(attr, base_class):
                    if attr_name != base_class.__name__ if base_class else True:
                        classes.append(attr)
        
        return classes
    
    def execute_module_function(self, module_name: str, function_name: str, 
                               *args, **kwargs) -> Any:
        """Կատարել մոդուլի ֆունկցիա"""
        if module_name not in self.loaded_modules:
            loaded = self.load_module(module_name)
            if not loaded:
                return {'error': f'Module {module_name} not found'}
        
        module = self.loaded_modules[module_name]
        
        if not hasattr(module, function_name):
            return {'error': f'Function {function_name} not found in {module_name}'}
        
        func = getattr(module, function_name)
        
        try:
            result = func(*args, **kwargs)
            return {
                'success': True,
                'result': result,
                'module': module_name,
                'function': function_name,
                'executed_at': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'module': module_name,
                'function': function_name
            }
    
    def get_registry(self) -> Dict:
        """Ստանալ մոդուլների ռեեստրը"""
        return self.module_registry
    
    def get_loaded_modules(self) -> List[str]:
        """Ստանալ բեռնված մոդուլների ցանկը"""
        return list(self.loaded_modules.keys())
    
    def export_registry(self, filename: str = "module_registry.json") -> str:
        """Արտահանել մոդուլների ռեեստրը"""
        registry_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_modules': len(self.module_registry),
            'loaded_count': len(self.loaded_modules),
            'registry': self.module_registry
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)
        
        return filename


class PluginSystem:
    """Պլագինների համակարգ"""
    
    def __init__(self):
        self.plugins = {}
        self.plugin_hooks = {}
        self.execution_order = []
        
    def register_plugin(self, name: str, plugin_instance: Any, 
                       priority: int = 0) -> bool:
        """Գրանցել պլագին"""
        if name in self.plugins:
            return False
        
        self.plugins[name] = {
            'instance': plugin_instance,
            'priority': priority,
            'registered_at': datetime.now().isoformat(),
            'active': True
        }
        
        self._sort_by_priority()
        
        if hasattr(plugin_instance, 'on_register'):
            plugin_instance.on_register()
        
        return True
    
    def unregister_plugin(self, name: str) -> bool:
        """Ապագրանցել պլագին"""
        if name not in self.plugins:
            return False
        
        plugin = self.plugins[name]
        
        if hasattr(plugin['instance'], 'on_unregister'):
            plugin['instance'].on_unregister()
        
        del self.plugins[name]
        
        if name in self.execution_order:
            self.execution_order.remove(name)
        
        return True
    
    def register_hook(self, hook_name: str, plugin_name: str, 
                     callback: Callable) -> bool:
        """Գրանցել հուկ"""
        if plugin_name not in self.plugins:
            return False
        
        if hook_name not in self.plugin_hooks:
            self.plugin_hooks[hook_name] = []
        
        self.plugin_hooks[hook_name].append({
            'plugin_name': plugin_name,
            'callback': callback,
            'priority': self.plugins[plugin_name]['priority']
        })
        
        self.plugin_hooks[hook_name].sort(key=lambda x: x['priority'], reverse=True)
        
        return True
    
    def trigger_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Ակտիվացնել հուկ"""
        if hook_name not in self.plugin_hooks:
            return []
        
        results = []
        
        for hook in self.plugin_hooks[hook_name]:
            try:
                result = hook['callback'](*args, **kwargs)
                results.append({
                    'plugin': hook['plugin_name'],
                    'result': result
                })
            except Exception as e:
                results.append({
                    'plugin': hook['plugin_name'],
                    'error': str(e)
                })
        
        return results
    
    def _sort_by_priority(self):
        """Տեսակավորել ըստ առաջնահերթության"""
        sorted_plugins = sorted(
            self.plugins.items(),
            key=lambda x: x[1]['priority'],
            reverse=True
        )
        self.execution_order = [name for name, _ in sorted_plugins]
    
    def get_active_plugins(self) -> List[str]:
        """Ստանալ ակտիվ պլագինները"""
        return [name for name, info in self.plugins.items() if info['active']]
    
    def enable_plugin(self, name: str) -> bool:
        """Միացնել պլագին"""
        if name not in self.plugins:
            return False
        
        self.plugins[name]['active'] = True
        
        if hasattr(self.plugins[name]['instance'], 'on_enable'):
            self.plugins[name]['instance'].on_enable()
        
        return True
    
    def disable_plugin(self, name: str) -> bool:
        """Անջատել պլագին"""
        if name not in self.plugins:
            return False
        
        self.plugins[name]['active'] = False
        
        if hasattr(self.plugins[name]['instance'], 'on_disable'):
            self.plugins[name]['instance'].on_disable()
        
        return True
    
    def get_plugin_info(self, name: str) -> Optional[Dict]:
        """Ստանալ պլագինի ինֆորմացիա"""
        if name not in self.plugins:
            return None
        
        plugin = self.plugins[name]
        return {
            'name': name,
            'priority': plugin['priority'],
            'registered_at': plugin['registered_at'],
            'active': plugin['active'],
            'has_on_register': hasattr(plugin['instance'], 'on_register'),
            'has_on_unregister': hasattr(plugin['instance'], 'on_unregister'),
            'has_on_enable': hasattr(plugin['instance'], 'on_enable'),
            'has_on_disable': hasattr(plugin['instance'], 'on_disable')
        }
    
    def list_plugins(self) -> List[Dict]:
        """Ցուցակել բոլոր պլագինները"""
        return [
            {
                'name': name,
                'info': self.get_plugin_info(name)
            }
            for name in self.plugins
        ]


__all__ = ['ModuleManager', 'PluginSystem']
