#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analytics Utilities - Վերլուծություն և վիզուալիզացիա
Data analysis, visualization and report generation
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class DataAnalyzer:
    """Տվյալների վերլուծության գործիք"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def descriptive_statistics(self, data: List[Union[int, float]]) -> Dict:
        """Նկարագրական վիճակագրություն"""
        if not data:
            return {'error': 'Empty data'}
        
        n = len(data)
        mean = sum(data) / n
        
        sorted_data = sorted(data)
        median = sorted_data[n // 2] if n % 2 == 1 else (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
        
        variance = sum((x - mean) ** 2 for x in data) / n
        std_dev = variance ** 0.5
        
        min_val = min(data)
        max_val = max(data)
        
        q1_idx = n // 4
        q3_idx = 3 * n // 4
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        
        return {
            'count': n,
            'mean': float(mean),
            'median': float(median),
            'std_deviation': float(std_dev),
            'variance': float(variance),
            'min': float(min_val),
            'max': float(max_val),
            'range': float(max_val - min_val),
            'q1': float(q1),
            'q3': float(q3),
            'iqr': float(q3 - q1)
        }
    
    def correlation_matrix(self, data_dict: Dict[str, List[Union[int, float]]]) -> Dict:
        """Կորելյացիայի մատրից"""
        keys = list(data_dict.keys())
        n = len(keys)
        
        matrix = {}
        
        for i, key1 in enumerate(keys):
            matrix[key1] = {}
            for j, key2 in enumerate(keys):
                if i == j:
                    matrix[key1][key2] = 1.0
                elif j < i:
                    matrix[key1][key2] = matrix[key2][key1]
                else:
                    corr = self._pearson_correlation(data_dict[key1], data_dict[key2])
                    matrix[key1][key2] = corr
        
        return {
            'keys': keys,
            'matrix': matrix
        }
    
    def _pearson_correlation(self, x: List[float], y: List[float]) -> float:
        """Pearson կորելյացիա"""
        n = min(len(x), len(y))
        if n == 0:
            return 0.0
        
        x = x[:n]
        y = y[:n]
        
        mean_x = sum(x) / n
        mean_y = sum(y) / n
        
        numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
        
        denom_x = sum((xi - mean_x) ** 2 for xi in x) ** 0.5
        denom_y = sum((yi - mean_y) ** 2 for yi in y) ** 0.5
        
        if denom_x == 0 or denom_y == 0:
            return 0.0
        
        return numerator / (denom_x * denom_y)
    
    def frequency_distribution(self, data: List[Any], bins: int = 10) -> Dict:
        """Հաճախականության բաշխում"""
        if not data:
            return {'error': 'Empty data'}
        
        numeric_data = [x for x in data if isinstance(x, (int, float))]
        
        if not numeric_data:
            counts = {}
            for item in data:
                counts[str(item)] = counts.get(str(item), 0) + 1
            
            return {
                'type': 'categorical',
                'distribution': counts,
                'unique_count': len(counts)
            }
        
        min_val = min(numeric_data)
        max_val = max(numeric_data)
        bin_width = (max_val - min_val) / bins if bins > 0 else 1
        
        distribution = []
        for i in range(bins):
            bin_start = min_val + i * bin_width
            bin_end = min_val + (i + 1) * bin_width
            count = sum(1 for x in numeric_data if bin_start <= x < bin_end)
            
            distribution.append({
                'bin_start': bin_start,
                'bin_end': bin_end,
                'count': count,
                'percentage': count / len(numeric_data) * 100
            })
        
        return {
            'type': 'numeric',
            'bins': bins,
            'min': min_val,
            'max': max_val,
            'distribution': distribution
        }
    
    def detect_outliers(self, data: List[Union[int, float]], 
                       method: str = 'iqr') -> Dict:
        """Հայտնաբերել շեղումները"""
        if not data:
            return {'error': 'Empty data'}
        
        sorted_data = sorted(data)
        n = len(data)
        
        q1 = sorted_data[n // 4]
        q3 = sorted_data[3 * n // 4]
        iqr = q3 - q1
        
        if method == 'iqr':
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        elif method == 'zscore':
            mean = sum(data) / n
            std = (sum((x - mean) ** 2 for x in data) / n) ** 0.5
            lower_bound = mean - 3 * std
            upper_bound = mean + 3 * std
        else:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
        
        outliers = [x for x in data if x < lower_bound or x > upper_bound]
        
        return {
            'method': method,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'outliers': outliers,
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / n * 100
        }
    
    def analyze_dataset(self, dataset: Dict[str, List]) -> Dict:
        """Համապարփակ վերլուծություն"""
        results = {}
        
        for column, data in dataset.items():
            if all(isinstance(x, (int, float)) for x in data):
                results[column] = {
                    'statistics': self.descriptive_statistics(data),
                    'distribution': self.frequency_distribution(data),
                    'outliers': self.detect_outliers(data)
                }
            else:
                results[column] = {
                    'type': 'categorical',
                    'unique_values': len(set(str(x) for x in data)),
                    'distribution': self.frequency_distribution(data)
                }
        
        self.analysis_results = results
        return results


class VisualizationTools:
    """Վիզուալիզացիայի գործիքներ"""
    
    def __init__(self, output_dir: str = "./output/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_charts = []
    
    def create_bar_chart_data(self, labels: List[str], values: List[float],
                             title: str = "Bar Chart") -> Dict:
        """Ստեղծել bar chart տվյալներ"""
        chart_data = {
            'type': 'bar',
            'title': title,
            'labels': labels,
            'values': values,
            'created_at': datetime.now().isoformat()
        }
        
        self.generated_charts.append(chart_data)
        return chart_data
    
    def create_line_chart_data(self, x_values: List, y_values: List,
                              title: str = "Line Chart",
                              x_label: str = "X", y_label: str = "Y") -> Dict:
        """Ստեղծել line chart տվյալներ"""
        chart_data = {
            'type': 'line',
            'title': title,
            'x_values': x_values,
            'y_values': y_values,
            'x_label': x_label,
            'y_label': y_label,
            'created_at': datetime.now().isoformat()
        }
        
        self.generated_charts.append(chart_data)
        return chart_data
    
    def create_pie_chart_data(self, labels: List[str], values: List[float],
                             title: str = "Pie Chart") -> Dict:
        """Ստեղծել pie chart տվյալներ"""
        total = sum(values)
        percentages = [v / total * 100 if total > 0 else 0 for v in values]
        
        chart_data = {
            'type': 'pie',
            'title': title,
            'labels': labels,
            'values': values,
            'percentages': percentages,
            'created_at': datetime.now().isoformat()
        }
        
        self.generated_charts.append(chart_data)
        return chart_data
    
    def create_scatter_plot_data(self, x_values: List[float], 
                                y_values: List[float],
                                title: str = "Scatter Plot",
                                labels: Optional[List[str]] = None) -> Dict:
        """Ստեղծել scatter plot տվյալներ"""
        chart_data = {
            'type': 'scatter',
            'title': title,
            'x_values': x_values,
            'y_values': y_values,
            'labels': labels,
            'created_at': datetime.now().isoformat()
        }
        
        self.generated_charts.append(chart_data)
        return chart_data
    
    def create_heatmap_data(self, matrix: List[List[float]],
                           row_labels: Optional[List[str]] = None,
                           col_labels: Optional[List[str]] = None,
                           title: str = "Heatmap") -> Dict:
        """Ստեղծել heatmap տվյալներ"""
        chart_data = {
            'type': 'heatmap',
            'title': title,
            'matrix': matrix,
            'row_labels': row_labels,
            'col_labels': col_labels,
            'rows': len(matrix),
            'cols': len(matrix[0]) if matrix else 0,
            'created_at': datetime.now().isoformat()
        }
        
        self.generated_charts.append(chart_data)
        return chart_data
    
    def export_chart_data(self, chart_index: int, 
                         filename: Optional[str] = None) -> str:
        """Արտահանել chart տվյալները"""
        if chart_index >= len(self.generated_charts):
            return ""
        
        chart_data = self.generated_charts[chart_index]
        
        if not filename:
            filename = f"chart_{chart_index}_{chart_data['type']}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(chart_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def export_all_charts(self, filename: str = "all_charts.json") -> str:
        """Արտահանել բոլոր chart-երը"""
        output_path = self.output_dir / filename
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_charts': len(self.generated_charts),
            'charts': self.generated_charts
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def generate_html_report(self, filename: str = "visualization_report.html") -> str:
        """Ստեղծել HTML ռեպորտ"""
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Visualization Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .chart {{ border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; }}
        pre {{ background: #f5f5f5; padding: 10px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>📊 Visualization Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p>Total Charts: {len(self.generated_charts)}</p>
"""
        
        for i, chart in enumerate(self.generated_charts):
            html_content += f"""
    <div class="chart">
        <h2>{chart['type'].upper()} - {chart['title']}</h2>
        <pre>{json.dumps(chart, indent=2, ensure_ascii=False)}</pre>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)


class ReportGenerator:
    """Ռեպորտների գեներացում"""
    
    def __init__(self, output_dir: str = "./output/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reports = []
    
    def create_summary_report(self, title: str, sections: Dict[str, Any],
                             author: Optional[str] = None) -> Dict:
        """Ստեղծել ամփոփ ռեպորտ"""
        report = {
            'type': 'summary',
            'title': title,
            'author': author,
            'generated_at': datetime.now().isoformat(),
            'sections': sections
        }
        
        self.reports.append(report)
        return report
    
    def create_analysis_report(self, title: str, 
                              data_summary: Dict,
                              findings: List[str],
                              recommendations: List[str],
                              charts: Optional[List[Dict]] = None) -> Dict:
        """Ստեղծել վերլուծական ռեպորտ"""
        report = {
            'type': 'analysis',
            'title': title,
            'generated_at': datetime.now().isoformat(),
            'data_summary': data_summary,
            'findings': findings,
            'recommendations': recommendations,
            'charts': charts or []
        }
        
        self.reports.append(report)
        return report
    
    def create_progress_report(self, project_name: str,
                              tasks_completed: int,
                              tasks_total: int,
                              milestones: List[Dict],
                              issues: Optional[List[str]] = None) -> Dict:
        """Ստեղծել առաջընթացի ռեպորտ"""
        progress_percentage = (tasks_completed / tasks_total * 100) if tasks_total > 0 else 0
        
        report = {
            'type': 'progress',
            'project_name': project_name,
            'generated_at': datetime.now().isoformat(),
            'tasks_completed': tasks_completed,
            'tasks_total': tasks_total,
            'progress_percentage': progress_percentage,
            'milestones': milestones,
            'issues': issues or []
        }
        
        self.reports.append(report)
        return report
    
    def export_report_json(self, report_index: int,
                          filename: Optional[str] = None) -> str:
        """Արտահանել ռեպորտը JSON"""
        if report_index >= len(self.reports):
            return ""
        
        report = self.reports[report_index]
        
        if not filename:
            filename = f"report_{report_index}_{report['type']}.json"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def export_report_markdown(self, report_index: int,
                              filename: Optional[str] = None) -> str:
        """Արտահանել ռեպորտը Markdown"""
        if report_index >= len(self.reports):
            return ""
        
        report = self.reports[report_index]
        
        md_content = f"# {report['title']}\n\n"
        md_content += f"*Generated: {report['generated_at']}*\n\n"
        
        if report['type'] == 'summary':
            for section_name, section_content in report.get('sections', {}).items():
                md_content += f"## {section_name}\n\n"
                md_content += f"{section_content}\n\n"
        
        elif report['type'] == 'analysis':
            md_content += "## Data Summary\n\n"
            md_content += f"```json\n{json.dumps(report.get('data_summary', {}), indent=2)}\n```\n\n"
            
            md_content += "## Findings\n\n"
            for finding in report.get('findings', []):
                md_content += f"- {finding}\n"
            
            md_content += "\n## Recommendations\n\n"
            for rec in report.get('recommendations', []):
                md_content += f"- {rec}\n"
        
        elif report['type'] == 'progress':
            md_content += f"### Progress: {report['progress_percentage']:.1f}%\n\n"
            md_content += f"Tasks: {report['tasks_completed']}/{report['tasks_total']}\n\n"
            
            md_content += "## Milestones\n\n"
            for milestone in report.get('milestones', []):
                status = "✅" if milestone.get('completed', False) else "⏳"
                md_content += f"- {status} {milestone.get('name', 'Unknown')}\n"
        
        if not filename:
            filename = f"report_{report_index}_{report['type']}.md"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(output_path)
    
    def get_all_reports(self) -> List[Dict]:
        """Ստանալ բոլոր ռեպորտները"""
        return self.reports


__all__ = ['DataAnalyzer', 'VisualizationTools', 'ReportGenerator']
