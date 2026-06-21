#!/usr/bin/env python3
"""
Comprehensive EDA for S2TLD Traffic Light Detection Datasets
Analyzes both S2TLD1080x1920 and S2TLD720x1280 datasets
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict, Counter
import json
from typing import Dict, List, Tuple
import statistics

class TrafficLightEDA:
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.datasets = {
            "S2TLD1080x1920": {
                "path": self.base_path / "S2TLD1080x1920" / "S2TLD（1080x1920）",
                "resolution": (1920, 1080),
                "subdirs": ["."]
            },
            "S2TLD720x1280": {
                "path": self.base_path / "S2TLD720x1280" / "S2TLD（720x1280）",
                "resolution": (1280, 720),
                "subdirs": ["normal_1", "normal_2"]
            }
        }
        self.results = {}
        
    def parse_xml_annotation(self, xml_path: Path) -> Dict | None:
        """Parse a single XML annotation file"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Extract image info
            filename_elem = root.find('filename')
            filename = filename_elem.text if filename_elem is not None and filename_elem.text else ""
            size = root.find('size')
            width_elem = size.find('width') if size is not None else None
            height_elem = size.find('height') if size is not None else None
            width = int(width_elem.text) if width_elem is not None and width_elem.text else 0
            height = int(height_elem.text) if height_elem is not None and height_elem.text else 0
            
            # Extract objects
            objects = []
            for obj in root.findall('object'):
                name_elem = obj.find('name')
                name = name_elem.text if name_elem is not None and name_elem.text else ""
                truncated = obj.find('truncated')
                truncated_val = int(truncated.text) if truncated is not None and truncated.text else 0
                difficult = obj.find('difficult')
                difficult_val = int(difficult.text) if difficult is not None and difficult.text else 0
                
                bndbox = obj.find('bndbox')
                if bndbox is not None:
                    xmin_elem = bndbox.find('xmin')
                    ymin_elem = bndbox.find('ymin')
                    xmax_elem = bndbox.find('xmax')
                    ymax_elem = bndbox.find('ymax')
                    
                    if xmin_elem is not None and xmin_elem.text and \
                       ymin_elem is not None and ymin_elem.text and \
                       xmax_elem is not None and xmax_elem.text and \
                       ymax_elem is not None and ymax_elem.text:
                        xmin = int(float(xmin_elem.text))
                        ymin = int(float(ymin_elem.text))
                        xmax = int(float(xmax_elem.text))
                        ymax = int(float(ymax_elem.text))
                    else:
                        continue
                    
                    bbox_width = xmax - xmin
                    bbox_height = ymax - ymin
                    bbox_area = bbox_width * bbox_height
                    aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
                    
                    # Calculate center position
                    center_x = (xmin + xmax) / 2
                    center_y = (ymin + ymax) / 2
                    
                    # Normalize positions (0-1)
                    norm_center_x = center_x / width if width > 0 else 0
                    norm_center_y = center_y / height if height > 0 else 0
                    
                    objects.append({
                        'class': name,
                        'bbox': {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax},
                        'bbox_width': bbox_width,
                        'bbox_height': bbox_height,
                        'bbox_area': bbox_area,
                        'aspect_ratio': aspect_ratio,
                        'center_x': center_x,
                        'center_y': center_y,
                        'norm_center_x': norm_center_x,
                        'norm_center_y': norm_center_y,
                        'truncated': truncated_val,
                        'difficult': difficult_val
                    })
            
            return {
                'filename': filename,
                'width': width,
                'height': height,
                'objects': objects,
                'num_objects': len(objects)
            }
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            return None
    
    def analyze_dataset(self, dataset_name: str) -> Dict:
        """Analyze a complete dataset"""
        dataset_info = self.datasets[dataset_name]
        dataset_path = dataset_info["path"]
        subdirs = dataset_info["subdirs"]
        
        all_annotations = []
        all_images = []
        
        for subdir in subdirs:
            if subdir == ".":
                ann_path = dataset_path / "Annotations"
                img_path = dataset_path / "JPEGImages"
            else:
                ann_path = dataset_path / subdir / "Annotations"
                img_path = dataset_path / subdir / "JPEGImages"
            
            # Count annotations
            if ann_path.exists():
                xml_files = list(ann_path.glob("*.xml"))
                all_annotations.extend(xml_files)
                
                # Parse each annotation
                for xml_file in xml_files:
                    parsed = self.parse_xml_annotation(xml_file)
                    if parsed:
                        parsed['subdir'] = subdir
                        parsed['xml_path'] = str(xml_file)
                        all_annotations.append(parsed)
            
            # Count images
            if img_path.exists():
                jpg_files = list(img_path.glob("*.jpg")) + list(img_path.glob("*.jpeg"))
                all_images.extend(jpg_files)
        
        # Compute statistics
        stats = self.compute_statistics(all_annotations, dataset_name)
        stats['total_images'] = len(all_images)
        stats['total_annotations'] = len([a for a in all_annotations if isinstance(a, dict)])
        stats['annotation_coverage'] = (stats['total_annotations'] / stats['total_images'] * 100) if stats['total_images'] > 0 else 0
        
        return stats
    
    def compute_statistics(self, annotations: List[Dict], dataset_name: str) -> Dict:
        """Compute comprehensive statistics from parsed annotations"""
        # Filter valid annotations
        valid_annotations = [a for a in annotations if isinstance(a, dict) and 'objects' in a]
        
        if not valid_annotations:
            return {
                'dataset_name': dataset_name,
                'total_annotations': 0,
                'total_objects': 0,
                'class_distribution': {},
                'error': 'No valid annotations found'
            }
        
        # Class distribution
        class_counts = Counter()
        bbox_widths = []
        bbox_heights = []
        bbox_areas = []
        aspect_ratios = []
        objects_per_image = []
        truncated_count = 0
        difficult_count = 0
        center_positions_x = []
        center_positions_y = []
        
        for ann in valid_annotations:
            objects_per_image.append(ann['num_objects'])
            
            for obj in ann['objects']:
                class_counts[obj['class']] += 1
                bbox_widths.append(obj['bbox_width'])
                bbox_heights.append(obj['bbox_height'])
                bbox_areas.append(obj['bbox_area'])
                aspect_ratios.append(obj['aspect_ratio'])
                center_positions_x.append(obj['norm_center_x'])
                center_positions_y.append(obj['norm_center_y'])
                
                if obj['truncated'] == 1:
                    truncated_count += 1
                if obj['difficult'] == 1:
                    difficult_count += 1
        
        total_objects = sum(class_counts.values())
        
        # Calculate statistics
        stats = {
            'dataset_name': dataset_name,
            'resolution': self.datasets[dataset_name]['resolution'],
            'total_objects': total_objects,
            'class_distribution': dict(class_counts),
            'class_percentages': {cls: (count / total_objects * 100) for cls, count in class_counts.items()},
            'objects_per_image': {
                'min': min(objects_per_image) if objects_per_image else 0,
                'max': max(objects_per_image) if objects_per_image else 0,
                'mean': statistics.mean(objects_per_image) if objects_per_image else 0,
                'median': statistics.median(objects_per_image) if objects_per_image else 0,
                'stdev': statistics.stdev(objects_per_image) if len(objects_per_image) > 1 else 0
            },
            'bbox_statistics': {
                'width': {
                    'min': min(bbox_widths) if bbox_widths else 0,
                    'max': max(bbox_widths) if bbox_widths else 0,
                    'mean': statistics.mean(bbox_widths) if bbox_widths else 0,
                    'median': statistics.median(bbox_widths) if bbox_widths else 0
                },
                'height': {
                    'min': min(bbox_heights) if bbox_heights else 0,
                    'max': max(bbox_heights) if bbox_heights else 0,
                    'mean': statistics.mean(bbox_heights) if bbox_heights else 0,
                    'median': statistics.median(bbox_heights) if bbox_heights else 0
                },
                'area': {
                    'min': min(bbox_areas) if bbox_areas else 0,
                    'max': max(bbox_areas) if bbox_areas else 0,
                    'mean': statistics.mean(bbox_areas) if bbox_areas else 0,
                    'median': statistics.median(bbox_areas) if bbox_areas else 0
                },
                'aspect_ratio': {
                    'min': min(aspect_ratios) if aspect_ratios else 0,
                    'max': max(aspect_ratios) if aspect_ratios else 0,
                    'mean': statistics.mean(aspect_ratios) if aspect_ratios else 0,
                    'median': statistics.median(aspect_ratios) if aspect_ratios else 0
                }
            },
            'quality_flags': {
                'truncated_objects': truncated_count,
                'difficult_objects': difficult_count,
                'truncated_percentage': (truncated_count / total_objects * 100) if total_objects > 0 else 0,
                'difficult_percentage': (difficult_count / total_objects * 100) if total_objects > 0 else 0
            },
            'spatial_distribution': {
                'center_x_mean': statistics.mean(center_positions_x) if center_positions_x else 0,
                'center_y_mean': statistics.mean(center_positions_y) if center_positions_y else 0,
                'center_x_stdev': statistics.stdev(center_positions_x) if len(center_positions_x) > 1 else 0,
                'center_y_stdev': statistics.stdev(center_positions_y) if len(center_positions_y) > 1 else 0
            }
        }
        
        return stats
    
    def run_analysis(self):
        """Run complete EDA on all datasets"""
        print("=" * 80)
        print("S2TLD Traffic Light Detection Dataset - Comprehensive EDA")
        print("=" * 80)
        print()
        
        for dataset_name in self.datasets.keys():
            print(f"\nAnalyzing {dataset_name}...")
            self.results[dataset_name] = self.analyze_dataset(dataset_name)
        
        return self.results
    
    def generate_report(self, output_file: str = "EDA_Report.md"):
        """Generate a comprehensive markdown report"""
        report = []
        report.append("# S2TLD Traffic Light Detection Dataset - EDA Report\n")
        report.append(f"*Generated: {Path.cwd()}*\n\n")
        
        report.append("## Executive Summary\n")
        report.append("This report provides a comprehensive exploratory data analysis of two S2TLD (Smart Traffic Light Detection) datasets with different resolutions.\n\n")
        
        # Dataset Overview
        report.append("## 1. Dataset Overview\n\n")
        for dataset_name, stats in self.results.items():
            report.append(f"### {dataset_name}\n")
            report.append(f"- **Resolution**: {stats['resolution'][0]}x{stats['resolution'][1]} pixels\n")
            report.append(f"- **Total Images**: {stats.get('total_images', 'N/A')}\n")
            report.append(f"- **Total Annotations**: {stats.get('total_annotations', 'N/A')}\n")
            report.append(f"- **Annotation Coverage**: {stats.get('annotation_coverage', 0):.2f}%\n")
            report.append(f"- **Total Objects**: {stats['total_objects']}\n")
            report.append(f"- **Classes**: {len(stats['class_distribution'])}\n\n")
        
        # Class Distribution
        report.append("## 2. Class Distribution Analysis\n\n")
        for dataset_name, stats in self.results.items():
            report.append(f"### {dataset_name}\n\n")
            report.append("| Class | Count | Percentage |\n")
            report.append("|-------|-------|------------|\n")
            for cls in sorted(stats['class_distribution'].keys()):
                count = stats['class_distribution'][cls]
                pct = stats['class_percentages'][cls]
                report.append(f"| {cls} | {count} | {pct:.2f}% |\n")
            report.append("\n")
            
            # Class imbalance analysis
            if stats['class_distribution']:
                max_class = max(stats['class_distribution'].items(), key=lambda x: x[1])
                min_class = min(stats['class_distribution'].items(), key=lambda x: x[1])
                imbalance_ratio = max_class[1] / min_class[1] if min_class[1] > 0 else float('inf')
                report.append(f"**Class Imbalance Ratio**: {imbalance_ratio:.2f}:1 (Most: {max_class[0]}, Least: {min_class[0]})\n\n")
        
        # Bounding Box Statistics
        report.append("## 3. Bounding Box Characteristics\n\n")
        for dataset_name, stats in self.results.items():
            report.append(f"### {dataset_name}\n\n")
            bbox_stats = stats['bbox_statistics']
            
            report.append("#### Width Statistics (pixels)\n")
            report.append(f"- Min: {bbox_stats['width']['min']:.2f}\n")
            report.append(f"- Max: {bbox_stats['width']['max']:.2f}\n")
            report.append(f"- Mean: {bbox_stats['width']['mean']:.2f}\n")
            report.append(f"- Median: {bbox_stats['width']['median']:.2f}\n\n")
            
            report.append("#### Height Statistics (pixels)\n")
            report.append(f"- Min: {bbox_stats['height']['min']:.2f}\n")
            report.append(f"- Max: {bbox_stats['height']['max']:.2f}\n")
            report.append(f"- Mean: {bbox_stats['height']['mean']:.2f}\n")
            report.append(f"- Median: {bbox_stats['height']['median']:.2f}\n\n")
            
            report.append("#### Area Statistics (pixels²)\n")
            report.append(f"- Min: {bbox_stats['area']['min']:.2f}\n")
            report.append(f"- Max: {bbox_stats['area']['max']:.2f}\n")
            report.append(f"- Mean: {bbox_stats['area']['mean']:.2f}\n")
            report.append(f"- Median: {bbox_stats['area']['median']:.2f}\n\n")
            
            report.append("#### Aspect Ratio Statistics (width/height)\n")
            report.append(f"- Min: {bbox_stats['aspect_ratio']['min']:.2f}\n")
            report.append(f"- Max: {bbox_stats['aspect_ratio']['max']:.2f}\n")
            report.append(f"- Mean: {bbox_stats['aspect_ratio']['mean']:.2f}\n")
            report.append(f"- Median: {bbox_stats['aspect_ratio']['median']:.2f}\n\n")
        
        # Objects per Image
        report.append("## 4. Objects per Image Statistics\n\n")
        for dataset_name, stats in self.results.items():
            report.append(f"### {dataset_name}\n\n")
            obj_stats = stats['objects_per_image']
            report.append(f"- Min: {obj_stats['min']}\n")
            report.append(f"- Max: {obj_stats['max']}\n")
            report.append(f"- Mean: {obj_stats['mean']:.2f}\n")
            report.append(f"- Median: {obj_stats['median']:.2f}\n")
            report.append(f"- Std Dev: {obj_stats['stdev']:.2f}\n\n")
        
        # Data Quality
        report.append("## 5. Data Quality Assessment\n\n")
        for dataset_name, stats in self.results.items():
            report.append(f"### {dataset_name}\n\n")
            quality = stats['quality_flags']
            report.append(f"- **Truncated Objects**: {quality['truncated_objects']} ({quality['truncated_percentage']:.2f}%)\n")
            report.append(f"- **Difficult Objects**: {quality['difficult_objects']} ({quality['difficult_percentage']:.2f}%)\n\n")
        
        # Spatial Distribution
        report.append("## 6. Spatial Distribution\n\n")
        for dataset_name, stats in self.results.items():
            report.append(f"### {dataset_name}\n\n")
            spatial = stats['spatial_distribution']
            report.append(f"- **Center X (normalized)**: Mean={spatial['center_x_mean']:.3f}, StdDev={spatial['center_x_stdev']:.3f}\n")
            report.append(f"- **Center Y (normalized)**: Mean={spatial['center_y_mean']:.3f}, StdDev={spatial['center_y_stdev']:.3f}\n\n")
        
        # Dataset Comparison
        report.append("## 7. Dataset Comparison\n\n")
        report.append("| Metric | S2TLD1080x1920 | S2TLD720x1280 |\n")
        report.append("|--------|----------------|---------------|\n")
        
        metrics = [
            ('Resolution', lambda s: f"{s['resolution'][0]}x{s['resolution'][1]}"),
            ('Total Images', lambda s: s.get('total_images', 'N/A')),
            ('Total Objects', lambda s: s['total_objects']),
            ('Avg Objects/Image', lambda s: f"{s['objects_per_image']['mean']:.2f}"),
            ('Number of Classes', lambda s: len(s['class_distribution'])),
            ('Avg BBox Width', lambda s: f"{s['bbox_statistics']['width']['mean']:.1f}px"),
            ('Avg BBox Height', lambda s: f"{s['bbox_statistics']['height']['mean']:.1f}px"),
            ('Truncated %', lambda s: f"{s['quality_flags']['truncated_percentage']:.2f}%")
        ]
        
        for metric_name, metric_func in metrics:
            row = f"| {metric_name} |"
            for dataset_name in ['S2TLD1080x1920', 'S2TLD720x1280']:
                if dataset_name in self.results:
                    row += f" {metric_func(self.results[dataset_name])} |"
                else:
                    row += " N/A |"
            report.append(row + "\n")
        report.append("\n")
        
        # Key Findings
        report.append("## 8. Key Findings\n\n")
        report.append("### Class Distribution\n")
        for dataset_name, stats in self.results.items():
            if stats['class_distribution']:
                most_common = max(stats['class_distribution'].items(), key=lambda x: x[1])
                least_common = min(stats['class_distribution'].items(), key=lambda x: x[1])
                report.append(f"- **{dataset_name}**: Most common class is '{most_common[0]}' ({most_common[1]} instances), ")
                report.append(f"least common is '{least_common[0]}' ({least_common[1]} instances)\n")
        report.append("\n")
        
        report.append("### Object Size Characteristics\n")
        for dataset_name, stats in self.results.items():
            avg_width = stats['bbox_statistics']['width']['mean']
            avg_height = stats['bbox_statistics']['height']['mean']
            res_width, res_height = stats['resolution']
            width_pct = (avg_width / res_width) * 100
            height_pct = (avg_height / res_height) * 100
            report.append(f"- **{dataset_name}**: Average object occupies {width_pct:.2f}% of image width ")
            report.append(f"and {height_pct:.2f}% of image height\n")
        report.append("\n")
        
        # Recommendations
        report.append("## 9. Recommendations\n\n")
        report.append("### Data Augmentation\n")
        report.append("- Apply horizontal flipping to increase dataset diversity\n")
        report.append("- Use brightness/contrast adjustments to simulate different lighting conditions\n")
        report.append("- Consider random cropping while preserving traffic light visibility\n\n")
        
        report.append("### Class Balancing\n")
        for dataset_name, stats in self.results.items():
            if stats['class_distribution']:
                max_class = max(stats['class_distribution'].items(), key=lambda x: x[1])
                min_class = min(stats['class_distribution'].items(), key=lambda x: x[1])
                imbalance_ratio = max_class[1] / min_class[1] if min_class[1] > 0 else float('inf')
                if imbalance_ratio > 3:
                    report.append(f"- **{dataset_name}**: Significant class imbalance detected (ratio: {imbalance_ratio:.2f}:1)\n")
                    report.append(f"  - Consider oversampling '{min_class[0]}' class or undersampling '{max_class[0]}' class\n")
                    report.append(f"  - Use weighted loss functions during training\n")
        report.append("\n")
        
        report.append("### Model Training Considerations\n")
        report.append("- Small object detection: Traffic lights are relatively small objects in the images\n")
        report.append("- Use anchor boxes appropriate for the average bounding box sizes\n")
        report.append("- Consider multi-scale feature pyramids for better small object detection\n")
        report.append("- Pay attention to truncated objects during training\n\n")
        
        report.append("### Dataset Split Recommendations\n")
        report.append("- Train: 70%, Validation: 15%, Test: 15%\n")
        report.append("- Ensure balanced class distribution across all splits\n")
        report.append("- Consider stratified splitting to maintain class proportions\n\n")
        
        # Write report
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print(f"\n✓ Report generated: {output_file}")
        return output_file
    
    def save_json_results(self, output_file: str = "eda_results.json"):
        """Save detailed results as JSON"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ JSON results saved: {output_file}")
        return output_file


def main():
    """Main execution function"""
    print("Starting Traffic Light Dataset EDA...\n")
    
    # Initialize EDA
    eda = TrafficLightEDA()
    
    # Run analysis
    results = eda.run_analysis()
    
    # Generate reports
    print("\n" + "=" * 80)
    print("Generating Reports...")
    print("=" * 80)
    
    markdown_report = eda.generate_report("EDA_Report.md")
    json_results = eda.save_json_results("eda_results.json")
    
    print("\n" + "=" * 80)
    print("EDA Complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  1. {markdown_report} - Comprehensive markdown report")
    print(f"  2. {json_results} - Detailed JSON results")
    print("\nNext steps:")
    print("  - Review the EDA report for insights")
    print("  - Use findings to inform data preprocessing")
    print("  - Plan model architecture based on object characteristics")
    print("  - Implement recommended data augmentation strategies")
    

if __name__ == "__main__":
    main()

# Made with Bob
