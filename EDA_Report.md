# S2TLD Traffic Light Detection Dataset - EDA Report
*Generated: /Users/mac/Downloads/archive-2*

## Executive Summary
This report provides a comprehensive exploratory data analysis of two S2TLD (Smart Traffic Light Detection) datasets with different resolutions.

## 1. Dataset Overview

### S2TLD1080x1920
- **Resolution**: 1920x1080 pixels
- **Total Images**: 1222
- **Total Annotations**: 1222
- **Annotation Coverage**: 100.00%
- **Total Objects**: 2436
- **Classes**: 5

### S2TLD720x1280
- **Resolution**: 1280x720 pixels
- **Total Images**: 4564
- **Total Annotations**: 4564
- **Annotation Coverage**: 100.00%
- **Total Objects**: 11694
- **Classes**: 4

## 2. Class Distribution Analysis

### S2TLD1080x1920

| Class | Count | Percentage |
|-------|-------|------------|
| green | 816 | 33.50% |
| off | 4 | 0.16% |
| red | 1235 | 50.70% |
| wait_on | 306 | 12.56% |
| yellow | 75 | 3.08% |

**Class Imbalance Ratio**: 308.75:1 (Most: red, Least: off)

### S2TLD720x1280

| Class | Count | Percentage |
|-------|-------|------------|
| green | 4528 | 38.72% |
| off | 503 | 4.30% |
| red | 6473 | 55.35% |
| yellow | 190 | 1.62% |

**Class Imbalance Ratio**: 34.07:1 (Most: red, Least: yellow)

## 3. Bounding Box Characteristics

### S2TLD1080x1920

#### Width Statistics (pixels)
- Min: 6.00
- Max: 320.00
- Mean: 54.54
- Median: 41.00

#### Height Statistics (pixels)
- Min: 0.00
- Max: 170.00
- Mean: 40.87
- Median: 32.00

#### Area Statistics (pixels²)
- Min: 0.00
- Max: 15850.00
- Mean: 2137.10
- Median: 1272.00

#### Aspect Ratio Statistics (width/height)
- Min: 0.00
- Max: 11.00
- Mean: 1.96
- Median: 1.57

### S2TLD720x1280

#### Width Statistics (pixels)
- Min: 5.00
- Max: 117.00
- Mean: 21.47
- Median: 20.00

#### Height Statistics (pixels)
- Min: 6.00
- Max: 272.00
- Mean: 48.78
- Median: 46.00

#### Area Statistics (pixels²)
- Min: 36.00
- Max: 31824.00
- Mean: 1250.21
- Median: 903.00

#### Aspect Ratio Statistics (width/height)
- Min: 0.18
- Max: 1.80
- Mean: 0.46
- Median: 0.42

## 4. Objects per Image Statistics

### S2TLD1080x1920

- Min: 0
- Max: 5
- Mean: 1.99
- Median: 2.00
- Std Dev: 0.78

### S2TLD720x1280

- Min: 1
- Max: 9
- Mean: 2.56
- Median: 2.00
- Std Dev: 1.26

## 5. Data Quality Assessment

### S2TLD1080x1920

- **Truncated Objects**: 0 (0.00%)
- **Difficult Objects**: 0 (0.00%)

### S2TLD720x1280

- **Truncated Objects**: 373 (3.19%)
- **Difficult Objects**: 0 (0.00%)

## 6. Spatial Distribution

### S2TLD1080x1920

- **Center X (normalized)**: Mean=0.516, StdDev=0.174
- **Center Y (normalized)**: Mean=0.305, StdDev=0.119

### S2TLD720x1280

- **Center X (normalized)**: Mean=0.491, StdDev=0.181
- **Center Y (normalized)**: Mean=0.268, StdDev=0.208

## 7. Dataset Comparison

| Metric | S2TLD1080x1920 | S2TLD720x1280 |
|--------|----------------|---------------|
| Resolution | 1920x1080 | 1280x720 |
| Total Images | 1222 | 4564 |
| Total Objects | 2436 | 11694 |
| Avg Objects/Image | 1.99 | 2.56 |
| Number of Classes | 5 | 4 |
| Avg BBox Width | 54.5px | 21.5px |
| Avg BBox Height | 40.9px | 48.8px |
| Truncated % | 0.00% | 3.19% |

## 8. Key Findings

### Class Distribution
- **S2TLD1080x1920**: Most common class is 'red' (1235 instances), least common is 'off' (4 instances)
- **S2TLD720x1280**: Most common class is 'red' (6473 instances), least common is 'yellow' (190 instances)

### Object Size Characteristics
- **S2TLD1080x1920**: Average object occupies 2.84% of image width and 3.78% of image height
- **S2TLD720x1280**: Average object occupies 1.68% of image width and 6.77% of image height

## 9. Recommendations

### Data Augmentation
- Apply horizontal flipping to increase dataset diversity
- Use brightness/contrast adjustments to simulate different lighting conditions
- Consider random cropping while preserving traffic light visibility

### Class Balancing
- **S2TLD1080x1920**: Significant class imbalance detected (ratio: 308.75:1)
  - Consider oversampling 'off' class or undersampling 'red' class
  - Use weighted loss functions during training
- **S2TLD720x1280**: Significant class imbalance detected (ratio: 34.07:1)
  - Consider oversampling 'yellow' class or undersampling 'red' class
  - Use weighted loss functions during training

### Model Training Considerations
- Small object detection: Traffic lights are relatively small objects in the images
- Use anchor boxes appropriate for the average bounding box sizes
- Consider multi-scale feature pyramids for better small object detection
- Pay attention to truncated objects during training

### Dataset Split Recommendations
- Train: 70%, Validation: 15%, Test: 15%
- Ensure balanced class distribution across all splits
- Consider stratified splitting to maintain class proportions

