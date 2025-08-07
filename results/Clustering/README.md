# Clustering-based Deduplication Results

## Experiment Information

**Execution Time**: 2025-06-01 19:14:37
**Duration**: 6.48 seconds

## Input Parameters

- **Input File**: `results/MinHash/after_minhash.json`
- **Output Directory**: `results/Clustering`
- **Target Range**: 80-120 designs
- **Method**: K-Means Clustering with Adaptive Representative Selection

## Algorithm Configuration

- **Feature Extraction**: 
  - Structural features (11 dimensions)
  - TF-IDF semantic features (up to 200 dimensions)
  - Feature standardization applied
- **Clustering Method**: K-Means with silhouette score optimization
- **Representative Selection**: Adaptive selection with cluster center + farthest point sampling

## Results Summary

### Dataset Statistics
- **Input Designs**: 857 designs
- **Selected Designs**: 72 designs
- **Selection Ratio**: 8.40%

### Clustering Performance
- **Optimal K**: 15 clusters
- **Silhouette Score**: 0.2229
- **Total Clusters**: 15
- **Covered Clusters**: 15
- **Cluster Coverage**: 100.00%

### Cluster Distribution
```
Cluster 0: 5 designs
Cluster 1: 5 designs
Cluster 2: 5 designs
Cluster 3: 5 designs
Cluster 4: 5 designs
Cluster 5: 5 designs
Cluster 6: 5 designs
Cluster 7: 5 designs
Cluster 8: 5 designs
Cluster 9: 5 designs
Cluster 10: 5 designs
Cluster 11: 2 designs
Cluster 12: 5 designs
Cluster 13: 5 designs
Cluster 14: 5 designs
```
### File Type Distribution
```
cpp: 60 files
h: 42 files
c: 27 files
hpp: 12 files
```

## Output Files

- `after_cluster.json`: Selected representative designs
- `clustering_results.png`: Visualization of clustering results
- `README.md`: This summary report

## Quality Metrics

- **Diversity**: 100.00% of clusters represented
- **Efficiency**: 91.6% reduction in dataset size
- **Clustering Quality**: Silhouette score of 0.2229

## Notes

The clustering-based approach groups similar designs and selects representative samples from each cluster. This ensures both diversity (covering different types of designs) and efficiency (reducing redundancy within similar designs).

The selection process prioritizes:
1. Cluster centers (most representative of each group)
2. Diverse samples within clusters (using farthest point sampling)
3. Proportional representation based on cluster sizes
