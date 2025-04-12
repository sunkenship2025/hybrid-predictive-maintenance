# Elbow Curve Summary: WCSS for Each Dataset

This table lists the WCSS values for different values of k (from 2 to 10) across all four datasets.

| Dataset | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| FD001 | 3246.17 | 2510.97 | 2315.82 | 2100.38 | 1724.82 | 1625.12 | 1564.63 | 1510.38 | 1479.32 |
| FD002 | 41150.21 | 19887.15 | 6537.17 | 1479.66 | 812.84 | 159.83 | 151.57 | 142.24 | 132.28 |
| FD003 | 4709.9 | 2911.12 | 2545.37 | 2339.51 | 1879.34 | 1741.17 | 1577.68 | 1492.31 | 1437.51 |
| FD004 | 46489.35 | 33708.72 | 7374.29 | 1481.33 | 740.49 | 241.32 | 213.28 | 198.29 | 188.09 |

## Notes
- As k increases, WCSS generally decreases.
- We typically look for the 'elbow point' — the spot where adding more clusters doesn’t improve WCSS much.
- For most of these datasets, k=5 gives a good balance between simplicity and separation.
