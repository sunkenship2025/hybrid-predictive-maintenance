# Elbow Curve Summary: WCSS for Each Dataset

This table lists the WCSS values for k=2 to 10.

| Dataset | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 | k=9 | k=10 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| FD001 | 3246.17 | 2510.98 | 2315.82 | 2100.38 | 1724.82 | 1625.12 | 1564.63 | 1510.38 | 1479.32 |
| FD002 | 41150.21 | 19887.15 | 6537.17 | 1479.66 | 812.84 | 159.83 | 151.57 | 142.24 | 132.28 |
| FD003 | 4709.9 | 2911.13 | 2545.37 | 2339.51 | 1879.34 | 1741.18 | 1577.69 | 1492.31 | 1437.51 |
| FD004 | 46489.35 | 33708.72 | 7374.29 | 1481.33 | 740.49 | 241.32 | 213.28 | 198.29 | 188.09 |

## Notes
- WCSS decreases with increasing clusters.
- The 'elbow point' helps choose the optimal k.
- For this project, k=5 is a good balance to reflect 5 degradation stages.
