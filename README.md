# Predicting Sea Ice Extent in Distinct Artic Regions using Time Series Forecasting
Author: Sam Nicholls

CIND820 Big Data Analytics Project

Dr. Sedef Akinli Kocak

# Repository Contents:

This repository contains code to produce time series analyses of four distinct regions of the Arctic Ocean, using three different modelling approaches (SARIMAX, N-Beats Generic, N-Beats Interpretable). The forecasts were done based on Daily (N-Beats), Monthly (SARIMAX) and Quarterly (N-Beats + SARIMAX) periods to identify if any meaningful predictions could be made season-to-season, vs shorter-term predictions. As well, this also includes code for a variety of charts comparing the decomposed trend, seasonality and residuals of each region, along with comparisons of the seasons by region, and regions by season. These plots are used to visually compare the variation by season in each region over the time series. The sea ice extent datasets from the National Sea Ice Data Center (NSIDC) are also included in this repository.
## Final Results
|                     | **Baffin**   |              |           | **Beaufort Sea** |            |             | **Canadian Archipelago** |              |             | **Hudson Bay** |              |             |
|---------------------|--------------|--------------|-----------|------------------|------------|-------------|--------------------------|--------------|-------------|----------------|--------------|-------------|
|                     |  Auto-ARIMA  |   N-Beats-G  | N-Beats-I |    Auto-ARIMA    |  N-Beats-G |  N-Beats-I  |        Auto-ARIMA        |   N-Beats-G  |  N-Beats-I  |   Auto-ARIMA   |   N-Beats-G  |  N-Beats-I  |
|               **Daily** |              |              |           |                  |            |             |                          |              |             |                |              |             |
|                  R2 |      n/a     |  **0.98833** |  0.98695  |        n/a       |   0.89839  | **0.96073** |            n/a           | **0.954019** |   0.94915   |       n/a      |    0.99131   | **0.99292** |
|                MAPE |      n/a     | **16.94432** |  19.79779 |        n/a       |   8.52432  | **4.14381** |            n/a           |    2.98166   | **2.74478** |       n/a      |   18.92285   | **15.2363** |
|               sMAPE |      n/a     | **13.27464** |  15.11459 |        n/a       |   6.84649  | **3.78023** |            n/a           |    2.92814   | **2.63073** |       n/a      | **15.04069** |   15.67656  |
|               MARRE |      n/a     |  **2.65579** |  2.91796  |        n/a       |   4.87445  | **2.61312** |            n/a           |    3.19834   | **2.69333** |       n/a      |    2.63837   | **2.39122** |
|   Training Time (s) |      n/a     |   904.04113  | 559.02918 |        n/a       | 890.230859 |  553.00243  |            n/a           |   887.92348  |  561.76325  |       n/a      |   883.83436  |  584.03656  |
| Prediction Time (s) |      n/a     |   46.46027   |  35.81377 |        n/a       |  46.20545  |   33.80518  |            n/a           |   46.10252   |   34.10997  |       n/a      |   45.97361   |   59.71793  |
|             **Monthly** |              |              |           |                  |            |             |                          |              |             |                |              |             |
|                  R2 |    0.95925   |  **0.95971** |  0.94553  |    **0.77103**   |   0.4025   |   -0.07359  |        **0.88915**       |    0.77882   |   0.85948   |   **0.97985**  |    0.97699   |   0.97553   |
|                MAPE | **21.45724** |   23.90144   |  34.38548 |    **8.95861**   |  16.50034  |   21.08673  |        **3.93129**       |    5.85326   |   5.37962   |   **16.6964**  |   22.38528   |   22.44499  |
|               sMAPE |  **18.8881** |   20.09608   |  25.21789 |    **8.40088**   |  13.03654  |   15.58699  |        **3.86471**       |    5.38639   |   4.96282   |  **14.81207**  |   16.34717   |   19.99001  |
|               MARRE |    5.3604    |  **5.22964** |  6.53592  |    **8.04704**   |  14.61446  |   17.95432  |        **5.71048**       |    8.43689   |   5.92602   |   **3.10644**  |    3.7128    |   3.80279   |
|   Training Time (s) |   17.03464   |   36.91299   |  22.54911 |      2.62215     |  36.18951  |   21.31398  |          3.92723         |   36.23147   |   23.96711  |     2.92094    |   36.31641   |   22.03048  |
| Prediction Time (s) |   173.52375  |    4.21103   |  2.92394  |     219.92207    |   3.87826  |   3.02287   |         226.59936        |    3.85528   |   5.83589   |    151.52126   |    3.83629   |   2.90995   |
|           **Quarterly** |              |              |           |                  |            |             |                          |              |             |                |              |             |
|                  R2 |    0.95506   |  **0.95607** |  0.95419  |    **0.80756**   |   0.60143  |   0.38439   |        **0.95154**       |    0.82832   |   0.62669   |   **0.98761**  |    0.98513   |   0.98271   |
|                MAPE |  **16.0923** |   17.05139   |  16.32683 |    **7.51416**   |  10.67066  |   14.36183  |        **2.87633**       |    4.68692   |   7.60562   |  **10.78148**  |   12.11734   |   17.11309  |
|               sMAPE |   15.24705   | **15.07831** |  16.44092 |    **7.26027**   |   8.95738  |   11.87659  |        **2.93557**       |    4.40848   |   7.00642   |  **10.43096**  |   11.10885   |   14.09589  |
|               MARRE |    6.14296   |  **5.85148** |  6.27611  |    **8.57557**   |  11.16317  |   15.90653  |        **5.23136**       |    9.20753   |   15.0499   |   **2.89498**  |    3.29576   |   4.40523   |
|   Training Time (s) |    1.77375   |   24.04406   |  12.64409 |      1.21414     |  23.97211  |   12.76301  |          1.60387         |   25.69189   |   12.73502  |     1.17916    |   25.20724   |   13.38257  |
| Prediction Time (s) |   72.92862   |    2.19645   |  1.61386  |     30.47853     |   2.22143  |   1.60586   |         37.49559         |    2.27839   |   1.54591   |    35.03232    |    2.18146   |   1.59387   |
