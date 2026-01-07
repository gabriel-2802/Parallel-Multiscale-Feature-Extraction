## Times for different threads/processes (ms) for big image


| N (threads/processes) | mpi (ms) | openmp (ms) | mpi_cuda (ms) | pthreads (ms) | mpi_openmp (ms) | pthreads_openmp (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1  | 181029 | 91605 | 20225 | 146977 | 154798 | 147870 |
| 2  | 105763 | 57762 | 20985 | 94933  | 90499  | 88111  |
| 4  | 71356  | 43541 | 21946 | 59475  | 67553  | 59663  |
| 8  | 52213  | 33967 | 24814 | 43736  | 50878  | 43531  |
| 16 | 45073  | 28735 | 25899 | 36262  | 46467  | 35890  |