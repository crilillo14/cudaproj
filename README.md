## CUDA Projects for a parallel computing course

Had to SSH into UM server to access a GPU so some work is still on a remote server, but this is like half of what I worked on.

1. Parallelized bitonic sort, because Batcher's odd even merge sort was impossible.
2. Lenia in parallel; actually a very useful implementation of parallelism because running this on my cpu was not giving great performance. When in parallel, time complexity is O(k), where k is the size of the kernel; is independent of cell number.

### Compilation with GNU Make

To compile and run any project, cd to the dir and enter ```make```

