cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from numpy cimport ndarray
import cython
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.boundscheck(False)
def minimum(ndarray[double,ndim=2] distances,ndarray[np.int64_t,ndim=2] sorted_indices,int N,int k):
    cdef int size=N
    cdef int counter=0
    cdef int m,min_pos,s,L
    cdef size_t i,j
    cdef float dist_i_sorted_j,dist_min_sorted_j
    cdef double[:,:] dis=distances
    cdef np.int64_t[:,:] sorted=sorted_indices
    L=size-k
    cdef int *removable = <int *>malloc(L * sizeof(int))
    while size > k:
        # Search for minimal distance
        min_pos = 0
        for i in range(1, N):
            for j in range(1, size):
                s=sorted[i,j]
                dist_i_sorted_j = dis[i,s]
                s=sorted[min_pos,j]
                dist_min_sorted_j = dis[min_pos,s]

                if dist_i_sorted_j < dist_min_sorted_j:
                    min_pos = i
                    break
                elif dist_i_sorted_j > dist_min_sorted_j:
                    break
        for i in range(N):
            dis[i,min_pos] = float("inf")
            dis[min_pos,i] = float("inf")

            for j in range(1, size - 1):
                if sorted[i,j] == min_pos:
                    m=j+1
                    sorted[i,j] = sorted[i,m]
                    sorted[i,m] = min_pos

        removable[counter]=min_pos
        counter+=1
        size -= 1
    try:
        return [ removable[i] for i in range(L) ]
    finally:
        free(removable)


