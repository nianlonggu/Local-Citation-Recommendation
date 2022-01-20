import cupy as cp
import numpy as np
from cupy.cuda import Device
import threading

class DPBool:
    ## The DPBool computes the Hamming Similarity of two sequences
    def __init__(self, explicit_vector_dim):
        ## the input explict_vector_dim is the explicit vector dim
        ## for example, if our embedding is 200 dimension bool, then the explicit vector dim is 200
        ## while the implicit vector dim is 200/8 = 25, the implicit vector dim is used for cupy kernel
        ## here it's not 200/32, as the unit input to matmul_dpbool is char, not int
        self.explicit_vector_dim = explicit_vector_dim
        self.implicit_vector_dim = int(np.ceil(explicit_vector_dim/8))
        self.raw_module = cp.RawModule(code=
        """
        #define VEC_DIM %d
        """ %( self.implicit_vector_dim ) + """
            extern "C"{
            __forceinline__ __device__
            // first version
            void dpbool( const unsigned char &a, const unsigned char &b, int &c,  int *lookup_table) {
                c += lookup_table[ (unsigned) a^b ];
            }           

            __global__ void matmul_dpbool( unsigned char *A, unsigned char *B, int *C, int A_height,  int A_width,  int *table ){           

                int start_row = threadIdx.y + blockIdx.y * blockDim.y;
                int stride = blockDim.y * gridDim.y;            

                __shared__ unsigned char Bs[VEC_DIM];
                __shared__ int lookup_table[256];
                // here we must use a blocksize of 256, to make sure it's consistent!
                lookup_table[ threadIdx.y ] = table[ threadIdx.y ];
                for(int currect_c = threadIdx.y ; currect_c< A_width; currect_c += blockDim.y ){
                    Bs[ currect_c] = B[ currect_c];
                }
                __syncthreads();            

                for( int current_row = start_row; current_row < A_height ;  current_row += stride  ){
                    
                    int Cvalue = 0;
                    for( int c = 0 ; c< A_width; c++ ){  
                        dpbool( A[ current_row * A_width + c ], Bs[c], Cvalue, lookup_table); 
                    }
                    C[current_row] = Cvalue;
                }
            }
            }

            """ ) 
        self.dp_bool = self.raw_module.get_function( "matmul_dpbool" )
        
        ## define a look up (bit set) table which can map uint8 to count of 1-bits
        lookup_table = np.zeros( 256, dtype= np.int32 )
        for i in range(256):
            lookup_table[i] = np.bitwise_and( i, 1 ) + lookup_table[  int(i/2) ]
        
        self.lookup_table = lookup_table            
    
    def dot(self, a, b ):
        assert a.shape[1] == b.shape[0] and a.shape[1] == self.implicit_vector_dim
        c = cp.zeros( a.shape[0], dtype= cp.int32 )
        lookup_table = cp.asarray(self.lookup_table)
        self.dp_bool( (1,68,1), (1,256,1), ( a, b, c, a.shape[0], a.shape[1], lookup_table   )  )
        
        ## We convert hamming distance to hamming similarity, as we always rank using simialrity
        c = self.explicit_vector_dim - c
        return c
    
    def pack_to_int8(self, a):
        packed_a = np.packbits( a, axis =1 ).astype(np.int8)
        return packed_a
    

class DPInt4:
    def __init__(self, explicit_vector_dim):
        ## the input explict_vector_dim is the explicit vector dim
        ## for example, if our embedding is 200 dimension int4, then the explicit vector dim is 200
        ## while the implicit vector dim is 200/8 = 25, the implicit vector dim is used for cupy kernel
        
        self.implicit_vector_dim = int(np.ceil(explicit_vector_dim/8))
        self.raw_module = cp.RawModule(code=
        """
        #define VEC_DIM %d
        """ %( self.implicit_vector_dim ) + """
        extern "C"{
        __forceinline__ __device__
        void dpint4( const int &a, const int &b, int &c, int *lookup_table) {
            for(int i = 0; i<8; ++i){
                c += lookup_table[((a >> ((8-i-1)*4)) & 0XF )] * lookup_table[((b >>( (8-i-1)*4)) & 0XF)];
            }
        }   
        
        __global__ void matmul_dpint4( int *A, int *B, int *C, int A_height,  int A_width, int *table ){        

            int start_row = threadIdx.y + blockIdx.y * blockDim.y;
            int stride = blockDim.y * gridDim.y;        

            __shared__ int Bs[VEC_DIM];
            __shared__ int lookup_table[256];
            
            //if ( threadIdx.y < 256 ){
                lookup_table[ threadIdx.y ] = table[ threadIdx.y ];
            //}
            
            for(int currect_c = threadIdx.y ; currect_c< A_width; currect_c += blockDim.y ){
                Bs[ currect_c] = B[ currect_c];
            }
            __syncthreads();        

            for( int current_row = start_row; current_row < A_height;  current_row += stride  ){
                        
                int Cvalue = 0;
                for( int c = 0 ; c< A_width; c++ ){
                    dpint4( A[ current_row * A_width + c ], Bs[c], Cvalue, lookup_table); 
                }
                C[current_row] = Cvalue;
            }
        }
                
        }
            """ ) # , backend= u'nvcc')        
        self.dp_int4 = self.raw_module.get_function( "matmul_dpint4" )
        self.lookup_table = np.zeros( 256, dtype = np.int32)
        for i in range(16):
            if i <= 7:
                self.lookup_table[i] = i
            else:
                self.lookup_table[i] = i - 16
            
    
    ## a and b are cupy int8 arrays, in each int8 value, the upper four-bits represent a value
    ## and a lower four-bits represent another value
    ## a has the shape [number_of_vectors, explicit_vector_dim/2], and b has the shape [explicit_vector_dim/2]
    ## return cupy int32 array c which has the shape [number_of_vectors]
    def dot(self, a, b ):
        assert a.shape[1] == b.shape[0] and a.dtype == "int8" and b.dtype == "int8"
        residual_dim = a.shape[1] % 4
        if residual_dim != 0:
            a = cp.concatenate([ a, cp.zeros( ( a.shape[0], 4-residual_dim ), dtype = cp.int8 ) ], axis = 1)
            b = cp.concatenate([ b, cp.zeros( ( 4 - residual_dim, ), dtype = cp.int8 ) ], axis = 0 )
        assert a.shape[1]/4 == self.implicit_vector_dim
        c = cp.zeros( a.shape[0], dtype= cp.int32 )
        lookup_table = cp.asarray(self.lookup_table)
        self.dp_int4( (1,68,1), (1,256,1), ( a, b, c, a.shape[0], int( a.shape[1]/4 ), lookup_table   )  )
        return c
    
    def pack_to_int8(self, a):
        ## assume that a has the type np.int32, and has 2 dimension
        if a.shape[1] % 2 != 0:
            a = np.concatenate([ a, np.zeros( ( a.shape[0], 1), dtype = a.dtype) ], axis = 1)
        packed_a = np.zeros( ( a.shape[0], int( a.shape[1]/2) ), dtype = np.int8 )
    
        for i in range(2):
            packed_a += np.bitwise_and(a[ :,i::2], 15)
            packed_a = packed_a << 4*(i<1)
        return packed_a
    

class DPInt8:
    def __init__(self, explicit_vector_dim):
        ## the input explict_vector_dim is the explicit vector dim
        ## for example, if our embedding is 200 dimension int8, then the explicit vector dim is 200
        ## while the implicit vector dim is 200/4 = 50, the implicit vector dim is used for cupy kernel
        
        self.implicit_vector_dim = int(np.ceil(explicit_vector_dim/4))
        self.raw_module = cp.RawModule(code=
        """
        #define VEC_DIM %d
        """ %( self.implicit_vector_dim ) + """
        extern "C"{
        __forceinline__ __device__
        void dp4a( const int &a, const int &b, int &c) {
            #if __CUDA_ARCH__ >= 610
              asm("dp4a.s32.s32 %0, %1, %2, %3;" : "+r"(c) : "r"(a), "r"(b), "r"(c)); 
            #else
              char4 &a4 = *((char4*)&a);
              char4 &b4 = *((char4*)&b);
              c += a4.x*b4.x;
              c += a4.y*b4.y;
              c += a4.z*b4.z;
              c += a4.w*b4.w;
            #endif
        }   
        

        __global__ void matmul_dp4a( int *A, int *B, int *C, int A_height,  int A_width ){        

            int start_row = threadIdx.y + blockIdx.y * blockDim.y;
            int stride = blockDim.y * gridDim.y;        

            __shared__ int Bs[VEC_DIM];
            for(int currect_c = threadIdx.y ; currect_c< A_width; currect_c += blockDim.y ){
                Bs[ currect_c] = B[ currect_c];
            }
            __syncthreads();        

            for( int current_row = start_row; current_row < A_height;  current_row += stride  ){
                        

                int Cvalue = 0;
                for( int c = 0 ; c< A_width; c++ ){
                    dp4a( A[ current_row * A_width + c ], Bs[c], Cvalue); 
                    //Cvalue =  A[ current_row * A_width + c ]* Bs[c] + Cvalue;
                }
                C[current_row] = Cvalue;
            }
        }
                
        }
            """ ) #, backend= u'nvcc')        
        self.dp_int8 = self.raw_module.get_function( "matmul_dp4a" )
    
    ## a and b are cupy int8 arrays
    ## a has the shape [number_of_vectors, explicit_vector_dim], and b has the shape [explicit_vector_dim]
    ## return cupy int32 array c which has the shape [number_of_vectors]
    def dot(self, a, b ):
        assert a.shape[1] == b.shape[0] and a.dtype == "int8" and b.dtype == "int8"
        residual_dim = a.shape[1] % 4
        if residual_dim != 0:
            a = cp.concatenate([ a, cp.zeros( ( a.shape[0], 4-residual_dim ), dtype = cp.int8 ) ], axis = 1)
            b = cp.concatenate([ b, cp.zeros( ( 4 - residual_dim, ), dtype = cp.int8 ) ], axis = 0 )
        assert a.shape[1]/4 == self.implicit_vector_dim 
        
        
        c = cp.zeros( a.shape[0], dtype= cp.int32 )
        self.dp_int8( (1,68,1), (1,256,1), ( a, b, c, a.shape[0], int( a.shape[1]/4 )  )  )
        return c



class BFIndexIPGPU:
    def __init__(self, embeddings, vector_dim, gpu_list = [], internal_precision = "float32", requires_precision_conversion = True):
        ## Here we assume that the embeddings is float32 by deafult, we internally convert it to internal precision
        ## for low precision computation
        ##  manually specify the vector_dim! As the precision may not be float32, so embeddings.shape[1] is not always the dimension size 
        self.internal_precision = internal_precision
        self.gpu_list = gpu_list
        assert len(gpu_list)>0
        
        self.dp_pool = {}
        for device_id in gpu_list:
            with Device(device_id):
                self.dp_pool[device_id] = {
                    "bool":DPBool(vector_dim),
                    "int4":DPInt4(vector_dim),
                    "int8":DPInt8(vector_dim)
                }
        if requires_precision_conversion:
            embeddings = self.convert_precision(embeddings, internal_precision )
        ## sharding
        self.total_num_embeddings = embeddings.shape[0]
        batch_size = int(np.ceil( embeddings.shape[0]/len(gpu_list) ))
        self.batch_size = batch_size
        self.embedding_list = []
        self.offset_list = []
        for i in range( len(gpu_list) ):
            with Device(gpu_list[i] ):
                subset_embeddings = embeddings[ batch_size * i : min(batch_size *(i+1), embeddings.shape[0] )  ]
                self.embedding_list.append( cp.asarray(subset_embeddings) )
                self.offset_list.append( cp.asarray( i * batch_size )  )
                
    # def convert_precision(self, embeddings, precision):
    #     if precision == "bool":
    #         embeddings = self.dp_pool[self.gpu_list[0]]["bool"].pack_to_int8(embeddings > 0)
    #     elif precision == "int4":
    #         embeddings = embeddings / max( np.abs(np.max(embeddings)), np.abs(np.min(embeddings)) ) * 8
    #         embeddings[ embeddings> 7 ] = 7
    #         embeddings[ embeddings < -8 ] = -8
    #         embeddings = self.dp_pool[self.gpu_list[0]]["int4"].pack_to_int8( embeddings.astype(np.int8) )
    #     elif precision == "int8":
    #         embeddings = embeddings / max( np.abs(np.max(embeddings)), np.abs(np.min(embeddings)) ) * 128
    #         embeddings[ embeddings> 127 ] = 127
    #         embeddings[ embeddings < -128 ] = -128
    #         embeddings = embeddings.astype(np.int8)
    #     else:
    #         embeddings = embeddings
    #     return embeddings

    def convert_precision(self, embeddings, precision):
        if precision == "bool":
            embeddings = self.dp_pool[self.gpu_list[0]]["bool"].pack_to_int8(embeddings > 0)
        elif precision == "int4":
            max_abs_value = np.std(np.abs(embeddings)) * 4
            if max_abs_value < 1e-12:
                embeddings = np.zeros_like(embeddings, dtype = np.int8 )
            else:
                embeddings = (embeddings/max_abs_value * 8).round()
                embeddings[ embeddings> 7 ] = 7
                embeddings[ embeddings < -8 ] = -8
                embeddings = embeddings.astype(np.int8)
            embeddings = self.dp_pool[self.gpu_list[0]]["int4"].pack_to_int8( embeddings )
        elif precision == "int8":
            max_abs_value = np.std(np.abs(embeddings)) * 4
            if max_abs_value < 1e-12:
                embeddings = np.zeros_like(embeddings, dtype = np.int8 )
            else:
                embeddings = (embeddings / max_abs_value * 128).round()
                embeddings[ embeddings> 127 ] = 127
                embeddings[ embeddings < -128 ] = -128
                embeddings = embeddings.astype(np.int8)
        else:
            embeddings = embeddings
        return embeddings


    ## Here the query_embedding has two dimensions, but there is only one vectors due to the limitation of current implementation
    def dp( self, embeddings, query_embedding,  precision, device_id ):
        if precision == "bool":
            product = self.dp_pool[device_id]["bool"].dot( embeddings, query_embedding[0] )
        elif precision == "int4":
            product = self.dp_pool[device_id]["int4"].dot( embeddings, query_embedding[0] )
        elif precision == "int8":
            product = self.dp_pool[device_id]["int8"].dot( embeddings, query_embedding[0] )
        else:
            product = cp.dot(embeddings, query_embedding[0]  )
        return product[cp.newaxis,:]
        

    ## Here the query_embedding has two dimensions, but there is only one vectors due to the limitation of current implementation
    def gpu_ranking_kernel( self, query_embedding, embeddings, n, device_id, indices_range= None):
        if indices_range is None:
            ### Here the distances are actually similarity, as we rank according to rge descending order of the similarity
            distances = self.dp( embeddings, query_embedding, self.internal_precision ,device_id)
            I = cp.argsort( -distances, axis = -1 )[:,:n]
            D = distances[ cp.arange(I.shape[0])[:,cp.newaxis].repeat(I.shape[1],axis = 1),  I ]
        else:
            if len(indices_range) > 0:
                distances = self.dp( embeddings[indices_range], query_embedding, self.internal_precision,device_id )
                I = cp.argsort( -distances, axis = -1 )[:,:n]
                D = distances[ cp.arange(I.shape[0])[:,cp.newaxis].repeat(I.shape[1],axis = 1),  I ]
                I = indices_range[I]
            else:
                I = cp.array([]).reshape( query_embedding.shape[0], 0 )
                D = cp.array([]).reshape( query_embedding.shape[0], 0 )
        return I, D
    
    ## Here the query_embedding has two dimensions, but there is only one vectors due to the limitation of current implementation
    def search(self, query_embedding, n, indices_range = None, requires_precision_conversion = True ):     
        assert len(query_embedding.shape)==2 and query_embedding.shape[0] == 1
        
        query_embedding_list = []
        if requires_precision_conversion:
            query_embedding = self.convert_precision( query_embedding, self.internal_precision)
        
        with Device( self.gpu_list[0] ):
            query_embedding = cp.asarray( query_embedding )
        
        I_list = []
        D_list = []
        
        for i in range( len(self.gpu_list) ):
            with Device( self.gpu_list[i] ):
                query_embedding_list.append( cp.asarray( query_embedding ) )
                
        if indices_range is None:
            indices_range_list = [None] * len(self.gpu_list)
        else:
            indices_range_list = []

            with Device( self.gpu_list[0] ):
                indices_range_gpu = cp.array(indices_range)

            for i in range( len(self.gpu_list) ):
                with Device( self.gpu_list[i] ):
                    indices_range = cp.asarray(indices_range_gpu)
                    sub_indices_pos = cp.logical_and( indices_range >= i * self.batch_size, indices_range< min( (i+1) * self.batch_size, self.total_num_embeddings  )    )
                    indices_range_list.append( indices_range[sub_indices_pos] - i * self.batch_size  )  
    
        for i in range( len(self.gpu_list) ):
            with Device( self.gpu_list[i] ):
                I, D = self.gpu_ranking_kernel(query_embedding_list[i],self.embedding_list[i], n, self.gpu_list[i], indices_range_list[i])
                I_list.append(I + self.offset_list[i] )
                D_list.append(D)
                
        if len(self.gpu_list) == 1:
            with Device( self.gpu_list[0] ):
                I = cp.asnumpy(I_list[0])
                D = cp.asnumpy(D_list[0])
        else:
            with Device( self.gpu_list[0] ):
                concated_I = cp.concatenate( [ cp.asarray(I) for I in I_list], axis = 1 )
                concated_D = cp.concatenate( [ cp.asarray(D) for D in D_list], axis = 1 )
                I = cp.argsort( -concated_D, axis = -1 )[:,:n] 
                row_indices = cp.arange(I.shape[0])[:,cp.newaxis].repeat(I.shape[1],axis = 1)
                D = concated_D[ row_indices ,  I]
                I = concated_I[ row_indices ,  I]

                I = cp.asnumpy(I)
                D = cp.asnumpy(D)                    
        return D, I


class BFIndexIPCPU:
    
    def __init__( self, embeddings, vector_dim , num_shards = 1 ):
        self.total_num_embeddings = embeddings.shape[0]
        shard_size = int(np.ceil( embeddings.shape[0]/num_shards ))
        self.shard_size = shard_size
        self.embedding_shards = []
        self.num_shards = num_shards
        for i in range( 0, embeddings.shape[0],shard_size ):
            self.embedding_shards.append( embeddings[i: i+shard_size] )
    
    def cpu_ranking_kernel( self, query_embedding,  n, shard_number, results, indices_range = None ):
        if indices_range is None:
            distances = np.matmul( query_embedding, self.embedding_shards[shard_number].T )
            I = np.argpartition( -distances, n-1, axis = -1 )[:,:n]
            D = distances[ np.arange(I.shape[0])[:,np.newaxis].repeat(I.shape[1],axis = 1),  I ]
        else:
            ## the parameters indices_range is the indices_range for the whole embedding matrix,
            ## in each thead, we need to convert it to the indices_range specially for current shard
            sub_indices_pos = np.logical_and( indices_range >= shard_number * self.shard_size, indices_range< min( (shard_number+1) * self.shard_size, self.total_num_embeddings  )    )
            indices_range = indices_range[sub_indices_pos] - shard_number * self.shard_size
            if len(indices_range) > 0:
                distances = np.matmul( query_embedding, self.embedding_shards[shard_number][indices_range].T )
                if len(indices_range) <= n:
                    I = np.tile(indices_range,query_embedding.shape[0]).reshape(query_embedding.shape[0], len(indices_range))
                    D = distances
                else:
                    I = np.argpartition( -distances, n-1, axis = -1 )[:,:n]
                    D = distances[ np.arange(I.shape[0])[:,np.newaxis].repeat(I.shape[1],axis = 1),  I ]
                    I = indices_range[I]
            else:
                I = np.array([]).reshape( query_embedding.shape[0], 0 )
                D = np.array([]).reshape( query_embedding.shape[0], 0 )
        ## return the real I by adding the corresponding shift
        results[shard_number] = (I + shard_number * self.shard_size ,D)
        
    def search( self, query_embedding, n , indices_range = None , requires_precision_conversion = None ):
        ## requires_precision_conversion is not used in CPU mode!
        if indices_range is not None:
            indices_range = np.array(indices_range)
        results = [None] * self.num_shards
        threads = []
        for shard_number in range( self.num_shards ):
            t = threading.Thread( target = self.cpu_ranking_kernel, args = ( query_embedding, n, shard_number, results, indices_range )  )
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
            
#         if self.num_shards == 1:
#             final_I, final_D = results[0]
#         else:
        I, D = list(zip( *results ))
        I = np.concatenate(I, axis = 1)
        D = np.concatenate(D, axis = 1)
        
        new_I = np.argsort( -D, axis = -1 )[:,:n]
        final_D = D[ np.arange(new_I.shape[0])[:,np.newaxis].repeat(new_I.shape[1],axis = 1),  new_I ]
        final_I = I[ np.arange(new_I.shape[0])[:,np.newaxis].repeat(new_I.shape[1],axis = 1),  new_I ]
        return  final_D, final_I.astype(np.int64)



class BFIndexIP:
    def __init__(self, embeddings, vector_dim, gpu_list = [], internal_precision = "float32", requires_precision_conversion = True ,num_shards = 1 ):
        ## num_shards is used for CPU mode
        ## internal_precision and requires_precision_conversion are only used for GPU mode
        if len(gpu_list) == 0:
            self.index = BFIndexIPCPU( embeddings, vector_dim , num_shards )
        else:
            self.index = BFIndexIPGPU( embeddings, vector_dim , gpu_list, internal_precision, requires_precision_conversion )
    def search( self, query_embedding, n , indices_range = None , requires_precision_conversion = True ):
        return self.index.search( query_embedding, n , indices_range, requires_precision_conversion )












"""
code backup Old version

"""



# import cupy as cp
# import numpy as np

# from cupy.cuda import Device


# class BFIndexIP:
#     def __init__(self, embeddings, gpu_list = [] ):
#         self.gpu_list = gpu_list
#         if len(gpu_list)>0:
#             ## sharding
#             self.total_num_embeddings = embeddings.shape[0]
#             batch_size = int(np.ceil( embeddings.shape[0]/len(gpu_list) ))
#             self.batch_size = batch_size
#             self.embedding_list = []
#             self.offset_list = []
#             for i in range( len(gpu_list) ):
#                 with Device(gpu_list[i] ):
#                     self.embedding_list.append( cp.asarray(embeddings[ batch_size * i : min(batch_size *(i+1), embeddings.shape[0] )  ], dtype = cp.float32) )
#                     self.offset_list.append( cp.asarray( i * batch_size )  )
#         else:
#             self.embeddings = embeddings
    
#     def gpu_ranking_kernel( self, query_embedding, embeddings, n, indices_range= None):
#         if indices_range is None:
#             distances = cp.matmul( query_embedding, embeddings.T )
#             I = cp.argsort( -distances, axis = -1 )[:,:n]
#             D = distances[ cp.arange(I.shape[0])[:,cp.newaxis].repeat(I.shape[1],axis = 1),  I ]
#         else:
#             if len(indices_range) > 0:
#                 distances = cp.matmul( query_embedding, embeddings[indices_range].T )
#                 I = cp.argsort( -distances, axis = -1 )[:,:n]
#                 D = distances[ cp.arange(I.shape[0])[:,cp.newaxis].repeat(I.shape[1],axis = 1),  I ]
#                 I = indices_range[I]
#             else:
#                 I = cp.array([]).reshape( query_embedding.shape[0], 0 )
#                 D = cp.array([]).reshape( query_embedding.shape[0], 0 )
#         return I, D
    
#     def search(self, query_embedding, n, indices_range = None):        
#         if len(self.gpu_list) == 0 :  ## using CPU (single thread for now)
#             if indices_range is None:
#                 distances = np.matmul( query_embedding, self.embeddings.T )
#                 I = np.argsort( -distances, axis = -1 )[:,:n]
#                 D = distances[ np.arange(I.shape[0])[:,np.newaxis].repeat(I.shape[1],axis = 1),  I ]
#             else:
#                 indices_range = np.array(indices_range)
#                 distances = np.matmul( query_embedding, self.embeddings[indices_range].T )
#                 I = np.argsort( -distances, axis = -1 )[:,:n]
#                 D = distances[ np.arange(I.shape[0])[:,np.newaxis].repeat(I.shape[1],axis = 1),  I ]
#                 I = indices_range[I]
#         else:
#             I_list = []
#             D_list = []
#             query_embedding_list = []
#             query_embedding = cp.asarray( query_embedding, dtype = cp.float32 )
#             for i in range( len(self.gpu_list) ):
#                 with Device( self.gpu_list[i] ):
#                     query_embedding_list.append( cp.asarray( query_embedding ) )
                    
#             if indices_range is None:
#                 indices_range_list = [None] * len(self.gpu_list)
#             else:
#                 indices_range_list = []
#                 indices_range_gpu = cp.array(indices_range)
#                 for i in range( len(self.gpu_list) ):
#                     with Device( self.gpu_list[i] ):
#                         indices_range = cp.asarray(indices_range_gpu)
#                         sub_indices_pos = cp.logical_and( indices_range >= i * self.batch_size, indices_range< min( (i+1) * self.batch_size, self.total_num_embeddings  )    )
#                         indices_range_list.append( indices_range[sub_indices_pos] - i * self.batch_size  )  
        
#             for i in range( len(self.gpu_list) ):
#                 with Device( self.gpu_list[i] ):
#                     I, D = self.gpu_ranking_kernel(query_embedding_list[i],self.embedding_list[i], n, indices_range_list[i])
#                     I_list.append(I + self.offset_list[i] )
#                     D_list.append(D)
                    
#             if len(self.gpu_list) == 1:
#                 I = cp.asnumpy(I_list[0])
#                 D = cp.asnumpy(D_list[0])
#             else:
#                 with Device( self.gpu_list[0] ):
#                     concated_I = cp.concatenate( [ cp.asarray(I) for I in I_list], axis = 1 )
#                     concated_D = cp.concatenate( [ cp.asarray(D) for D in D_list], axis = 1 )
#                     I = cp.argsort( -concated_D, axis = -1 )[:,:n] 
#                     row_indices = cp.arange(I.shape[0])[:,cp.newaxis].repeat(I.shape[1],axis = 1)
#                     D = concated_D[ row_indices ,  I]
#                     I = concated_I[ row_indices ,  I]
#                 I = cp.asnumpy(I)
#                 D = cp.asnumpy(D)                    
#         return D, I



# """
# The following code is for backup, which does not support multiple gpus
# """

# # class BFIndexIP:
# #     def __init__(self, embeddings, gpu = None ):
# #         self.gpu = gpu
# #         if gpu is not None:
# #             with Device(gpu):
# #                 self.embeddings = cp.asarray(embeddings, dtype = cp.float32)
# #         else:
# #             self.embeddings = embeddings
            
# #     def search(self, query_embedding, n, indices_range = None):        
# #         if self.gpu is None:
# #             if indices_range is None:
# #                 distances = np.matmul( query_embedding, self.embeddings.T )
# #                 I = np.argsort( -distances, axis = -1 )[:,:n]
# #                 D = distances[ np.arange(I.shape[0])[:,np.newaxis].repeat(I.shape[1],axis = 1),  I ]
# #             else:
# #                 indices_range = np.array(indices_range)
# #                 distances = np.matmul( query_embedding, self.embeddings[indices_range].T )
# #                 I = np.argsort( -distances, axis = -1 )[:,:n]
# #                 D = distances[ np.arange(I.shape[0])[:,np.newaxis].repeat(I.shape[1],axis = 1),  I ]
# #                 I = indices_range[I]
# #         else:
# #             with Device(self.gpu):
# #                 query_embedding = cp.asarray( query_embedding, dtype = cp.float32 )
# #                 if indices_range is None:
# #                     distances = cp.matmul( query_embedding,  self.embeddings.T )
# #                     I = cp.argsort( -distances, axis = -1 )[:,:n]
# #                     D = distances[ cp.arange(I.shape[0])[:,cp.newaxis].repeat(I.shape[1],axis = 1),  I ]
# #                 else:
# #                     indices_range = cp.array(indices_range)
# #                     distances = cp.matmul( query_embedding, self.embeddings[indices_range].T )
# #                     I = cp.argsort( -distances, axis = -1 )[:,:n]
# #                     D = distances[ cp.arange(I.shape[0])[:,cp.newaxis].repeat(I.shape[1],axis = 1),  I ]
# #                     I = indices_range[I]
# #                 I = cp.asnumpy(I)
# #                 D = cp.asnumpy(D)
                    
# #         return D, I