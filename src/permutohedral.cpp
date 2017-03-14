/*
    Copyright (c) 2013, Philipp Krähenbühl
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
        * Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
        * Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.
        * Neither the name of the Stanford University nor the
        names of its contributors may be used to endorse or promote products
        derived from this software without specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY Philipp Krähenbühl ''AS IS'' AND ANY
    EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL Philipp Krähenbühl BE LIABLE FOR ANY
    DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
    ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
    SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "permutohedral.h"

#ifdef WIN32
inline int round(double X) {
	return int(X+.5);
}
#endif

#ifdef __SSE__
// SSE Permutoheral lattice
# define SSE_PERMUTOHEDRAL
#endif

#if defined(SSE_PERMUTOHEDRAL)
# include <emmintrin.h>
# include <xmmintrin.h>
# ifdef __SSE4_1__
#  include <smmintrin.h>
# endif
#endif

#define STRICT_INEQUALITY false  // used in "_ord" functions: if true, consider strict inequalty 

/************************************************/
/***                Hash Table                ***/
/************************************************/

class HashTable{
protected:
	size_t key_size_, filled_, capacity_;
	std::vector< short > keys_;
	std::vector< int > table_;
	void grow(){
		// Create the new memory and copy the values in
		int old_capacity = capacity_;
		capacity_ *= 2;
		std::vector<short> old_keys( (old_capacity+10)*key_size_ );
		std::copy( keys_.begin(), keys_.end(), old_keys.begin() );
		std::vector<int> old_table( capacity_, -1 );
		
		// Swap the memory
		table_.swap( old_table );
		keys_.swap( old_keys );
		
		// Reinsert each element
		for( int i=0; i<old_capacity; i++ )
			if (old_table[i] >= 0){
				int e = old_table[i];
				size_t h = hash( getKey(e) ) % capacity_;
				for(; table_[h] >= 0; h = h<capacity_-1 ? h+1 : 0);
				table_[h] = e;
			}
	}
	size_t hash( const short * k ) {
		size_t r = 0;
		for( size_t i=0; i<key_size_; i++ ){
			r += k[i];
			r *= 1664525;
		}
		return r;
	}
public:
	explicit HashTable( int key_size, int n_elements ) : key_size_ ( key_size ), filled_(0), capacity_(2*n_elements), keys_((capacity_/2+10)*key_size_), table_(2*n_elements,-1) {
	}
	int size() const {
		return filled_;
	}
	void reset() {
		filled_ = 0;
		std::fill( table_.begin(), table_.end(), -1 );
	}
	int find( const short * k, bool create = false ){
		if (2*filled_ >= capacity_) grow();
		// Get the hash value
		size_t h = hash( k ) % capacity_;
		// Find the element with he right key, using linear probing
		while(1){
			int e = table_[h];
			if (e==-1){
				if (create){
					// Insert a new key and return the new id
					for( size_t i=0; i<key_size_; i++ )
						keys_[ filled_*key_size_+i ] = k[i];
					return table_[h] = filled_++;
				}
				else
					return -1;
			}
			// Check if the current key is The One
			bool good = true;
			for( size_t i=0; i<key_size_ && good; i++ )
				if (keys_[ e*key_size_+i ] != k[i])
					good = false;
			if (good)
				return e;
			// Continue searching
			h++;
			if (h==capacity_) h = 0;
		}
	}
	const short * getKey( int i ) const{
		return &keys_[i*key_size_];
	}

};

/************************************************/
/***          Permutohedral Lattice           ***/
/************************************************/

Permutohedral::Permutohedral():N_( 0 ), M_( 0 ), d_( 0 ) {
}
#ifdef SSE_PERMUTOHEDRAL
void Permutohedral::init ( const MatrixXf & feature )
{
	// Compute the lattice coordinates for each feature [there is going to be a lot of magic here
	N_ = feature.cols();
	d_ = feature.rows();
	HashTable hash_table( d_, N_/**(d_+1)*/ );
	
	const int blocksize = sizeof(__m128) / sizeof(float);
	const __m128 invdplus1   = _mm_set1_ps( 1.0f / (d_+1) );
	const __m128 dplus1      = _mm_set1_ps( d_+1 );
	const __m128 Zero        = _mm_set1_ps( 0 );
	const __m128 One         = _mm_set1_ps( 1 );

	// Allocate the class memory
	offset_.resize( (d_+1)*(N_+16) );
	std::fill( offset_.begin(), offset_.end(), 0 );
	barycentric_.resize( (d_+1)*(N_+16) );
	std::fill( barycentric_.begin(), barycentric_.end(), 0 );
	rank_.resize( (d_+1)*(N_+16) );
	
	// Allocate the local memory
	__m128 * scale_factor = (__m128*) _mm_malloc( (d_  )*sizeof(__m128) , 16 );
	__m128 * f            = (__m128*) _mm_malloc( (d_  )*sizeof(__m128) , 16 );
	__m128 * elevated     = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128) , 16 );
	__m128 * rem0         = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128) , 16 );
	__m128 * rank         = (__m128*) _mm_malloc( (d_+1)*sizeof(__m128), 16 );
	float * barycentric = new float[(d_+2)*blocksize];
	short * canonical = new short[(d_+1)*(d_+1)];
	short * key = new short[d_+1];
	
	// Compute the canonical simplex
	for( int i=0; i<=d_; i++ ){
		for( int j=0; j<=d_-i; j++ )
			canonical[i*(d_+1)+j] = i;
		for( int j=d_-i+1; j<=d_; j++ )
			canonical[i*(d_+1)+j] = i - (d_+1);
	}
	
	// Expected standard deviation of our filter (p.6 in [Adams etal 2010])
	float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
	// Compute the diagonal part of E (p.5 in [Adams etal 2010])
	for( int i=0; i<d_; i++ )
		scale_factor[i] = _mm_set1_ps( 1.0 / sqrt( (i+2)*(i+1) ) * inv_std_dev );
	
	// Setup the SSE rounding
#ifndef __SSE4_1__
	const unsigned int old_rounding = _mm_getcsr();
	_mm_setcsr( (old_rounding&~_MM_ROUND_MASK) | _MM_ROUND_NEAREST );
#endif

	// Compute the simplex each feature lies in
	for( int k=0; k<N_; k+=blocksize ){
		// Load the feature from memory
		float * ff = (float*)f;
		for( int j=0; j<d_; j++ )
			for( int i=0; i<blocksize; i++ )
				ff[ j*blocksize + i ] = k+i < N_ ? feature(j,k+i) : 0.0;
		
		// Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
		
		// sm contains the sum of 1..n of our faeture vector
		__m128 sm = Zero;
		for( int j=d_; j>0; j-- ){
			__m128 cf = f[j-1]*scale_factor[j-1];
			elevated[j] = sm - _mm_set1_ps(j)*cf;
			sm += cf;
		}
		elevated[0] = sm;
		
		// Find the closest 0-colored simplex through rounding
		__m128 sum = Zero;
		for( int i=0; i<=d_; i++ ){
			__m128 v = invdplus1 * elevated[i];
#ifdef __SSE4_1__
			v = _mm_round_ps( v, _MM_FROUND_TO_NEAREST_INT );
#else
			v = _mm_cvtepi32_ps( _mm_cvtps_epi32( v ) );
#endif
			rem0[i] = v*dplus1;
			sum += v;
		}
		
		// Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
		for( int i=0; i<=d_; i++ )
			rank[i] = Zero;
		for( int i=0; i<d_; i++ ){
			__m128 di = elevated[i] - rem0[i];
			for( int j=i+1; j<=d_; j++ ){
				__m128 dj = elevated[j] - rem0[j];
				__m128 c = _mm_and_ps( One, _mm_cmplt_ps( di, dj ) );
				rank[i] += c;
				rank[j] += One-c;
			}
		}
		
		// If the point doesn't lie on the plane (sum != 0) bring it back
		for( int i=0; i<=d_; i++ ){
			rank[i] += sum;
			__m128 add = _mm_and_ps( dplus1, _mm_cmplt_ps( rank[i], Zero ) );
			__m128 sub = _mm_and_ps( dplus1, _mm_cmpge_ps( rank[i], dplus1 ) );
			rank[i] += add-sub;
			rem0[i] += add-sub;
		}
		
		// Compute the barycentric coordinates (p.10 in [Adams etal 2010])
		for( int i=0; i<(d_+2)*blocksize; i++ )
			barycentric[ i ] = 0;
		for( int i=0; i<=d_; i++ ){
			__m128 v = (elevated[i] - rem0[i])*invdplus1;
			
			// Didn't figure out how to SSE this
			float * fv = (float*)&v;
			float * frank = (float*)&rank[i];
			for( int j=0; j<blocksize; j++ ){
				int p = d_-frank[j];
				barycentric[j*(d_+2)+p  ] += fv[j];
				barycentric[j*(d_+2)+p+1] -= fv[j];
			}
		}
		
		// The rest is not SSE'd
		for( int j=0; j<blocksize; j++ ){
			// Wrap around
			barycentric[j*(d_+2)+0]+= 1 + barycentric[j*(d_+2)+d_+1];
			
			float * frank = (float*)rank;
			float * frem0 = (float*)rem0;
			// Compute all vertices and their offset
			for( int remainder=0; remainder<=d_; remainder++ ){
				for( int i=0; i<d_; i++ ){
					key[i] = frem0[i*blocksize+j] + canonical[ remainder*(d_+1) + (int)frank[i*blocksize+j] ];
				}
				offset_[ (j+k)*(d_+1)+remainder ] = hash_table.find( key, true );
				rank_[ (j+k)*(d_+1)+remainder ] = frank[remainder*blocksize+j];
				barycentric_[ (j+k)*(d_+1)+remainder ] = barycentric[ j*(d_+2)+remainder ];
			}
		}
	}
	_mm_free( scale_factor );
	_mm_free( f );
	_mm_free( elevated );
	_mm_free( rem0 );
	_mm_free( rank );
	delete [] barycentric;
	delete [] canonical;
	delete [] key;
	
	// Reset the SSE rounding
#ifndef __SSE4_1__
	_mm_setcsr( old_rounding );
#endif
	
	// This is normally fast enough so no SSE needed here
	// Find the Neighbors of each lattice point
	
	// Get the number of vertices in the lattice
	M_ = hash_table.size();
	
	// Create the neighborhood structure
	blur_neighbors_.resize( (d_+1)*M_ );
	
	short * n1 = new short[d_+1];
	short * n2 = new short[d_+1];
	
	// For each of d+1 axes,
	for( int j = 0; j <= d_; j++ ){
		for( int i=0; i<M_; i++ ){
			const short * key = hash_table.getKey( i );
			for( int k=0; k<d_; k++ ){
				n1[k] = key[k] - 1;
				n2[k] = key[k] + 1;
			}
			n1[j] = key[j] + d_;
			n2[j] = key[j] - d_;
			
			blur_neighbors_[j*M_+i].n1 = hash_table.find( n1 );
			blur_neighbors_[j*M_+i].n2 = hash_table.find( n2 );
		}
	}
	delete[] n1;
	delete[] n2;
}
#else
void Permutohedral::init ( const MatrixXf & feature )
{
	// Compute the lattice coordinates for each feature [there is going to be a lot of magic here
	N_ = feature.cols();
	d_ = feature.rows();
	HashTable hash_table( d_, N_*(d_+1) );

	// Allocate the class memory
	offset_.resize( (d_+1)*N_ );
	rank_.resize( (d_+1)*N_ );
	barycentric_.resize( (d_+1)*N_ );
	
	// Allocate the local memory
	float * scale_factor = new float[d_];
	float * elevated = new float[d_+1];
	float * rem0 = new float[d_+1];
	float * barycentric = new float[d_+2];
	short * rank = new short[d_+1];
	short * canonical = new short[(d_+1)*(d_+1)];
	short * key = new short[d_+1];
	
	// Compute the canonical simplex
	for( int i=0; i<=d_; i++ ){
		for( int j=0; j<=d_-i; j++ )
			canonical[i*(d_+1)+j] = i;
		for( int j=d_-i+1; j<=d_; j++ )
			canonical[i*(d_+1)+j] = i - (d_+1);
	}
	
	// Expected standard deviation of our filter (p.6 in [Adams etal 2010])
	float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
	// Compute the diagonal part of E (p.5 in [Adams etal 2010])
	for( int i=0; i<d_; i++ )
		scale_factor[i] = 1.0 / sqrt( double((i+2)*(i+1)) ) * inv_std_dev;
	
	// Compute the simplex each feature lies in
	for( int k=0; k<N_; k++ ){
		// Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
		const float * f = &feature(0,k);
		
		// sm contains the sum of 1..n of our faeture vector
		float sm = 0;
		for( int j=d_; j>0; j-- ){
			float cf = f[j-1]*scale_factor[j-1];
			elevated[j] = sm - j*cf;
			sm += cf;
		}
		elevated[0] = sm;
		
		// Find the closest 0-colored simplex through rounding
		float down_factor = 1.0f / (d_+1);
		float up_factor = (d_+1);
		int sum = 0;
		for( int i=0; i<=d_; i++ ){
			//int rd1 = round( down_factor * elevated[i]);
			int rd2;
			float v = down_factor * elevated[i];
			float up = ceilf(v)*up_factor;
			float down = floorf(v)*up_factor;
			if (up - elevated[i] < elevated[i] - down) rd2 = (short)up;
			else rd2 = (short)down;

			//if(rd1!=rd2)
			//	break;

			rem0[i] = rd2;
			sum += rd2*down_factor;
		}
		
		// Find the simplex we are in and store it in rank (where rank describes what position coorinate i has in the sorted order of the features values)
		for( int i=0; i<=d_; i++ )
			rank[i] = 0;
		for( int i=0; i<d_; i++ ){
			double di = elevated[i] - rem0[i];
			for( int j=i+1; j<=d_; j++ )
				if ( di < elevated[j] - rem0[j])
					rank[i]++;
				else
					rank[j]++;
		}
		
		// If the point doesn't lie on the plane (sum != 0) bring it back
		for( int i=0; i<=d_; i++ ){
			rank[i] += sum;
			if ( rank[i] < 0 ){
				rank[i] += d_+1;
				rem0[i] += d_+1;
			}
			else if ( rank[i] > d_ ){
				rank[i] -= d_+1;
				rem0[i] -= d_+1;
			}
		}
		
		// Compute the barycentric coordinates (p.10 in [Adams etal 2010])
		for( int i=0; i<=d_+1; i++ )
			barycentric[i] = 0;
		for( int i=0; i<=d_; i++ ){
			float v = (elevated[i] - rem0[i])*down_factor;
			barycentric[d_-rank[i]  ] += v;
			barycentric[d_-rank[i]+1] -= v;
		}
		// Wrap around
		barycentric[0] += 1.0 + barycentric[d_+1];
		
		// Compute all vertices and their offset
		for( int remainder=0; remainder<=d_; remainder++ ){
			for( int i=0; i<d_; i++ )
				key[i] = rem0[i] + canonical[ remainder*(d_+1) + rank[i] ];
			offset_[ k*(d_+1)+remainder ] = hash_table.find( key, true );
			rank_[ k*(d_+1)+remainder ] = rank[remainder];
			barycentric_[ k*(d_+1)+remainder ] = barycentric[ remainder ];
		}
	}
	delete [] scale_factor;
	delete [] elevated;
	delete [] rem0;
	delete [] barycentric;
	delete [] rank;
	delete [] canonical;
	delete [] key;
	
	
	// Find the Neighbors of each lattice point
	
	// Get the number of vertices in the lattice
	M_ = hash_table.size();
	
	// Create the neighborhood structure
	blur_neighbors_.resize( (d_+1)*M_ );
	
	short * n1 = new short[d_+1];
	short * n2 = new short[d_+1];
	
	// For each of d+1 axes,
	for( int j = 0; j <= d_; j++ ){
		for( int i=0; i<M_; i++ ){
			const short * key = hash_table.getKey( i );
			for( int k=0; k<d_; k++ ){
				n1[k] = key[k] - 1;
				n2[k] = key[k] + 1;
			}
			n1[j] = key[j] + d_;
			n2[j] = key[j] - d_;
			
			blur_neighbors_[j*M_+i].n1 = hash_table.find( n1 );
			blur_neighbors_[j*M_+i].n2 = hash_table.find( n2 );
		}
	}
	delete[] n1;
	delete[] n2;
}
#endif
void Permutohedral::seqCompute ( float* out, const float* in, int value_size, bool reverse ) const
{
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	float * values = new float[ (M_+2)*value_size ];
	float * new_values = new float[ (M_+2)*value_size ];
	
	for( int i=0; i<(M_+2)*value_size; i++ )
		values[i] = new_values[i] = 0;
	
	// Splatting
	for( int i=0;  i<N_; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ )
				values[ o*value_size+k ] += w * in[ i*value_size+k ];
		}
	}

	// Blurring
	for( int j=reverse?d_:0; j<=d_ && j>=0; reverse?j--:j++ ){
		for( int i=0; i<M_; i++ ){
			float * old_val = values + (i+1)*value_size;
			float * new_val = new_values + (i+1)*value_size;
			
			int n1 = blur_neighbors_[j*M_+i].n1+1;
			int n2 = blur_neighbors_[j*M_+i].n2+1;
			float * n1_val = values + n1*value_size;
			float * n2_val = values + n2*value_size;
			for( int k=0; k<value_size; k++ )
				new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
		}
		std::swap( values, new_values );
	}
	// Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
	float alpha = 1.0f / (1+powf(2, -d_)); // 0.8 in 2D / 0.97 in 5D
	
	// Slicing
	for( int i=0; i<N_; i++ ){
		for( int k=0; k<value_size; k++ )
			out[i*value_size+k] = 0;
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ )
				out[ i*value_size+k ] += w * values[ o*value_size+k ] * alpha;
		}
	}
	
	
	delete[] values;
	delete[] new_values;
}
void printSplitArray(split_array *in) {
	float *in_f = (float *)in;
	int precision = 10;
	for(int i=precision-1; i>=0; --i) {
		printf("%3d: %f\n", i*RESOLUTION/precision, in_f[int(i*RESOLUTION/precision)]);
	}
}
void addSplitArray(split_array *out, float alpha, float up_to, bool from_top, int *k_bin_count) {
	float *out_f = (float *)out;
    if (from_top) {	// the pixels that have lesser Q value than the current one influence the current pixel
        int coeff = std::max(int(floor((up_to-1e-9)*RESOLUTION)), 0);
        assert(coeff >= 0 && coeff < RESOLUTION);
        k_bin_count[coeff] += 1;
#if STRICT_INEQUALITY   // don't influence pixels that belongs to the same bin ++coeff;
#endif
		for(int i=coeff; i<RESOLUTION; ++i) {
			out_f[i] += alpha;
		}
	} else {	// the pixels that have greater Q value than the current one influence the current pixel
		int coeff = std::min(int(floor((up_to)*RESOLUTION)), RESOLUTION-1);
		assert(coeff >= 0 && coeff < RESOLUTION);
                k_bin_count[coeff] += 1;
#if STRICT_INEQUALITY   // don't influence pixels that belongs to the same bin
        --coeff;
#endif
		for(int i=0; i<=coeff; ++i) {
			out_f[i] += alpha;
		}
	}
}
void weightedAddSplitArray(split_array *out, split_array *in1, float alpha, split_array *in2, split_array *in3) {
	float *out_f = (float *)out;
	float *in1_f = (float *)in1;
	float *in2_f = (float *)in2;
	float *in3_f = (float *)in3;
	for(int i=0; i<RESOLUTION; ++i) {
		out_f[i] = in1_f[i] + alpha * (in2_f[i] + in3_f[i]);
	}
}
void sliceSplitArray(float *out, float alpha, float up_to, split_array *in, bool from_top, int *k_bin_count) {
	float *in_f = (float *)in;
	int coeff;
	if (from_top) {	
		coeff = std::max(int(floor((up_to-1e-9)*RESOLUTION)), 0);
	} else {	
		coeff = std::min(int(floor((up_to)*RESOLUTION)), RESOLUTION-1);
	}
	assert(coeff >= 0 && coeff < RESOLUTION);
//	*out += in_f[coeff] * alpha/k_bin_count[coeff];
       *out += in_f[coeff] * alpha;
}

void Permutohedral::seqCompute_upper_minus_lower_ord (float* out, const float* in, int value_size) const { 
        const int bins = RESOLUTION;
        int binCount[value_size][bins];
        memset(binCount, 0, value_size*sizeof(split_array));
   
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	split_array * values = new split_array[ (M_+2)*value_size ];
	split_array * new_values = new split_array[ (M_+2)*value_size ];

	// Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
	float alpha = 1.0f / (1+powf(2, -d_)); // 0.8 in 2D / 0.97 in 5D
	
	memset(values, 0, (M_+2)*value_size*sizeof(split_array));
	memset(new_values, 0, (M_+2)*value_size*sizeof(split_array));

	// Lower
	// Splatting
	for( int i=0;  i<N_; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ ) {
				addSplitArray(&values[ o*value_size+k ], w, in[ i*value_size+k ], false, binCount[k]);
                // values[ o*value_size+k ] += w * in[ i*value_size+k ];
                //printf("#%d\n", o*value_size+k);
                //printSplitArray(&values[ o*value_size+k ]);
			}
		}
	}

        for( int k=0; k<value_size; k++ ) {
            for( int i=0;  i<RESOLUTION; i++ ){
                binCount[k][i] = binCount[k][i]/(d_+1);
               binCount[k][i] = 1;
            }
        }

       // std::cout << "Printing bin counts" << std::endl;
       // for(int i = 0; i < value_size; i ++){
       //     for(int j = 0; j < bins; j ++)
       //         std::cout << binCount[i][j] << " ";
       //     std::cout << std::endl;
       // }

	// Blurring
	for( int j=0; j<=d_; ++j ){
		for( int i=0; i<M_; i++ ){
			split_array * old_val = values + (i+1)*value_size;
			split_array * new_val = new_values + (i+1)*value_size;
			
			int n1 = blur_neighbors_[j*M_+i].n1+1;
			int n2 = blur_neighbors_[j*M_+i].n2+1;
			split_array * n1_val = values + n1*value_size;
			split_array * n2_val = values + n2*value_size;
			for( int k=0; k<value_size; k++ ) {
				weightedAddSplitArray(&new_val[k], &old_val[k], 0.5, &n1_val[k], &n2_val[k]);
                        // new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
                //printf("#%d\n", (i+1)*value_size+k);
                //printSplitArray(&new_val[ k ]);

//                if (i == 0 && k == 0) {
//                printf("#%d\n", (i+1)*value_size+k);
//                printf("new_val\n"); printSplitArray(&new_val[ k ]);
//                printf("old_val\n"); printSplitArray(&old_val[ k ]);
//                printf("n1_val\n"); printSplitArray(&n1_val[ k ]);
//                printf("n2_val\n"); printSplitArray(&n2_val[ k ]);
//                }
			}
		}
		std::swap( values, new_values );
	}

	// Slicing
	for( int i=0; i<N_; i++ ){
		for( int k=0; k<value_size; k++ )
			out[i*value_size+k] = 0;
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ ) {
				sliceSplitArray(&out[ i*value_size+k ], -w*alpha, in[ i*value_size+k ], &values[ o*value_size+k ], false, binCount[k]);
				//out[ i*value_size+k ] += w * values[ o*value_size+k ] * alpha;
			}
		}
	}
	
	memset(values, 0, (M_+2)*value_size*sizeof(split_array));
	memset(new_values, 0, (M_+2)*value_size*sizeof(split_array));
        memset(binCount, 0, value_size*sizeof(split_array));
	
	// Upper
	// Splatting
	for( int i=0;  i<N_; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ ) {
				addSplitArray(&values[ o*value_size+k ], w, in[ i*value_size+k ], true,binCount[k]);
				// values[ o*value_size+k ] += w * in[ i*value_size+k ];
                //printf("#%d\n", o*value_size+k);
                //printSplitArray(&values[ o*value_size+k ]);
			}
		}
	}

        for( int k=0; k<value_size; k++ ) {
            for( int i=0;  i<RESOLUTION; i++ ){
                binCount[k][i] = binCount[k][i]/(d_+1);
                binCount[k][i] = 1;
            }
        }

        // Blurring
	for( int j=0; j<=d_; ++j ){
		for( int i=0; i<M_; i++ ){
			split_array * old_val = values + (i+1)*value_size;
			split_array * new_val = new_values + (i+1)*value_size;
			
			int n1 = blur_neighbors_[j*M_+i].n1+1;
			int n2 = blur_neighbors_[j*M_+i].n2+1;
			split_array * n1_val = values + n1*value_size;
			split_array * n2_val = values + n2*value_size;
			for( int k=0; k<value_size; k++ ) {
				weightedAddSplitArray(&new_val[k], &old_val[k], 0.5, &n1_val[k], &n2_val[k]);
				// new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
                //printf("#%d\n", (i+1)*value_size+k);
                //printSplitArray(&new_val[ k ]);
			}
		}
		std::swap( values, new_values );
	}
	
	// Slicing
	for( int i=0; i<N_; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ ) {
				sliceSplitArray(&out[ i*value_size+k ], w*alpha, in[ i*value_size+k ], &values[ o*value_size+k ], true, binCount[k]);
				// out[ i*value_size+k ] += w * values[ o*value_size+k ] * alpha;
			}
		}
	}
	
	delete[] values;
	delete[] new_values;
}

void printContSplitArray(cont_split_array *array) {
	printf("Array of size %d out of %d\n", array->size, array->total_size);
	for(int i=array->size-1; i>=0; --i) {
		printf("%.3f / %.3f\n", array->values[i], array->indices[i]);
	}
}
void hasNanContSplitArray(cont_split_array *array) {
	if(array->size > 0) {
		float ind = array->indices[0];
		for(int i=1; i<array->size; ++i) {
			if(ind >= array->indices[i]) {
				printf("%f, %f, ==%d\n", array->indices[i-1], array->indices[i], array->indices[i-1]==array->indices[i]);
				printf("Wrong order in indices.\n");
				printContSplitArray(array);
				exit(1);
			}
			ind = array->indices[i];
		}
	}
	for(int i=0; i<array->size; ++i) {
		if(array->values[i] != array->values[i]) {
			printf("Nan in values for i=%d\n", i);
			printContSplitArray(array);
			exit(1);
		}
		if(array->indices[i] != array->indices[i]) {
			printf("Nan in indices for i=%d\n", i);
			printContSplitArray(array);
			exit(1);
		}
	}
}
void growContSplitArray(cont_split_array *array, int needed) {
	if(array->total_size == 0) {
		array->total_size = ORIG_SIZE;
		array->values = (float*) malloc(array->total_size*sizeof(float));
		array->indices = (float*) malloc(array->total_size*sizeof(float));
	}
	while(needed > array->total_size) {
		array->total_size = array->total_size*GROW_FACTOR;
		array->values = (float*) realloc(array->values, array->total_size*sizeof(float));
		array->indices = (float*) realloc(array->indices, array->total_size*sizeof(float));
	}
}
void copyContSplitArray(cont_split_array *to, cont_split_array *from) {
	growContSplitArray(to, from->size);
	to->size = from->size;
	memcpy(to->values, from->values, to->size*sizeof(float));
	memcpy(to->indices, from->indices, to->size*sizeof(float));
}
cont_split_array * cloneContSplitArray(cont_split_array * in) {
	cont_split_array *out = (cont_split_array*) malloc(sizeof(cont_split_array));
	out->values = (float*) malloc(in->total_size*sizeof(float));
	out->indices = (float*) malloc(in->total_size*sizeof(float));
	out->size = in->size;
	out->total_size = in->total_size;
	memcpy(out->values, in->values, in->size*sizeof(float));
	memcpy(out->indices, in->indices, in->size*sizeof(float));
	return out;
}
cont_split_array * newContSplitArray() {
	cont_split_array *out = (cont_split_array*) malloc(sizeof(cont_split_array));
	out->size = 0;
	out->total_size = 0;
	out->values = NULL;
	out->indices = NULL;
	return out;
}
void freeContSplitArray(cont_split_array *in, bool all) {
	in->total_size = 0;
	in->size = 0;
	if(in->values) {
		free(in->values);
	}
	if(in->indices) {
		free(in->indices);
	}
	if(all) {
		free(in);
	}
}
void addContSplitArrayCst(cont_split_array *out, float alpha, float up_to, bool from_top) {
	if (0 == out->size) {
		growContSplitArray(out, 2);
		out->size = 2;
		out->indices[0] = -1;
		out->indices[1] = up_to;
		if (from_top) {
			out->values[0] = 0;
			out->values[1] = alpha;
		} else {
			out->values[0] = alpha;
			out->values[1] = 0;
		}
		return;
	}
	if (from_top) {
		int index = 0;
		while(index < out->size && out->indices[index] < up_to) {
			++index;
		}
		if(index == out->size) {
			// The entry is at the top of the double list
			growContSplitArray(out, out->size+1);
			++out->size;
			out->indices[index] = up_to;
			out->values[index] = out->values[index-1] + alpha;
		} else if(out->indices[index] - up_to < 1e-5) {
			// This index already exist in the double list
			for(int i=index; i<out->size; ++i) {
				out->values[i] += alpha;
			}
		} else {
			// A new index need to be inserted inside the double list
			growContSplitArray(out, out->size+1);
			++out->size;
			float tmp, prev;
			prev = up_to;
			for(int i=index; i<out->size; ++i) {
				tmp = out->indices[i];
				out->indices[i] = prev;
				prev = tmp;
			}
			prev = out->values[index-1] + alpha;
			for(int i=index; i<out->size; ++i) {
				tmp = out->values[i] + alpha;
				out->values[i] = prev;
				prev = tmp;
			}
 		}
	} else {
		int index = 0;
		while(index < out->size && out->indices[index] < up_to) {
			++index;
		}
		if(index == out->size) {
			// The entry is at the top of the double list
			growContSplitArray(out, out->size+1);
			++out->size;
			out->indices[index] = up_to;
			out->values[index] = out->values[index-1];
			for(int i=0; i<out->size-1; ++i) {
				out->values[i] += alpha;
			}
		} else if(out->indices[index] - up_to < 1e-5) {
			// This index already exist in the double list
			for(int i=0; i<index; ++i) {
				out->values[i] += alpha;
			}
		} else {
			// A new index need to be inserted inside the double list
			growContSplitArray(out, out->size+1);
			++out->size;
			float tmp, prev;
			prev = up_to;
			for(int i=index; i<out->size; ++i) {
				tmp = out->indices[i];
				out->indices[i] = prev;
				prev = tmp;
			}
			prev = out->values[index-1];
			for(int i=index; i<out->size; ++i) {
				tmp = out->values[i];
				out->values[i] = prev;
				prev = tmp;
			}
			for(int i=0; i<index; ++i) {
				out->values[i] += alpha;
			}
 		}
	}
}
void scalarMulContSplitArray(cont_split_array *out, float scalar) {
	for(int i=0; i<out->size; ++i) {
		out->values[i] *= scalar;
	}
}
void addContSplitArray(cont_split_array *out, cont_split_array *in1, cont_split_array *in2) {
	if(0 == in1->size) {
		copyContSplitArray(out, in2);
		return;
	}
	if(0 == in2->size) {
		copyContSplitArray(out, in1);
		return;
	}
	// Reinitialize out
	growContSplitArray(out, 1);
	out->size = 1;
	out->indices[0] = -1;
	out->values[0] = in1->values[0] + in2->values[0];

	int index1=1, index2=1;
	float indice1, indice2;
	bool done1, done2;
	while(true) {
		growContSplitArray(out, out->size+1);
		if(index1<in1->size) {
			indice1 = in1->indices[index1];
			done1 = false;
		} else {
			done1 = true;
		}
		if(index2<in2->size) {
			indice2 = in2->indices[index2];
			done2 = false;
		} else {
			done2 = true;
		}
		if(done1 && done2) {
			return;
		}

		if(done2 || (!done1 && indice1 < indice2)) {
			out->indices[out->size] = indice1;
			out->values[out->size] = in1->values[index1] + in2->values[index2-1];
			++out->size;
			++index1;
		} else if(done1 || (!done2 && indice2 < indice1)) {
			out->indices[out->size] = indice2;
			out->values[out->size] = in1->values[index1-1] + in2->values[index2];
			++out->size;
			++index2;
		} else {
			out->indices[out->size] = indice1;
			out->values[out->size] = in1->values[index1] + in2->values[index2];
			++out->size;
			++index1;
			++index2;
		}
	}
}
void weightedAddContSplitArray(cont_split_array *out, cont_split_array *in1, float alpha, 
        cont_split_array *in2, cont_split_array *in3) {
	cont_split_array * tmp = newContSplitArray();
	//cont_split_array * tmp2 = newContSplitArray();
	addContSplitArray(tmp, in2, in3); // tmp = in2 + in3
	scalarMulContSplitArray(tmp, alpha); // alpha * (in2+in3)
	//addContSplitArray(tmp2, in1, tmp); // tmp2 = in1 + alpha*(in2+in3)
	//cont_split_array * out_clone = cloneContSplitArray(out);
	//addContSplitArray(out, out_clone, tmp2); // out = out + in1+alpha*(in2+in3)
        freeContSplitArray(out, false);
	addContSplitArray(out, in1, tmp); // out = in1 + alpha*(in2+in3) // this is what we want!
	freeContSplitArray(tmp, true);
	//freeContSplitArray(tmp2, true);
	//freeContSplitArray(out_clone, true);
}
void sliceContSplitArray(float *out, float alpha, float up_to, cont_split_array *in, bool from_top) {
    if (in->size == 0) return;
    if (from_top) {
    	for(int i=0; i<in->size; ++i) {
    		if (in->indices[i] == up_to) {
                *out += in->values[i] * alpha;
    		    return;
            } else if (in->indices[i] > up_to) {
                *out += in->values[i-1] * alpha;
    		    return;
            }
    	}
        *out += in->values[in->size-1] * alpha;
        return;
    } else {
    	for(int i=0; i<in->size; ++i) {
    		if (in->indices[i] >= up_to) {
                *out += in->values[i-1] * alpha;
    		    return;
            }
    	}
        *out += in->values[in->size-1] * alpha;
        return;
    }
}
void printContSplitArray(cont_split_array *array, bool from_top) {
	int precision = 10;
    for(int i=precision-1; i>=0; --i) {
        float out = 0;
        if (i == precision-1) sliceContSplitArray(&out, 1, 1, array, from_top);
        else sliceContSplitArray(&out, 1, float(i)/precision, array, from_top);
		printf("%3d: %f\n", i*RESOLUTION/precision, out);
	}
}
//void testSplitArrays() {
//	// To compare the discrete and continuous implementations
//	split_array disc = {0};
//	split_array disc1 = {0};
//	split_array disc2 = {0};
//	split_array disc3 = {0};
//	cont_split_array * cont = newContSplitArray();
//	cont_split_array * cont1 = newContSplitArray();
//	cont_split_array * cont2 = newContSplitArray();
//	cont_split_array * cont3 = newContSplitArray();
//	float out_disc, out_cont;
//
//    bool upper = false;
//
//	addSplitArray(&disc1, 1, 0., upper);
//	addContSplitArrayCst(cont1, 1, 0., upper);
//	printSplitArray(&disc1);
//	printContSplitArray(cont1);
//	printContSplitArray(cont1, upper);
//
//    out_disc = 0; out_cont = 0;
//	sliceSplitArray(&out_disc, 1, 1, &disc1, upper);
//	sliceContSplitArray(&out_cont, 1, 1, cont1, upper);
//	printf("%f vs %f\n", out_disc, out_cont);
//
//    out_disc = 0; out_cont = 0;
//	sliceSplitArray(&out_disc, 1, 0, &disc1, upper);
//	sliceContSplitArray(&out_cont, 1, 0, cont1, upper);
//	printf("%f vs %f\n", out_disc, out_cont);
//
//    out_disc = 0; out_cont = 0;
//	sliceSplitArray(&out_disc, 1, 0.1, &disc1, upper);
//	sliceContSplitArray(&out_cont, 1, 0.1, cont1, upper);
//	printf("%f vs %f\n", out_disc, out_cont);
//
////	addSplitArray(&disc1, 1, 0., upper);
////	addContSplitArrayCst(cont1, 1, 0., upper);
////	printSplitArray(&disc1);
////	printContSplitArray(cont1);
////
////	addSplitArray(&disc1, 1, 1, upper);
////	addContSplitArrayCst(cont1, 1, 1, upper);
////	printSplitArray(&disc1);
////	printContSplitArray(cont1);
////
////	addSplitArray(&disc1, 1, 0, upper);
////	addContSplitArrayCst(cont1, 1, 0, upper);
////	printSplitArray(&disc1);
////	printContSplitArray(cont1);
////
////	addSplitArray(&disc1, 1, 1, upper);
////	addContSplitArrayCst(cont1, 1, 1, upper);
////	printSplitArray(&disc1);
////	printContSplitArray(cont1);
////
//////	addSplitArray(&disc2, 0.7, 0.2, upper);
//////	addContSplitArrayCst(cont2, 0.7, 0.2, upper);
//////	addSplitArray(&disc2, 0.4, 0.5, upper);
//////	addContSplitArrayCst(cont2, 0.4, 0.5, upper);
////
////	addSplitArray(&disc2, 1, 1, upper);
////	addContSplitArrayCst(cont2, 1, 1, upper);
////	printSplitArray(&disc2);
////	printContSplitArray(cont2);
////
////	addSplitArray(&disc2, 1, 1, upper);
////	addContSplitArrayCst(cont2, 1, 1, upper);
////	printSplitArray(&disc2);
////	printContSplitArray(cont2);
////    
////	addSplitArray(&disc2, 1, 1, upper);
////	addContSplitArrayCst(cont2, 1, 1, upper);
////	printSplitArray(&disc2);
////	printContSplitArray(cont2);
////
////	weightedAddSplitArray(&disc, &disc1, 1, &disc2, &disc2);
////	weightedAddContSplitArray(cont, cont1, 1, cont2, cont2);
////	printSplitArray(&disc);
////	printContSplitArray(cont);
////	printContSplitArray(cont, upper);
//
////    out_disc = 0; out_cont = 0;
////	sliceSplitArray(&out_disc, 1, 1, &disc, upper);
////	sliceContSplitArray(&out_cont, 1, 1, cont, upper);
////	printf("%f vs %f\n", out_disc, out_cont);
////
////    out_disc = 0; out_cont = 0;
////	sliceSplitArray(&out_disc, 1, 0, &disc, upper);
////	sliceContSplitArray(&out_cont, 1, 0, cont, upper);
////	printf("%f vs %f\n", out_disc, out_cont);
////
////    out_disc = 0; out_cont = 0;
////	sliceSplitArray(&out_disc, 1, 0.1, &disc, upper);
////	sliceContSplitArray(&out_cont, 1, 0.1, cont, upper);
////	printf("%f vs %f\n", out_disc, out_cont);
//
//	freeContSplitArray(cont, true);
//	freeContSplitArray(cont1, true);
//	freeContSplitArray(cont2, true);
//	freeContSplitArray(cont3, true);
//	exit(0);
//}
void Permutohedral::seqCompute_upper_minus_lower_ord_cont (float* out, const float* in, int value_size) const {
    //testSplitArrays();
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	cont_split_array * values = new cont_split_array[ (M_+2)*value_size ]();
	cont_split_array * new_values = new cont_split_array[ (M_+2)*value_size ]();

	// Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
	float alpha = 1.0f / (1+powf(2, -d_)); // 0.8 in 2D / 0.97 in 5D
	
    //printf("\ncont\n");
	// Lower
	// Splatting
	for( int i=0;  i<N_; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			//if (w!=0) {
				for( int k=0; k<value_size; k++ ) {
					addContSplitArrayCst(&values[ o*value_size+k ], w, in[ i*value_size+k ], false);
					// values[ o*value_size+k ] += w * in[ i*value_size+k ];
                    //printf("#%d\n", o*value_size+k);
	                //printContSplitArray(&values[ o*value_size+k ], false);
				}
			//}
		}
	}

	// Blurring
	for( int j=0; j<=d_; ++j ){
		for( int i=0; i<M_; i++ ){
			cont_split_array * old_val = values + (i+1)*value_size;
			cont_split_array * new_val = new_values + (i+1)*value_size;
			
			int n1 = blur_neighbors_[j*M_+i].n1+1;
			int n2 = blur_neighbors_[j*M_+i].n2+1;
			cont_split_array * n1_val = values + n1*value_size;
			cont_split_array * n2_val = values + n2*value_size;
			for( int k=0; k<value_size; k++ ) {
				weightedAddContSplitArray(&new_val[k], &old_val[k], 0.5, &n1_val[k], &n2_val[k]);
				// new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
                //printf("#%d\n", (i+1)*value_size+k);
                //printContSplitArray(&new_val[ k ], false);
//                if (i == 0 && k == 0) {
//                printf("#%d\n", (i+1)*value_size+k);
//                printf("new_val\n"); printContSplitArray(&new_val[ k ], false);
//                printf("old_val\n"); printContSplitArray(&old_val[ k ], false);
//                printf("n1_val\n"); printContSplitArray(&n1_val[ k ], false);
//                printf("n2_val\n"); printContSplitArray(&n2_val[ k ], false);
//                }
			}
		}
		std::swap( values, new_values );
	}

	// Slicing
	for( int i=0; i<N_; i++ ){
		for( int k=0; k<value_size; k++ )
			out[i*value_size+k] = 0;
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ ) {
				sliceContSplitArray(&out[ i*value_size+k ], -w*alpha, in[ i*value_size+k ], &values[ o*value_size+k ], false);
				//out[ i*value_size+k ] += w * values[ o*value_size+k ] * alpha;
			}
		}
	}

	for(int i=0; i<(M_+2)*value_size; ++i) {
		values[i].size = 0;
		new_values[i].size = 0;
	}
	
	// Upper
	// Splatting
	for( int i=0;  i<N_; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ ) {
				addContSplitArrayCst(&values[ o*value_size+k ], w, in[ i*value_size+k ], true);
				// values[ o*value_size+k ] += w * in[ i*value_size+k ];
                //printf("#%d\n", o*value_size+k);
	            //printContSplitArray(&values[ o*value_size+k ], true);
	            //printContSplitArray(&values[ o*value_size+k ]);
			}
		}
	}

	// Blurring
	for( int j=0; j<=d_; ++j ){
		for( int i=0; i<M_; i++ ){
			cont_split_array * old_val = values + (i+1)*value_size;
			cont_split_array * new_val = new_values + (i+1)*value_size;
			
			int n1 = blur_neighbors_[j*M_+i].n1+1;
			int n2 = blur_neighbors_[j*M_+i].n2+1;
			cont_split_array * n1_val = values + n1*value_size;
			cont_split_array * n2_val = values + n2*value_size;
			for( int k=0; k<value_size; k++ ) {
				weightedAddContSplitArray(&new_val[k], &old_val[k], 0.5, &n1_val[k], &n2_val[k]);
				// new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
                //printf("#%d\n", (i+1)*value_size+k);
	            //printContSplitArray(&new_val[ k ], true);
			}
		}
		std::swap( values, new_values );
	}
	
	// Slicing
	for( int i=0; i<N_; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			for( int k=0; k<value_size; k++ ) {
				sliceContSplitArray(&out[ i*value_size+k ], w*alpha, in[ i*value_size+k ], &values[ o*value_size+k ], true);
				// out[ i*value_size+k ] += w * values[ o*value_size+k ] * alpha;
			}
		}
	}
	
	for(int i=0; i<(M_+2)*value_size; ++i) {
		freeContSplitArray(values+i, false);
		freeContSplitArray(new_values+i, false);
	}

	delete[] values;
	delete[] new_values;
}
void Permutohedral::seqCompute_upper_minus_lower_dc ( float* out, int low, int middle_low, int middle_high, int high ) const
{
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	float * values = new float[ M_+2 ];
	float * new_values = new float[ M_+2 ];
	bool * activated = new bool[M_];
	std::vector<int> list;
	list.reserve(d_*M_);
	
	//// Upper
	memset(values, 0, (M_+2)*sizeof(float));
	memset(new_values, 0, (M_+2)*sizeof(float));
	memset(activated, 0, M_*sizeof(bool));
	
	// Splatting
	for( int i=middle_high; i<high; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j];
			float w = barycentric_[i*(d_+1)+j];
			values[ o+1 ] += w;
			if(!activated[o]) {
				activated[o] = true;
				list.push_back(o);
			}
		}
	}

	for( int i=low; i<middle_low; ++i) {
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j];
			if(!activated[o]) {
				activated[o] = true;
				list.push_back(o);
			}
		}
	}

	// Blurring
	for( int j=0; j<=d_ && j>=0; j++ ){
		for( int i:list ){
			int n1 = blur_neighbors_[j*M_+i].n1;
			int n2 = blur_neighbors_[j*M_+i].n2;
			new_values[i+1] = values[i+1]+0.5*(values[n1+1] + values[n2+1]);
		}
		std::swap( values, new_values );
	}
	// Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
	float alpha = 1.0f / (1+powf(2, -d_));
	
	// Slicing
	for( int i=low;  i<middle_low; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			out[ i ] += w * values[ o ] * alpha;
		}
	}
	
	//// Lower
	memset(values, 0, (M_+2)*sizeof(float));
	memset(new_values, 0, (M_+2)*sizeof(float));
	
	// Splatting
	for( int i=low; i<middle_low; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j];
			float w = barycentric_[i*(d_+1)+j];
			values[ o+1 ] += w;
		}
	}

	// Blurring
	for( int j=0; j<=d_ && j>=0; j++ ){
		for( int i:list ){
			int n1 = blur_neighbors_[j*M_+i].n1;
			int n2 = blur_neighbors_[j*M_+i].n2;
			new_values[i+1] = values[i+1]+0.5*(values[n1+1] + values[n2+1]);
		}
		std::swap( values, new_values );
	}
	
	// Slicing
	for( int i=middle_high;  i<high; i++ ){
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			float w = barycentric_[i*(d_+1)+j];
			out[ i ] -= w * values[ o ] * alpha;
		}
	}
	
	
	delete[] values;
	delete[] new_values;
	delete[] activated;
}
#ifdef SSE_PERMUTOHEDRAL
void Permutohedral::sseCompute ( float* out, const float* in, int value_size, bool reverse ) const
{
	const int sse_value_size = (value_size-1)*sizeof(float) / sizeof(__m128) + 1;
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	__m128 * sse_val    = (__m128*) _mm_malloc( sse_value_size*sizeof(__m128), 16 );
	__m128 * values     = (__m128*) _mm_malloc( (M_+2)*sse_value_size*sizeof(__m128), 16 );
	__m128 * new_values = (__m128*) _mm_malloc( (M_+2)*sse_value_size*sizeof(__m128), 16 );
	
	__m128 Zero = _mm_set1_ps( 0 );
	
	for( int i=0; i<(M_+2)*sse_value_size; i++ )
		values[i] = new_values[i] = Zero;
	for( int i=0; i<sse_value_size; i++ )
		sse_val[i] = Zero;
	
	// Splatting
	for( int i=0;  i<N_; i++ ){
		memcpy( sse_val, in+i*value_size, value_size*sizeof(float) );
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			__m128 w = _mm_set1_ps( barycentric_[i*(d_+1)+j] );
			for( int k=0; k<sse_value_size; k++ )
				values[ o*sse_value_size+k ] += w * sse_val[k];
		}
	}
	// Blurring
	__m128 half = _mm_set1_ps(0.5);
	for( int j=reverse?d_:0; j<=d_ && j>=0; reverse?j--:j++ ){
		for( int i=0; i<M_; i++ ){
			__m128 * old_val = values + (i+1)*sse_value_size;
			__m128 * new_val = new_values + (i+1)*sse_value_size;
			
			int n1 = blur_neighbors_[j*M_+i].n1+1;
			int n2 = blur_neighbors_[j*M_+i].n2+1;
			__m128 * n1_val = values + n1*sse_value_size;
			__m128 * n2_val = values + n2*sse_value_size;
			for( int k=0; k<sse_value_size; k++ )
				new_val[k] = old_val[k]+half*(n1_val[k] + n2_val[k]);
		}
		std::swap( values, new_values );
	}
	// Alpha is a magic scaling constant (write Andrew if you really wanna understand this)
	float alpha = 1.0f / (1+powf(2, -d_));
	
	// Slicing
	for( int i=0; i<N_; i++ ){
		for( int k=0; k<sse_value_size; k++ )
			sse_val[ k ] = Zero;
		for( int j=0; j<=d_; j++ ){
			int o = offset_[i*(d_+1)+j]+1;
			__m128 w = _mm_set1_ps( barycentric_[i*(d_+1)+j] * alpha );
			for( int k=0; k<sse_value_size; k++ )
				sse_val[ k ] += w * values[ o*sse_value_size+k ];
		}
		memcpy( out+i*value_size, sse_val, value_size*sizeof(float) );
	}
	
	_mm_free( sse_val );
	_mm_free( values );
	_mm_free( new_values );
}
#else
void Permutohedral::sseCompute ( float* out, const float* in, int value_size, bool reverse ) const
{
	seqCompute( out, in, value_size, reverse );
}
#endif
void Permutohedral::compute_upper_minus_lower_dc ( MatrixXf & out, int low, int middle_low, int middle_high, int high ) const
{
	// Here anly one label at a time so always seq
	assert(out.cols()==N_);
	seqCompute_upper_minus_lower_dc(out.data(), low, middle_low, middle_high, high);
}
void Permutohedral::compute_upper_minus_lower_ord ( MatrixXf & out, const MatrixXf & Q) const {
	seqCompute_upper_minus_lower_ord(out.data(), Q.data(), Q.rows());
}
void Permutohedral::compute_upper_minus_lower_ord_cont ( MatrixXf & out, const MatrixXf & Q) const {
	seqCompute_upper_minus_lower_ord_cont(out.data(), Q.data(), Q.rows());
}
void Permutohedral::compute ( MatrixXf & out, const MatrixXf & in, bool reverse ) const
{
	if( out.cols() != in.cols() || out.rows() != in.rows() )
		out = 0*in;
	if( in.rows() <= 2 )
		seqCompute( out.data(), in.data(), in.rows(), reverse );
	else
		sseCompute( out.data(), in.data(), in.rows(), reverse );
}
MatrixXf Permutohedral::compute ( const MatrixXf & in, bool reverse ) const
{
	MatrixXf r;
	compute( r, in, reverse );
	return r;
}
// Compute the gradient of a^T K b
void Permutohedral::gradient ( float* df, const float * a, const float* b, int value_size ) const
{
	// Shift all values by 1 such that -1 -> 0 (used for blurring)
	float * values = new float[ (M_+2)*value_size ];
	float * new_values = new float[ (M_+2)*value_size ];
	
	// Set the results to 0
	std::fill( df, df+N_*d_, 0.f );
	
	// Initialize some constants
	std::vector<float> scale_factor( d_ );
	float inv_std_dev = sqrt(2.0 / 3.0)*(d_+1);
	for( int i=0; i<d_; i++ )
		scale_factor[i] = 1.0 / sqrt( double((i+2)*(i+1)) ) * inv_std_dev;
	
	// Alpha is a magic scaling constant multiplied by down_factor
	float alpha = 1.0f / (1+powf(2, -d_)) / (d_+1);
	
	for( int dir=0; dir<2; dir++ ) {
		for( int i=0; i<(M_+2)*value_size; i++ )
			values[i] = new_values[i] = 0;
	
		// Splatting
		for( int i=0;  i<N_; i++ ){
			for( int j=0; j<=d_; j++ ){
				int o = offset_[i*(d_+1)+j]+1;
				float w = barycentric_[i*(d_+1)+j];
				for( int k=0; k<value_size; k++ )
					values[ o*value_size+k ] += w * (dir?b:a)[ i*value_size+k ];
			}
		}
		
		// BLUR
		for( int j=dir?d_:0; j<=d_ && j>=0; dir?j--:j++ ){
			for( int i=0; i<M_; i++ ){
				float * old_val = values + (i+1)*value_size;
				float * new_val = new_values + (i+1)*value_size;
			
				int n1 = blur_neighbors_[j*M_+i].n1+1;
				int n2 = blur_neighbors_[j*M_+i].n2+1;
				float * n1_val = values + n1*value_size;
				float * n2_val = values + n2*value_size;
				for( int k=0; k<value_size; k++ )
					new_val[k] = old_val[k]+0.5*(n1_val[k] + n2_val[k]);
			}
			std::swap( values, new_values );
		}
	
		// Slicing gradient computation
		std::vector<float> r_a( (d_+1)*value_size ), sm( value_size );
	
		for( int i=0; i<N_; i++ ){
			// Rotate a
			std::fill( r_a.begin(), r_a.end(), 0.f );
			for( int j=0; j<=d_; j++ ){
				int r0 = d_ - rank_[i*(d_+1)+j];
				int r1 = r0+1>d_?0:r0+1;
				int o0 = offset_[i*(d_+1)+r0]+1;
				int o1 = offset_[i*(d_+1)+r1]+1;
				for( int k=0; k<value_size; k++ ) {
					r_a[ j*value_size+k ] += alpha*values[ o0*value_size+k ];
					r_a[ j*value_size+k ] -= alpha*values[ o1*value_size+k ];
				}
			}
			// Multiply by the elevation matrix
			std::copy( r_a.begin(), r_a.begin()+value_size, sm.begin() );
			for( int j=1; j<=d_; j++ ) {
				float grad = 0;
				for( int k=0; k<value_size; k++ ) {
					// Elevate ...
					float v = scale_factor[j-1]*(sm[k]-j*r_a[j*value_size+k]);
					// ... and add
					grad += (dir?a:b)[ i*value_size+k ]*v;
				
					sm[k] += r_a[j*value_size+k];
				}
				// Store the gradient
				df[i*d_+j-1] += grad;
			}
		}
	}		
	delete[] values;
	delete[] new_values;
}
