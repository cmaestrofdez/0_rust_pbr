use std::{ops::{Add, Sub, Mul, Div}, sync::Once};

use cgmath::{Point2, Vector2, AbsDiffEq};
use hexf::hexf32;
use once_cell::sync::Lazy;
use rand::{prelude::ThreadRng, Rng};
 
use crate::{    sampler::{custom_rng, Sampler}};
 
use crate::Point2i;
pub const ONE_MINUS_EPSILON_f64: f64 =  hexf::hexf32!("0x1.fffffep-1") as f64;

const PRIME_TABLE_SIZE  : u16 = 1000_u16;
pub fn reverse_bits_32(n: u32) -> u32 {
    let mut n = (n << 16) | (n >> 16);
    n = ((n & 0x00ff_00ff) << 8) | ((n & 0xff00_ff00) >> 8);
    n = ((n & 0x0f0f_0f0f) << 4) | ((n & 0xf0f0_f0f0) >> 4);
    n = ((n & 0x3333_3333) << 2) | ((n & 0xcccc_cccc) >> 2);
    n = ((n & 0x5555_5555) << 1) | ((n & 0xaaaa_aaaa) >> 1);
    n
}
pub fn reverse_bits_64(n: u64) -> u64 {
    let n0: u64 = reverse_bits_32(n as u32) as u64;
    let n1: u64 = reverse_bits_32((n >> 32) as u32) as u64;
    (n0 << 32) | n1
}
pub fn radical_inverse_pow2(base : u16, a : u64)->f64{
    reverse_bits_64(a) as f64 * hexf32!("0x1.0p-64") as  f64 
}

pub fn find_pixel_coords(pmin:&Point2<i64>, pmax:&Point2<i64>,  kMaxResolution : i32, scales:&mut Point2i, expons:&mut Point2i, sample_stride_:&mut i64){
  let res :Vector2<i64> =   pmax - pmin;
  let mut base_scales : Point2i = Point2::new(0,0);
  let mut  base_exponent : Point2i  = Point2::new(0,0);
  for i in 0..2{
    let base = if i as u8 == 0 {
        2
    }else{
        3
    };
    let mut scale = 1_i64;
    let mut exponet = 0_i64;
    while scale < res[i].min(kMaxResolution as i64) {
        scale*=base;
        exponet+=1;
    }
    base_scales[i]=scale;
    base_exponent[i]=exponet;
  }
 let sample_stride =  (base_scales[0]*base_scales[1]) as i64;
 *scales = base_scales;
 *expons = base_exponent;
 *sample_stride_=sample_stride;
}
/**
 *  v1 = d1
 *  v2 = bd1 + d2 b=base , inv_base_n = b^n-1... b^0
 *  vn = b^n-1 * d1  + b^n-2*d2 ... + dn
 * 
 */
pub fn radical_inverse_different_pow2(base : u16, a : u64)->f64{
    let invBase = 1.0 as f64 / base as f64;
    let mut  reversedigits = 0_u64;
    let mut inv_base_n  = 1_f64;
    let mut a = a;
    while  a != 0_u64 {
        let next = a / base as u64 ;
        let digit = a - next * base as u64  ;
        reversedigits = reversedigits * base as u64 + digit;
        inv_base_n *=invBase;
        a = next ;
    }
    (reversedigits as f64 * inv_base_n).min(ONE_MINUS_EPSILON_f64)
}

pub fn suffle_test<T>(samp : &mut [T], count : i32 , n_dimensions : i32, mut rng : & mut custom_rng::CustomRng){
    for i in 0..count {
        let other = i as u32 + rng.uniform_uint32_bounded(count as u32 - i as u32);
        for j in 0..n_dimensions{
            samp.swap(n_dimensions as usize *i as usize+j as usize, n_dimensions as usize *other as usize + j as usize  );
        }
        
    }
}
pub fn suffle<T>(samp : &mut [T], count : i32 , n_dimensions : i32, mut rng : & mut ThreadRng){
    for i in 0..count {
        let other = i + rng.gen_range(0..count-i);
        for j in 0..n_dimensions{
            samp.swap(n_dimensions as usize *i as usize+j as usize, n_dimensions as usize *other as usize + j as usize  );
        }
        
    }
}
// https://github.com/wahn/rs_pbrt/blob/bf924c9179ab54e62ea1a945e0b5f37676b0478c/src/core/pbrt.rs#L125
pub fn mod_t<T>(a: T, b: T) -> T
where
    T: num::Zero
        + Copy
        + PartialOrd
        + Add<T, Output = T>
        + Sub<T, Output = T>
        + Mul<T, Output = T>
        + Div<T, Output = T>,
{
    
    let result: T = a - (a / b) * b;
    if result < num::Zero::zero() {
        result + b
    } else {
        result
    }
}
fn multiplicative_inverse(a: i64, n: i64) -> u64 {
    let mut x: i64 = 0;
    let mut y: i64 = 0;
    extended_gcd(a as u64, n as u64, &mut x, &mut y);
    mod_t(x, n) as u64
}

fn extended_gcd(a: u64, b: u64, x: &mut i64, y: &mut i64) {
    if b == 0_u64 {
        *x = 1_i64;
        *y = 0_i64;
    } else {
        let d: i64 = a as i64 / b as i64;
        let mut xp: i64 = 0;
        let mut yp: i64 = 0;
        extended_gcd(b, a % b, &mut xp, &mut yp);
        *x = yp;
        *y = xp - (d * yp);
    }
}

pub fn compute_radical_inverse_permutations_for_test(  inverse_permutations:&mut Vec<u16>,rng :& mut  custom_rng::CustomRng){

    let mut  permu_array = 0_usize;
    for i in 0..PRIME_TABLE_SIZE{
       permu_array += PRIMES[i as usize] as usize;
    }
    let mut perms : Vec<u16> = vec![0_u16; permu_array];

    let mut p : usize = 0;
    for i in 0..PRIME_TABLE_SIZE{
        for j in 0..PRIMES[i as usize]{
            perms[p + j as usize]=j as u16;
        }
        suffle_test(&mut perms[p..(p+PRIMES[i as usize] as usize)], PRIMES[i as usize] as i32, 1, rng);
        p+=PRIMES[i as usize] as usize;
    }
// println!("{}", perms.len());
    *inverse_permutations = perms;

}

pub fn compute_radical_inverse_permutations(  inverse_permutations:&mut Vec<u16>,rng : & mut ThreadRng){
    let mut  permu_array = 0_usize;
    for i in 0..PRIME_TABLE_SIZE{
       permu_array += PRIMES[i as usize] as usize;
    }
    let mut perms : Vec<u16> = vec![0_u16; permu_array];

    let mut p : usize = 0;
    for i in 0..PRIME_TABLE_SIZE{
        for j in 0..PRIMES[i as usize]{
            perms[p + j as usize]=j as u16;
        }
        suffle(&mut perms[p..(p+PRIMES[i as usize] as usize)], PRIMES[i as usize] as i32, 1, rng);
        p+=PRIMES[i as usize] as usize;
    }
    *inverse_permutations = perms;
}

pub fn inverse_radical_inverse(base: u8, inverse: u64, n_digits: u64) -> u64 {
    let mut inverse: u64 = inverse;
    let mut index: u64 = 0;
    for _i in 0..n_digits {
        let digit: u64 = inverse % base as u64;
        inverse /= base as u64;
        index = index * base as u64 + digit;
    }
    index
}






pub 
const k_Max_Resolution : i32 = 128_i32;
// https://stackoverflow.com/questions/27791532/how-do-i-create-a-global-mutable-singleton
static   RADICAL_INVERSE_PERMUTATIONS  : Lazy<Vec<u16>>=  Lazy::new(||{
    let mut perms :Vec<u16>  = vec![];
   ;
    compute_radical_inverse_permutations_for_test(&mut perms, &mut custom_rng::CustomRng::new());
    println!("Lazy inialization for RADICAL_INVERSE_PERMUTATIONS");
    perms
});
// static  mut RADICAL_INVERSE_PERMUTATIONS   :Vec<u16> = vec![];
// static INIT: Once = Once::new();
// fn get_cached_val() -> &Vec<u16> {
//     unsafe {
//         INIT.call_once(|| {
//             compute_radical_inverse_permutations_for_test(&mut RADICAL_INVERSE_PERMUTATIONS, &mut test_rng::Rng::new());
//         });
//         &RADICAL_INVERSE_PERMUTATIONS
//     }
// }
 
pub struct SamplerHalton{
    spp : i64,
    base_scales : Point2i, 
    base_exponent : Point2i,
    sample_stride : u64,
    multiplicative_inverse : [u64;2],
      sample_pixel_center : bool, 
   current_dimension : i64,
   current_pixel_sample_index : i64,
    interval_sample_index : i64,
    permutation_per_dimension : Vec<u16>,
    permutation_per_dimension_lazy :Lazy<Vec<u16>>,
    pixel_for_offset  : Point2i,
    current_pixel   : Point2i,
    offset_per_current_pixel :u64,
    pixel_for_offset_x: i32,
    pixel_for_offset_y: i32,
    array_start_dim: i64,
    array_end_dim: i64,
    // 
    samples_array_1d : Vec<Vec<f64>>,
    samples_array_2d : Vec<Vec<(f64, f64)>>,
    array_1d_offset  : usize,
    array_2d_offset  : usize,
    samples_1d_array_sizes : Vec<usize>,
    samples_2d_array_sizes: Vec<usize>,
}
impl SamplerHalton {
    pub  fn new(bpmin: &Point2i,  bpmax:&Point2i, spp : i64, sample_center : bool)->Self{
        let mut  scales = Point2i::new(0,0 );
        let mut  exponent = Point2i::new(0,0);
        let mut stride : i64 = 0;
        find_pixel_coords(bpmin, bpmax, k_Max_Resolution,    & mut scales,  & mut exponent , & mut stride);
         let a = multiplicative_inverse(scales[1], scales[0]);
         let b =  multiplicative_inverse(scales[0], scales[1]);
         let mut perms :Vec<u16>  = vec![];
         
        compute_radical_inverse_permutations_for_test(&mut perms, &mut custom_rng::CustomRng::new());

        SamplerHalton{
            spp, 
            base_scales:scales,
            base_exponent:exponent,
            sample_stride:stride as u64,
            multiplicative_inverse:[a, b],
            sample_pixel_center:false, 
            current_dimension: 0_i64, 
            interval_sample_index: 0_i64, 
            permutation_per_dimension:perms,
            permutation_per_dimension_lazy:Lazy::new(||{
                let mut perms :Vec<u16>  = vec![];
                compute_radical_inverse_permutations_for_test(&mut perms, &mut custom_rng::CustomRng::new());
                println!("Lazy inialization for RADICAL_INVERSE_PERMUTATIONS");
                perms
            }),
            pixel_for_offset :  Point2i::new(0,0), 
            current_pixel :  Point2i::new(0,0), 
            offset_per_current_pixel : 0_u64,
            pixel_for_offset_x : 0_i32,
            pixel_for_offset_y: 0_i32,
            array_start_dim: 5_i64, 
            array_end_dim: 0_i64,
            current_pixel_sample_index:0_i64,

            // request samples 
            samples_array_1d: Vec::new(),
            samples_array_2d: Vec::new(),
            array_1d_offset:0, 
            array_2d_offset:0,
            samples_1d_array_sizes:Vec::new(), 
            samples_2d_array_sizes:Vec::new(), 
        }
    }
    pub fn compute_array_size_end(& mut self) {
        self.array_end_dim  = self.array_start_dim + self.samples_array_1d.len() as i64 + 2_i64 * self.samples_array_2d.len() as i64  
       
    }
     fn compute_array_1d_request_samples(& mut self){
    
        for i in 0..self.samples_1d_array_sizes.len(){
           let nsamples =  self.samples_1d_array_sizes[i] as i64 * self.spp ;
           for s in 0..nsamples{
               let ithsample =  self.get_index_for_sample(s as u64);
               self.samples_array_1d[i as usize][s as usize] = self.sample_dimesion(ithsample, self.array_start_dim+s);
           }
        }
    }
    fn compute_array_2d_request_samples(& mut self){
        let mut dimension = self.array_start_dim + self.samples_1d_array_sizes.len() as i64;
        for i in 0..self.samples_2d_array_sizes.len(){
           let nsamples =  self.samples_2d_array_sizes[i] as i64 * self.spp ;
           for s in 0..nsamples{

               let ithsample =  self.get_index_for_sample(s as u64);
               let x =self.sample_dimesion(ithsample, dimension);
               let y =self.sample_dimesion(ithsample, dimension+1);
               self.samples_array_2d[i as usize][s as usize] = (x, y);
           }
           dimension += 2;
        }
        assert_eq!(self.array_end_dim , dimension);
    }
    pub fn get_request_2d(& mut self, nsamples:i64){
        self.samples_2d_array_sizes.push(nsamples as usize);
        let nsamplesperpixel = nsamples * self.spp;
        let newsamplesbucket = vec![(0 as f64,  0 as f64) ; nsamplesperpixel as usize];
        self.samples_array_2d.push(newsamplesbucket);

    }
    pub fn sample_dimesion(&self, index : u64, dim : i64)->f64{
        if dim == 0{
            radical_inverse(dim as  u16, index>>self.base_exponent[0] as u64)
        }else if dim == 1 {
            radical_inverse(dim as  u16, index / self.base_scales[1] as u64)
        }else{
            let perm = self.get_permutation_per_dimension(dim);
            
            scrambled_radical_inverse(dim as  u16, index, perm)
        }
    }
    pub fn get_index_for_sample(&mut self, sample_num : u64)->u64{
      
      
      if   self.pixel_for_offset != self.current_pixel {
       self.offset_per_current_pixel = 0;
        if self.sample_stride > 1 {
            let pointmod = Point2i::new(mod_t(self.current_pixel[0], k_Max_Resolution as i64),mod_t(self.current_pixel[1], k_Max_Resolution as i64));
            for i in 0..2{
                
                let dim_offset  =  if i == 0{
                    // ? InverseRadicalInverse<2>(pm[i], baseExponents[i]): InverseRadicalInverse<3>(pm[i], baseExponents[i]);
                    inverse_radical_inverse(2 as u8, pointmod[i] as u64, self.base_exponent[i]  as u64)
                }else {
                    inverse_radical_inverse(3 as u8, pointmod[i]  as u64, self.base_exponent[i] as u64)
                };
                
                self.offset_per_current_pixel += 
                dim_offset
                            * (self.sample_stride / self.base_scales[i] as u64) as u64
                            * self.multiplicative_inverse[i as usize] as u64 ;
                            
                
            }
            self.offset_per_current_pixel = self.offset_per_current_pixel % self.sample_stride  ; 
        } //  if self.sample_stride > 1
        // self.pixel_for_offset_x = self.current_pixel.x as i32;
        // self.pixel_for_offset_y = self.current_pixel.y as i32; 
        self.pixel_for_offset.x = self.current_pixel.x ;
        self.pixel_for_offset.y = self.current_pixel.y;
      }// if   self.pixel_for_offset == self.current_pixel
       
        self.offset_per_current_pixel + sample_num * self.sample_stride
    }// get_index_for_sample
    pub fn get2d(&mut self)->(f64, f64){
        if self.current_dimension + 1 >= self.array_start_dim && self.current_dimension < self.array_end_dim {
            self.current_dimension = self.array_end_dim;
        }
       
        let mut y = self.sample_dimesion(self.interval_sample_index as u64, self.current_dimension +1  );
        let mut  x = self.sample_dimesion(self.interval_sample_index as u64, self.current_dimension +0 );
        self.current_dimension+=2;

        // correct first and  seconds dimension . sample pixels
        if self.current_dimension ==2{
           x+= self.current_pixel.x as f64;
           y+= self.current_pixel.y as f64;
        }
         (x, y)
    }
    pub fn get1d(&mut self)-> f64{ 
        if self.current_dimension + 1 >= self.array_start_dim && self.current_dimension < self.array_end_dim {
            self.current_dimension = self.array_end_dim;
        }
       let x =  self.sample_dimesion(self.interval_sample_index as u64, self.current_dimension  );
        self.current_dimension+=1;

        x   
    }
    pub fn start_pixel(&mut self, p : Point2i){
        self.current_pixel = p;
        self.current_pixel_sample_index = 0_i64;
        self.current_dimension = 0_i64;
        self.interval_sample_index = self.get_index_for_sample(0_u64) as i64;
        self.array_end_dim = 5;
        self.compute_array_size_end();
        self.compute_array_1d_request_samples();
        self.compute_array_2d_request_samples();
    }
    pub fn start_next_sample(&mut self)->bool{
        self.current_dimension = 0_i64;
       
        self.interval_sample_index = self.get_index_for_sample(self.current_pixel_sample_index as u64 + 1) as i64;
        // reset_array_offsets();
       self.array_2d_offset = 0;
       self.array_1d_offset = 0;
        self.current_pixel_sample_index +=1;
        self.current_pixel_sample_index < self.spp
    }
    fn get_permutation_per_dimension(&self, dim : i64)->&[u16]{
        if  dim>PRIME_TABLE_SIZE  as i64{
            panic!("get_permutation_per_dimension panic!");
        }
        // &RADICAL_INVERSE_PERMUTATIONS[PRIME_SUMS[dim as usize]]
        &self.permutation_per_dimension [PRIME_SUMS[dim as usize] as usize..]
    }
    pub fn get_spp(&self)->u32 {
        self.spp as u32
    }
}


impl Sampler for SamplerHalton{
    fn has_samples(&self) -> bool {
        todo!()
    }

    fn get2d(&mut self) -> (f64, f64)  {
       self.get2d()
    }

    fn get1d(&mut self) ->  f64  {
        self.get1d()
    }

    fn get_current_pixel(&self) -> (u32, u32) {
        todo!()
    }

    fn get_spp(&self) -> u32 {
        self.get_spp()
    }

    fn start_pixel(&mut self, p: Point2i) {
        self.start_pixel(p)
    }

    fn start_next_sample(&mut self) -> bool {
        self.start_next_sample()
    }
}




/**
 *  v1 = d1
 *  v2 = bd1 + d2 b=base , inv_base_n = b^n-1... b^0
 *  vn = b^n-1 * d1  + b^n-2*d2 ... + dn
 * 
 */
pub fn scrambled_radical_inverse_different_pow2(base : u16, perm:&[u16] , a : u64)->f64{
    let invBase = 1.0 as f64 / base as f64;
    let mut  reversedigits = 0_u64;
    let mut inv_base_n  = 1_f64;
    let mut a = a;
    while  a != 0_u64 {
        let next = a / base as u64 ;
        let digit = a - next * base as u64  ;
        reversedigits = reversedigits * base as u64 +   perm[digit as usize] as u64;
        inv_base_n *=invBase;
        a = next ;
    }
   
    (inv_base_n *
    (reversedigits as f64 + invBase * perm[0]as f64 / (1.0 - invBase))).min(ONE_MINUS_EPSILON_f64)

}


pub fn scrambled_radical_inverse_different_pow2_lazy_permutation(base : u16, perm:Lazy<Vec<u16>>, a : u64)->f64{
    let invBase = 1.0 as f64 / base as f64;
    let mut  reversedigits = 0_u64;
    let mut inv_base_n  = 1_f64;
    let mut a = a;
    while  a != 0_u64 {
        let next = a / base as u64 ;
        let digit = a - next * base as u64  ;
        reversedigits = reversedigits * base as u64 +   perm[digit as usize] as u64;
        inv_base_n *=invBase;
        a = next ;
    }
    (inv_base_n *
    (reversedigits as f64 * invBase * perm[0]as f64 / (1.0 - invBase))).min(ONE_MINUS_EPSILON_f64)

}










pub fn scrambled_radical_inverse(base_index: u16, a: u64, perm: &[u16]) -> f64 {
    match base_index {
        0 =>scrambled_radical_inverse_different_pow2(2_u16, perm, a),
        1 =>scrambled_radical_inverse_different_pow2(3_u16, perm, a),
        2 =>scrambled_radical_inverse_different_pow2(5_u16, perm, a),
        3 =>scrambled_radical_inverse_different_pow2(7_u16, perm, a),
        4 =>scrambled_radical_inverse_different_pow2(11_u16, perm, a),
        5 =>scrambled_radical_inverse_different_pow2(13_u16, perm, a),
        6 =>scrambled_radical_inverse_different_pow2(17_u16, perm, a),
        7 =>scrambled_radical_inverse_different_pow2(19_u16, perm, a),
        8 =>scrambled_radical_inverse_different_pow2(23_u16, perm, a),
        9 =>scrambled_radical_inverse_different_pow2(29_u16, perm, a),
        10 =>scrambled_radical_inverse_different_pow2(31_u16, perm, a),
        11 =>scrambled_radical_inverse_different_pow2(37_u16, perm, a),

        12 =>scrambled_radical_inverse_different_pow2(41_u16, perm, a),

        13 =>scrambled_radical_inverse_different_pow2(43_u16, perm, a),

        14 =>scrambled_radical_inverse_different_pow2(47_u16, perm, a),

        15 =>scrambled_radical_inverse_different_pow2(53_u16, perm, a),

        16 =>scrambled_radical_inverse_different_pow2(59_u16, perm, a),

        17 =>scrambled_radical_inverse_different_pow2(61_u16, perm, a),

        18 =>scrambled_radical_inverse_different_pow2(67_u16, perm, a),

        19 =>scrambled_radical_inverse_different_pow2(71_u16, perm, a),

        20 =>scrambled_radical_inverse_different_pow2(73_u16, perm, a),

        21 =>scrambled_radical_inverse_different_pow2(79_u16, perm, a),

        22 =>scrambled_radical_inverse_different_pow2(83_u16, perm, a),

        23 =>scrambled_radical_inverse_different_pow2(89_u16, perm, a),

        24 =>scrambled_radical_inverse_different_pow2(97_u16, perm, a),

        25 =>scrambled_radical_inverse_different_pow2(101_u16, perm, a),

        26 =>scrambled_radical_inverse_different_pow2(103_u16, perm, a),

        27 =>scrambled_radical_inverse_different_pow2(107_u16, perm, a),

        28 =>scrambled_radical_inverse_different_pow2(109_u16, perm, a),

        29 =>scrambled_radical_inverse_different_pow2(113_u16, perm, a),

        30 =>scrambled_radical_inverse_different_pow2(127_u16, perm, a),

        31 =>scrambled_radical_inverse_different_pow2(131_u16, perm, a),

        32 =>scrambled_radical_inverse_different_pow2(137_u16, perm, a),

        33 =>scrambled_radical_inverse_different_pow2(139_u16, perm, a),

        34 =>scrambled_radical_inverse_different_pow2(149_u16, perm, a),

        35 =>scrambled_radical_inverse_different_pow2(151_u16, perm, a),

        36 =>scrambled_radical_inverse_different_pow2(157_u16, perm, a),

        37 =>scrambled_radical_inverse_different_pow2(163_u16, perm, a),

        38 =>scrambled_radical_inverse_different_pow2(167_u16, perm, a),

        39 =>scrambled_radical_inverse_different_pow2(173_u16, perm, a),

        40 =>scrambled_radical_inverse_different_pow2(179_u16, perm, a),

        41 =>scrambled_radical_inverse_different_pow2(181_u16, perm, a),

        42 =>scrambled_radical_inverse_different_pow2(191_u16, perm, a),

        43 =>scrambled_radical_inverse_different_pow2(193_u16, perm, a),

        44 =>scrambled_radical_inverse_different_pow2(197_u16, perm, a),

        45 =>scrambled_radical_inverse_different_pow2(199_u16, perm, a),

        46 =>scrambled_radical_inverse_different_pow2(211_u16, perm, a),

        47 =>scrambled_radical_inverse_different_pow2(223_u16, perm, a),

        48 =>scrambled_radical_inverse_different_pow2(227_u16, perm, a),

        49 =>scrambled_radical_inverse_different_pow2(229_u16, perm, a),

        50 =>scrambled_radical_inverse_different_pow2(233_u16, perm, a),

        51 =>scrambled_radical_inverse_different_pow2(239_u16, perm, a),

        52 =>scrambled_radical_inverse_different_pow2(241_u16, perm, a),

        53 =>scrambled_radical_inverse_different_pow2(251_u16, perm, a),

        54 =>scrambled_radical_inverse_different_pow2(257_u16, perm, a),

        55 =>scrambled_radical_inverse_different_pow2(263_u16, perm, a),

        56 =>scrambled_radical_inverse_different_pow2(269_u16, perm, a),

        57 =>scrambled_radical_inverse_different_pow2(271_u16, perm, a),

        58 =>scrambled_radical_inverse_different_pow2(277_u16, perm, a),

        59 =>scrambled_radical_inverse_different_pow2(281_u16, perm, a),

        60 =>scrambled_radical_inverse_different_pow2(283_u16, perm, a),

        61 =>scrambled_radical_inverse_different_pow2(293_u16, perm, a),

        62 =>scrambled_radical_inverse_different_pow2(307_u16, perm, a),

        63 =>scrambled_radical_inverse_different_pow2(311_u16, perm, a),

        64 =>scrambled_radical_inverse_different_pow2(313_u16, perm, a),

        65 =>scrambled_radical_inverse_different_pow2(317_u16, perm, a),

        66 =>scrambled_radical_inverse_different_pow2(331_u16, perm, a),

        67 =>scrambled_radical_inverse_different_pow2(337_u16, perm, a),

        68 =>scrambled_radical_inverse_different_pow2(347_u16, perm, a),

        69 =>scrambled_radical_inverse_different_pow2(349_u16, perm, a),

        70 =>scrambled_radical_inverse_different_pow2(353_u16, perm, a),

        71 =>scrambled_radical_inverse_different_pow2(359_u16, perm, a),

        72 =>scrambled_radical_inverse_different_pow2(367_u16, perm, a),

        73 =>scrambled_radical_inverse_different_pow2(373_u16, perm, a),

        74 =>scrambled_radical_inverse_different_pow2(379_u16, perm, a),

        75 =>scrambled_radical_inverse_different_pow2(383_u16, perm, a),

        76 =>scrambled_radical_inverse_different_pow2(389_u16, perm, a),

        77 =>scrambled_radical_inverse_different_pow2(397_u16, perm, a),

        78 =>scrambled_radical_inverse_different_pow2(401_u16, perm, a),

        79 =>scrambled_radical_inverse_different_pow2(409_u16, perm, a),

        80 =>scrambled_radical_inverse_different_pow2(419_u16, perm, a),

        81 =>scrambled_radical_inverse_different_pow2(421_u16, perm, a),

        82 =>scrambled_radical_inverse_different_pow2(431_u16, perm, a),

        83 =>scrambled_radical_inverse_different_pow2(433_u16, perm, a),

        84 =>scrambled_radical_inverse_different_pow2(439_u16, perm, a),

        85 =>scrambled_radical_inverse_different_pow2(443_u16, perm, a),

        86 =>scrambled_radical_inverse_different_pow2(449_u16, perm, a),

        87 =>scrambled_radical_inverse_different_pow2(457_u16, perm, a),

        88 =>scrambled_radical_inverse_different_pow2(461_u16, perm, a),

        89 =>scrambled_radical_inverse_different_pow2(463_u16, perm, a),

        90 =>scrambled_radical_inverse_different_pow2(467_u16, perm, a),

        91 =>scrambled_radical_inverse_different_pow2(479_u16, perm, a),

        92 =>scrambled_radical_inverse_different_pow2(487_u16, perm, a),

        93 =>scrambled_radical_inverse_different_pow2(491_u16, perm, a),

        94 =>scrambled_radical_inverse_different_pow2(499_u16, perm, a),

        95 =>scrambled_radical_inverse_different_pow2(503_u16, perm, a),

        96 =>scrambled_radical_inverse_different_pow2(509_u16, perm, a),

        97 =>scrambled_radical_inverse_different_pow2(521_u16, perm, a),

        98 =>scrambled_radical_inverse_different_pow2(523_u16, perm, a),

        99 =>scrambled_radical_inverse_different_pow2(541_u16, perm, a),

        100 =>scrambled_radical_inverse_different_pow2(547_u16, perm, a),

        101 =>scrambled_radical_inverse_different_pow2(557_u16, perm, a),

        102 =>scrambled_radical_inverse_different_pow2(563_u16, perm, a),

        103 =>scrambled_radical_inverse_different_pow2(569_u16, perm, a),

        104 =>scrambled_radical_inverse_different_pow2(571_u16, perm, a),

        105 =>scrambled_radical_inverse_different_pow2(577_u16, perm, a),

        106 =>scrambled_radical_inverse_different_pow2(587_u16, perm, a),

        107 =>scrambled_radical_inverse_different_pow2(593_u16, perm, a),

        108 =>scrambled_radical_inverse_different_pow2(599_u16, perm, a),

        109 =>scrambled_radical_inverse_different_pow2(601_u16, perm, a),

        110 =>scrambled_radical_inverse_different_pow2(607_u16, perm, a),

        111 =>scrambled_radical_inverse_different_pow2(613_u16, perm, a),

        112 =>scrambled_radical_inverse_different_pow2(617_u16, perm, a),

        113 =>scrambled_radical_inverse_different_pow2(619_u16, perm, a),

        114 =>scrambled_radical_inverse_different_pow2(631_u16, perm, a),

        115 =>scrambled_radical_inverse_different_pow2(641_u16, perm, a),

        116 =>scrambled_radical_inverse_different_pow2(643_u16, perm, a),

        117 =>scrambled_radical_inverse_different_pow2(647_u16, perm, a),

        118 =>scrambled_radical_inverse_different_pow2(653_u16, perm, a),

        119 =>scrambled_radical_inverse_different_pow2(659_u16, perm, a),

        120 =>scrambled_radical_inverse_different_pow2(661_u16, perm, a),

        121 =>scrambled_radical_inverse_different_pow2(673_u16, perm, a),

        122 =>scrambled_radical_inverse_different_pow2(677_u16, perm, a),

        123 =>scrambled_radical_inverse_different_pow2(683_u16, perm, a),

        124 =>scrambled_radical_inverse_different_pow2(691_u16, perm, a),

        125 =>scrambled_radical_inverse_different_pow2(701_u16, perm, a),

        126 =>scrambled_radical_inverse_different_pow2(709_u16, perm, a),

        127 =>scrambled_radical_inverse_different_pow2(719_u16, perm, a),

        128 =>scrambled_radical_inverse_different_pow2(727_u16, perm, a),

        129 =>scrambled_radical_inverse_different_pow2(733_u16, perm, a),

        130 =>scrambled_radical_inverse_different_pow2(739_u16, perm, a),

        131 =>scrambled_radical_inverse_different_pow2(743_u16, perm, a),

        132 =>scrambled_radical_inverse_different_pow2(751_u16, perm, a),

        133 =>scrambled_radical_inverse_different_pow2(757_u16, perm, a),

        134 =>scrambled_radical_inverse_different_pow2(761_u16, perm, a),

        135 =>scrambled_radical_inverse_different_pow2(769_u16, perm, a),

        136 =>scrambled_radical_inverse_different_pow2(773_u16, perm, a),

        137 =>scrambled_radical_inverse_different_pow2(787_u16, perm, a),

        138 =>scrambled_radical_inverse_different_pow2(797_u16, perm, a),

        139 =>scrambled_radical_inverse_different_pow2(809_u16, perm, a),

        140 =>scrambled_radical_inverse_different_pow2(811_u16, perm, a),

        141 =>scrambled_radical_inverse_different_pow2(821_u16, perm, a),

        142 =>scrambled_radical_inverse_different_pow2(823_u16, perm, a),

        143 =>scrambled_radical_inverse_different_pow2(827_u16, perm, a),

        144 =>scrambled_radical_inverse_different_pow2(829_u16, perm, a),

        145 =>scrambled_radical_inverse_different_pow2(839_u16, perm, a),

        146 =>scrambled_radical_inverse_different_pow2(853_u16, perm, a),

        147 =>scrambled_radical_inverse_different_pow2(857_u16, perm, a),

        148 =>scrambled_radical_inverse_different_pow2(859_u16, perm, a),

        149 =>scrambled_radical_inverse_different_pow2(863_u16, perm, a),

        150 =>scrambled_radical_inverse_different_pow2(877_u16, perm, a),

        151 =>scrambled_radical_inverse_different_pow2(881_u16, perm, a),

        152 =>scrambled_radical_inverse_different_pow2(883_u16, perm, a),

        153 =>scrambled_radical_inverse_different_pow2(887_u16, perm, a),

        154 =>scrambled_radical_inverse_different_pow2(907_u16, perm, a),

        155 =>scrambled_radical_inverse_different_pow2(911_u16, perm, a),

        156 =>scrambled_radical_inverse_different_pow2(919_u16, perm, a),

        157 =>scrambled_radical_inverse_different_pow2(929_u16, perm, a),

        158 =>scrambled_radical_inverse_different_pow2(937_u16, perm, a),

        159 =>scrambled_radical_inverse_different_pow2(941_u16, perm, a),

        160 =>scrambled_radical_inverse_different_pow2(947_u16, perm, a),

        161 =>scrambled_radical_inverse_different_pow2(953_u16, perm, a),

        162 =>scrambled_radical_inverse_different_pow2(967_u16, perm, a),

        163 =>scrambled_radical_inverse_different_pow2(971_u16, perm, a),

        164 =>scrambled_radical_inverse_different_pow2(977_u16, perm, a),

        165 =>scrambled_radical_inverse_different_pow2(983_u16, perm, a),

        166 =>scrambled_radical_inverse_different_pow2(991_u16, perm, a),

        167 =>scrambled_radical_inverse_different_pow2(997_u16, perm, a),

        168 =>scrambled_radical_inverse_different_pow2(1009_u16, perm, a),

        169 =>scrambled_radical_inverse_different_pow2(1013_u16, perm, a),

        170 =>scrambled_radical_inverse_different_pow2(1019_u16, perm, a),

        171 =>scrambled_radical_inverse_different_pow2(1021_u16, perm, a),

        172 =>scrambled_radical_inverse_different_pow2(1031_u16, perm, a),

        173 =>scrambled_radical_inverse_different_pow2(1033_u16, perm, a),

        174 =>scrambled_radical_inverse_different_pow2(1039_u16, perm, a),

        175 =>scrambled_radical_inverse_different_pow2(1049_u16, perm, a),

        176 =>scrambled_radical_inverse_different_pow2(1051_u16, perm, a),

        177 =>scrambled_radical_inverse_different_pow2(1061_u16, perm, a),

        178 =>scrambled_radical_inverse_different_pow2(1063_u16, perm, a),

        179 =>scrambled_radical_inverse_different_pow2(1069_u16, perm, a),

        180 =>scrambled_radical_inverse_different_pow2(1087_u16, perm, a),

        181 =>scrambled_radical_inverse_different_pow2(1091_u16, perm, a),

        182 =>scrambled_radical_inverse_different_pow2(1093_u16, perm, a),

        183 =>scrambled_radical_inverse_different_pow2(1097_u16, perm, a),

        184 =>scrambled_radical_inverse_different_pow2(1103_u16, perm, a),

        185 =>scrambled_radical_inverse_different_pow2(1109_u16, perm, a),

        186 =>scrambled_radical_inverse_different_pow2(1117_u16, perm, a),

        187 =>scrambled_radical_inverse_different_pow2(1123_u16, perm, a),

        188 =>scrambled_radical_inverse_different_pow2(1129_u16, perm, a),

        189 =>scrambled_radical_inverse_different_pow2(1151_u16, perm, a),

        190 =>scrambled_radical_inverse_different_pow2(1153_u16, perm, a),

        191 =>scrambled_radical_inverse_different_pow2(1163_u16, perm, a),

        192 =>scrambled_radical_inverse_different_pow2(1171_u16, perm, a),

        193 =>scrambled_radical_inverse_different_pow2(1181_u16, perm, a),

        194 =>scrambled_radical_inverse_different_pow2(1187_u16, perm, a),

        195 =>scrambled_radical_inverse_different_pow2(1193_u16, perm, a),

        196 =>scrambled_radical_inverse_different_pow2(1201_u16, perm, a),

        197 =>scrambled_radical_inverse_different_pow2(1213_u16, perm, a),

        198 =>scrambled_radical_inverse_different_pow2(1217_u16, perm, a),

        199 =>scrambled_radical_inverse_different_pow2(1223_u16, perm, a),

        200 =>scrambled_radical_inverse_different_pow2(1229_u16, perm, a),

        201 =>scrambled_radical_inverse_different_pow2(1231_u16, perm, a),

        202 =>scrambled_radical_inverse_different_pow2(1237_u16, perm, a),

        203 =>scrambled_radical_inverse_different_pow2(1249_u16, perm, a),

        204 =>scrambled_radical_inverse_different_pow2(1259_u16, perm, a),

        205 =>scrambled_radical_inverse_different_pow2(1277_u16, perm, a),

        206 =>scrambled_radical_inverse_different_pow2(1279_u16, perm, a),

        207 =>scrambled_radical_inverse_different_pow2(1283_u16, perm, a),

        208 =>scrambled_radical_inverse_different_pow2(1289_u16, perm, a),

        209 =>scrambled_radical_inverse_different_pow2(1291_u16, perm, a),

        210 =>scrambled_radical_inverse_different_pow2(1297_u16, perm, a),

        211 =>scrambled_radical_inverse_different_pow2(1301_u16, perm, a),

        212 =>scrambled_radical_inverse_different_pow2(1303_u16, perm, a),

        213 =>scrambled_radical_inverse_different_pow2(1307_u16, perm, a),

        214 =>scrambled_radical_inverse_different_pow2(1319_u16, perm, a),

        215 =>scrambled_radical_inverse_different_pow2(1321_u16, perm, a),

        216 =>scrambled_radical_inverse_different_pow2(1327_u16, perm, a),

        217 =>scrambled_radical_inverse_different_pow2(1361_u16, perm, a),

        218 =>scrambled_radical_inverse_different_pow2(1367_u16, perm, a),

        219 =>scrambled_radical_inverse_different_pow2(1373_u16, perm, a),

        220 =>scrambled_radical_inverse_different_pow2(1381_u16, perm, a),

        221 =>scrambled_radical_inverse_different_pow2(1399_u16, perm, a),

        222 =>scrambled_radical_inverse_different_pow2(1409_u16, perm, a),

        223 =>scrambled_radical_inverse_different_pow2(1423_u16, perm, a),

        224 =>scrambled_radical_inverse_different_pow2(1427_u16, perm, a),

        225 =>scrambled_radical_inverse_different_pow2(1429_u16, perm, a),

        226 =>scrambled_radical_inverse_different_pow2(1433_u16, perm, a),

        227 =>scrambled_radical_inverse_different_pow2(1439_u16, perm, a),

        228 =>scrambled_radical_inverse_different_pow2(1447_u16, perm, a),

        229 =>scrambled_radical_inverse_different_pow2(1451_u16, perm, a),

        230 =>scrambled_radical_inverse_different_pow2(1453_u16, perm, a),

        231 =>scrambled_radical_inverse_different_pow2(1459_u16, perm, a),

        232 =>scrambled_radical_inverse_different_pow2(1471_u16, perm, a),

        233 =>scrambled_radical_inverse_different_pow2(1481_u16, perm, a),

        234 =>scrambled_radical_inverse_different_pow2(1483_u16, perm, a),

        235 =>scrambled_radical_inverse_different_pow2(1487_u16, perm, a),

        236 =>scrambled_radical_inverse_different_pow2(1489_u16, perm, a),

        237 =>scrambled_radical_inverse_different_pow2(1493_u16, perm, a),

        238 =>scrambled_radical_inverse_different_pow2(1499_u16, perm, a),

        239 =>scrambled_radical_inverse_different_pow2(1511_u16, perm, a),

        240 =>scrambled_radical_inverse_different_pow2(1523_u16, perm, a),

        241 =>scrambled_radical_inverse_different_pow2(1531_u16, perm, a),

        242 =>scrambled_radical_inverse_different_pow2(1543_u16, perm, a),

        243 =>scrambled_radical_inverse_different_pow2(1549_u16, perm, a),

        244 =>scrambled_radical_inverse_different_pow2(1553_u16, perm, a),

        245 =>scrambled_radical_inverse_different_pow2(1559_u16, perm, a),

        246 =>scrambled_radical_inverse_different_pow2(1567_u16, perm, a),

        247 =>scrambled_radical_inverse_different_pow2(1571_u16, perm, a),

        248 =>scrambled_radical_inverse_different_pow2(1579_u16, perm, a),

        249 =>scrambled_radical_inverse_different_pow2(1583_u16, perm, a),

        250 =>scrambled_radical_inverse_different_pow2(1597_u16, perm, a),

        251 =>scrambled_radical_inverse_different_pow2(1601_u16, perm, a),

        252 =>scrambled_radical_inverse_different_pow2(1607_u16, perm, a),

        253 =>scrambled_radical_inverse_different_pow2(1609_u16, perm, a),

        254 =>scrambled_radical_inverse_different_pow2(1613_u16, perm, a),

        255 =>scrambled_radical_inverse_different_pow2(1619_u16, perm, a),

        256 =>scrambled_radical_inverse_different_pow2(1621_u16, perm, a),

        257 =>scrambled_radical_inverse_different_pow2(1627_u16, perm, a),

        258 =>scrambled_radical_inverse_different_pow2(1637_u16, perm, a),

        259 =>scrambled_radical_inverse_different_pow2(1657_u16, perm, a),

        260 =>scrambled_radical_inverse_different_pow2(1663_u16, perm, a),

        261 =>scrambled_radical_inverse_different_pow2(1667_u16, perm, a),

        262 =>scrambled_radical_inverse_different_pow2(1669_u16, perm, a),

        263 =>scrambled_radical_inverse_different_pow2(1693_u16, perm, a),

        264 =>scrambled_radical_inverse_different_pow2(1697_u16, perm, a),

        265 =>scrambled_radical_inverse_different_pow2(1699_u16, perm, a),

        266 =>scrambled_radical_inverse_different_pow2(1709_u16, perm, a),

        267 =>scrambled_radical_inverse_different_pow2(1721_u16, perm, a),

        268 =>scrambled_radical_inverse_different_pow2(1723_u16, perm, a),

        269 =>scrambled_radical_inverse_different_pow2(1733_u16, perm, a),

        270 =>scrambled_radical_inverse_different_pow2(1741_u16, perm, a),

        271 =>scrambled_radical_inverse_different_pow2(1747_u16, perm, a),

        272 =>scrambled_radical_inverse_different_pow2(1753_u16, perm, a),

        273 =>scrambled_radical_inverse_different_pow2(1759_u16, perm, a),

        274 =>scrambled_radical_inverse_different_pow2(1777_u16, perm, a),

        275 =>scrambled_radical_inverse_different_pow2(1783_u16, perm, a),

        276 =>scrambled_radical_inverse_different_pow2(1787_u16, perm, a),

        277 =>scrambled_radical_inverse_different_pow2(1789_u16, perm, a),

        278 =>scrambled_radical_inverse_different_pow2(1801_u16, perm, a),

        279 =>scrambled_radical_inverse_different_pow2(1811_u16, perm, a),

        280 =>scrambled_radical_inverse_different_pow2(1823_u16, perm, a),

        281 =>scrambled_radical_inverse_different_pow2(1831_u16, perm, a),

        282 =>scrambled_radical_inverse_different_pow2(1847_u16, perm, a),

        283 =>scrambled_radical_inverse_different_pow2(1861_u16, perm, a),

        284 =>scrambled_radical_inverse_different_pow2(1867_u16, perm, a),

        285 =>scrambled_radical_inverse_different_pow2(1871_u16, perm, a),

        286 =>scrambled_radical_inverse_different_pow2(1873_u16, perm, a),

        287 =>scrambled_radical_inverse_different_pow2(1877_u16, perm, a),

        288 =>scrambled_radical_inverse_different_pow2(1879_u16, perm, a),

        289 =>scrambled_radical_inverse_different_pow2(1889_u16, perm, a),

        290 =>scrambled_radical_inverse_different_pow2(1901_u16, perm, a),

        291 =>scrambled_radical_inverse_different_pow2(1907_u16, perm, a),

        292 =>scrambled_radical_inverse_different_pow2(1913_u16, perm, a),

        293 =>scrambled_radical_inverse_different_pow2(1931_u16, perm, a),

        294 =>scrambled_radical_inverse_different_pow2(1933_u16, perm, a),

        295 =>scrambled_radical_inverse_different_pow2(1949_u16, perm, a),

        296 =>scrambled_radical_inverse_different_pow2(1951_u16, perm, a),

        297 =>scrambled_radical_inverse_different_pow2(1973_u16, perm, a),

        298 =>scrambled_radical_inverse_different_pow2(1979_u16, perm, a),

        299 =>scrambled_radical_inverse_different_pow2(1987_u16, perm, a),

        300 =>scrambled_radical_inverse_different_pow2(1993_u16, perm, a),

        301 =>scrambled_radical_inverse_different_pow2(1997_u16, perm, a),

        302 =>scrambled_radical_inverse_different_pow2(1999_u16, perm, a),

        303 =>scrambled_radical_inverse_different_pow2(2003_u16, perm, a),

        304 =>scrambled_radical_inverse_different_pow2(2011_u16, perm, a),

        305 =>scrambled_radical_inverse_different_pow2(2017_u16, perm, a),

        306 =>scrambled_radical_inverse_different_pow2(2027_u16, perm, a),

        307 =>scrambled_radical_inverse_different_pow2(2029_u16, perm, a),

        308 =>scrambled_radical_inverse_different_pow2(2039_u16, perm, a),

        309 =>scrambled_radical_inverse_different_pow2(2053_u16, perm, a),

        310 =>scrambled_radical_inverse_different_pow2(2063_u16, perm, a),

        311 =>scrambled_radical_inverse_different_pow2(2069_u16, perm, a),

        312 =>scrambled_radical_inverse_different_pow2(2081_u16, perm, a),

        313 =>scrambled_radical_inverse_different_pow2(2083_u16, perm, a),

        314 =>scrambled_radical_inverse_different_pow2(2087_u16, perm, a),

        315 =>scrambled_radical_inverse_different_pow2(2089_u16, perm, a),

        316 =>scrambled_radical_inverse_different_pow2(2099_u16, perm, a),

        317 =>scrambled_radical_inverse_different_pow2(2111_u16, perm, a),

        318 =>scrambled_radical_inverse_different_pow2(2113_u16, perm, a),

        319 =>scrambled_radical_inverse_different_pow2(2129_u16, perm, a),

        320 =>scrambled_radical_inverse_different_pow2(2131_u16, perm, a),

        321 =>scrambled_radical_inverse_different_pow2(2137_u16, perm, a),

        322 =>scrambled_radical_inverse_different_pow2(2141_u16, perm, a),

        323 =>scrambled_radical_inverse_different_pow2(2143_u16, perm, a),

        324 =>scrambled_radical_inverse_different_pow2(2153_u16, perm, a),

        325 =>scrambled_radical_inverse_different_pow2(2161_u16, perm, a),

        326 =>scrambled_radical_inverse_different_pow2(2179_u16, perm, a),

        327 =>scrambled_radical_inverse_different_pow2(2203_u16, perm, a),

        328 =>scrambled_radical_inverse_different_pow2(2207_u16, perm, a),

        329 =>scrambled_radical_inverse_different_pow2(2213_u16, perm, a),

        330 =>scrambled_radical_inverse_different_pow2(2221_u16, perm, a),

        331 =>scrambled_radical_inverse_different_pow2(2237_u16, perm, a),

        332 =>scrambled_radical_inverse_different_pow2(2239_u16, perm, a),

        333 =>scrambled_radical_inverse_different_pow2(2243_u16, perm, a),

        334 =>scrambled_radical_inverse_different_pow2(2251_u16, perm, a),

        335 =>scrambled_radical_inverse_different_pow2(2267_u16, perm, a),

        336 =>scrambled_radical_inverse_different_pow2(2269_u16, perm, a),

        337 =>scrambled_radical_inverse_different_pow2(2273_u16, perm, a),

        338 =>scrambled_radical_inverse_different_pow2(2281_u16, perm, a),

        339 =>scrambled_radical_inverse_different_pow2(2287_u16, perm, a),

        340 =>scrambled_radical_inverse_different_pow2(2293_u16, perm, a),

        341 =>scrambled_radical_inverse_different_pow2(2297_u16, perm, a),

        342 =>scrambled_radical_inverse_different_pow2(2309_u16, perm, a),

        343 =>scrambled_radical_inverse_different_pow2(2311_u16, perm, a),

        344 =>scrambled_radical_inverse_different_pow2(2333_u16, perm, a),

        345 =>scrambled_radical_inverse_different_pow2(2339_u16, perm, a),

        346 =>scrambled_radical_inverse_different_pow2(2341_u16, perm, a),

        347 =>scrambled_radical_inverse_different_pow2(2347_u16, perm, a),

        348 =>scrambled_radical_inverse_different_pow2(2351_u16, perm, a),

        349 =>scrambled_radical_inverse_different_pow2(2357_u16, perm, a),

        350 =>scrambled_radical_inverse_different_pow2(2371_u16, perm, a),

        351 =>scrambled_radical_inverse_different_pow2(2377_u16, perm, a),

        352 =>scrambled_radical_inverse_different_pow2(2381_u16, perm, a),

        353 =>scrambled_radical_inverse_different_pow2(2383_u16, perm, a),

        354 =>scrambled_radical_inverse_different_pow2(2389_u16, perm, a),

        355 =>scrambled_radical_inverse_different_pow2(2393_u16, perm, a),

        356 =>scrambled_radical_inverse_different_pow2(2399_u16, perm, a),

        357 =>scrambled_radical_inverse_different_pow2(2411_u16, perm, a),

        358 =>scrambled_radical_inverse_different_pow2(2417_u16, perm, a),

        359 =>scrambled_radical_inverse_different_pow2(2423_u16, perm, a),

        360 =>scrambled_radical_inverse_different_pow2(2437_u16, perm, a),

        361 =>scrambled_radical_inverse_different_pow2(2441_u16, perm, a),

        362 =>scrambled_radical_inverse_different_pow2(2447_u16, perm, a),

        363 =>scrambled_radical_inverse_different_pow2(2459_u16, perm, a),

        364 =>scrambled_radical_inverse_different_pow2(2467_u16, perm, a),

        365 =>scrambled_radical_inverse_different_pow2(2473_u16, perm, a),

        366 =>scrambled_radical_inverse_different_pow2(2477_u16, perm, a),

        367 =>scrambled_radical_inverse_different_pow2(2503_u16, perm, a),

        368 =>scrambled_radical_inverse_different_pow2(2521_u16, perm, a),

        369 =>scrambled_radical_inverse_different_pow2(2531_u16, perm, a),

        370 =>scrambled_radical_inverse_different_pow2(2539_u16, perm, a),

        371 =>scrambled_radical_inverse_different_pow2(2543_u16, perm, a),

        372 =>scrambled_radical_inverse_different_pow2(2549_u16, perm, a),

        373 =>scrambled_radical_inverse_different_pow2(2551_u16, perm, a),

        374 =>scrambled_radical_inverse_different_pow2(2557_u16, perm, a),

        375 =>scrambled_radical_inverse_different_pow2(2579_u16, perm, a),

        376 =>scrambled_radical_inverse_different_pow2(2591_u16, perm, a),

        377 =>scrambled_radical_inverse_different_pow2(2593_u16, perm, a),

        378 =>scrambled_radical_inverse_different_pow2(2609_u16, perm, a),

        379 =>scrambled_radical_inverse_different_pow2(2617_u16, perm, a),

        380 =>scrambled_radical_inverse_different_pow2(2621_u16, perm, a),

        381 =>scrambled_radical_inverse_different_pow2(2633_u16, perm, a),

        382 =>scrambled_radical_inverse_different_pow2(2647_u16, perm, a),

        383 =>scrambled_radical_inverse_different_pow2(2657_u16, perm, a),

        384 =>scrambled_radical_inverse_different_pow2(2659_u16, perm, a),

        385 =>scrambled_radical_inverse_different_pow2(2663_u16, perm, a),

        386 =>scrambled_radical_inverse_different_pow2(2671_u16, perm, a),

        387 =>scrambled_radical_inverse_different_pow2(2677_u16, perm, a),

        388 =>scrambled_radical_inverse_different_pow2(2683_u16, perm, a),

        389 =>scrambled_radical_inverse_different_pow2(2687_u16, perm, a),

        390 =>scrambled_radical_inverse_different_pow2(2689_u16, perm, a),

        391 =>scrambled_radical_inverse_different_pow2(2693_u16, perm, a),

        392 =>scrambled_radical_inverse_different_pow2(2699_u16, perm, a),

        393 =>scrambled_radical_inverse_different_pow2(2707_u16, perm, a),

        394 =>scrambled_radical_inverse_different_pow2(2711_u16, perm, a),

        395 =>scrambled_radical_inverse_different_pow2(2713_u16, perm, a),

        396 =>scrambled_radical_inverse_different_pow2(2719_u16, perm, a),

        397 =>scrambled_radical_inverse_different_pow2(2729_u16, perm, a),

        398 =>scrambled_radical_inverse_different_pow2(2731_u16, perm, a),

        399 =>scrambled_radical_inverse_different_pow2(2741_u16, perm, a),

        400 =>scrambled_radical_inverse_different_pow2(2749_u16, perm, a),

        401 =>scrambled_radical_inverse_different_pow2(2753_u16, perm, a),

        402 =>scrambled_radical_inverse_different_pow2(2767_u16, perm, a),

        403 =>scrambled_radical_inverse_different_pow2(2777_u16, perm, a),

        404 =>scrambled_radical_inverse_different_pow2(2789_u16, perm, a),

        405 =>scrambled_radical_inverse_different_pow2(2791_u16, perm, a),

        406 =>scrambled_radical_inverse_different_pow2(2797_u16, perm, a),

        407 =>scrambled_radical_inverse_different_pow2(2801_u16, perm, a),

        408 =>scrambled_radical_inverse_different_pow2(2803_u16, perm, a),

        409 =>scrambled_radical_inverse_different_pow2(2819_u16, perm, a),

        410 =>scrambled_radical_inverse_different_pow2(2833_u16, perm, a),

        411 =>scrambled_radical_inverse_different_pow2(2837_u16, perm, a),

        412 =>scrambled_radical_inverse_different_pow2(2843_u16, perm, a),

        413 =>scrambled_radical_inverse_different_pow2(2851_u16, perm, a),

        414 =>scrambled_radical_inverse_different_pow2(2857_u16, perm, a),

        415 =>scrambled_radical_inverse_different_pow2(2861_u16, perm, a),

        416 =>scrambled_radical_inverse_different_pow2(2879_u16, perm, a),

        417 =>scrambled_radical_inverse_different_pow2(2887_u16, perm, a),

        418 =>scrambled_radical_inverse_different_pow2(2897_u16, perm, a),

        419 =>scrambled_radical_inverse_different_pow2(2903_u16, perm, a),

        420 =>scrambled_radical_inverse_different_pow2(2909_u16, perm, a),

        421 =>scrambled_radical_inverse_different_pow2(2917_u16, perm, a),

        422 =>scrambled_radical_inverse_different_pow2(2927_u16, perm, a),

        423 =>scrambled_radical_inverse_different_pow2(2939_u16, perm, a),

        424 =>scrambled_radical_inverse_different_pow2(2953_u16, perm, a),

        425 =>scrambled_radical_inverse_different_pow2(2957_u16, perm, a),

        426 =>scrambled_radical_inverse_different_pow2(2963_u16, perm, a),

        427 =>scrambled_radical_inverse_different_pow2(2969_u16, perm, a),

        428 =>scrambled_radical_inverse_different_pow2(2971_u16, perm, a),

        429 =>scrambled_radical_inverse_different_pow2(2999_u16, perm, a),

        430 =>scrambled_radical_inverse_different_pow2(3001_u16, perm, a),

        431 =>scrambled_radical_inverse_different_pow2(3011_u16, perm, a),

        432 =>scrambled_radical_inverse_different_pow2(3019_u16, perm, a),

        433 =>scrambled_radical_inverse_different_pow2(3023_u16, perm, a),

        434 =>scrambled_radical_inverse_different_pow2(3037_u16, perm, a),

        435 =>scrambled_radical_inverse_different_pow2(3041_u16, perm, a),

        436 =>scrambled_radical_inverse_different_pow2(3049_u16, perm, a),

        437 =>scrambled_radical_inverse_different_pow2(3061_u16, perm, a),

        438 =>scrambled_radical_inverse_different_pow2(3067_u16, perm, a),

        439 =>scrambled_radical_inverse_different_pow2(3079_u16, perm, a),

        440 =>scrambled_radical_inverse_different_pow2(3083_u16, perm, a),

        441 =>scrambled_radical_inverse_different_pow2(3089_u16, perm, a),

        442 =>scrambled_radical_inverse_different_pow2(3109_u16, perm, a),

        443 =>scrambled_radical_inverse_different_pow2(3119_u16, perm, a),

        444 =>scrambled_radical_inverse_different_pow2(3121_u16, perm, a),

        445 =>scrambled_radical_inverse_different_pow2(3137_u16, perm, a),

        446 =>scrambled_radical_inverse_different_pow2(3163_u16, perm, a),

        447 =>scrambled_radical_inverse_different_pow2(3167_u16, perm, a),

        448 =>scrambled_radical_inverse_different_pow2(3169_u16, perm, a),

        449 =>scrambled_radical_inverse_different_pow2(3181_u16, perm, a),

        450 =>scrambled_radical_inverse_different_pow2(3187_u16, perm, a),

        451 =>scrambled_radical_inverse_different_pow2(3191_u16, perm, a),

        452 =>scrambled_radical_inverse_different_pow2(3203_u16, perm, a),

        453 =>scrambled_radical_inverse_different_pow2(3209_u16, perm, a),

        454 =>scrambled_radical_inverse_different_pow2(3217_u16, perm, a),

        455 =>scrambled_radical_inverse_different_pow2(3221_u16, perm, a),

        456 =>scrambled_radical_inverse_different_pow2(3229_u16, perm, a),

        457 =>scrambled_radical_inverse_different_pow2(3251_u16, perm, a),

        458 =>scrambled_radical_inverse_different_pow2(3253_u16, perm, a),

        459 =>scrambled_radical_inverse_different_pow2(3257_u16, perm, a),

        460 =>scrambled_radical_inverse_different_pow2(3259_u16, perm, a),

        461 =>scrambled_radical_inverse_different_pow2(3271_u16, perm, a),

        462 =>scrambled_radical_inverse_different_pow2(3299_u16, perm, a),

        463 =>scrambled_radical_inverse_different_pow2(3301_u16, perm, a),

        464 =>scrambled_radical_inverse_different_pow2(3307_u16, perm, a),

        465 =>scrambled_radical_inverse_different_pow2(3313_u16, perm, a),

        466 =>scrambled_radical_inverse_different_pow2(3319_u16, perm, a),

        467 =>scrambled_radical_inverse_different_pow2(3323_u16, perm, a),

        468 =>scrambled_radical_inverse_different_pow2(3329_u16, perm, a),

        469 =>scrambled_radical_inverse_different_pow2(3331_u16, perm, a),

        470 =>scrambled_radical_inverse_different_pow2(3343_u16, perm, a),

        471 =>scrambled_radical_inverse_different_pow2(3347_u16, perm, a),

        472 =>scrambled_radical_inverse_different_pow2(3359_u16, perm, a),

        473 =>scrambled_radical_inverse_different_pow2(3361_u16, perm, a),

        474 =>scrambled_radical_inverse_different_pow2(3371_u16, perm, a),

        475 =>scrambled_radical_inverse_different_pow2(3373_u16, perm, a),

        476 =>scrambled_radical_inverse_different_pow2(3389_u16, perm, a),

        477 =>scrambled_radical_inverse_different_pow2(3391_u16, perm, a),

        478 =>scrambled_radical_inverse_different_pow2(3407_u16, perm, a),

        479 =>scrambled_radical_inverse_different_pow2(3413_u16, perm, a),

        480 =>scrambled_radical_inverse_different_pow2(3433_u16, perm, a),

        481 =>scrambled_radical_inverse_different_pow2(3449_u16, perm, a),

        482 =>scrambled_radical_inverse_different_pow2(3457_u16, perm, a),

        483 =>scrambled_radical_inverse_different_pow2(3461_u16, perm, a),

        484 =>scrambled_radical_inverse_different_pow2(3463_u16, perm, a),

        485 =>scrambled_radical_inverse_different_pow2(3467_u16, perm, a),

        486 =>scrambled_radical_inverse_different_pow2(3469_u16, perm, a),

        487 =>scrambled_radical_inverse_different_pow2(3491_u16, perm, a),

        488 =>scrambled_radical_inverse_different_pow2(3499_u16, perm, a),

        489 =>scrambled_radical_inverse_different_pow2(3511_u16, perm, a),

        490 =>scrambled_radical_inverse_different_pow2(3517_u16, perm, a),

        491 =>scrambled_radical_inverse_different_pow2(3527_u16, perm, a),

        492 =>scrambled_radical_inverse_different_pow2(3529_u16, perm, a),

        493 =>scrambled_radical_inverse_different_pow2(3533_u16, perm, a),

        494 =>scrambled_radical_inverse_different_pow2(3539_u16, perm, a),

        495 =>scrambled_radical_inverse_different_pow2(3541_u16, perm, a),

        496 =>scrambled_radical_inverse_different_pow2(3547_u16, perm, a),

        497 =>scrambled_radical_inverse_different_pow2(3557_u16, perm, a),

        498 =>scrambled_radical_inverse_different_pow2(3559_u16, perm, a),

        499 =>scrambled_radical_inverse_different_pow2(3571_u16, perm, a),

        500 =>scrambled_radical_inverse_different_pow2(3581_u16, perm, a),

        501 =>scrambled_radical_inverse_different_pow2(3583_u16, perm, a),

        502 =>scrambled_radical_inverse_different_pow2(3593_u16, perm, a),

        503 =>scrambled_radical_inverse_different_pow2(3607_u16, perm, a),

        504 =>scrambled_radical_inverse_different_pow2(3613_u16, perm, a),

        505 =>scrambled_radical_inverse_different_pow2(3617_u16, perm, a),

        506 =>scrambled_radical_inverse_different_pow2(3623_u16, perm, a),

        507 =>scrambled_radical_inverse_different_pow2(3631_u16, perm, a),

        508 =>scrambled_radical_inverse_different_pow2(3637_u16, perm, a),

        509 =>scrambled_radical_inverse_different_pow2(3643_u16, perm, a),

        510 =>scrambled_radical_inverse_different_pow2(3659_u16, perm, a),

        511 =>scrambled_radical_inverse_different_pow2(3671_u16, perm, a),

        512 =>scrambled_radical_inverse_different_pow2(3673_u16, perm, a),

        513 =>scrambled_radical_inverse_different_pow2(3677_u16, perm, a),

        514 =>scrambled_radical_inverse_different_pow2(3691_u16, perm, a),

        515 =>scrambled_radical_inverse_different_pow2(3697_u16, perm, a),

        516 =>scrambled_radical_inverse_different_pow2(3701_u16, perm, a),

        517 =>scrambled_radical_inverse_different_pow2(3709_u16, perm, a),

        518 =>scrambled_radical_inverse_different_pow2(3719_u16, perm, a),

        519 =>scrambled_radical_inverse_different_pow2(3727_u16, perm, a),

        520 =>scrambled_radical_inverse_different_pow2(3733_u16, perm, a),

        521 =>scrambled_radical_inverse_different_pow2(3739_u16, perm, a),

        522 =>scrambled_radical_inverse_different_pow2(3761_u16, perm, a),

        523 =>scrambled_radical_inverse_different_pow2(3767_u16, perm, a),

        524 =>scrambled_radical_inverse_different_pow2(3769_u16, perm, a),

        525 =>scrambled_radical_inverse_different_pow2(3779_u16, perm, a),

        526 =>scrambled_radical_inverse_different_pow2(3793_u16, perm, a),

        527 =>scrambled_radical_inverse_different_pow2(3797_u16, perm, a),

        528 =>scrambled_radical_inverse_different_pow2(3803_u16, perm, a),

        529 =>scrambled_radical_inverse_different_pow2(3821_u16, perm, a),

        530 =>scrambled_radical_inverse_different_pow2(3823_u16, perm, a),

        531 =>scrambled_radical_inverse_different_pow2(3833_u16, perm, a),

        532 =>scrambled_radical_inverse_different_pow2(3847_u16, perm, a),

        533 =>scrambled_radical_inverse_different_pow2(3851_u16, perm, a),

        534 =>scrambled_radical_inverse_different_pow2(3853_u16, perm, a),

        535 =>scrambled_radical_inverse_different_pow2(3863_u16, perm, a),

        536 =>scrambled_radical_inverse_different_pow2(3877_u16, perm, a),

        537 =>scrambled_radical_inverse_different_pow2(3881_u16, perm, a),

        538 =>scrambled_radical_inverse_different_pow2(3889_u16, perm, a),

        539 =>scrambled_radical_inverse_different_pow2(3907_u16, perm, a),

        540 =>scrambled_radical_inverse_different_pow2(3911_u16, perm, a),

        541 =>scrambled_radical_inverse_different_pow2(3917_u16, perm, a),

        542 =>scrambled_radical_inverse_different_pow2(3919_u16, perm, a),

        543 =>scrambled_radical_inverse_different_pow2(3923_u16, perm, a),

        544 =>scrambled_radical_inverse_different_pow2(3929_u16, perm, a),

        545 =>scrambled_radical_inverse_different_pow2(3931_u16, perm, a),

        546 =>scrambled_radical_inverse_different_pow2(3943_u16, perm, a),

        547 =>scrambled_radical_inverse_different_pow2(3947_u16, perm, a),

        548 =>scrambled_radical_inverse_different_pow2(3967_u16, perm, a),

        549 =>scrambled_radical_inverse_different_pow2(3989_u16, perm, a),

        550 =>scrambled_radical_inverse_different_pow2(4001_u16, perm, a),

        551 =>scrambled_radical_inverse_different_pow2(4003_u16, perm, a),

        552 =>scrambled_radical_inverse_different_pow2(4007_u16, perm, a),

        553 =>scrambled_radical_inverse_different_pow2(4013_u16, perm, a),

        554 =>scrambled_radical_inverse_different_pow2(4019_u16, perm, a),

        555 =>scrambled_radical_inverse_different_pow2(4021_u16, perm, a),

        556 =>scrambled_radical_inverse_different_pow2(4027_u16, perm, a),

        557 =>scrambled_radical_inverse_different_pow2(4049_u16, perm, a),

        558 =>scrambled_radical_inverse_different_pow2(4051_u16, perm, a),

        559 =>scrambled_radical_inverse_different_pow2(4057_u16, perm, a),

        560 =>scrambled_radical_inverse_different_pow2(4073_u16, perm, a),

        561 =>scrambled_radical_inverse_different_pow2(4079_u16, perm, a),

        562 =>scrambled_radical_inverse_different_pow2(4091_u16, perm, a),

        563 =>scrambled_radical_inverse_different_pow2(4093_u16, perm, a),

        564 =>scrambled_radical_inverse_different_pow2(4099_u16, perm, a),

        565 =>scrambled_radical_inverse_different_pow2(4111_u16, perm, a),

        566 =>scrambled_radical_inverse_different_pow2(4127_u16, perm, a),

        567 =>scrambled_radical_inverse_different_pow2(4129_u16, perm, a),

        568 =>scrambled_radical_inverse_different_pow2(4133_u16, perm, a),

        569 =>scrambled_radical_inverse_different_pow2(4139_u16, perm, a),

        570 =>scrambled_radical_inverse_different_pow2(4153_u16, perm, a),

        571 =>scrambled_radical_inverse_different_pow2(4157_u16, perm, a),

        572 =>scrambled_radical_inverse_different_pow2(4159_u16, perm, a),

        573 =>scrambled_radical_inverse_different_pow2(4177_u16, perm, a),

        574 =>scrambled_radical_inverse_different_pow2(4201_u16, perm, a),

        575 =>scrambled_radical_inverse_different_pow2(4211_u16, perm, a),

        576 =>scrambled_radical_inverse_different_pow2(4217_u16, perm, a),

        577 =>scrambled_radical_inverse_different_pow2(4219_u16, perm, a),

        578 =>scrambled_radical_inverse_different_pow2(4229_u16, perm, a),

        579 =>scrambled_radical_inverse_different_pow2(4231_u16, perm, a),

        580 =>scrambled_radical_inverse_different_pow2(4241_u16, perm, a),

        581 =>scrambled_radical_inverse_different_pow2(4243_u16, perm, a),

        582 =>scrambled_radical_inverse_different_pow2(4253_u16, perm, a),

        583 =>scrambled_radical_inverse_different_pow2(4259_u16, perm, a),

        584 =>scrambled_radical_inverse_different_pow2(4261_u16, perm, a),

        585 =>scrambled_radical_inverse_different_pow2(4271_u16, perm, a),

        586 =>scrambled_radical_inverse_different_pow2(4273_u16, perm, a),

        587 =>scrambled_radical_inverse_different_pow2(4283_u16, perm, a),

        588 =>scrambled_radical_inverse_different_pow2(4289_u16, perm, a),

        589 =>scrambled_radical_inverse_different_pow2(4297_u16, perm, a),

        590 =>scrambled_radical_inverse_different_pow2(4327_u16, perm, a),

        591 =>scrambled_radical_inverse_different_pow2(4337_u16, perm, a),

        592 =>scrambled_radical_inverse_different_pow2(4339_u16, perm, a),

        593 =>scrambled_radical_inverse_different_pow2(4349_u16, perm, a),

        594 =>scrambled_radical_inverse_different_pow2(4357_u16, perm, a),

        595 =>scrambled_radical_inverse_different_pow2(4363_u16, perm, a),

        596 =>scrambled_radical_inverse_different_pow2(4373_u16, perm, a),

        597 =>scrambled_radical_inverse_different_pow2(4391_u16, perm, a),

        598 =>scrambled_radical_inverse_different_pow2(4397_u16, perm, a),

        599 =>scrambled_radical_inverse_different_pow2(4409_u16, perm, a),

        600 =>scrambled_radical_inverse_different_pow2(4421_u16, perm, a),

        601 =>scrambled_radical_inverse_different_pow2(4423_u16, perm, a),

        602 =>scrambled_radical_inverse_different_pow2(4441_u16, perm, a),

        603 =>scrambled_radical_inverse_different_pow2(4447_u16, perm, a),

        604 =>scrambled_radical_inverse_different_pow2(4451_u16, perm, a),

        605 =>scrambled_radical_inverse_different_pow2(4457_u16, perm, a),

        606 =>scrambled_radical_inverse_different_pow2(4463_u16, perm, a),

        607 =>scrambled_radical_inverse_different_pow2(4481_u16, perm, a),

        608 =>scrambled_radical_inverse_different_pow2(4483_u16, perm, a),

        609 =>scrambled_radical_inverse_different_pow2(4493_u16, perm, a),

        610 =>scrambled_radical_inverse_different_pow2(4507_u16, perm, a),

        611 =>scrambled_radical_inverse_different_pow2(4513_u16, perm, a),

        612 =>scrambled_radical_inverse_different_pow2(4517_u16, perm, a),

        613 =>scrambled_radical_inverse_different_pow2(4519_u16, perm, a),

        614 =>scrambled_radical_inverse_different_pow2(4523_u16, perm, a),

        615 =>scrambled_radical_inverse_different_pow2(4547_u16, perm, a),

        616 =>scrambled_radical_inverse_different_pow2(4549_u16, perm, a),

        617 =>scrambled_radical_inverse_different_pow2(4561_u16, perm, a),

        618 =>scrambled_radical_inverse_different_pow2(4567_u16, perm, a),

        619 =>scrambled_radical_inverse_different_pow2(4583_u16, perm, a),

        620 =>scrambled_radical_inverse_different_pow2(4591_u16, perm, a),

        621 =>scrambled_radical_inverse_different_pow2(4597_u16, perm, a),

        622 =>scrambled_radical_inverse_different_pow2(4603_u16, perm, a),

        623 =>scrambled_radical_inverse_different_pow2(4621_u16, perm, a),

        624 =>scrambled_radical_inverse_different_pow2(4637_u16, perm, a),

        625 =>scrambled_radical_inverse_different_pow2(4639_u16, perm, a),

        626 =>scrambled_radical_inverse_different_pow2(4643_u16, perm, a),

        627 =>scrambled_radical_inverse_different_pow2(4649_u16, perm, a),

        628 =>scrambled_radical_inverse_different_pow2(4651_u16, perm, a),

        629 =>scrambled_radical_inverse_different_pow2(4657_u16, perm, a),

        630 =>scrambled_radical_inverse_different_pow2(4663_u16, perm, a),

        631 =>scrambled_radical_inverse_different_pow2(4673_u16, perm, a),

        632 =>scrambled_radical_inverse_different_pow2(4679_u16, perm, a),

        633 =>scrambled_radical_inverse_different_pow2(4691_u16, perm, a),

        634 =>scrambled_radical_inverse_different_pow2(4703_u16, perm, a),

        635 =>scrambled_radical_inverse_different_pow2(4721_u16, perm, a),

        636 =>scrambled_radical_inverse_different_pow2(4723_u16, perm, a),

        637 =>scrambled_radical_inverse_different_pow2(4729_u16, perm, a),

        638 =>scrambled_radical_inverse_different_pow2(4733_u16, perm, a),

        639 =>scrambled_radical_inverse_different_pow2(4751_u16, perm, a),

        640 =>scrambled_radical_inverse_different_pow2(4759_u16, perm, a),

        641 =>scrambled_radical_inverse_different_pow2(4783_u16, perm, a),

        642 =>scrambled_radical_inverse_different_pow2(4787_u16, perm, a),

        643 =>scrambled_radical_inverse_different_pow2(4789_u16, perm, a),

        644 =>scrambled_radical_inverse_different_pow2(4793_u16, perm, a),

        645 =>scrambled_radical_inverse_different_pow2(4799_u16, perm, a),

        646 =>scrambled_radical_inverse_different_pow2(4801_u16, perm, a),

        647 =>scrambled_radical_inverse_different_pow2(4813_u16, perm, a),

        648 =>scrambled_radical_inverse_different_pow2(4817_u16, perm, a),

        649 =>scrambled_radical_inverse_different_pow2(4831_u16, perm, a),

        650 =>scrambled_radical_inverse_different_pow2(4861_u16, perm, a),

        651 =>scrambled_radical_inverse_different_pow2(4871_u16, perm, a),

        652 =>scrambled_radical_inverse_different_pow2(4877_u16, perm, a),

        653 =>scrambled_radical_inverse_different_pow2(4889_u16, perm, a),

        654 =>scrambled_radical_inverse_different_pow2(4903_u16, perm, a),

        655 =>scrambled_radical_inverse_different_pow2(4909_u16, perm, a),

        656 =>scrambled_radical_inverse_different_pow2(4919_u16, perm, a),

        657 =>scrambled_radical_inverse_different_pow2(4931_u16, perm, a),

        658 =>scrambled_radical_inverse_different_pow2(4933_u16, perm, a),

        659 =>scrambled_radical_inverse_different_pow2(4937_u16, perm, a),

        660 =>scrambled_radical_inverse_different_pow2(4943_u16, perm, a),

        661 =>scrambled_radical_inverse_different_pow2(4951_u16, perm, a),

        662 =>scrambled_radical_inverse_different_pow2(4957_u16, perm, a),

        663 =>scrambled_radical_inverse_different_pow2(4967_u16, perm, a),

        664 =>scrambled_radical_inverse_different_pow2(4969_u16, perm, a),

        665 =>scrambled_radical_inverse_different_pow2(4973_u16, perm, a),

        666 =>scrambled_radical_inverse_different_pow2(4987_u16, perm, a),

        667 =>scrambled_radical_inverse_different_pow2(4993_u16, perm, a),

        668 =>scrambled_radical_inverse_different_pow2(4999_u16, perm, a),

        669 =>scrambled_radical_inverse_different_pow2(5003_u16, perm, a),

        670 =>scrambled_radical_inverse_different_pow2(5009_u16, perm, a),

        671 =>scrambled_radical_inverse_different_pow2(5011_u16, perm, a),

        672 =>scrambled_radical_inverse_different_pow2(5021_u16, perm, a),

        673 =>scrambled_radical_inverse_different_pow2(5023_u16, perm, a),

        674 =>scrambled_radical_inverse_different_pow2(5039_u16, perm, a),

        675 =>scrambled_radical_inverse_different_pow2(5051_u16, perm, a),

        676 =>scrambled_radical_inverse_different_pow2(5059_u16, perm, a),

        677 =>scrambled_radical_inverse_different_pow2(5077_u16, perm, a),

        678 =>scrambled_radical_inverse_different_pow2(5081_u16, perm, a),

        679 =>scrambled_radical_inverse_different_pow2(5087_u16, perm, a),

        680 =>scrambled_radical_inverse_different_pow2(5099_u16, perm, a),

        681 =>scrambled_radical_inverse_different_pow2(5101_u16, perm, a),

        682 =>scrambled_radical_inverse_different_pow2(5107_u16, perm, a),

        683 =>scrambled_radical_inverse_different_pow2(5113_u16, perm, a),

        684 =>scrambled_radical_inverse_different_pow2(5119_u16, perm, a),

        685 =>scrambled_radical_inverse_different_pow2(5147_u16, perm, a),

        686 =>scrambled_radical_inverse_different_pow2(5153_u16, perm, a),

        687 =>scrambled_radical_inverse_different_pow2(5167_u16, perm, a),

        688 =>scrambled_radical_inverse_different_pow2(5171_u16, perm, a),

        689 =>scrambled_radical_inverse_different_pow2(5179_u16, perm, a),

        690 =>scrambled_radical_inverse_different_pow2(5189_u16, perm, a),

        691 =>scrambled_radical_inverse_different_pow2(5197_u16, perm, a),

        692 =>scrambled_radical_inverse_different_pow2(5209_u16, perm, a),

        693 =>scrambled_radical_inverse_different_pow2(5227_u16, perm, a),

        694 =>scrambled_radical_inverse_different_pow2(5231_u16, perm, a),

        695 =>scrambled_radical_inverse_different_pow2(5233_u16, perm, a),

        696 =>scrambled_radical_inverse_different_pow2(5237_u16, perm, a),

        697 =>scrambled_radical_inverse_different_pow2(5261_u16, perm, a),

        698 =>scrambled_radical_inverse_different_pow2(5273_u16, perm, a),

        699 =>scrambled_radical_inverse_different_pow2(5279_u16, perm, a),

        700 =>scrambled_radical_inverse_different_pow2(5281_u16, perm, a),

        701 =>scrambled_radical_inverse_different_pow2(5297_u16, perm, a),

        702 =>scrambled_radical_inverse_different_pow2(5303_u16, perm, a),

        703 =>scrambled_radical_inverse_different_pow2(5309_u16, perm, a),

        704 =>scrambled_radical_inverse_different_pow2(5323_u16, perm, a),

        705 =>scrambled_radical_inverse_different_pow2(5333_u16, perm, a),

        706 =>scrambled_radical_inverse_different_pow2(5347_u16, perm, a),

        707 =>scrambled_radical_inverse_different_pow2(5351_u16, perm, a),

        708 =>scrambled_radical_inverse_different_pow2(5381_u16, perm, a),

        709 =>scrambled_radical_inverse_different_pow2(5387_u16, perm, a),

        710 =>scrambled_radical_inverse_different_pow2(5393_u16, perm, a),

        711 =>scrambled_radical_inverse_different_pow2(5399_u16, perm, a),

        712 =>scrambled_radical_inverse_different_pow2(5407_u16, perm, a),

        713 =>scrambled_radical_inverse_different_pow2(5413_u16, perm, a),

        714 =>scrambled_radical_inverse_different_pow2(5417_u16, perm, a),

        715 =>scrambled_radical_inverse_different_pow2(5419_u16, perm, a),

        716 =>scrambled_radical_inverse_different_pow2(5431_u16, perm, a),

        717 =>scrambled_radical_inverse_different_pow2(5437_u16, perm, a),

        718 =>scrambled_radical_inverse_different_pow2(5441_u16, perm, a),

        719 =>scrambled_radical_inverse_different_pow2(5443_u16, perm, a),

        720 =>scrambled_radical_inverse_different_pow2(5449_u16, perm, a),

        721 =>scrambled_radical_inverse_different_pow2(5471_u16, perm, a),

        722 =>scrambled_radical_inverse_different_pow2(5477_u16, perm, a),

        723 =>scrambled_radical_inverse_different_pow2(5479_u16, perm, a),

        724 =>scrambled_radical_inverse_different_pow2(5483_u16, perm, a),

        725 =>scrambled_radical_inverse_different_pow2(5501_u16, perm, a),

        726 =>scrambled_radical_inverse_different_pow2(5503_u16, perm, a),

        727 =>scrambled_radical_inverse_different_pow2(5507_u16, perm, a),

        728 =>scrambled_radical_inverse_different_pow2(5519_u16, perm, a),

        729 =>scrambled_radical_inverse_different_pow2(5521_u16, perm, a),

        730 =>scrambled_radical_inverse_different_pow2(5527_u16, perm, a),

        731 =>scrambled_radical_inverse_different_pow2(5531_u16, perm, a),

        732 =>scrambled_radical_inverse_different_pow2(5557_u16, perm, a),

        733 =>scrambled_radical_inverse_different_pow2(5563_u16, perm, a),

        734 =>scrambled_radical_inverse_different_pow2(5569_u16, perm, a),

        735 =>scrambled_radical_inverse_different_pow2(5573_u16, perm, a),

        736 =>scrambled_radical_inverse_different_pow2(5581_u16, perm, a),

        737 =>scrambled_radical_inverse_different_pow2(5591_u16, perm, a),

        738 =>scrambled_radical_inverse_different_pow2(5623_u16, perm, a),

        739 =>scrambled_radical_inverse_different_pow2(5639_u16, perm, a),

        740 =>scrambled_radical_inverse_different_pow2(5641_u16, perm, a),

        741 =>scrambled_radical_inverse_different_pow2(5647_u16, perm, a),

        742 =>scrambled_radical_inverse_different_pow2(5651_u16, perm, a),

        743 =>scrambled_radical_inverse_different_pow2(5653_u16, perm, a),

        744 =>scrambled_radical_inverse_different_pow2(5657_u16, perm, a),

        745 =>scrambled_radical_inverse_different_pow2(5659_u16, perm, a),

        746 =>scrambled_radical_inverse_different_pow2(5669_u16, perm, a),

        747 =>scrambled_radical_inverse_different_pow2(5683_u16, perm, a),

        748 =>scrambled_radical_inverse_different_pow2(5689_u16, perm, a),

        749 =>scrambled_radical_inverse_different_pow2(5693_u16, perm, a),

        750 =>scrambled_radical_inverse_different_pow2(5701_u16, perm, a),

        751 =>scrambled_radical_inverse_different_pow2(5711_u16, perm, a),

        752 =>scrambled_radical_inverse_different_pow2(5717_u16, perm, a),

        753 =>scrambled_radical_inverse_different_pow2(5737_u16, perm, a),

        754 =>scrambled_radical_inverse_different_pow2(5741_u16, perm, a),

        755 =>scrambled_radical_inverse_different_pow2(5743_u16, perm, a),

        756 =>scrambled_radical_inverse_different_pow2(5749_u16, perm, a),

        757 =>scrambled_radical_inverse_different_pow2(5779_u16, perm, a),

        758 =>scrambled_radical_inverse_different_pow2(5783_u16, perm, a),

        759 =>scrambled_radical_inverse_different_pow2(5791_u16, perm, a),

        760 =>scrambled_radical_inverse_different_pow2(5801_u16, perm, a),

        761 =>scrambled_radical_inverse_different_pow2(5807_u16, perm, a),

        762 =>scrambled_radical_inverse_different_pow2(5813_u16, perm, a),

        763 =>scrambled_radical_inverse_different_pow2(5821_u16, perm, a),

        764 =>scrambled_radical_inverse_different_pow2(5827_u16, perm, a),

        765 =>scrambled_radical_inverse_different_pow2(5839_u16, perm, a),

        766 =>scrambled_radical_inverse_different_pow2(5843_u16, perm, a),

        767 =>scrambled_radical_inverse_different_pow2(5849_u16, perm, a),

        768 =>scrambled_radical_inverse_different_pow2(5851_u16, perm, a),

        769 =>scrambled_radical_inverse_different_pow2(5857_u16, perm, a),

        770 =>scrambled_radical_inverse_different_pow2(5861_u16, perm, a),

        771 =>scrambled_radical_inverse_different_pow2(5867_u16, perm, a),

        772 =>scrambled_radical_inverse_different_pow2(5869_u16, perm, a),

        773 =>scrambled_radical_inverse_different_pow2(5879_u16, perm, a),

        774 =>scrambled_radical_inverse_different_pow2(5881_u16, perm, a),

        775 =>scrambled_radical_inverse_different_pow2(5897_u16, perm, a),

        776 =>scrambled_radical_inverse_different_pow2(5903_u16, perm, a),

        777 =>scrambled_radical_inverse_different_pow2(5923_u16, perm, a),

        778 =>scrambled_radical_inverse_different_pow2(5927_u16, perm, a),

        779 =>scrambled_radical_inverse_different_pow2(5939_u16, perm, a),

        780 =>scrambled_radical_inverse_different_pow2(5953_u16, perm, a),

        781 =>scrambled_radical_inverse_different_pow2(5981_u16, perm, a),

        782 =>scrambled_radical_inverse_different_pow2(5987_u16, perm, a),

        783 =>scrambled_radical_inverse_different_pow2(6007_u16, perm, a),

        784 =>scrambled_radical_inverse_different_pow2(6011_u16, perm, a),

        785 =>scrambled_radical_inverse_different_pow2(6029_u16, perm, a),

        786 =>scrambled_radical_inverse_different_pow2(6037_u16, perm, a),

        787 =>scrambled_radical_inverse_different_pow2(6043_u16, perm, a),

        788 =>scrambled_radical_inverse_different_pow2(6047_u16, perm, a),

        789 =>scrambled_radical_inverse_different_pow2(6053_u16, perm, a),

        790 =>scrambled_radical_inverse_different_pow2(6067_u16, perm, a),

        791 =>scrambled_radical_inverse_different_pow2(6073_u16, perm, a),

        792 =>scrambled_radical_inverse_different_pow2(6079_u16, perm, a),

        793 =>scrambled_radical_inverse_different_pow2(6089_u16, perm, a),

        794 =>scrambled_radical_inverse_different_pow2(6091_u16, perm, a),

        795 =>scrambled_radical_inverse_different_pow2(6101_u16, perm, a),

        796 =>scrambled_radical_inverse_different_pow2(6113_u16, perm, a),

        797 =>scrambled_radical_inverse_different_pow2(6121_u16, perm, a),

        798 =>scrambled_radical_inverse_different_pow2(6131_u16, perm, a),

        799 =>scrambled_radical_inverse_different_pow2(6133_u16, perm, a),

        800 =>scrambled_radical_inverse_different_pow2(6143_u16, perm, a),

        801 =>scrambled_radical_inverse_different_pow2(6151_u16, perm, a),

        802 =>scrambled_radical_inverse_different_pow2(6163_u16, perm, a),

        803 =>scrambled_radical_inverse_different_pow2(6173_u16, perm, a),

        804 =>scrambled_radical_inverse_different_pow2(6197_u16, perm, a),

        805 =>scrambled_radical_inverse_different_pow2(6199_u16, perm, a),

        806 =>scrambled_radical_inverse_different_pow2(6203_u16, perm, a),

        807 =>scrambled_radical_inverse_different_pow2(6211_u16, perm, a),

        808 =>scrambled_radical_inverse_different_pow2(6217_u16, perm, a),

        809 =>scrambled_radical_inverse_different_pow2(6221_u16, perm, a),

        810 =>scrambled_radical_inverse_different_pow2(6229_u16, perm, a),

        811 =>scrambled_radical_inverse_different_pow2(6247_u16, perm, a),

        812 =>scrambled_radical_inverse_different_pow2(6257_u16, perm, a),

        813 =>scrambled_radical_inverse_different_pow2(6263_u16, perm, a),

        814 =>scrambled_radical_inverse_different_pow2(6269_u16, perm, a),

        815 =>scrambled_radical_inverse_different_pow2(6271_u16, perm, a),

        816 =>scrambled_radical_inverse_different_pow2(6277_u16, perm, a),

        817 =>scrambled_radical_inverse_different_pow2(6287_u16, perm, a),

        818 =>scrambled_radical_inverse_different_pow2(6299_u16, perm, a),

        819 =>scrambled_radical_inverse_different_pow2(6301_u16, perm, a),

        820 =>scrambled_radical_inverse_different_pow2(6311_u16, perm, a),

        821 =>scrambled_radical_inverse_different_pow2(6317_u16, perm, a),

        822 =>scrambled_radical_inverse_different_pow2(6323_u16, perm, a),

        823 =>scrambled_radical_inverse_different_pow2(6329_u16, perm, a),

        824 =>scrambled_radical_inverse_different_pow2(6337_u16, perm, a),

        825 =>scrambled_radical_inverse_different_pow2(6343_u16, perm, a),

        826 =>scrambled_radical_inverse_different_pow2(6353_u16, perm, a),

        827 =>scrambled_radical_inverse_different_pow2(6359_u16, perm, a),

        828 =>scrambled_radical_inverse_different_pow2(6361_u16, perm, a),

        829 =>scrambled_radical_inverse_different_pow2(6367_u16, perm, a),

        830 =>scrambled_radical_inverse_different_pow2(6373_u16, perm, a),

        831 =>scrambled_radical_inverse_different_pow2(6379_u16, perm, a),

        832 =>scrambled_radical_inverse_different_pow2(6389_u16, perm, a),

        833 =>scrambled_radical_inverse_different_pow2(6397_u16, perm, a),

        834 =>scrambled_radical_inverse_different_pow2(6421_u16, perm, a),

        835 =>scrambled_radical_inverse_different_pow2(6427_u16, perm, a),

        836 =>scrambled_radical_inverse_different_pow2(6449_u16, perm, a),

        837 =>scrambled_radical_inverse_different_pow2(6451_u16, perm, a),

        838 =>scrambled_radical_inverse_different_pow2(6469_u16, perm, a),

        839 =>scrambled_radical_inverse_different_pow2(6473_u16, perm, a),

        840 =>scrambled_radical_inverse_different_pow2(6481_u16, perm, a),

        841 =>scrambled_radical_inverse_different_pow2(6491_u16, perm, a),

        842 =>scrambled_radical_inverse_different_pow2(6521_u16, perm, a),

        843 =>scrambled_radical_inverse_different_pow2(6529_u16, perm, a),

        844 =>scrambled_radical_inverse_different_pow2(6547_u16, perm, a),

        845 =>scrambled_radical_inverse_different_pow2(6551_u16, perm, a),

        846 =>scrambled_radical_inverse_different_pow2(6553_u16, perm, a),

        847 =>scrambled_radical_inverse_different_pow2(6563_u16, perm, a),

        848 =>scrambled_radical_inverse_different_pow2(6569_u16, perm, a),

        849 =>scrambled_radical_inverse_different_pow2(6571_u16, perm, a),

        850 =>scrambled_radical_inverse_different_pow2(6577_u16, perm, a),

        851 =>scrambled_radical_inverse_different_pow2(6581_u16, perm, a),

        852 =>scrambled_radical_inverse_different_pow2(6599_u16, perm, a),

        853 =>scrambled_radical_inverse_different_pow2(6607_u16, perm, a),

        854 =>scrambled_radical_inverse_different_pow2(6619_u16, perm, a),

        855 =>scrambled_radical_inverse_different_pow2(6637_u16, perm, a),

        856 =>scrambled_radical_inverse_different_pow2(6653_u16, perm, a),

        857 =>scrambled_radical_inverse_different_pow2(6659_u16, perm, a),

        858 =>scrambled_radical_inverse_different_pow2(6661_u16, perm, a),

        859 =>scrambled_radical_inverse_different_pow2(6673_u16, perm, a),

        860 =>scrambled_radical_inverse_different_pow2(6679_u16, perm, a),

        861 =>scrambled_radical_inverse_different_pow2(6689_u16, perm, a),

        862 =>scrambled_radical_inverse_different_pow2(6691_u16, perm, a),

        863 =>scrambled_radical_inverse_different_pow2(6701_u16, perm, a),

        864 =>scrambled_radical_inverse_different_pow2(6703_u16, perm, a),

        865 =>scrambled_radical_inverse_different_pow2(6709_u16, perm, a),

        866 =>scrambled_radical_inverse_different_pow2(6719_u16, perm, a),

        867 =>scrambled_radical_inverse_different_pow2(6733_u16, perm, a),

        868 =>scrambled_radical_inverse_different_pow2(6737_u16, perm, a),

        869 =>scrambled_radical_inverse_different_pow2(6761_u16, perm, a),

        870 =>scrambled_radical_inverse_different_pow2(6763_u16, perm, a),

        871 =>scrambled_radical_inverse_different_pow2(6779_u16, perm, a),

        872 =>scrambled_radical_inverse_different_pow2(6781_u16, perm, a),

        873 =>scrambled_radical_inverse_different_pow2(6791_u16, perm, a),

        874 =>scrambled_radical_inverse_different_pow2(6793_u16, perm, a),

        875 =>scrambled_radical_inverse_different_pow2(6803_u16, perm, a),

        876 =>scrambled_radical_inverse_different_pow2(6823_u16, perm, a),

        877 =>scrambled_radical_inverse_different_pow2(6827_u16, perm, a),

        878 =>scrambled_radical_inverse_different_pow2(6829_u16, perm, a),

        879 =>scrambled_radical_inverse_different_pow2(6833_u16, perm, a),

        880 =>scrambled_radical_inverse_different_pow2(6841_u16, perm, a),

        881 =>scrambled_radical_inverse_different_pow2(6857_u16, perm, a),

        882 =>scrambled_radical_inverse_different_pow2(6863_u16, perm, a),

        883 =>scrambled_radical_inverse_different_pow2(6869_u16, perm, a),

        884 =>scrambled_radical_inverse_different_pow2(6871_u16, perm, a),

        885 =>scrambled_radical_inverse_different_pow2(6883_u16, perm, a),

        886 =>scrambled_radical_inverse_different_pow2(6899_u16, perm, a),

        887 =>scrambled_radical_inverse_different_pow2(6907_u16, perm, a),

        888 =>scrambled_radical_inverse_different_pow2(6911_u16, perm, a),

        889 =>scrambled_radical_inverse_different_pow2(6917_u16, perm, a),

        890 =>scrambled_radical_inverse_different_pow2(6947_u16, perm, a),

        891 =>scrambled_radical_inverse_different_pow2(6949_u16, perm, a),

        892 =>scrambled_radical_inverse_different_pow2(6959_u16, perm, a),

        893 =>scrambled_radical_inverse_different_pow2(6961_u16, perm, a),

        894 =>scrambled_radical_inverse_different_pow2(6967_u16, perm, a),

        895 =>scrambled_radical_inverse_different_pow2(6971_u16, perm, a),

        896 =>scrambled_radical_inverse_different_pow2(6977_u16, perm, a),

        897 =>scrambled_radical_inverse_different_pow2(6983_u16, perm, a),

        898 =>scrambled_radical_inverse_different_pow2(6991_u16, perm, a),

        899 =>scrambled_radical_inverse_different_pow2(6997_u16, perm, a),

        900 =>scrambled_radical_inverse_different_pow2(7001_u16, perm, a),

        901 =>scrambled_radical_inverse_different_pow2(7013_u16, perm, a),

        902 =>scrambled_radical_inverse_different_pow2(7019_u16, perm, a),

        903 =>scrambled_radical_inverse_different_pow2(7027_u16, perm, a),

        904 =>scrambled_radical_inverse_different_pow2(7039_u16, perm, a),

        905 =>scrambled_radical_inverse_different_pow2(7043_u16, perm, a),

        906 =>scrambled_radical_inverse_different_pow2(7057_u16, perm, a),

        907 =>scrambled_radical_inverse_different_pow2(7069_u16, perm, a),

        908 =>scrambled_radical_inverse_different_pow2(7079_u16, perm, a),

        909 =>scrambled_radical_inverse_different_pow2(7103_u16, perm, a),

        910 =>scrambled_radical_inverse_different_pow2(7109_u16, perm, a),

        911 =>scrambled_radical_inverse_different_pow2(7121_u16, perm, a),

        912 =>scrambled_radical_inverse_different_pow2(7127_u16, perm, a),

        913 =>scrambled_radical_inverse_different_pow2(7129_u16, perm, a),

        914 =>scrambled_radical_inverse_different_pow2(7151_u16, perm, a),

        915 =>scrambled_radical_inverse_different_pow2(7159_u16, perm, a),

        916 =>scrambled_radical_inverse_different_pow2(7177_u16, perm, a),

        917 =>scrambled_radical_inverse_different_pow2(7187_u16, perm, a),

        918 =>scrambled_radical_inverse_different_pow2(7193_u16, perm, a),

        919 =>scrambled_radical_inverse_different_pow2(7207_u16, perm, a),

        920 =>scrambled_radical_inverse_different_pow2(7211_u16, perm, a),

        921 =>scrambled_radical_inverse_different_pow2(7213_u16, perm, a),

        922 =>scrambled_radical_inverse_different_pow2(7219_u16, perm, a),

        923 =>scrambled_radical_inverse_different_pow2(7229_u16, perm, a),

        924 =>scrambled_radical_inverse_different_pow2(7237_u16, perm, a),

        925 =>scrambled_radical_inverse_different_pow2(7243_u16, perm, a),

        926 =>scrambled_radical_inverse_different_pow2(7247_u16, perm, a),

        927 =>scrambled_radical_inverse_different_pow2(7253_u16, perm, a),

        928 =>scrambled_radical_inverse_different_pow2(7283_u16, perm, a),

        929 =>scrambled_radical_inverse_different_pow2(7297_u16, perm, a),

        930 =>scrambled_radical_inverse_different_pow2(7307_u16, perm, a),

        931 =>scrambled_radical_inverse_different_pow2(7309_u16, perm, a),

        932 =>scrambled_radical_inverse_different_pow2(7321_u16, perm, a),

        933 =>scrambled_radical_inverse_different_pow2(7331_u16, perm, a),

        934 =>scrambled_radical_inverse_different_pow2(7333_u16, perm, a),

        935 =>scrambled_radical_inverse_different_pow2(7349_u16, perm, a),

        936 =>scrambled_radical_inverse_different_pow2(7351_u16, perm, a),

        937 =>scrambled_radical_inverse_different_pow2(7369_u16, perm, a),

        938 =>scrambled_radical_inverse_different_pow2(7393_u16, perm, a),

        939 =>scrambled_radical_inverse_different_pow2(7411_u16, perm, a),

        940 =>scrambled_radical_inverse_different_pow2(7417_u16, perm, a),

        941 =>scrambled_radical_inverse_different_pow2(7433_u16, perm, a),

        942 =>scrambled_radical_inverse_different_pow2(7451_u16, perm, a),

        943 =>scrambled_radical_inverse_different_pow2(7457_u16, perm, a),

        944 =>scrambled_radical_inverse_different_pow2(7459_u16, perm, a),

        945 =>scrambled_radical_inverse_different_pow2(7477_u16, perm, a),

        946 =>scrambled_radical_inverse_different_pow2(7481_u16, perm, a),

        947 =>scrambled_radical_inverse_different_pow2(7487_u16, perm, a),

        948 =>scrambled_radical_inverse_different_pow2(7489_u16, perm, a),

        949 =>scrambled_radical_inverse_different_pow2(7499_u16, perm, a),

        950 =>scrambled_radical_inverse_different_pow2(7507_u16, perm, a),

        951 =>scrambled_radical_inverse_different_pow2(7517_u16, perm, a),

        952 =>scrambled_radical_inverse_different_pow2(7523_u16, perm, a),

        953 =>scrambled_radical_inverse_different_pow2(7529_u16, perm, a),

        954 =>scrambled_radical_inverse_different_pow2(7537_u16, perm, a),

        955 =>scrambled_radical_inverse_different_pow2(7541_u16, perm, a),

        956 =>scrambled_radical_inverse_different_pow2(7547_u16, perm, a),

        957 =>scrambled_radical_inverse_different_pow2(7549_u16, perm, a),

        958 =>scrambled_radical_inverse_different_pow2(7559_u16, perm, a),

        959 =>scrambled_radical_inverse_different_pow2(7561_u16, perm, a),

        960 =>scrambled_radical_inverse_different_pow2(7573_u16, perm, a),

        961 =>scrambled_radical_inverse_different_pow2(7577_u16, perm, a),

        962 =>scrambled_radical_inverse_different_pow2(7583_u16, perm, a),

        963 =>scrambled_radical_inverse_different_pow2(7589_u16, perm, a),

        964 =>scrambled_radical_inverse_different_pow2(7591_u16, perm, a),

        965 =>scrambled_radical_inverse_different_pow2(7603_u16, perm, a),

        966 =>scrambled_radical_inverse_different_pow2(7607_u16, perm, a),

        967 =>scrambled_radical_inverse_different_pow2(7621_u16, perm, a),

        968 =>scrambled_radical_inverse_different_pow2(7639_u16, perm, a),

        969 =>scrambled_radical_inverse_different_pow2(7643_u16, perm, a),

        970 =>scrambled_radical_inverse_different_pow2(7649_u16, perm, a),

        971 =>scrambled_radical_inverse_different_pow2(7669_u16, perm, a),

        972 =>scrambled_radical_inverse_different_pow2(7673_u16, perm, a),

        973 =>scrambled_radical_inverse_different_pow2(7681_u16, perm, a),

        974 =>scrambled_radical_inverse_different_pow2(7687_u16, perm, a),

        975 =>scrambled_radical_inverse_different_pow2(7691_u16, perm, a),

        976 =>scrambled_radical_inverse_different_pow2(7699_u16, perm, a),

        977 =>scrambled_radical_inverse_different_pow2(7703_u16, perm, a),

        978 =>scrambled_radical_inverse_different_pow2(7717_u16, perm, a),

        979 =>scrambled_radical_inverse_different_pow2(7723_u16, perm, a),

        980 =>scrambled_radical_inverse_different_pow2(7727_u16, perm, a),

        981 =>scrambled_radical_inverse_different_pow2(7741_u16, perm, a),

        982 =>scrambled_radical_inverse_different_pow2(7753_u16, perm, a),

        983 =>scrambled_radical_inverse_different_pow2(7757_u16, perm, a),

        984 =>scrambled_radical_inverse_different_pow2(7759_u16, perm, a),

        985 =>scrambled_radical_inverse_different_pow2(7789_u16, perm, a),

        986 =>scrambled_radical_inverse_different_pow2(7793_u16, perm, a),

        987 =>scrambled_radical_inverse_different_pow2(7817_u16, perm, a),

        988 =>scrambled_radical_inverse_different_pow2(7823_u16, perm, a),

        989 =>scrambled_radical_inverse_different_pow2(7829_u16, perm, a),

        990 =>scrambled_radical_inverse_different_pow2(7841_u16, perm, a),

        991 =>scrambled_radical_inverse_different_pow2(7853_u16, perm, a),

        992 =>scrambled_radical_inverse_different_pow2(7867_u16, perm, a),

        993 =>scrambled_radical_inverse_different_pow2(7873_u16, perm, a),

        994 =>scrambled_radical_inverse_different_pow2(7877_u16, perm, a),

        995 =>scrambled_radical_inverse_different_pow2(7879_u16, perm, a),

        996 =>scrambled_radical_inverse_different_pow2(7883_u16, perm, a),

        997 =>scrambled_radical_inverse_different_pow2(7901_u16, perm, a),

        998 =>scrambled_radical_inverse_different_pow2(7907_u16, perm, a),

        999 =>scrambled_radical_inverse_different_pow2(7919_u16, perm, a),

        1000 =>scrambled_radical_inverse_different_pow2(7927_u16, perm, a),

        1001 =>scrambled_radical_inverse_different_pow2(7933_u16, perm, a),

        1002 =>scrambled_radical_inverse_different_pow2(7937_u16, perm, a),

        1003 =>scrambled_radical_inverse_different_pow2(7949_u16, perm, a),

        1004 =>scrambled_radical_inverse_different_pow2(7951_u16, perm, a),

        1005 =>scrambled_radical_inverse_different_pow2(7963_u16, perm, a),

        1006 =>scrambled_radical_inverse_different_pow2(7993_u16, perm, a),

        1007 =>scrambled_radical_inverse_different_pow2(8009_u16, perm, a),

        1008 =>scrambled_radical_inverse_different_pow2(8011_u16, perm, a),

        1009 =>scrambled_radical_inverse_different_pow2(8017_u16, perm, a),

        1010 =>scrambled_radical_inverse_different_pow2(8039_u16, perm, a),

        1011 =>scrambled_radical_inverse_different_pow2(8053_u16, perm, a),

        1012 =>scrambled_radical_inverse_different_pow2(8059_u16, perm, a),

        1013 =>scrambled_radical_inverse_different_pow2(8069_u16, perm, a),

        1014 =>scrambled_radical_inverse_different_pow2(8081_u16, perm, a),

        1015 =>scrambled_radical_inverse_different_pow2(8087_u16, perm, a),

        1016 =>scrambled_radical_inverse_different_pow2(8089_u16, perm, a),

        1017 =>scrambled_radical_inverse_different_pow2(8093_u16, perm, a),

        1018 =>scrambled_radical_inverse_different_pow2(8101_u16, perm, a),

        1019 =>scrambled_radical_inverse_different_pow2(8111_u16, perm, a),

        1020 =>scrambled_radical_inverse_different_pow2(8117_u16, perm, a),

        1021 =>scrambled_radical_inverse_different_pow2(8123_u16, perm, a),

        1022 =>scrambled_radical_inverse_different_pow2(8147_u16, perm, a),

        1023 =>scrambled_radical_inverse_different_pow2(8161_u16, perm, a),
        _ => {
            panic!("TODO: scrambled_radical_inverse({:?}, {:?})", base_index, a);
        }
    }

}





pub fn radical_inverse(base : u16, a : u64)-> f64{

    match base  {
        0=>radical_inverse_pow2(base, a),
        1 => radical_inverse_different_pow2(3_u16, a),
        2 => radical_inverse_different_pow2(5_u16, a),
        3 => radical_inverse_different_pow2(7_u16, a),
        4 => radical_inverse_different_pow2(11_u16, a),
        5 => radical_inverse_different_pow2(13_u16, a),
        6 => radical_inverse_different_pow2(17_u16, a),
        7 => radical_inverse_different_pow2(19_u16, a),
        8 => radical_inverse_different_pow2(23_u16, a),
        9 => radical_inverse_different_pow2(29_u16, a),
        10 => radical_inverse_different_pow2(31_u16, a),
        11 => radical_inverse_different_pow2(37_u16, a),
        12 => radical_inverse_different_pow2(41_u16, a),
        13 => radical_inverse_different_pow2(43_u16, a),
        14 => radical_inverse_different_pow2(47_u16, a),
        15 => radical_inverse_different_pow2(53_u16, a),
        16 => radical_inverse_different_pow2(59_u16, a),
        17 => radical_inverse_different_pow2(61_u16, a),
        18 => radical_inverse_different_pow2(67_u16, a),
        19 => radical_inverse_different_pow2(71_u16, a),
        20 => radical_inverse_different_pow2(73_u16, a),
        21 => radical_inverse_different_pow2(79_u16, a),
        22 => radical_inverse_different_pow2(83_u16, a),
        23 => radical_inverse_different_pow2(89_u16, a),
        24 => radical_inverse_different_pow2(97_u16, a),
        25 => radical_inverse_different_pow2(101_u16, a),
        26 => radical_inverse_different_pow2(103_u16, a),
        27 => radical_inverse_different_pow2(107_u16, a),
        28 => radical_inverse_different_pow2(109_u16, a),
        29 => radical_inverse_different_pow2(113_u16, a),
        30 => radical_inverse_different_pow2(127_u16, a),
        31 => radical_inverse_different_pow2(131_u16, a),
        32 => radical_inverse_different_pow2(137_u16, a),
        33 => radical_inverse_different_pow2(139_u16, a),
        34 => radical_inverse_different_pow2(149_u16, a),
        35 => radical_inverse_different_pow2(151_u16, a),
        36 => radical_inverse_different_pow2(157_u16, a),
        37 => radical_inverse_different_pow2(163_u16, a),
        38 => radical_inverse_different_pow2(167_u16, a),
        39 => radical_inverse_different_pow2(173_u16, a),
        40 => radical_inverse_different_pow2(179_u16, a),
        41 => radical_inverse_different_pow2(181_u16, a),
        42 => radical_inverse_different_pow2(191_u16, a),
        43 => radical_inverse_different_pow2(193_u16, a),
        44 => radical_inverse_different_pow2(197_u16, a),
        45 => radical_inverse_different_pow2(199_u16, a),
        46 => radical_inverse_different_pow2(211_u16, a),
        47 => radical_inverse_different_pow2(223_u16, a),
        48 => radical_inverse_different_pow2(227_u16, a),
        49 => radical_inverse_different_pow2(229_u16, a),
        50 => radical_inverse_different_pow2(233_u16, a),
        51 => radical_inverse_different_pow2(239_u16, a),
        52 => radical_inverse_different_pow2(241_u16, a),
        53 => radical_inverse_different_pow2(251_u16, a),
        54 => radical_inverse_different_pow2(257_u16, a),
        55 => radical_inverse_different_pow2(263_u16, a),
        56 => radical_inverse_different_pow2(269_u16, a),
        57 => radical_inverse_different_pow2(271_u16, a),
        58 => radical_inverse_different_pow2(277_u16, a),
        59 => radical_inverse_different_pow2(281_u16, a),
        60 => radical_inverse_different_pow2(283_u16, a),
        61 => radical_inverse_different_pow2(293_u16, a),
        62 => radical_inverse_different_pow2(307_u16, a),
        63 => radical_inverse_different_pow2(311_u16, a),
        64 => radical_inverse_different_pow2(313_u16, a),
        65 => radical_inverse_different_pow2(317_u16, a),
        66 => radical_inverse_different_pow2(331_u16, a),
        67 => radical_inverse_different_pow2(337_u16, a),
        68 => radical_inverse_different_pow2(347_u16, a),
        69 => radical_inverse_different_pow2(349_u16, a),
        70 => radical_inverse_different_pow2(353_u16, a),
        71 => radical_inverse_different_pow2(359_u16, a),
        72 => radical_inverse_different_pow2(367_u16, a),
        73 => radical_inverse_different_pow2(373_u16, a),
        74 => radical_inverse_different_pow2(379_u16, a),
        75 => radical_inverse_different_pow2(383_u16, a),
        76 => radical_inverse_different_pow2(389_u16, a),
        77 => radical_inverse_different_pow2(397_u16, a),
        78 => radical_inverse_different_pow2(401_u16, a),
        79 => radical_inverse_different_pow2(409_u16, a),
        80 => radical_inverse_different_pow2(419_u16, a),
        81 => radical_inverse_different_pow2(421_u16, a),
        82 => radical_inverse_different_pow2(431_u16, a),
        83 => radical_inverse_different_pow2(433_u16, a),
        84 => radical_inverse_different_pow2(439_u16, a),
        85 => radical_inverse_different_pow2(443_u16, a),
        86 => radical_inverse_different_pow2(449_u16, a),
        87 => radical_inverse_different_pow2(457_u16, a),
        88 => radical_inverse_different_pow2(461_u16, a),
        89 => radical_inverse_different_pow2(463_u16, a),
        90 => radical_inverse_different_pow2(467_u16, a),
        91 => radical_inverse_different_pow2(479_u16, a),
        92 => radical_inverse_different_pow2(487_u16, a),
        93 => radical_inverse_different_pow2(491_u16, a),
        94 => radical_inverse_different_pow2(499_u16, a),
        95 => radical_inverse_different_pow2(503_u16, a),
        96 => radical_inverse_different_pow2(509_u16, a),
        97 => radical_inverse_different_pow2(521_u16, a),
        98 => radical_inverse_different_pow2(523_u16, a),
        99 => radical_inverse_different_pow2(541_u16, a),
        100 => radical_inverse_different_pow2(547_u16, a),
        101 => radical_inverse_different_pow2(557_u16, a),
        102 => radical_inverse_different_pow2(563_u16, a),
        103 => radical_inverse_different_pow2(569_u16, a),
        104 => radical_inverse_different_pow2(571_u16, a),
        105 => radical_inverse_different_pow2(577_u16, a),
        106 => radical_inverse_different_pow2(587_u16, a),
        107 => radical_inverse_different_pow2(593_u16, a),
        108 => radical_inverse_different_pow2(599_u16, a),
        109 => radical_inverse_different_pow2(601_u16, a),
        110 => radical_inverse_different_pow2(607_u16, a),
        111 => radical_inverse_different_pow2(613_u16, a),
        112 => radical_inverse_different_pow2(617_u16, a),
        113 => radical_inverse_different_pow2(619_u16, a),
        114 => radical_inverse_different_pow2(631_u16, a),
        115 => radical_inverse_different_pow2(641_u16, a),
        116 => radical_inverse_different_pow2(643_u16, a),
        117 => radical_inverse_different_pow2(647_u16, a),
        118 => radical_inverse_different_pow2(653_u16, a),
        119 => radical_inverse_different_pow2(659_u16, a),
        120 => radical_inverse_different_pow2(661_u16, a),
        121 => radical_inverse_different_pow2(673_u16, a),
        122 => radical_inverse_different_pow2(677_u16, a),
        123 => radical_inverse_different_pow2(683_u16, a),
        124 => radical_inverse_different_pow2(691_u16, a),
        125 => radical_inverse_different_pow2(701_u16, a),
        126 => radical_inverse_different_pow2(709_u16, a),
        127 => radical_inverse_different_pow2(719_u16, a),
        128 => radical_inverse_different_pow2(727_u16, a),
        129 => radical_inverse_different_pow2(733_u16, a),
        130 => radical_inverse_different_pow2(739_u16, a),
        131 => radical_inverse_different_pow2(743_u16, a),
        132 => radical_inverse_different_pow2(751_u16, a),
        133 => radical_inverse_different_pow2(757_u16, a),
        134 => radical_inverse_different_pow2(761_u16, a),
        135 => radical_inverse_different_pow2(769_u16, a),
        136 => radical_inverse_different_pow2(773_u16, a),
        137 => radical_inverse_different_pow2(787_u16, a),
        138 => radical_inverse_different_pow2(797_u16, a),
        139 => radical_inverse_different_pow2(809_u16, a),
        140 => radical_inverse_different_pow2(811_u16, a),
        141 => radical_inverse_different_pow2(821_u16, a),
        142 => radical_inverse_different_pow2(823_u16, a),
        143 => radical_inverse_different_pow2(827_u16, a),
        144 => radical_inverse_different_pow2(829_u16, a),
        145 => radical_inverse_different_pow2(839_u16, a),
        146 => radical_inverse_different_pow2(853_u16, a),
        147 => radical_inverse_different_pow2(857_u16, a),
        148 => radical_inverse_different_pow2(859_u16, a),
        149 => radical_inverse_different_pow2(863_u16, a),
        150 => radical_inverse_different_pow2(877_u16, a),
        151 => radical_inverse_different_pow2(881_u16, a),
        152 => radical_inverse_different_pow2(883_u16, a),
        153 => radical_inverse_different_pow2(887_u16, a),
        154 => radical_inverse_different_pow2(907_u16, a),
        155 => radical_inverse_different_pow2(911_u16, a),
        156 => radical_inverse_different_pow2(919_u16, a),
        157 => radical_inverse_different_pow2(929_u16, a),
        158 => radical_inverse_different_pow2(937_u16, a),
        159 => radical_inverse_different_pow2(941_u16, a),
        160 => radical_inverse_different_pow2(947_u16, a),
        161 => radical_inverse_different_pow2(953_u16, a),
        162 => radical_inverse_different_pow2(967_u16, a),
        163 => radical_inverse_different_pow2(971_u16, a),
        164 => radical_inverse_different_pow2(977_u16, a),
        165 => radical_inverse_different_pow2(983_u16, a),
        166 => radical_inverse_different_pow2(991_u16, a),
        167 => radical_inverse_different_pow2(997_u16, a),
        168 => radical_inverse_different_pow2(1009_u16, a),
        169 => radical_inverse_different_pow2(1013_u16, a),
        170 => radical_inverse_different_pow2(1019_u16, a),
        171 => radical_inverse_different_pow2(1021_u16, a),
        172 => radical_inverse_different_pow2(1031_u16, a),
        173 => radical_inverse_different_pow2(1033_u16, a),
        174 => radical_inverse_different_pow2(1039_u16, a),
        175 => radical_inverse_different_pow2(1049_u16, a),
        176 => radical_inverse_different_pow2(1051_u16, a),
        177 => radical_inverse_different_pow2(1061_u16, a),
        178 => radical_inverse_different_pow2(1063_u16, a),
        179 => radical_inverse_different_pow2(1069_u16, a),
        180 => radical_inverse_different_pow2(1087_u16, a),
        181 => radical_inverse_different_pow2(1091_u16, a),
        182 => radical_inverse_different_pow2(1093_u16, a),
        183 => radical_inverse_different_pow2(1097_u16, a),
        184 => radical_inverse_different_pow2(1103_u16, a),
        185 => radical_inverse_different_pow2(1109_u16, a),
        186 => radical_inverse_different_pow2(1117_u16, a),
        187 => radical_inverse_different_pow2(1123_u16, a),
        188 => radical_inverse_different_pow2(1129_u16, a),
        189 => radical_inverse_different_pow2(1151_u16, a),
        190 => radical_inverse_different_pow2(1153_u16, a),
        191 => radical_inverse_different_pow2(1163_u16, a),
        192 => radical_inverse_different_pow2(1171_u16, a),
        193 => radical_inverse_different_pow2(1181_u16, a),
        194 => radical_inverse_different_pow2(1187_u16, a),
        195 => radical_inverse_different_pow2(1193_u16, a),
        196 => radical_inverse_different_pow2(1201_u16, a),
        197 => radical_inverse_different_pow2(1213_u16, a),
        198 => radical_inverse_different_pow2(1217_u16, a),
        199 => radical_inverse_different_pow2(1223_u16, a),
        200 => radical_inverse_different_pow2(1229_u16, a),
        201 => radical_inverse_different_pow2(1231_u16, a),
        202 => radical_inverse_different_pow2(1237_u16, a),
        203 => radical_inverse_different_pow2(1249_u16, a),
        204 => radical_inverse_different_pow2(1259_u16, a),
        205 => radical_inverse_different_pow2(1277_u16, a),
        206 => radical_inverse_different_pow2(1279_u16, a),
        207 => radical_inverse_different_pow2(1283_u16, a),
        208 => radical_inverse_different_pow2(1289_u16, a),
        209 => radical_inverse_different_pow2(1291_u16, a),
        210 => radical_inverse_different_pow2(1297_u16, a),
        211 => radical_inverse_different_pow2(1301_u16, a),
        212 => radical_inverse_different_pow2(1303_u16, a),
        213 => radical_inverse_different_pow2(1307_u16, a),
        214 => radical_inverse_different_pow2(1319_u16, a),
        215 => radical_inverse_different_pow2(1321_u16, a),
        216 => radical_inverse_different_pow2(1327_u16, a),
        217 => radical_inverse_different_pow2(1361_u16, a),
        218 => radical_inverse_different_pow2(1367_u16, a),
        219 => radical_inverse_different_pow2(1373_u16, a),
        220 => radical_inverse_different_pow2(1381_u16, a),
        221 => radical_inverse_different_pow2(1399_u16, a),
        222 => radical_inverse_different_pow2(1409_u16, a),
        223 => radical_inverse_different_pow2(1423_u16, a),
        224 => radical_inverse_different_pow2(1427_u16, a),
        225 => radical_inverse_different_pow2(1429_u16, a),
        226 => radical_inverse_different_pow2(1433_u16, a),
        227 => radical_inverse_different_pow2(1439_u16, a),
        228 => radical_inverse_different_pow2(1447_u16, a),
        229 => radical_inverse_different_pow2(1451_u16, a),
        230 => radical_inverse_different_pow2(1453_u16, a),
        231 => radical_inverse_different_pow2(1459_u16, a),
        232 => radical_inverse_different_pow2(1471_u16, a),
        233 => radical_inverse_different_pow2(1481_u16, a),
        234 => radical_inverse_different_pow2(1483_u16, a),
        235 => radical_inverse_different_pow2(1487_u16, a),
        236 => radical_inverse_different_pow2(1489_u16, a),
        237 => radical_inverse_different_pow2(1493_u16, a),
        238 => radical_inverse_different_pow2(1499_u16, a),
        239 => radical_inverse_different_pow2(1511_u16, a),
        240 => radical_inverse_different_pow2(1523_u16, a),
        241 => radical_inverse_different_pow2(1531_u16, a),
        242 => radical_inverse_different_pow2(1543_u16, a),
        243 => radical_inverse_different_pow2(1549_u16, a),
        244 => radical_inverse_different_pow2(1553_u16, a),
        245 => radical_inverse_different_pow2(1559_u16, a),
        246 => radical_inverse_different_pow2(1567_u16, a),
        247 => radical_inverse_different_pow2(1571_u16, a),
        248 => radical_inverse_different_pow2(1579_u16, a),
        249 => radical_inverse_different_pow2(1583_u16, a),
        250 => radical_inverse_different_pow2(1597_u16, a),
        251 => radical_inverse_different_pow2(1601_u16, a),
        252 => radical_inverse_different_pow2(1607_u16, a),
        253 => radical_inverse_different_pow2(1609_u16, a),
        254 => radical_inverse_different_pow2(1613_u16, a),
        255 => radical_inverse_different_pow2(1619_u16, a),
        256 => radical_inverse_different_pow2(1621_u16, a),
        257 => radical_inverse_different_pow2(1627_u16, a),
        258 => radical_inverse_different_pow2(1637_u16, a),
        259 => radical_inverse_different_pow2(1657_u16, a),
        260 => radical_inverse_different_pow2(1663_u16, a),
        261 => radical_inverse_different_pow2(1667_u16, a),
        262 => radical_inverse_different_pow2(1669_u16, a),
        263 => radical_inverse_different_pow2(1693_u16, a),
        264 => radical_inverse_different_pow2(1697_u16, a),
        265 => radical_inverse_different_pow2(1699_u16, a),
        266 => radical_inverse_different_pow2(1709_u16, a),
        267 => radical_inverse_different_pow2(1721_u16, a),
        268 => radical_inverse_different_pow2(1723_u16, a),
        269 => radical_inverse_different_pow2(1733_u16, a),
        270 => radical_inverse_different_pow2(1741_u16, a),
        271 => radical_inverse_different_pow2(1747_u16, a),
        272 => radical_inverse_different_pow2(1753_u16, a),
        273 => radical_inverse_different_pow2(1759_u16, a),
        274 => radical_inverse_different_pow2(1777_u16, a),
        275 => radical_inverse_different_pow2(1783_u16, a),
        276 => radical_inverse_different_pow2(1787_u16, a),
        277 => radical_inverse_different_pow2(1789_u16, a),
        278 => radical_inverse_different_pow2(1801_u16, a),
        279 => radical_inverse_different_pow2(1811_u16, a),
        280 => radical_inverse_different_pow2(1823_u16, a),
        281 => radical_inverse_different_pow2(1831_u16, a),
        282 => radical_inverse_different_pow2(1847_u16, a),
        283 => radical_inverse_different_pow2(1861_u16, a),
        284 => radical_inverse_different_pow2(1867_u16, a),
        285 => radical_inverse_different_pow2(1871_u16, a),
        286 => radical_inverse_different_pow2(1873_u16, a),
        287 => radical_inverse_different_pow2(1877_u16, a),
        288 => radical_inverse_different_pow2(1879_u16, a),
        289 => radical_inverse_different_pow2(1889_u16, a),
        290 => radical_inverse_different_pow2(1901_u16, a),
        291 => radical_inverse_different_pow2(1907_u16, a),
        292 => radical_inverse_different_pow2(1913_u16, a),
        293 => radical_inverse_different_pow2(1931_u16, a),
        294 => radical_inverse_different_pow2(1933_u16, a),
        295 => radical_inverse_different_pow2(1949_u16, a),
        296 => radical_inverse_different_pow2(1951_u16, a),
        297 => radical_inverse_different_pow2(1973_u16, a),
        298 => radical_inverse_different_pow2(1979_u16, a),
        299 => radical_inverse_different_pow2(1987_u16, a),
        300 => radical_inverse_different_pow2(1993_u16, a),
        301 => radical_inverse_different_pow2(1997_u16, a),
        302 => radical_inverse_different_pow2(1999_u16, a),
        303 => radical_inverse_different_pow2(2003_u16, a),
        304 => radical_inverse_different_pow2(2011_u16, a),
        305 => radical_inverse_different_pow2(2017_u16, a),
        306 => radical_inverse_different_pow2(2027_u16, a),
        307 => radical_inverse_different_pow2(2029_u16, a),
        308 => radical_inverse_different_pow2(2039_u16, a),
        309 => radical_inverse_different_pow2(2053_u16, a),
        310 => radical_inverse_different_pow2(2063_u16, a),
        311 => radical_inverse_different_pow2(2069_u16, a),
        312 => radical_inverse_different_pow2(2081_u16, a),
        313 => radical_inverse_different_pow2(2083_u16, a),
        314 => radical_inverse_different_pow2(2087_u16, a),
        315 => radical_inverse_different_pow2(2089_u16, a),
        316 => radical_inverse_different_pow2(2099_u16, a),
        317 => radical_inverse_different_pow2(2111_u16, a),
        318 => radical_inverse_different_pow2(2113_u16, a),
        319 => radical_inverse_different_pow2(2129_u16, a),
        320 => radical_inverse_different_pow2(2131_u16, a),
        321 => radical_inverse_different_pow2(2137_u16, a),
        322 => radical_inverse_different_pow2(2141_u16, a),
        323 => radical_inverse_different_pow2(2143_u16, a),
        324 => radical_inverse_different_pow2(2153_u16, a),
        325 => radical_inverse_different_pow2(2161_u16, a),
        326 => radical_inverse_different_pow2(2179_u16, a),
        327 => radical_inverse_different_pow2(2203_u16, a),
        328 => radical_inverse_different_pow2(2207_u16, a),
        329 => radical_inverse_different_pow2(2213_u16, a),
        330 => radical_inverse_different_pow2(2221_u16, a),
        331 => radical_inverse_different_pow2(2237_u16, a),
        332 => radical_inverse_different_pow2(2239_u16, a),
        333 => radical_inverse_different_pow2(2243_u16, a),
        334 => radical_inverse_different_pow2(2251_u16, a),
        335 => radical_inverse_different_pow2(2267_u16, a),
        336 => radical_inverse_different_pow2(2269_u16, a),
        337 => radical_inverse_different_pow2(2273_u16, a),
        338 => radical_inverse_different_pow2(2281_u16, a),
        339 => radical_inverse_different_pow2(2287_u16, a),
        340 => radical_inverse_different_pow2(2293_u16, a),
        341 => radical_inverse_different_pow2(2297_u16, a),
        342 => radical_inverse_different_pow2(2309_u16, a),
        343 => radical_inverse_different_pow2(2311_u16, a),
        344 => radical_inverse_different_pow2(2333_u16, a),
        345 => radical_inverse_different_pow2(2339_u16, a),
        346 => radical_inverse_different_pow2(2341_u16, a),
        347 => radical_inverse_different_pow2(2347_u16, a),
        348 => radical_inverse_different_pow2(2351_u16, a),
        349 => radical_inverse_different_pow2(2357_u16, a),
        350 => radical_inverse_different_pow2(2371_u16, a),
        351 => radical_inverse_different_pow2(2377_u16, a),
        352 => radical_inverse_different_pow2(2381_u16, a),
        353 => radical_inverse_different_pow2(2383_u16, a),
        354 => radical_inverse_different_pow2(2389_u16, a),
        355 => radical_inverse_different_pow2(2393_u16, a),
        356 => radical_inverse_different_pow2(2399_u16, a),
        357 => radical_inverse_different_pow2(2411_u16, a),
        358 => radical_inverse_different_pow2(2417_u16, a),
        359 => radical_inverse_different_pow2(2423_u16, a),
        360 => radical_inverse_different_pow2(2437_u16, a),
        361 => radical_inverse_different_pow2(2441_u16, a),
        362 => radical_inverse_different_pow2(2447_u16, a),
        363 => radical_inverse_different_pow2(2459_u16, a),
        364 => radical_inverse_different_pow2(2467_u16, a),
        365 => radical_inverse_different_pow2(2473_u16, a),
        366 => radical_inverse_different_pow2(2477_u16, a),
        367 => radical_inverse_different_pow2(2503_u16, a),
        368 => radical_inverse_different_pow2(2521_u16, a),
        369 => radical_inverse_different_pow2(2531_u16, a),
        370 => radical_inverse_different_pow2(2539_u16, a),
        371 => radical_inverse_different_pow2(2543_u16, a),
        372 => radical_inverse_different_pow2(2549_u16, a),
        373 => radical_inverse_different_pow2(2551_u16, a),
        374 => radical_inverse_different_pow2(2557_u16, a),
        375 => radical_inverse_different_pow2(2579_u16, a),
        376 => radical_inverse_different_pow2(2591_u16, a),
        377 => radical_inverse_different_pow2(2593_u16, a),
        378 => radical_inverse_different_pow2(2609_u16, a),
        379 => radical_inverse_different_pow2(2617_u16, a),
        380 => radical_inverse_different_pow2(2621_u16, a),
        381 => radical_inverse_different_pow2(2633_u16, a),
        382 => radical_inverse_different_pow2(2647_u16, a),
        383 => radical_inverse_different_pow2(2657_u16, a),
        384 => radical_inverse_different_pow2(2659_u16, a),
        385 => radical_inverse_different_pow2(2663_u16, a),
        386 => radical_inverse_different_pow2(2671_u16, a),
        387 => radical_inverse_different_pow2(2677_u16, a),
        388 => radical_inverse_different_pow2(2683_u16, a),
        389 => radical_inverse_different_pow2(2687_u16, a),
        390 => radical_inverse_different_pow2(2689_u16, a),
        391 => radical_inverse_different_pow2(2693_u16, a),
        392 => radical_inverse_different_pow2(2699_u16, a),
        393 => radical_inverse_different_pow2(2707_u16, a),
        394 => radical_inverse_different_pow2(2711_u16, a),
        395 => radical_inverse_different_pow2(2713_u16, a),
        396 => radical_inverse_different_pow2(2719_u16, a),
        397 => radical_inverse_different_pow2(2729_u16, a),
        398 => radical_inverse_different_pow2(2731_u16, a),
        399 => radical_inverse_different_pow2(2741_u16, a),
        400 => radical_inverse_different_pow2(2749_u16, a),
        401 => radical_inverse_different_pow2(2753_u16, a),
        402 => radical_inverse_different_pow2(2767_u16, a),
        403 => radical_inverse_different_pow2(2777_u16, a),
        404 => radical_inverse_different_pow2(2789_u16, a),
        405 => radical_inverse_different_pow2(2791_u16, a),
        406 => radical_inverse_different_pow2(2797_u16, a),
        407 => radical_inverse_different_pow2(2801_u16, a),
        408 => radical_inverse_different_pow2(2803_u16, a),
        409 => radical_inverse_different_pow2(2819_u16, a),
        410 => radical_inverse_different_pow2(2833_u16, a),
        411 => radical_inverse_different_pow2(2837_u16, a),
        412 => radical_inverse_different_pow2(2843_u16, a),
        413 => radical_inverse_different_pow2(2851_u16, a),
        414 => radical_inverse_different_pow2(2857_u16, a),
        415 => radical_inverse_different_pow2(2861_u16, a),
        416 => radical_inverse_different_pow2(2879_u16, a),
        417 => radical_inverse_different_pow2(2887_u16, a),
        418 => radical_inverse_different_pow2(2897_u16, a),
        419 => radical_inverse_different_pow2(2903_u16, a),
        420 => radical_inverse_different_pow2(2909_u16, a),
        421 => radical_inverse_different_pow2(2917_u16, a),
        422 => radical_inverse_different_pow2(2927_u16, a),
        423 => radical_inverse_different_pow2(2939_u16, a),
        424 => radical_inverse_different_pow2(2953_u16, a),
        425 => radical_inverse_different_pow2(2957_u16, a),
        426 => radical_inverse_different_pow2(2963_u16, a),
        427 => radical_inverse_different_pow2(2969_u16, a),
        428 => radical_inverse_different_pow2(2971_u16, a),
        429 => radical_inverse_different_pow2(2999_u16, a),
        430 => radical_inverse_different_pow2(3001_u16, a),
        431 => radical_inverse_different_pow2(3011_u16, a),
        432 => radical_inverse_different_pow2(3019_u16, a),
        433 => radical_inverse_different_pow2(3023_u16, a),
        434 => radical_inverse_different_pow2(3037_u16, a),
        435 => radical_inverse_different_pow2(3041_u16, a),
        436 => radical_inverse_different_pow2(3049_u16, a),
        437 => radical_inverse_different_pow2(3061_u16, a),
        438 => radical_inverse_different_pow2(3067_u16, a),
        439 => radical_inverse_different_pow2(3079_u16, a),
        440 => radical_inverse_different_pow2(3083_u16, a),
        441 => radical_inverse_different_pow2(3089_u16, a),
        442 => radical_inverse_different_pow2(3109_u16, a),
        443 => radical_inverse_different_pow2(3119_u16, a),
        444 => radical_inverse_different_pow2(3121_u16, a),
        445 => radical_inverse_different_pow2(3137_u16, a),
        446 => radical_inverse_different_pow2(3163_u16, a),
        447 => radical_inverse_different_pow2(3167_u16, a),
        448 => radical_inverse_different_pow2(3169_u16, a),
        449 => radical_inverse_different_pow2(3181_u16, a),
        450 => radical_inverse_different_pow2(3187_u16, a),
        451 => radical_inverse_different_pow2(3191_u16, a),
        452 => radical_inverse_different_pow2(3203_u16, a),
        453 => radical_inverse_different_pow2(3209_u16, a),
        454 => radical_inverse_different_pow2(3217_u16, a),
        455 => radical_inverse_different_pow2(3221_u16, a),
        456 => radical_inverse_different_pow2(3229_u16, a),
        457 => radical_inverse_different_pow2(3251_u16, a),
        458 => radical_inverse_different_pow2(3253_u16, a),
        459 => radical_inverse_different_pow2(3257_u16, a),
        460 => radical_inverse_different_pow2(3259_u16, a),
        461 => radical_inverse_different_pow2(3271_u16, a),
        462 => radical_inverse_different_pow2(3299_u16, a),
        463 => radical_inverse_different_pow2(3301_u16, a),
        464 => radical_inverse_different_pow2(3307_u16, a),
        465 => radical_inverse_different_pow2(3313_u16, a),
        466 => radical_inverse_different_pow2(3319_u16, a),
        467 => radical_inverse_different_pow2(3323_u16, a),
        468 => radical_inverse_different_pow2(3329_u16, a),
        469 => radical_inverse_different_pow2(3331_u16, a),
        470 => radical_inverse_different_pow2(3343_u16, a),
        471 => radical_inverse_different_pow2(3347_u16, a),
        472 => radical_inverse_different_pow2(3359_u16, a),
        473 => radical_inverse_different_pow2(3361_u16, a),
        474 => radical_inverse_different_pow2(3371_u16, a),
        475 => radical_inverse_different_pow2(3373_u16, a),
        476 => radical_inverse_different_pow2(3389_u16, a),
        477 => radical_inverse_different_pow2(3391_u16, a),
        478 => radical_inverse_different_pow2(3407_u16, a),
        479 => radical_inverse_different_pow2(3413_u16, a),
        480 => radical_inverse_different_pow2(3433_u16, a),
        481 => radical_inverse_different_pow2(3449_u16, a),
        482 => radical_inverse_different_pow2(3457_u16, a),
        483 => radical_inverse_different_pow2(3461_u16, a),
        484 => radical_inverse_different_pow2(3463_u16, a),
        485 => radical_inverse_different_pow2(3467_u16, a),
        486 => radical_inverse_different_pow2(3469_u16, a),
        487 => radical_inverse_different_pow2(3491_u16, a),
        488 => radical_inverse_different_pow2(3499_u16, a),
        489 => radical_inverse_different_pow2(3511_u16, a),
        490 => radical_inverse_different_pow2(3517_u16, a),
        491 => radical_inverse_different_pow2(3527_u16, a),
        492 => radical_inverse_different_pow2(3529_u16, a),
        493 => radical_inverse_different_pow2(3533_u16, a),
        494 => radical_inverse_different_pow2(3539_u16, a),
        495 => radical_inverse_different_pow2(3541_u16, a),
        496 => radical_inverse_different_pow2(3547_u16, a),
        497 => radical_inverse_different_pow2(3557_u16, a),
        498 => radical_inverse_different_pow2(3559_u16, a),
        499 => radical_inverse_different_pow2(3571_u16, a),
        500 => radical_inverse_different_pow2(3581_u16, a),
        501 => radical_inverse_different_pow2(3583_u16, a),
        502 => radical_inverse_different_pow2(3593_u16, a),
        503 => radical_inverse_different_pow2(3607_u16, a),
        504 => radical_inverse_different_pow2(3613_u16, a),
        505 => radical_inverse_different_pow2(3617_u16, a),
        506 => radical_inverse_different_pow2(3623_u16, a),
        507 => radical_inverse_different_pow2(3631_u16, a),
        508 => radical_inverse_different_pow2(3637_u16, a),
        509 => radical_inverse_different_pow2(3643_u16, a),
        510 => radical_inverse_different_pow2(3659_u16, a),
        511 => radical_inverse_different_pow2(3671_u16, a),
        512 => radical_inverse_different_pow2(3673_u16, a),
        513 => radical_inverse_different_pow2(3677_u16, a),
        514 => radical_inverse_different_pow2(3691_u16, a),
        515 => radical_inverse_different_pow2(3697_u16, a),
        516 => radical_inverse_different_pow2(3701_u16, a),
        517 => radical_inverse_different_pow2(3709_u16, a),
        518 => radical_inverse_different_pow2(3719_u16, a),
        519 => radical_inverse_different_pow2(3727_u16, a),
        520 => radical_inverse_different_pow2(3733_u16, a),
        521 => radical_inverse_different_pow2(3739_u16, a),
        522 => radical_inverse_different_pow2(3761_u16, a),
        523 => radical_inverse_different_pow2(3767_u16, a),
        524 => radical_inverse_different_pow2(3769_u16, a),
        525 => radical_inverse_different_pow2(3779_u16, a),
        526 => radical_inverse_different_pow2(3793_u16, a),
        527 => radical_inverse_different_pow2(3797_u16, a),
        528 => radical_inverse_different_pow2(3803_u16, a),
        529 => radical_inverse_different_pow2(3821_u16, a),
        530 => radical_inverse_different_pow2(3823_u16, a),
        531 => radical_inverse_different_pow2(3833_u16, a),
        532 => radical_inverse_different_pow2(3847_u16, a),
        533 => radical_inverse_different_pow2(3851_u16, a),
        534 => radical_inverse_different_pow2(3853_u16, a),
        535 => radical_inverse_different_pow2(3863_u16, a),
        536 => radical_inverse_different_pow2(3877_u16, a),
        537 => radical_inverse_different_pow2(3881_u16, a),
        538 => radical_inverse_different_pow2(3889_u16, a),
        539 => radical_inverse_different_pow2(3907_u16, a),
        540 => radical_inverse_different_pow2(3911_u16, a),
        541 => radical_inverse_different_pow2(3917_u16, a),
        542 => radical_inverse_different_pow2(3919_u16, a),
        543 => radical_inverse_different_pow2(3923_u16, a),
        544 => radical_inverse_different_pow2(3929_u16, a),
        545 => radical_inverse_different_pow2(3931_u16, a),
        546 => radical_inverse_different_pow2(3943_u16, a),
        547 => radical_inverse_different_pow2(3947_u16, a),
        548 => radical_inverse_different_pow2(3967_u16, a),
        549 => radical_inverse_different_pow2(3989_u16, a),
        550 => radical_inverse_different_pow2(4001_u16, a),
        551 => radical_inverse_different_pow2(4003_u16, a),
        552 => radical_inverse_different_pow2(4007_u16, a),
        553 => radical_inverse_different_pow2(4013_u16, a),
        554 => radical_inverse_different_pow2(4019_u16, a),
        555 => radical_inverse_different_pow2(4021_u16, a),
        556 => radical_inverse_different_pow2(4027_u16, a),
        557 => radical_inverse_different_pow2(4049_u16, a),
        558 => radical_inverse_different_pow2(4051_u16, a),
        559 => radical_inverse_different_pow2(4057_u16, a),
        560 => radical_inverse_different_pow2(4073_u16, a),
        561 => radical_inverse_different_pow2(4079_u16, a),
        562 => radical_inverse_different_pow2(4091_u16, a),
        563 => radical_inverse_different_pow2(4093_u16, a),
        564 => radical_inverse_different_pow2(4099_u16, a),
        565 => radical_inverse_different_pow2(4111_u16, a),
        566 => radical_inverse_different_pow2(4127_u16, a),
        567 => radical_inverse_different_pow2(4129_u16, a),
        568 => radical_inverse_different_pow2(4133_u16, a),
        569 => radical_inverse_different_pow2(4139_u16, a),
        570 => radical_inverse_different_pow2(4153_u16, a),
        571 => radical_inverse_different_pow2(4157_u16, a),
        572 => radical_inverse_different_pow2(4159_u16, a),
        573 => radical_inverse_different_pow2(4177_u16, a),
        574 => radical_inverse_different_pow2(4201_u16, a),
        575 => radical_inverse_different_pow2(4211_u16, a),
        576 => radical_inverse_different_pow2(4217_u16, a),
        577 => radical_inverse_different_pow2(4219_u16, a),
        578 => radical_inverse_different_pow2(4229_u16, a),
        579 => radical_inverse_different_pow2(4231_u16, a),
        580 => radical_inverse_different_pow2(4241_u16, a),
        581 => radical_inverse_different_pow2(4243_u16, a),
        582 => radical_inverse_different_pow2(4253_u16, a),
        583 => radical_inverse_different_pow2(4259_u16, a),
        584 => radical_inverse_different_pow2(4261_u16, a),
        585 => radical_inverse_different_pow2(4271_u16, a),
        586 => radical_inverse_different_pow2(4273_u16, a),
        587 => radical_inverse_different_pow2(4283_u16, a),
        588 => radical_inverse_different_pow2(4289_u16, a),
        589 => radical_inverse_different_pow2(4297_u16, a),
        590 => radical_inverse_different_pow2(4327_u16, a),
        591 => radical_inverse_different_pow2(4337_u16, a),
        592 => radical_inverse_different_pow2(4339_u16, a),
        593 => radical_inverse_different_pow2(4349_u16, a),
        594 => radical_inverse_different_pow2(4357_u16, a),
        595 => radical_inverse_different_pow2(4363_u16, a),
        596 => radical_inverse_different_pow2(4373_u16, a),
        597 => radical_inverse_different_pow2(4391_u16, a),
        598 => radical_inverse_different_pow2(4397_u16, a),
        599 => radical_inverse_different_pow2(4409_u16, a),
        600 => radical_inverse_different_pow2(4421_u16, a),
        601 => radical_inverse_different_pow2(4423_u16, a),
        602 => radical_inverse_different_pow2(4441_u16, a),
        603 => radical_inverse_different_pow2(4447_u16, a),
        604 => radical_inverse_different_pow2(4451_u16, a),
        605 => radical_inverse_different_pow2(4457_u16, a),
        606 => radical_inverse_different_pow2(4463_u16, a),
        607 => radical_inverse_different_pow2(4481_u16, a),
        608 => radical_inverse_different_pow2(4483_u16, a),
        609 => radical_inverse_different_pow2(4493_u16, a),
        610 => radical_inverse_different_pow2(4507_u16, a),
        611 => radical_inverse_different_pow2(4513_u16, a),
        612 => radical_inverse_different_pow2(4517_u16, a),
        613 => radical_inverse_different_pow2(4519_u16, a),
        614 => radical_inverse_different_pow2(4523_u16, a),
        615 => radical_inverse_different_pow2(4547_u16, a),
        616 => radical_inverse_different_pow2(4549_u16, a),
        617 => radical_inverse_different_pow2(4561_u16, a),
        618 => radical_inverse_different_pow2(4567_u16, a),
        619 => radical_inverse_different_pow2(4583_u16, a),
        620 => radical_inverse_different_pow2(4591_u16, a),
        621 => radical_inverse_different_pow2(4597_u16, a),
        622 => radical_inverse_different_pow2(4603_u16, a),
        623 => radical_inverse_different_pow2(4621_u16, a),
        624 => radical_inverse_different_pow2(4637_u16, a),
        625 => radical_inverse_different_pow2(4639_u16, a),
        626 => radical_inverse_different_pow2(4643_u16, a),
        627 => radical_inverse_different_pow2(4649_u16, a),
        628 => radical_inverse_different_pow2(4651_u16, a),
        629 => radical_inverse_different_pow2(4657_u16, a),
        630 => radical_inverse_different_pow2(4663_u16, a),
        631 => radical_inverse_different_pow2(4673_u16, a),
        632 => radical_inverse_different_pow2(4679_u16, a),
        633 => radical_inverse_different_pow2(4691_u16, a),
        634 => radical_inverse_different_pow2(4703_u16, a),
        635 => radical_inverse_different_pow2(4721_u16, a),
        636 => radical_inverse_different_pow2(4723_u16, a),
        637 => radical_inverse_different_pow2(4729_u16, a),
        638 => radical_inverse_different_pow2(4733_u16, a),
        639 => radical_inverse_different_pow2(4751_u16, a),
        640 => radical_inverse_different_pow2(4759_u16, a),
        641 => radical_inverse_different_pow2(4783_u16, a),
        642 => radical_inverse_different_pow2(4787_u16, a),
        643 => radical_inverse_different_pow2(4789_u16, a),
        644 => radical_inverse_different_pow2(4793_u16, a),
        645 => radical_inverse_different_pow2(4799_u16, a),
        646 => radical_inverse_different_pow2(4801_u16, a),
        647 => radical_inverse_different_pow2(4813_u16, a),
        648 => radical_inverse_different_pow2(4817_u16, a),
        649 => radical_inverse_different_pow2(4831_u16, a),
        650 => radical_inverse_different_pow2(4861_u16, a),
        651 => radical_inverse_different_pow2(4871_u16, a),
        652 => radical_inverse_different_pow2(4877_u16, a),
        653 => radical_inverse_different_pow2(4889_u16, a),
        654 => radical_inverse_different_pow2(4903_u16, a),
        655 => radical_inverse_different_pow2(4909_u16, a),
        656 => radical_inverse_different_pow2(4919_u16, a),
        657 => radical_inverse_different_pow2(4931_u16, a),
        658 => radical_inverse_different_pow2(4933_u16, a),
        659 => radical_inverse_different_pow2(4937_u16, a),
        660 => radical_inverse_different_pow2(4943_u16, a),
        661 => radical_inverse_different_pow2(4951_u16, a),
        662 => radical_inverse_different_pow2(4957_u16, a),
        663 => radical_inverse_different_pow2(4967_u16, a),
        664 => radical_inverse_different_pow2(4969_u16, a),
        665 => radical_inverse_different_pow2(4973_u16, a),
        666 => radical_inverse_different_pow2(4987_u16, a),
        667 => radical_inverse_different_pow2(4993_u16, a),
        668 => radical_inverse_different_pow2(4999_u16, a),
        669 => radical_inverse_different_pow2(5003_u16, a),
        670 => radical_inverse_different_pow2(5009_u16, a),
        671 => radical_inverse_different_pow2(5011_u16, a),
        672 => radical_inverse_different_pow2(5021_u16, a),
        673 => radical_inverse_different_pow2(5023_u16, a),
        674 => radical_inverse_different_pow2(5039_u16, a),
        675 => radical_inverse_different_pow2(5051_u16, a),
        676 => radical_inverse_different_pow2(5059_u16, a),
        677 => radical_inverse_different_pow2(5077_u16, a),
        678 => radical_inverse_different_pow2(5081_u16, a),
        679 => radical_inverse_different_pow2(5087_u16, a),
        680 => radical_inverse_different_pow2(5099_u16, a),
        681 => radical_inverse_different_pow2(5101_u16, a),
        682 => radical_inverse_different_pow2(5107_u16, a),
        683 => radical_inverse_different_pow2(5113_u16, a),
        684 => radical_inverse_different_pow2(5119_u16, a),
        685 => radical_inverse_different_pow2(5147_u16, a),
        686 => radical_inverse_different_pow2(5153_u16, a),
        687 => radical_inverse_different_pow2(5167_u16, a),
        688 => radical_inverse_different_pow2(5171_u16, a),
        689 => radical_inverse_different_pow2(5179_u16, a),
        690 => radical_inverse_different_pow2(5189_u16, a),
        691 => radical_inverse_different_pow2(5197_u16, a),
        692 => radical_inverse_different_pow2(5209_u16, a),
        693 => radical_inverse_different_pow2(5227_u16, a),
        694 => radical_inverse_different_pow2(5231_u16, a),
        695 => radical_inverse_different_pow2(5233_u16, a),
        696 => radical_inverse_different_pow2(5237_u16, a),
        697 => radical_inverse_different_pow2(5261_u16, a),
        698 => radical_inverse_different_pow2(5273_u16, a),
        699 => radical_inverse_different_pow2(5279_u16, a),
        700 => radical_inverse_different_pow2(5281_u16, a),
        701 => radical_inverse_different_pow2(5297_u16, a),
        702 => radical_inverse_different_pow2(5303_u16, a),
        703 => radical_inverse_different_pow2(5309_u16, a),
        704 => radical_inverse_different_pow2(5323_u16, a),
        705 => radical_inverse_different_pow2(5333_u16, a),
        706 => radical_inverse_different_pow2(5347_u16, a),
        707 => radical_inverse_different_pow2(5351_u16, a),
        708 => radical_inverse_different_pow2(5381_u16, a),
        709 => radical_inverse_different_pow2(5387_u16, a),
        710 => radical_inverse_different_pow2(5393_u16, a),
        711 => radical_inverse_different_pow2(5399_u16, a),
        712 => radical_inverse_different_pow2(5407_u16, a),
        713 => radical_inverse_different_pow2(5413_u16, a),
        714 => radical_inverse_different_pow2(5417_u16, a),
        715 => radical_inverse_different_pow2(5419_u16, a),
        716 => radical_inverse_different_pow2(5431_u16, a),
        717 => radical_inverse_different_pow2(5437_u16, a),
        718 => radical_inverse_different_pow2(5441_u16, a),
        719 => radical_inverse_different_pow2(5443_u16, a),
        720 => radical_inverse_different_pow2(5449_u16, a),
        721 => radical_inverse_different_pow2(5471_u16, a),
        722 => radical_inverse_different_pow2(5477_u16, a),
        723 => radical_inverse_different_pow2(5479_u16, a),
        724 => radical_inverse_different_pow2(5483_u16, a),
        725 => radical_inverse_different_pow2(5501_u16, a),
        726 => radical_inverse_different_pow2(5503_u16, a),
        727 => radical_inverse_different_pow2(5507_u16, a),
        728 => radical_inverse_different_pow2(5519_u16, a),
        729 => radical_inverse_different_pow2(5521_u16, a),
        730 => radical_inverse_different_pow2(5527_u16, a),
        731 => radical_inverse_different_pow2(5531_u16, a),
        732 => radical_inverse_different_pow2(5557_u16, a),
        733 => radical_inverse_different_pow2(5563_u16, a),
        734 => radical_inverse_different_pow2(5569_u16, a),
        735 => radical_inverse_different_pow2(5573_u16, a),
        736 => radical_inverse_different_pow2(5581_u16, a),
        737 => radical_inverse_different_pow2(5591_u16, a),
        738 => radical_inverse_different_pow2(5623_u16, a),
        739 => radical_inverse_different_pow2(5639_u16, a),
        740 => radical_inverse_different_pow2(5641_u16, a),
        741 => radical_inverse_different_pow2(5647_u16, a),
        742 => radical_inverse_different_pow2(5651_u16, a),
        743 => radical_inverse_different_pow2(5653_u16, a),
        744 => radical_inverse_different_pow2(5657_u16, a),
        745 => radical_inverse_different_pow2(5659_u16, a),
        746 => radical_inverse_different_pow2(5669_u16, a),
        747 => radical_inverse_different_pow2(5683_u16, a),
        748 => radical_inverse_different_pow2(5689_u16, a),
        749 => radical_inverse_different_pow2(5693_u16, a),
        750 => radical_inverse_different_pow2(5701_u16, a),
        751 => radical_inverse_different_pow2(5711_u16, a),
        752 => radical_inverse_different_pow2(5717_u16, a),
        753 => radical_inverse_different_pow2(5737_u16, a),
        754 => radical_inverse_different_pow2(5741_u16, a),
        755 => radical_inverse_different_pow2(5743_u16, a),
        756 => radical_inverse_different_pow2(5749_u16, a),
        757 => radical_inverse_different_pow2(5779_u16, a),
        758 => radical_inverse_different_pow2(5783_u16, a),
        759 => radical_inverse_different_pow2(5791_u16, a),
        760 => radical_inverse_different_pow2(5801_u16, a),
        761 => radical_inverse_different_pow2(5807_u16, a),
        762 => radical_inverse_different_pow2(5813_u16, a),
        763 => radical_inverse_different_pow2(5821_u16, a),
        764 => radical_inverse_different_pow2(5827_u16, a),
        765 => radical_inverse_different_pow2(5839_u16, a),
        766 => radical_inverse_different_pow2(5843_u16, a),
        767 => radical_inverse_different_pow2(5849_u16, a),
        768 => radical_inverse_different_pow2(5851_u16, a),
        769 => radical_inverse_different_pow2(5857_u16, a),
        770 => radical_inverse_different_pow2(5861_u16, a),
        771 => radical_inverse_different_pow2(5867_u16, a),
        772 => radical_inverse_different_pow2(5869_u16, a),
        773 => radical_inverse_different_pow2(5879_u16, a),
        774 => radical_inverse_different_pow2(5881_u16, a),
        775 => radical_inverse_different_pow2(5897_u16, a),
        776 => radical_inverse_different_pow2(5903_u16, a),
        777 => radical_inverse_different_pow2(5923_u16, a),
        778 => radical_inverse_different_pow2(5927_u16, a),
        779 => radical_inverse_different_pow2(5939_u16, a),
        780 => radical_inverse_different_pow2(5953_u16, a),
        781 => radical_inverse_different_pow2(5981_u16, a),
        782 => radical_inverse_different_pow2(5987_u16, a),
        783 => radical_inverse_different_pow2(6007_u16, a),
        784 => radical_inverse_different_pow2(6011_u16, a),
        785 => radical_inverse_different_pow2(6029_u16, a),
        786 => radical_inverse_different_pow2(6037_u16, a),
        787 => radical_inverse_different_pow2(6043_u16, a),
        788 => radical_inverse_different_pow2(6047_u16, a),
        789 => radical_inverse_different_pow2(6053_u16, a),
        790 => radical_inverse_different_pow2(6067_u16, a),
        791 => radical_inverse_different_pow2(6073_u16, a),
        792 => radical_inverse_different_pow2(6079_u16, a),
        793 => radical_inverse_different_pow2(6089_u16, a),
        794 => radical_inverse_different_pow2(6091_u16, a),
        795 => radical_inverse_different_pow2(6101_u16, a),
        796 => radical_inverse_different_pow2(6113_u16, a),
        797 => radical_inverse_different_pow2(6121_u16, a),
        798 => radical_inverse_different_pow2(6131_u16, a),
        799 => radical_inverse_different_pow2(6133_u16, a),
        800 => radical_inverse_different_pow2(6143_u16, a),
        801 => radical_inverse_different_pow2(6151_u16, a),
        802 => radical_inverse_different_pow2(6163_u16, a),
        803 => radical_inverse_different_pow2(6173_u16, a),
        804 => radical_inverse_different_pow2(6197_u16, a),
        805 => radical_inverse_different_pow2(6199_u16, a),
        806 => radical_inverse_different_pow2(6203_u16, a),
        807 => radical_inverse_different_pow2(6211_u16, a),
        808 => radical_inverse_different_pow2(6217_u16, a),
        809 => radical_inverse_different_pow2(6221_u16, a),
        810 => radical_inverse_different_pow2(6229_u16, a),
        811 => radical_inverse_different_pow2(6247_u16, a),
        812 => radical_inverse_different_pow2(6257_u16, a),
        813 => radical_inverse_different_pow2(6263_u16, a),
        814 => radical_inverse_different_pow2(6269_u16, a),
        815 => radical_inverse_different_pow2(6271_u16, a),
        816 => radical_inverse_different_pow2(6277_u16, a),
        817 => radical_inverse_different_pow2(6287_u16, a),
        818 => radical_inverse_different_pow2(6299_u16, a),
        819 => radical_inverse_different_pow2(6301_u16, a),
        820 => radical_inverse_different_pow2(6311_u16, a),
        821 => radical_inverse_different_pow2(6317_u16, a),
        822 => radical_inverse_different_pow2(6323_u16, a),
        823 => radical_inverse_different_pow2(6329_u16, a),
        824 => radical_inverse_different_pow2(6337_u16, a),
        825 => radical_inverse_different_pow2(6343_u16, a),
        826 => radical_inverse_different_pow2(6353_u16, a),
        827 => radical_inverse_different_pow2(6359_u16, a),
        828 => radical_inverse_different_pow2(6361_u16, a),
        829 => radical_inverse_different_pow2(6367_u16, a),
        830 => radical_inverse_different_pow2(6373_u16, a),
        831 => radical_inverse_different_pow2(6379_u16, a),
        832 => radical_inverse_different_pow2(6389_u16, a),
        833 => radical_inverse_different_pow2(6397_u16, a),
        834 => radical_inverse_different_pow2(6421_u16, a),
        835 => radical_inverse_different_pow2(6427_u16, a),
        836 => radical_inverse_different_pow2(6449_u16, a),
        837 => radical_inverse_different_pow2(6451_u16, a),
        838 => radical_inverse_different_pow2(6469_u16, a),
        839 => radical_inverse_different_pow2(6473_u16, a),
        840 => radical_inverse_different_pow2(6481_u16, a),
        841 => radical_inverse_different_pow2(6491_u16, a),
        842 => radical_inverse_different_pow2(6521_u16, a),
        843 => radical_inverse_different_pow2(6529_u16, a),
        844 => radical_inverse_different_pow2(6547_u16, a),
        845 => radical_inverse_different_pow2(6551_u16, a),
        846 => radical_inverse_different_pow2(6553_u16, a),
        847 => radical_inverse_different_pow2(6563_u16, a),
        848 => radical_inverse_different_pow2(6569_u16, a),
        849 => radical_inverse_different_pow2(6571_u16, a),
        850 => radical_inverse_different_pow2(6577_u16, a),
        851 => radical_inverse_different_pow2(6581_u16, a),
        852 => radical_inverse_different_pow2(6599_u16, a),
        853 => radical_inverse_different_pow2(6607_u16, a),
        854 => radical_inverse_different_pow2(6619_u16, a),
        855 => radical_inverse_different_pow2(6637_u16, a),
        856 => radical_inverse_different_pow2(6653_u16, a),
        857 => radical_inverse_different_pow2(6659_u16, a),
        858 => radical_inverse_different_pow2(6661_u16, a),
        859 => radical_inverse_different_pow2(6673_u16, a),
        860 => radical_inverse_different_pow2(6679_u16, a),
        861 => radical_inverse_different_pow2(6689_u16, a),
        862 => radical_inverse_different_pow2(6691_u16, a),
        863 => radical_inverse_different_pow2(6701_u16, a),
        864 => radical_inverse_different_pow2(6703_u16, a),
        865 => radical_inverse_different_pow2(6709_u16, a),
        866 => radical_inverse_different_pow2(6719_u16, a),
        867 => radical_inverse_different_pow2(6733_u16, a),
        868 => radical_inverse_different_pow2(6737_u16, a),
        869 => radical_inverse_different_pow2(6761_u16, a),
        870 => radical_inverse_different_pow2(6763_u16, a),
        871 => radical_inverse_different_pow2(6779_u16, a),
        872 => radical_inverse_different_pow2(6781_u16, a),
        873 => radical_inverse_different_pow2(6791_u16, a),
        874 => radical_inverse_different_pow2(6793_u16, a),
        875 => radical_inverse_different_pow2(6803_u16, a),
        876 => radical_inverse_different_pow2(6823_u16, a),
        877 => radical_inverse_different_pow2(6827_u16, a),
        878 => radical_inverse_different_pow2(6829_u16, a),
        879 => radical_inverse_different_pow2(6833_u16, a),
        880 => radical_inverse_different_pow2(6841_u16, a),
        881 => radical_inverse_different_pow2(6857_u16, a),
        882 => radical_inverse_different_pow2(6863_u16, a),
        883 => radical_inverse_different_pow2(6869_u16, a),
        884 => radical_inverse_different_pow2(6871_u16, a),
        885 => radical_inverse_different_pow2(6883_u16, a),
        886 => radical_inverse_different_pow2(6899_u16, a),
        887 => radical_inverse_different_pow2(6907_u16, a),
        888 => radical_inverse_different_pow2(6911_u16, a),
        889 => radical_inverse_different_pow2(6917_u16, a),
        890 => radical_inverse_different_pow2(6947_u16, a),
        891 => radical_inverse_different_pow2(6949_u16, a),
        892 => radical_inverse_different_pow2(6959_u16, a),
        893 => radical_inverse_different_pow2(6961_u16, a),
        894 => radical_inverse_different_pow2(6967_u16, a),
        895 => radical_inverse_different_pow2(6971_u16, a),
        896 => radical_inverse_different_pow2(6977_u16, a),
        897 => radical_inverse_different_pow2(6983_u16, a),
        898 => radical_inverse_different_pow2(6991_u16, a),
        899 => radical_inverse_different_pow2(6997_u16, a),
        900 => radical_inverse_different_pow2(7001_u16, a),
        901 => radical_inverse_different_pow2(7013_u16, a),
        902 => radical_inverse_different_pow2(7019_u16, a),
        903 => radical_inverse_different_pow2(7027_u16, a),
        904 => radical_inverse_different_pow2(7039_u16, a),
        905 => radical_inverse_different_pow2(7043_u16, a),
        906 => radical_inverse_different_pow2(7057_u16, a),
        907 => radical_inverse_different_pow2(7069_u16, a),
        908 => radical_inverse_different_pow2(7079_u16, a),
        909 => radical_inverse_different_pow2(7103_u16, a),
        910 => radical_inverse_different_pow2(7109_u16, a),
        911 => radical_inverse_different_pow2(7121_u16, a),
        912 => radical_inverse_different_pow2(7127_u16, a),
        913 => radical_inverse_different_pow2(7129_u16, a),
        914 => radical_inverse_different_pow2(7151_u16, a),
        915 => radical_inverse_different_pow2(7159_u16, a),
        916 => radical_inverse_different_pow2(7177_u16, a),
        917 => radical_inverse_different_pow2(7187_u16, a),
        918 => radical_inverse_different_pow2(7193_u16, a),
        919 => radical_inverse_different_pow2(7207_u16, a),
        920 => radical_inverse_different_pow2(7211_u16, a),
        921 => radical_inverse_different_pow2(7213_u16, a),
        922 => radical_inverse_different_pow2(7219_u16, a),
        923 => radical_inverse_different_pow2(7229_u16, a),
        924 => radical_inverse_different_pow2(7237_u16, a),
        925 => radical_inverse_different_pow2(7243_u16, a),
        926 => radical_inverse_different_pow2(7247_u16, a),
        927 => radical_inverse_different_pow2(7253_u16, a),
        928 => radical_inverse_different_pow2(7283_u16, a),
        929 => radical_inverse_different_pow2(7297_u16, a),
        930 => radical_inverse_different_pow2(7307_u16, a),
        931 => radical_inverse_different_pow2(7309_u16, a),
        932 => radical_inverse_different_pow2(7321_u16, a),
        933 => radical_inverse_different_pow2(7331_u16, a),
        934 => radical_inverse_different_pow2(7333_u16, a),
        935 => radical_inverse_different_pow2(7349_u16, a),
        936 => radical_inverse_different_pow2(7351_u16, a),
        937 => radical_inverse_different_pow2(7369_u16, a),
        938 => radical_inverse_different_pow2(7393_u16, a),
        939 => radical_inverse_different_pow2(7411_u16, a),
        940 => radical_inverse_different_pow2(7417_u16, a),
        941 => radical_inverse_different_pow2(7433_u16, a),
        942 => radical_inverse_different_pow2(7451_u16, a),
        943 => radical_inverse_different_pow2(7457_u16, a),
        944 => radical_inverse_different_pow2(7459_u16, a),
        945 => radical_inverse_different_pow2(7477_u16, a),
        946 => radical_inverse_different_pow2(7481_u16, a),
        947 => radical_inverse_different_pow2(7487_u16, a),
        948 => radical_inverse_different_pow2(7489_u16, a),
        949 => radical_inverse_different_pow2(7499_u16, a),
        950 => radical_inverse_different_pow2(7507_u16, a),
        951 => radical_inverse_different_pow2(7517_u16, a),
        952 => radical_inverse_different_pow2(7523_u16, a),
        953 => radical_inverse_different_pow2(7529_u16, a),
        954 => radical_inverse_different_pow2(7537_u16, a),
        955 => radical_inverse_different_pow2(7541_u16, a),
        956 => radical_inverse_different_pow2(7547_u16, a),
        957 => radical_inverse_different_pow2(7549_u16, a),
        958 => radical_inverse_different_pow2(7559_u16, a),
        959 => radical_inverse_different_pow2(7561_u16, a),
        960 => radical_inverse_different_pow2(7573_u16, a),
        961 => radical_inverse_different_pow2(7577_u16, a),
        962 => radical_inverse_different_pow2(7583_u16, a),
        963 => radical_inverse_different_pow2(7589_u16, a),
        964 => radical_inverse_different_pow2(7591_u16, a),
        965 => radical_inverse_different_pow2(7603_u16, a),
        966 => radical_inverse_different_pow2(7607_u16, a),
        967 => radical_inverse_different_pow2(7621_u16, a),
        968 => radical_inverse_different_pow2(7639_u16, a),
        969 => radical_inverse_different_pow2(7643_u16, a),
        970 => radical_inverse_different_pow2(7649_u16, a),
        971 => radical_inverse_different_pow2(7669_u16, a),
        972 => radical_inverse_different_pow2(7673_u16, a),
        973 => radical_inverse_different_pow2(7681_u16, a),
        974 => radical_inverse_different_pow2(7687_u16, a),
        975 => radical_inverse_different_pow2(7691_u16, a),
        976 => radical_inverse_different_pow2(7699_u16, a),
        977 => radical_inverse_different_pow2(7703_u16, a),
        978 => radical_inverse_different_pow2(7717_u16, a),
        979 => radical_inverse_different_pow2(7723_u16, a),
        980 => radical_inverse_different_pow2(7727_u16, a),
        981 => radical_inverse_different_pow2(7741_u16, a),
        982 => radical_inverse_different_pow2(7753_u16, a),
        983 => radical_inverse_different_pow2(7757_u16, a),
        984 => radical_inverse_different_pow2(7759_u16, a),
        985 => radical_inverse_different_pow2(7789_u16, a),
        986 => radical_inverse_different_pow2(7793_u16, a),
        987 => radical_inverse_different_pow2(7817_u16, a),
        988 => radical_inverse_different_pow2(7823_u16, a),
        989 => radical_inverse_different_pow2(7829_u16, a),
        990 => radical_inverse_different_pow2(7841_u16, a),
        991 => radical_inverse_different_pow2(7853_u16, a),
        992 => radical_inverse_different_pow2(7867_u16, a),
        993 => radical_inverse_different_pow2(7873_u16, a),
        994 => radical_inverse_different_pow2(7877_u16, a),
        995 => radical_inverse_different_pow2(7879_u16, a),
        996 => radical_inverse_different_pow2(7883_u16, a),
        997 => radical_inverse_different_pow2(7901_u16, a),
        998 => radical_inverse_different_pow2(7907_u16, a),
        999 => radical_inverse_different_pow2(7919_u16, a),
        1000 => radical_inverse_different_pow2(7927_u16, a),
        1001 => radical_inverse_different_pow2(7933_u16, a),
        1002 => radical_inverse_different_pow2(7937_u16, a),
        1003 => radical_inverse_different_pow2(7949_u16, a),
        1004 => radical_inverse_different_pow2(7951_u16, a),
        1005 => radical_inverse_different_pow2(7963_u16, a),
        1006 => radical_inverse_different_pow2(7993_u16, a),
        1007 => radical_inverse_different_pow2(8009_u16, a),
        1008 => radical_inverse_different_pow2(8011_u16, a),
        1009 => radical_inverse_different_pow2(8017_u16, a),
        1010 => radical_inverse_different_pow2(8039_u16, a),
        1011 => radical_inverse_different_pow2(8053_u16, a),
        1012 => radical_inverse_different_pow2(8059_u16, a),
        1013 => radical_inverse_different_pow2(8069_u16, a),
        1014 => radical_inverse_different_pow2(8081_u16, a),
        1015 => radical_inverse_different_pow2(8087_u16, a),
        1016 => radical_inverse_different_pow2(8089_u16, a),
        1017 => radical_inverse_different_pow2(8093_u16, a),
        1018 => radical_inverse_different_pow2(8101_u16, a),
        1019 => radical_inverse_different_pow2(8111_u16, a),
        1020 => radical_inverse_different_pow2(8117_u16, a),
        1021 => radical_inverse_different_pow2(8123_u16, a),
        1022 => radical_inverse_different_pow2(8147_u16, a),
        1023 => radical_inverse_different_pow2(8161_u16, a),
        _ => {
            panic!("TODO: radical_inverse({:?}, {:?})", base, a);
        }
    }
}


pub const PRIMES: [u32; PRIME_TABLE_SIZE as usize] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547,
    557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
    661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797,
    809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929,
    937, 941, 947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039,
    1049, 1051, 1061, 1063, 1069, 1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153,
    1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279,
    1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373, 1381, 1399, 1409,
    1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499,
    1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601, 1607, 1609, 1613,
    1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733, 1741,
    1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873,
    1877, 1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999,
    2003, 2011, 2017, 2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113,
    2129, 2131, 2137, 2141, 2143, 2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251,
    2267, 2269, 2273, 2281, 2287, 2293, 2297, 2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371,
    2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423, 2437, 2441, 2447, 2459, 2467, 2473, 2477,
    2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593, 2609, 2617, 2621, 2633, 2647,
    2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713, 2719, 2729, 2731,
    2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851, 2857,
    2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001,
    3011, 3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163,
    3167, 3169, 3181, 3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299,
    3301, 3307, 3313, 3319, 3323, 3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407,
    3413, 3433, 3449, 3457, 3461, 3463, 3467, 3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539,
    3541, 3547, 3557, 3559, 3571, 3581, 3583, 3593, 3607, 3613, 3617, 3623, 3631, 3637, 3643, 3659,
    3671, 3673, 3677, 3691, 3697, 3701, 3709, 3719, 3727, 3733, 3739, 3761, 3767, 3769, 3779, 3793,
    3797, 3803, 3821, 3823, 3833, 3847, 3851, 3853, 3863, 3877, 3881, 3889, 3907, 3911, 3917, 3919,
    3923, 3929, 3931, 3943, 3947, 3967, 3989, 4001, 4003, 4007, 4013, 4019, 4021, 4027, 4049, 4051,
    4057, 4073, 4079, 4091, 4093, 4099, 4111, 4127, 4129, 4133, 4139, 4153, 4157, 4159, 4177, 4201,
    4211, 4217, 4219, 4229, 4231, 4241, 4243, 4253, 4259, 4261, 4271, 4273, 4283, 4289, 4297, 4327,
    4337, 4339, 4349, 4357, 4363, 4373, 4391, 4397, 4409, 4421, 4423, 4441, 4447, 4451, 4457, 4463,
    4481, 4483, 4493, 4507, 4513, 4517, 4519, 4523, 4547, 4549, 4561, 4567, 4583, 4591, 4597, 4603,
    4621, 4637, 4639, 4643, 4649, 4651, 4657, 4663, 4673, 4679, 4691, 4703, 4721, 4723, 4729, 4733,
    4751, 4759, 4783, 4787, 4789, 4793, 4799, 4801, 4813, 4817, 4831, 4861, 4871, 4877, 4889, 4903,
    4909, 4919, 4931, 4933, 4937, 4943, 4951, 4957, 4967, 4969, 4973, 4987, 4993, 4999, 5003, 5009,
    5011, 5021, 5023, 5039, 5051, 5059, 5077, 5081, 5087, 5099, 5101, 5107, 5113, 5119, 5147, 5153,
    5167, 5171, 5179, 5189, 5197, 5209, 5227, 5231, 5233, 5237, 5261, 5273, 5279, 5281, 5297, 5303,
    5309, 5323, 5333, 5347, 5351, 5381, 5387, 5393, 5399, 5407, 5413, 5417, 5419, 5431, 5437, 5441,
    5443, 5449, 5471, 5477, 5479, 5483, 5501, 5503, 5507, 5519, 5521, 5527, 5531, 5557, 5563, 5569,
    5573, 5581, 5591, 5623, 5639, 5641, 5647, 5651, 5653, 5657, 5659, 5669, 5683, 5689, 5693, 5701,
    5711, 5717, 5737, 5741, 5743, 5749, 5779, 5783, 5791, 5801, 5807, 5813, 5821, 5827, 5839, 5843,
    5849, 5851, 5857, 5861, 5867, 5869, 5879, 5881, 5897, 5903, 5923, 5927, 5939, 5953, 5981, 5987,
    6007, 6011, 6029, 6037, 6043, 6047, 6053, 6067, 6073, 6079, 6089, 6091, 6101, 6113, 6121, 6131,
    6133, 6143, 6151, 6163, 6173, 6197, 6199, 6203, 6211, 6217, 6221, 6229, 6247, 6257, 6263, 6269,
    6271, 6277, 6287, 6299, 6301, 6311, 6317, 6323, 6329, 6337, 6343, 6353, 6359, 6361, 6367, 6373,
    6379, 6389, 6397, 6421, 6427, 6449, 6451, 6469, 6473, 6481, 6491, 6521, 6529, 6547, 6551, 6553,
    6563, 6569, 6571, 6577, 6581, 6599, 6607, 6619, 6637, 6653, 6659, 6661, 6673, 6679, 6689, 6691,
    6701, 6703, 6709, 6719, 6733, 6737, 6761, 6763, 6779, 6781, 6791, 6793, 6803, 6823, 6827, 6829,
    6833, 6841, 6857, 6863, 6869, 6871, 6883, 6899, 6907, 6911, 6917, 6947, 6949, 6959, 6961, 6967,
    6971, 6977, 6983, 6991, 6997, 7001, 7013, 7019, 7027, 7039, 7043, 7057, 7069, 7079, 7103, 7109,
    7121, 7127, 7129, 7151, 7159, 7177, 7187, 7193, 7207, 7211, 7213, 7219, 7229, 7237, 7243, 7247,
    7253, 7283, 7297, 7307, 7309, 7321, 7331, 7333, 7349, 7351, 7369, 7393, 7411, 7417, 7433, 7451,
    7457, 7459, 7477, 7481, 7487, 7489, 7499, 7507, 7517, 7523, 7529, 7537, 7541, 7547, 7549, 7559,
    7561, 7573, 7577, 7583, 7589, 7591, 7603, 7607, 7621, 7639, 7643, 7649, 7669, 7673, 7681, 7687,
    7691, 7699, 7703, 7717, 7723, 7727, 7741, 7753, 7757, 7759, 7789, 7793, 7817, 7823, 7829, 7841,
    7853, 7867, 7873, 7877, 7879, 7883, 7901, 7907, 7919,
];


pub const PRIME_SUMS: [u32; PRIME_TABLE_SIZE as usize] = [
    0, 2, 5, 10, 17, 28, 41, 58, 77, 100, 129, 160, 197, 238, 281, 328, 381, 440, 501, 568, 639,
    712, 791, 874, 963, 1060, 1161, 1264, 1371, 1480, 1593, 1720, 1851, 1988, 2127, 2276, 2427,
    2584, 2747, 2914, 3087, 3266, 3447, 3638, 3831, 4028, 4227, 4438, 4661, 4888, 5117, 5350, 5589,
    5830, 6081, 6338, 6601, 6870, 7141, 7418, 7699, 7982, 8275, 8582, 8893, 9206, 9523, 9854,
    10191, 10538, 10887, 11240, 11599, 11966, 12339, 12718, 13101, 13490, 13887, 14288, 14697,
    15116, 15537, 15968, 16401, 16840, 17283, 17732, 18189, 18650, 19113, 19580, 20059, 20546,
    21037, 21536, 22039, 22548, 23069, 23592, 24133, 24680, 25237, 25800, 26369, 26940, 27517,
    28104, 28697, 29296, 29897, 30504, 31117, 31734, 32353, 32984, 33625, 34268, 34915, 35568,
    36227, 36888, 37561, 38238, 38921, 39612, 40313, 41022, 41741, 42468, 43201, 43940, 44683,
    45434, 46191, 46952, 47721, 48494, 49281, 50078, 50887, 51698, 52519, 53342, 54169, 54998,
    55837, 56690, 57547, 58406, 59269, 60146, 61027, 61910, 62797, 63704, 64615, 65534, 66463,
    67400, 68341, 69288, 70241, 71208, 72179, 73156, 74139, 75130, 76127, 77136, 78149, 79168,
    80189, 81220, 82253, 83292, 84341, 85392, 86453, 87516, 88585, 89672, 90763, 91856, 92953,
    94056, 95165, 96282, 97405, 98534, 99685, 100_838, 102_001, 103_172, 104_353, 105_540, 106_733,
    107_934, 109_147, 110_364, 111_587, 112_816, 114_047, 115_284, 116_533, 117_792, 119_069,
    120_348, 121_631, 122_920, 124_211, 125_508, 126_809, 128_112, 129_419, 130_738, 132_059,
    133_386, 134_747, 136_114, 137_487, 138_868, 140_267, 141_676, 143_099, 144_526, 145_955,
    147_388, 148_827, 150_274, 151_725, 153_178, 154_637, 156_108, 157_589, 159_072, 160_559,
    162_048, 163_541, 165_040, 166_551, 168_074, 169_605, 171_148, 172_697, 174_250, 175_809,
    177_376, 178_947, 180_526, 182_109, 183_706, 185_307, 186_914, 188_523, 190_136, 191_755,
    193_376, 195_003, 196_640, 198_297, 199_960, 201_627, 203_296, 204_989, 206_686, 208_385,
    210_094, 211_815, 213_538, 215_271, 217_012, 218_759, 220_512, 222_271, 224_048, 225_831,
    227_618, 229_407, 231_208, 233_019, 234_842, 236_673, 238_520, 240_381, 242_248, 244_119,
    245_992, 247_869, 249_748, 251_637, 253_538, 255_445, 257_358, 259_289, 261_222, 263_171,
    265_122, 267_095, 269_074, 271_061, 273_054, 275_051, 277_050, 279_053, 281_064, 283_081,
    285_108, 287_137, 289_176, 291_229, 293_292, 295_361, 297_442, 299_525, 301_612, 303_701,
    305_800, 307_911, 310_024, 312_153, 314_284, 316_421, 318_562, 320_705, 322_858, 325_019,
    327_198, 329_401, 331_608, 333_821, 336_042, 338_279, 340_518, 342_761, 345_012, 347_279,
    349_548, 351_821, 354_102, 356_389, 358_682, 360_979, 363_288, 365_599, 367_932, 370_271,
    372_612, 374_959, 377_310, 379_667, 382_038, 384_415, 386_796, 389_179, 391_568, 393_961,
    396_360, 398_771, 401_188, 403_611, 406_048, 408_489, 410_936, 413_395, 415_862, 418_335,
    420_812, 423_315, 425_836, 428_367, 430_906, 433_449, 435_998, 438_549, 441_106, 443_685,
    446_276, 448_869, 451_478, 454_095, 456_716, 459_349, 461_996, 464_653, 467_312, 469_975,
    472_646, 475_323, 478_006, 480_693, 483_382, 486_075, 488_774, 491_481, 494_192, 496_905,
    499_624, 502_353, 505_084, 507_825, 510_574, 513_327, 516_094, 518_871, 521_660, 524_451,
    527_248, 530_049, 532_852, 535_671, 538_504, 541_341, 544_184, 547_035, 549_892, 552_753,
    555_632, 558_519, 561_416, 564_319, 567_228, 570_145, 573_072, 576_011, 578_964, 581_921,
    584_884, 587_853, 590_824, 593_823, 596_824, 599_835, 602_854, 605_877, 608_914, 611_955,
    615_004, 618_065, 621_132, 624_211, 627_294, 630_383, 633_492, 636_611, 639_732, 642_869,
    646_032, 649_199, 652_368, 655_549, 658_736, 661_927, 665_130, 668_339, 671_556, 674_777,
    678_006, 681_257, 684_510, 687_767, 691_026, 694_297, 697_596, 700_897, 704_204, 707_517,
    710_836, 714_159, 717_488, 720_819, 724_162, 727_509, 730_868, 734_229, 737_600, 740_973,
    744_362, 747_753, 751_160, 754_573, 758_006, 761_455, 764_912, 768_373, 771_836, 775_303,
    778_772, 782_263, 785_762, 789_273, 792_790, 796_317, 799_846, 803_379, 806_918, 810_459,
    814_006, 817_563, 821_122, 824_693, 828_274, 831_857, 835_450, 839_057, 842_670, 846_287,
    849_910, 853_541, 857_178, 860_821, 864_480, 868_151, 871_824, 875_501, 879_192, 882_889,
    886_590, 890_299, 894_018, 897_745, 901_478, 905_217, 908_978, 912_745, 916_514, 920_293,
    924_086, 927_883, 931_686, 935_507, 939_330, 943_163, 947_010, 950_861, 954_714, 958_577,
    962_454, 966_335, 970_224, 974_131, 978_042, 981_959, 985_878, 989_801, 993_730, 997_661,
    100_1604, 100_5551, 100_9518, 101_3507, 101_7508, 102_1511, 102_5518, 102_9531, 103_3550,
    103_7571, 104_1598, 104_5647, 104_9698, 105_3755, 105_7828, 106_1907, 106_5998, 107_0091,
    107_4190, 107_8301, 108_2428, 108_6557, 109_0690, 109_4829, 109_8982, 110_3139, 110_7298,
    111_1475, 111_5676, 111_9887, 112_4104, 112_8323, 113_2552, 113_6783, 114_1024, 114_5267,
    114_9520, 115_3779, 115_8040, 116_2311, 116_6584, 117_0867, 117_5156, 117_9453, 118_3780,
    118_8117, 119_2456, 119_6805, 120_1162, 120_5525, 120_9898, 121_4289, 121_8686, 122_3095,
    122_7516, 123_1939, 123_6380, 124_0827, 124_5278, 124_9735, 125_4198, 125_8679, 126_3162,
    126_7655, 127_2162, 127_6675, 128_1192, 128_5711, 129_0234, 129_4781, 129_9330, 130_3891,
    130_8458, 131_3041, 131_7632, 132_2229, 132_6832, 133_1453, 133_6090, 134_0729, 134_5372,
    135_0021, 135_4672, 135_9329, 136_3992, 136_8665, 137_3344, 137_8035, 138_2738, 138_7459,
    139_2182, 139_6911, 140_1644, 140_6395, 141_1154, 141_5937, 142_0724, 142_5513, 143_0306,
    143_5105, 143_9906, 144_4719, 144_9536, 145_4367, 145_9228, 146_4099, 146_8976, 147_3865,
    147_8768, 148_3677, 148_8596, 149_3527, 149_8460, 150_3397, 150_8340, 151_3291, 151_8248,
    152_3215, 152_8184, 153_3157, 153_8144, 154_3137, 154_8136, 155_3139, 155_8148, 156_3159,
    156_8180, 157_3203, 157_8242, 158_3293, 158_8352, 159_3429, 159_8510, 160_3597, 160_8696,
    161_3797, 161_8904, 162_4017, 162_9136, 163_4283, 163_9436, 164_4603, 164_9774, 165_4953,
    166_0142, 166_5339, 167_0548, 167_5775, 168_1006, 168_6239, 169_1476, 169_6737, 170_2010,
    170_7289, 171_2570, 171_7867, 172_3170, 172_8479, 173_3802, 173_9135, 174_4482, 174_9833,
    175_5214, 176_0601, 176_5994, 177_1393, 177_6800, 178_2213, 178_7630, 179_3049, 179_8480,
    180_3917, 180_9358, 181_4801, 182_0250, 182_5721, 183_1198, 183_6677, 184_2160, 184_7661,
    185_3164, 185_8671, 186_4190, 186_9711, 187_5238, 188_0769, 188_6326, 189_1889, 189_7458,
    190_3031, 190_8612, 191_4203, 191_9826, 192_5465, 193_1106, 193_6753, 194_2404, 194_8057,
    195_3714, 195_9373, 196_5042, 197_0725, 197_6414, 198_2107, 198_7808, 199_3519, 199_9236,
    200_4973, 201_0714, 201_6457, 202_2206, 202_7985, 203_3768, 203_9559, 204_5360, 205_1167,
    205_6980, 206_2801, 206_8628, 207_4467, 208_0310, 208_6159, 209_2010, 209_7867, 210_3728,
    210_9595, 211_5464, 212_1343, 212_7224, 213_3121, 213_9024, 214_4947, 215_0874, 215_6813,
    216_2766, 216_8747, 217_4734, 218_0741, 218_6752, 219_2781, 219_8818, 220_4861, 221_0908,
    221_6961, 222_3028, 222_9101, 223_5180, 224_1269, 224_7360, 225_3461, 225_9574, 226_5695,
    227_1826, 227_7959, 228_4102, 229_0253, 229_6416, 230_2589, 230_8786, 231_4985, 232_1188,
    232_7399, 233_3616, 233_9837, 234_6066, 235_2313, 235_8570, 236_4833, 237_1102, 237_7373,
    238_3650, 238_9937, 239_6236, 240_2537, 240_8848, 241_5165, 242_1488, 242_7817, 243_4154,
    244_0497, 244_6850, 245_3209, 245_9570, 246_5937, 247_2310, 247_8689, 248_5078, 249_1475,
    249_7896, 250_4323, 251_0772, 251_7223, 252_3692, 253_0165, 253_6646, 254_3137, 254_9658,
    255_6187, 256_2734, 256_9285, 257_5838, 258_2401, 258_8970, 259_5541, 260_2118, 260_8699,
    261_5298, 262_1905, 262_8524, 263_5161, 264_1814, 264_8473, 265_5134, 266_1807, 266_8486,
    267_5175, 268_1866, 268_8567, 269_5270, 270_1979, 270_8698, 271_5431, 272_2168, 272_8929,
    273_5692, 274_2471, 274_9252, 275_6043, 276_2836, 276_9639, 277_6462, 278_3289, 279_0118,
    279_6951, 280_3792, 281_0649, 281_7512, 282_4381, 283_1252, 283_8135, 284_5034, 285_1941,
    285_8852, 286_5769, 287_2716, 287_9665, 288_6624, 289_3585, 290_0552, 290_7523, 291_4500,
    292_1483, 292_8474, 293_5471, 294_2472, 294_9485, 295_6504, 296_3531, 297_0570, 297_7613,
    298_4670, 299_1739, 299_8818, 300_5921, 301_3030, 302_0151, 302_7278, 303_4407, 304_1558,
    304_8717, 305_5894, 306_3081, 307_0274, 307_7481, 308_4692, 309_1905, 309_9124, 310_6353,
    311_3590, 312_0833, 312_8080, 313_5333, 314_2616, 314_9913, 315_7220, 316_4529, 317_1850,
    317_9181, 318_6514, 319_3863, 320_1214, 320_8583, 321_5976, 322_3387, 323_0804, 323_8237,
    324_5688, 325_3145, 326_0604, 326_8081, 327_5562, 328_3049, 329_0538, 329_8037, 330_5544,
    331_3061, 332_0584, 332_8113, 333_5650, 334_3191, 335_0738, 335_8287, 336_5846, 337_3407,
    338_0980, 338_8557, 339_6140, 340_3729, 341_1320, 341_8923, 342_6530, 343_4151, 344_1790,
    344_9433, 345_7082, 346_4751, 347_2424, 348_0105, 348_7792, 349_5483, 350_3182, 351_0885,
    351_8602, 352_6325, 353_4052, 354_1793, 354_9546, 355_7303, 356_5062, 357_2851, 358_0644,
    358_8461, 359_6284, 360_4113, 361_1954, 361_9807, 362_7674, 363_5547, 364_3424, 365_1303,
    365_9186, 366_7087, 367_4994,
];