use rand::{
    distributions::{Standard, Uniform},
    prelude::{Distribution, SliceRandom, ThreadRng},
    thread_rng, Rng,
};
 

use crate::Point2i;

use self::custom_rng::CustomRng;

pub mod custom_rng {
    use hexf::hexf32;

    pub const FLOAT_ONE_MINUS_EPSILON: f64 = hexf32!("0x1.fffffep-1") as f64;

    pub const PCG32_DEFAULT_STATE: u64 = 0x853c_49e6_748f_ea9b;
    pub const PCG32_DEFAULT_STREAM: u64 = 0xda3e_39cb_94b9_5bdb;
    pub const PCG32_MULT: u64 = 0x5851_f42d_4c95_7f2d;
    #[derive(Debug, Default, Copy, Clone)]
    pub struct CustomRng {
        state: u64,
        inc: u64,
    }
    impl CustomRng {
        pub fn new() -> Self {
            CustomRng {
                state: PCG32_DEFAULT_STATE,
                inc: PCG32_DEFAULT_STREAM,
            }
        }
        pub fn set_sequence(&mut self, initseq: u64) {
            self.state = 0_u64;
            let (shl, _overflow) = initseq.overflowing_shl(1);
            self.inc = shl | 1;
            self.uniform_uint32();
            let (add, _overflow) = self.state.overflowing_add(PCG32_DEFAULT_STATE);
            self.state = add;
            self.uniform_uint32();
        }
        pub fn uniform_uint32(&mut self) -> u32 {
            let oldstate: u64 = self.state;
            // C++: state = oldstate * PCG32_MULT + inc;
            let (mul, _overflow) = oldstate.overflowing_mul(PCG32_MULT);
            let (add, _overflow) = mul.overflowing_add(self.inc);
            self.state = add;
            // C++: uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
            let (shr, _overflow) = oldstate.overflowing_shr(18);
            let combine = shr ^ oldstate;
            let (shr, _overflow) = combine.overflowing_shr(27);
            let xorshifted: u32 = shr as u32;
            // C++: uint32_t rot = (uint32_t)(oldstate >> 59u);
            let (shr, _overflow) = oldstate.overflowing_shr(59);
            let rot: u32 = shr as u32;
            // C++: return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
            let (shr, _overflow) = xorshifted.overflowing_shr(rot);
            // bitwise not in Rust is ! (not the ~ operator like in C)
            let neg = !rot;
            let (add, _overflow) = neg.overflowing_add(1_u32);
            let (shl, _overflow) = xorshifted.overflowing_shl(add & 31);
            shr | shl
        }
        pub fn uniform_uint32_bounded(&mut self, b: u32) -> u32 {
            // bitwise not in Rust is ! (not the ~ operator like in C)
            let threshold = (!b + 1) & b;
            loop {
                let r = self.uniform_uint32();
                if r >= threshold {
                    return r % b;
                }
            }
        }
        pub fn uniform_float(&mut self) -> f64 {
            //#ifndef PBRT_HAVE_HEX_FP_CONSTANTS
            // (self.uniform_uint32() as Float * 2.3283064365386963e-10 as Float)
            //     .min(FLOAT_ONE_MINUS_EPSILON)
            //#else
            (self.uniform_uint32() as f64 * hexf32!("0x1.0p-32") as f64)
                .min(FLOAT_ONE_MINUS_EPSILON)
            //#endif
        }
    }
} // end of  mod test_rng

pub trait Sampler {
    fn has_samples(&self) -> bool;
   // fn next_pixel_samples(&mut self, samples: &mut Vec<(f64, f64)>);
    // fn get_sample2d(&mut self, samples: &mut Vec<(f64, f64)>, rng: &mut ThreadRng);
    // fn get_sample1d(&mut self, samples: &mut Vec<f64>, rng: &mut ThreadRng);
    fn get2d(&mut self) -> (f64, f64) ;
    fn get1d(&mut self) ->  f64 ;
    fn get_current_pixel(&self) -> (u32, u32);
    fn get_spp(&self) -> u32;

    fn start_pixel(&mut self, p: Point2i);
    fn start_next_sample(&mut self) -> bool;
}
pub trait Get2D{ 
    fn get2d(&mut self)->(f64,f64);
}
pub trait Get1D{ 
    fn get1d(&mut self)->f64;
}



#[derive(Debug, Clone)]
pub struct SamplerUniform {
    // xmin, ymin, xmax, ymax
    pub bounds: ((u32, u32), (u32, u32)),
    pub current_pixel: (u32, u32),
    pub distribution: Uniform<f64>,
    current_index_sample: i64,
    current_offset_sample :u32,
    sample_pixel_center : bool,
    rng : Option<Box<CustomRng>>,
    pub spp: u32,
}
impl SamplerUniform {
    pub fn new(region: ((u32, u32), (u32, u32)),spp: u32, sample_pixel_center : bool, seed : Option<u64>) -> SamplerUniform {

        match seed {
            Some(seednumber)=>{
                let mut custom_rng = custom_rng::CustomRng::new();
                custom_rng.set_sequence(seed.unwrap());
                
                SamplerUniform {
                    sample_pixel_center ,
                    bounds: region,
                    distribution: Uniform::new(0.0, 1.0),
                    current_pixel: (region.0 .0, region.0 .1),
                    current_index_sample: 0_i64,
                    current_offset_sample :0_u32,
                    spp,
                    rng : Some(Box::new( custom_rng))
                }
            }
            None => {
                SamplerUniform {
                    sample_pixel_center ,
                    bounds: region,
                    distribution: Uniform::new(0.0, 1.0),
                    current_pixel: (region.0 .0, region.0 .1),
                    current_index_sample: 0_i64,
                    current_offset_sample :0_u32,
                    spp,
                    rng : None
                }
            }
        }
       
    }
    pub fn start_pixel(&mut self, p: Point2i) {
        self.current_pixel = (p.x as u32, p.y as u32);
        self.current_index_sample = 0_i64; // comienzo un nuevo sample
        self.current_offset_sample=1;
    }
    pub fn has_samples(&self) -> bool {
        self.bounds.1 .1 != self.current_pixel.1
    }
    pub fn start_next_sample(&mut self) -> bool {
        let ret = if self.current_index_sample < (self.spp as i64) -1  {
            self.current_index_sample += 1;
            // esto es un flag que avisa que es el primer sample , el de la camara
            self.current_offset_sample=1; 
           return  true;
        } else {
           return  false;
        };
        ret
    }

    pub fn next_pixel_samples(&mut self, samples: &mut Vec<(f64, f64)>) {
        samples.clear();
        if self.sample_pixel_center {
            samples.push((
                self.current_pixel.0 as f64 + 0.5,
                self.current_pixel.1 as f64 + 0.5,
            ));
        }else{
            let mut vreqsamples: Vec<(f64, f64)> = vec![(0.0, 0.0); 1];
            self.inner_get_sample2d(&mut vreqsamples, &mut rand::thread_rng());
            samples.push((
                self.current_pixel.0 as f64 + vreqsamples[0].0,
                self.current_pixel.1 as f64 + vreqsamples[0].1,
            ));
        }
        
         
        
    }
    pub fn get1d(&mut self)->f64{
        let mut vreqsamples: Vec<f64> = vec![ 0.0; 1];
        self.inner_get_sample1d(&mut vreqsamples,&mut rand::thread_rng());
        vreqsamples[0]
    }

    pub fn get2d(&mut self) -> (f64, f64) {
        let mut vreqsamples: Vec<(f64, f64)> = vec![(0.0, 0.0); 1];
        // && self.sample_pixel_center
        if self.current_offset_sample ==1 {
            // se ha llamado start pixel, tengo que retornar pixel coords
            self.next_pixel_samples(&mut vreqsamples);
            // self.current_index_sample += 2;
            self.current_offset_sample=0;
            vreqsamples[0]
        }  else {
            self.inner_get_sample2d(&mut vreqsamples, &mut rand::thread_rng());
            // self.current_index_sample += 2;
            vreqsamples[0]
        }
    }
     fn inner_get_sample2d(&mut self, samples: &mut Vec<(f64, f64)>, rng: &mut ThreadRng) {
        if let Some( custom_rng) =  &mut self.rng {
            for s in samples.iter_mut() {
                s.0 =custom_rng.uniform_float();
                s.1 =custom_rng.uniform_float();
            }
            
        }else{
            let mut iter = self.distribution.sample_iter(rng);
            for s in samples.iter_mut() {
                s.0 = iter.next().unwrap();
                s.1 = iter.next().unwrap();
            }
        }
        
     
        
        
    }
     fn inner_get_sample1d(&mut self, samples: &mut Vec<f64>, rng: &mut ThreadRng) {
        if let Some( custom_rng) =  &mut self.rng {
            for s in samples.iter_mut() {
                *s=  custom_rng.uniform_float();
            }
            
        }else{
            let mut iter = self.distribution.sample_iter(rng);
            for mut s in samples.iter_mut() {
                *s = iter.next().unwrap();
            }
        }
       
    }
    pub fn get_current_pixel(&self) -> (u32, u32) {
        self.current_pixel
    }
    pub fn get_spp(&self) -> u32 {
        1
    }
}
 
impl Sampler for SamplerUniform {
    fn start_pixel(&mut self, p: Point2i) {
        self.start_pixel(p)
    }
    fn start_next_sample(&mut self) -> bool {
       self.start_next_sample()
    }
    fn has_samples(&self) -> bool {
        self.has_samples()
    }

    // fn next_pixel_samples(&mut self, samples: &mut Vec<(f64, f64)>) {
    //     self.next_pixel_samples(samples)
    // }
 fn get1d(&mut self) ->  f64 {
    self.get1d()
 }
 fn get2d(&mut self) -> (f64, f64) {
    self.get2d()
 }
    // fn get_sample2d(&mut self, samples: &mut Vec<(f64, f64)>, rng: &mut ThreadRng) {
    //     self.get_sample2d(samples, rng)
    // }

    // fn get_sample1d(&mut self, samples: &mut Vec<f64>, rng: &mut ThreadRng) {
    //     self.get_sample1d(samples, rng)
    // }

    fn get_current_pixel(&self) -> (u32, u32) {
        self.get_current_pixel()
    }

    fn get_spp(&self) -> u32 {
        self.get_spp()
    }
}













pub struct SamplerLd2 {
    pub bounds: ((u32, u32), (u32, u32)),
    pub current_pixel: (u32, u32),
    pub current_index_sample: i64,
    current_offset_sample :u32,
    pub distribution: Uniform<u32>,
    sample_pixel_center : bool,
    pub spp: u32,
}
impl SamplerLd2 {
    fn sample_02_2d(samples: &mut Vec<(f64, f64)>, scramble: (u32, u32), offset: u32) {
        for s in samples.iter_mut().enumerate() {
            let t = Self::sample_02(s.0 as u32 + offset, scramble);
            *s.1 = (t.0 as f64, t.1 as f64);
        }
    }
    fn sample_van_der_corput_1d(samples: &mut [f64], scramble: u32, offset: u32) {
        for s in samples.iter_mut().enumerate() {
            let t = Self::van_der_corput(s.0 as u32 + offset, scramble);
            *s.1 = t as f64;
        }
    }
    fn sample_02(n: u32, scramble: (u32, u32)) -> (f32, f32) {
        (
            Self::van_der_corput(n, scramble.0),
            Self::sobol(n, scramble.1),
        )
    }
    fn van_der_corput(mut n: u32, scramble: u32) -> f32 {
        n = (n << 16) | (n >> 16);
        n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
        n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
        n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
        n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
        n ^= scramble;
        f32::min(
            ((n >> 8) & 0xffffff) as f32 / ((1 << 24) as f32),
            1.0 - f32::EPSILON,
        )
    }

    fn sobol(mut n: u32, mut scramble: u32) -> f32 {
        let mut i = 1 << 31;
        while n != 0 {
            if n & 0x1 != 0 {
                scramble ^= i;
            }
            n >>= 1;
            i ^= i >> 1;
        }
        f32::min(
            ((scramble >> 8) & 0xffffff) as f32 / ((1 << 24) as f32),
            1.0 - f32::EPSILON,
        )
    }

    pub fn has_samples(&self) -> bool {
        self.bounds.1 .1 != self.current_pixel.1
    }
    pub fn new(region: ((u32, u32), (u32, u32)), spp: u32,  sample_pixel_center : bool) -> Self {
        SamplerLd2 {
            bounds: region,
            current_index_sample: 0_i64,
            current_offset_sample :0_u32,
            distribution: Uniform::new(0, std::u32::MAX),
            current_pixel: (region.0 .0, region.0 .1),
            spp,
             sample_pixel_center
        }
    }
    pub fn start_pixel(&mut self, p: Point2i) {
    
         self. current_pixel = (p.x as u32, p.y as u32);
        self.current_index_sample = 0_i64; // comienzo un nuevo sample
        self.current_offset_sample=1;
    }
    pub fn start_next_sample(&mut self) -> bool {
        let ret = if self.current_index_sample < (self.spp as i64) -1  {
            self.current_index_sample += 1;
            self.current_offset_sample=1;
           return  true;
        } else {
           return  false;
        };
        ret
    }
    pub fn next_pixel_samples(&mut self, samples: &mut Vec<(f64, f64)>) {
        if samples.len() != 1 as usize {
            panic!("");
        }
        // samples.clear();

        if self.sample_pixel_center {
            //self.inner_get_sample2d(samples, &mut rand::thread_rng());
            for s in samples.iter_mut() {
                s.0 += self.current_pixel.0 as f64 + 0.5;
                s.1 += self.current_pixel.1 as f64+ 0.5;
            }
        }else{
            self.inner_get_sample2d(samples, &mut rand::thread_rng());
            for s in samples.iter_mut() {
                s.0 += self.current_pixel.0 as f64;
                s.1 += self.current_pixel.1 as f64;
            }
        }
        
        self.current_offset_sample = 0;
        
        
        
    }
    // si el  self.current_index_sample==0  significa que hemos llamado a start_pixel();
    //
    pub fn get2d(&mut self) -> (f64, f64) {
        let mut vreqsamples: Vec<(f64, f64)> = vec![(0.0, 0.0); 1];
        if self.current_offset_sample == 1 {
            // se ha llamado start pixel, tengo que retornar pixel coords
            self.next_pixel_samples(&mut vreqsamples);
            // self.current_index_sample += 2;
            vreqsamples[0]
        } else {
            self.inner_get_sample2d(&mut vreqsamples, &mut rand::thread_rng());
            // self.current_index_sample += 2;
            vreqsamples[0]
        }
    }
    pub fn get1d(&mut self) -> f64 {
        let mut vreqsamples: Vec<f64> = vec![0.0; 1];
        self.inner_get_sample1d(&mut vreqsamples, &mut rand::thread_rng());
        // self.current_index_sample += 1;
        vreqsamples[0]
    }
    pub fn inner_get_sample2d(&mut self, samples: &mut Vec<(f64, f64)>, _: &mut ThreadRng) {
        let mut iter = self.distribution.sample_iter(rand::thread_rng());
        let scrambled = (iter.next().unwrap(), iter.next().unwrap());
        Self::sample_02_2d(samples, scrambled, 0);
        samples.shuffle(&mut rand::thread_rng());
    }
    pub fn inner_get_sample1d(&mut self, samples: &mut Vec<f64>, _: &mut ThreadRng) {
        let mut iter = self.distribution.sample_iter(rand::thread_rng());
        let scrambled = iter.next().unwrap();
        Self::sample_van_der_corput_1d(samples, scrambled, 0);
        samples.shuffle(&mut rand::thread_rng());
    }
    pub fn get_spp(&self) -> u32 {
        self.spp
    }
    pub fn get_current_pixel(&self) -> (u32, u32) {
        self.current_pixel
    }
}
// tengo que cambiar el interface sampler,
//     +añado 
//         +start_pixel()
//         +start_next_sample()
 
impl Sampler for SamplerLd2 {
    fn start_pixel(&mut self, p: Point2i) {
           self.start_pixel(p)
    }
    fn start_next_sample(&mut self) -> bool {
        self.start_next_sample()
    }
    fn has_samples(&self) -> bool {
        self.has_samples()
    }

    // fn next_pixel_samples(&mut self, samples: &mut Vec<(f64, f64)>) {
    //     self.next_pixel_samples(samples)
    // }
fn get1d(&mut self) ->  f64 {
    self.get1d( )
}
fn get2d(&mut self) -> (f64, f64) {
    self.get2d()
}
    // fn get_sample2d(&mut self, samples: &mut Vec<(f64, f64)>, rng: &mut ThreadRng) {
    //     self.get2d( );
    // }

    // fn get_sample1d(&mut self, samples: &mut Vec<f64>, rng: &mut ThreadRng) {
    //     self.get1d( );
    // }

    fn get_current_pixel(&self) -> (u32, u32) {
        self.get_current_pixel()
    }

    fn get_spp(&self) -> u32 {
        self.get_spp()
    }
}

pub struct SamplerLd {
    pub bounds: ((u32, u32), (u32, u32)),
    pub current_pixel: (u32, u32),
    // pub current_index_sample : i64,
    pub distribution: Uniform<u32>,

    pub spp: u32,
}
impl SamplerLd {
    fn sample_02_2d(samples: &mut Vec<(f64, f64)>, scramble: (u32, u32), offset: u32) {
        for s in samples.iter_mut().enumerate() {
            let t = Self::sample_02(s.0 as u32 + offset, scramble);
            *s.1 = (t.0 as f64, t.1 as f64);
        }
    }
    fn sample_van_der_corput_1d(samples: &mut [f64], scramble: u32, offset: u32) {
        for s in samples.iter_mut().enumerate() {
            let t = Self::van_der_corput(s.0 as u32 + offset, scramble);
            *s.1 = t as f64;
        }
    }
    fn sample_02(n: u32, scramble: (u32, u32)) -> (f32, f32) {
        (
            Self::van_der_corput(n, scramble.0),
            Self::sobol(n, scramble.1),
        )
    }
    fn van_der_corput(mut n: u32, scramble: u32) -> f32 {
        n = (n << 16) | (n >> 16);
        n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8);
        n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4);
        n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2);
        n = ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1);
        n ^= scramble;
        f32::min(
            ((n >> 8) & 0xffffff) as f32 / ((1 << 24) as f32),
            1.0 - f32::EPSILON,
        )
    }

    fn sobol(mut n: u32, mut scramble: u32) -> f32 {
        let mut i = 1 << 31;
        while n != 0 {
            if n & 0x1 != 0 {
                scramble ^= i;
            }
            n >>= 1;
            i ^= i >> 1;
        }
        f32::min(
            ((scramble >> 8) & 0xffffff) as f32 / ((1 << 24) as f32),
            1.0 - f32::EPSILON,
        )
    }

    pub fn has_samples(&self) -> bool {
        self.bounds.1 .1 != self.current_pixel.1
    }
    pub fn new(region: ((u32, u32), (u32, u32)), spp: u32) -> SamplerLd {
        SamplerLd {
            bounds: region,
            // current_index_sample : 0_i64,
            distribution: Uniform::new(0, std::u32::MAX),
            current_pixel: (region.0 .0, region.0 .1),
            spp,
        }
    }

    pub fn next_pixel_samples(&mut self, samples: &mut Vec<(f64, f64)>) {
        if samples.len() != self.spp as usize {
            panic!("");
        }
        // samples.clear();

        self.inner_get_sample2d(samples, &mut rand::thread_rng());
        for s in samples.iter_mut() {
            s.0 += self.current_pixel.0 as f64;
            s.1 += self.current_pixel.1 as f64;
        }
        //
        self.current_pixel.0 += 1;
        if self.current_pixel.0 == self.bounds.1 .0 {
            self.current_pixel.0 = self.bounds.0 .0;
            self.current_pixel.1 += 1;
        }
    }
    pub fn  inner_get_sample2d(&mut self, samples: &mut Vec<(f64, f64)>, _: &mut ThreadRng) {
        let mut iter = self.distribution.sample_iter(rand::thread_rng());
        let scrambled = (iter.next().unwrap(), iter.next().unwrap());
        Self::sample_02_2d(samples, scrambled, 0);
        samples.shuffle(&mut rand::thread_rng());
    }
    pub fn inner_get_sample1d(&mut self, samples: &mut Vec<f64>, _: &mut ThreadRng) {
        let mut iter = self.distribution.sample_iter(rand::thread_rng());
        let scrambled = iter.next().unwrap();
        Self::sample_van_der_corput_1d(samples, scrambled, 0);
        samples.shuffle(&mut rand::thread_rng());
    }
    pub fn get_spp(&self) -> u32 {
        self.spp
    }
    pub fn get_current_pixel(&self) -> (u32, u32) {
        self.current_pixel
    }
}

impl Sampler for SamplerLd {
    fn start_pixel(&mut self, p: Point2i) {
        //     self.start_pixel(p)
    }
    fn start_next_sample(&mut self) -> bool {
        true
    }
    fn has_samples(&self) -> bool {
        self.has_samples()
    }

    // fn next_pixel_samples(&mut self, samples: &mut Vec<(f64, f64)>) {
    //     self.next_pixel_samples(samples)
    // }

    // fn get_sample2d(&mut self, samples: &mut Vec<(f64, f64)>, rng: &mut ThreadRng) {
    //     self.get_sample2d(samples, rng)
    // }

    // fn get_sample1d(&mut self, samples: &mut Vec<f64>, rng: &mut ThreadRng) {
    //     self.get_sample1d(samples, rng)
    // }
    fn get1d(&mut self) ->  f64 {
        let mut vreqsamples: Vec<f64> = vec![0.0; 1];
       self.inner_get_sample1d(&mut vreqsamples, &mut rand::thread_rng());
      
           vreqsamples[0]
    }
    fn get2d(&mut self) -> (f64, f64) {
        let mut vreqsamples: Vec<(f64, f64)> = vec![(0.0, 0.0); 1];
        self.inner_get_sample2d(&mut vreqsamples, &mut rand::thread_rng());
        vreqsamples[0]

    }

 
    fn get_current_pixel(&self) -> (u32, u32) {
        self.get_current_pixel()
    }

    fn get_spp(&self) -> u32 {
        self.get_spp()
    }
}
