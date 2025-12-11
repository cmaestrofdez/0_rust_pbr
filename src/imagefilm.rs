#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unused_parens)]
use std::{process::Output, rc::Rc, sync::mpsc::Receiver, fs::File, borrow::BorrowMut};

use actix_web::test::init_service;
use cgmath::{Point3, Vector3, InnerSpace, Decomposed, Quaternion, Transform, Point2};
use image::{codecs::png::PngEncoder, ImageEncoder};
use num_traits::Float;
use palette::Srgb;
use rand::Rng;
use crate::{primitives::{
    prims::{Ray, HitRecord, self, IntersectionOclussion}, 
    Plane, PrimitiveIntersection, Disk}, materials::{
    
        SampleIllumination, MaterialDescType, Plastic, MicroFacetReflection, FrameShading, MicroFaceDistribution, Fresnel, FresnelNop, BeckmannDistribution, TrowbridgeReitzDistribution, BsdfType, Fr, Pdf, RecordSampleIn, AbsCosTheta, SphPhi, SphTheta, SphericalCoords},

        Metal, raytracerv2::{Scene, interset_scene},
        raytracerv2::{Camera, powerHeuristic, clamp}, integrator::updateL, assert_delta, Bounds2f, Point2f
        };


pub trait FilterEval{
    fn radius(&self)->(f64, f64);
    fn eval(&self, xy:&(f64, f64))->f64;
    fn inv_radius(&self) ->(f64, f64) ;
}


#[derive(
    Debug,
    Clone,
    
    // Deserialize, Serialize
)]
pub 
enum  FilterFilm{
    FilterFilmBoxType(FilterFilmBox),
    FilterFilmTriangleType(FilterFilmTriangle),
    FilterFilmLanczosType(FilterFilmLanczos),
    FilterFilmGaussType(FilterFilmGauss),
    FilterFilmNoneType ,
    
}

impl FilterEval for FilterFilm{
     fn eval(&self,xy:&(f64, f64))->f64{
         match *self{
            FilterFilm:: FilterFilmBoxType(f)=>f.eval(xy),
            FilterFilm:: FilterFilmTriangleType(f)=>f.eval(xy),
            FilterFilm:: FilterFilmLanczosType(f)=>f.eval(xy),
            FilterFilm:: FilterFilmGaussType(f)=>f.eval(xy),
            FilterFilm::FilterFilmNoneType=>panic!("unknown impletacion of filter!--->!!!"),
          
         }
    }
     fn radius(&self) ->(f64, f64) {
        match *self{
            FilterFilm:: FilterFilmBoxType(f)=>f.radius(),
            FilterFilm:: FilterFilmTriangleType(f)=>f.radius(),
            FilterFilm::FilterFilmLanczosType(f)=>f.radius(),
            FilterFilm:: FilterFilmGaussType(f)=>f.radius(),
            FilterFilm::FilterFilmNoneType=>panic!("unknown impletacion of filter!--->!!!"),
         
        }
     }
     fn inv_radius(&self) ->(f64, f64) {
        match *self{
            FilterFilm:: FilterFilmBoxType(f)=>f.inv_radius(),
             FilterFilm:: FilterFilmTriangleType(f)=>f.inv_radius(),
             FilterFilm::FilterFilmLanczosType(f)=>f.inv_radius(),
             FilterFilm:: FilterFilmGaussType(f)=>f.inv_radius(),
            FilterFilm::FilterFilmNoneType=>panic!("unknown impletacion of filter!--->!!!"),
           
        }
     }
         
    
}






#[derive(
    Debug,
    Clone,
    Copy,  
)]
pub struct FilterFilmBox{
    pub filterRadius : (f64, f64) 
}
impl FilterFilmBox{
   pub fn new(filterRadius:(f64, f64) )->FilterFilmBox{
        FilterFilmBox{filterRadius}
    }
}
impl FilterEval for FilterFilmBox{
     fn eval(&self,xy:&(f64, f64))->f64{
        1.0
    }
     fn radius(&self) ->(f64, f64) {
        self.filterRadius
     }
     fn inv_radius(&self) ->(f64, f64) {
        (1.0/self.filterRadius.0, 1.0/self.filterRadius.1)
     }
}


#[derive(
    Debug,
    Clone,
    Copy,  
)]
pub struct FilterFilmTriangle{
    pub filterRadius : (f64, f64) 
}
impl FilterFilmTriangle{
    fn new(filterRadius:(f64, f64) )->FilterFilmTriangle{
        FilterFilmTriangle{filterRadius}
    }
}
impl FilterEval for FilterFilmTriangle{
     fn eval(&self,xy:&(f64, f64))->f64{
        (self.filterRadius.0 - xy.0.abs()).max(0.0) * (self.filterRadius.1 - xy.1.abs()).max(0.0) 
    }
     fn radius(&self) ->(f64, f64) {
        self.filterRadius
     }
     fn inv_radius(&self) ->(f64, f64) {
        (1.0/self.filterRadius.0, 1.0/self.filterRadius.1)
     }
}




#[derive(
    Debug,
    Clone,
    Copy,  
)]
pub struct FilterFilmLanczos{
    pub filterRadius : (f64, f64) , // 3,3
    pub tau :  f64,                 // 4

}
impl FilterFilmLanczos{
    fn sinc( x : f64)->f64{
        let x = x.abs();
        if x < 1e-5 {
            return 1.0;
        }
        (std::f64::consts::PI * x ).sin() / (std::f64::consts::PI * x) 
    }
    fn lanczos( r : f64, x : f64, tau : f64)->f64{
        let x  = x.abs();
        if x > r{
            return 0.0;
        }
       Self:: sinc(x/tau) * Self::sinc(x)

    }
    fn new(filterRadius:(f64, f64) , tau :f64 )->FilterFilmLanczos{
        FilterFilmLanczos{filterRadius, tau}
    }
}
impl FilterEval for FilterFilmLanczos{
     fn eval(&self,xy:&(f64, f64))->f64{
        Self::lanczos(self.radius().0, xy.0, self.tau)* Self::lanczos(self.radius().1, xy.1, self.tau)
    }
     fn radius(&self) ->(f64, f64) {
        self.filterRadius
     }
     fn inv_radius(&self) ->(f64, f64) {
        (1.0/self.filterRadius.0, 1.0/self.filterRadius.1)
     }
}



#[derive(
    Debug,
    Clone,
    Copy,  
)]
pub struct FilterFilmGauss{
    pub filterRadius : (f64, f64),
    pub factorexp :  (f64, f64), 
    pub a : f64, 
}
impl FilterFilmGauss{
   pub  fn new(filterRadius:(f64, f64) , a : f64)->FilterFilmGauss{
       let factorexp =( (-a * filterRadius.0 * filterRadius.0).exp(),(-a * filterRadius.1 * filterRadius.1).exp()); 
        FilterFilmGauss{filterRadius, factorexp, a} 
    }
}
impl FilterEval for FilterFilmGauss{
     fn eval(&self,xy:&(f64, f64))->f64{
     let fx =   ( ( -self.a * xy.0 * xy.0).exp() - self. factorexp.0).max(0.0);
     let fy =   (  ( -self.a * xy.1 * xy.1).exp() - self. factorexp.1).max(0.0);
        fx * fy
    }
     fn radius(&self) ->(f64, f64) {
        self.filterRadius
     }
     fn inv_radius(&self) ->(f64, f64) {
        (1.0/self.filterRadius.0, 1.0/self.filterRadius.1)
     }
}





#[derive(
    Debug,
    Clone,
    Copy,
    // Deserialize, Serialize
)]
pub struct Pix{
    p : Srgb,
    w : f64,
    stats_hit : u64,
}
impl Pix {
    pub fn value_zero()->Pix{
        Pix{p:Srgb::new(0.0,0.0,0.0), w:0.0, stats_hit:0}
    }
}

#[derive(
    Debug,
    Clone,
    
    // Deserialize, Serialize
)]
pub struct ImgFilm{
     pub  filterRadius : (f64, f64),
     pub pixels : Vec<Pix>,
      pub dims : Bounds2f,
    pub filterTable : Vec<f64>,
    pub filterTableSize : u32,
    pub w : usize,
    pub h :  usize,
    pub filter : FilterFilm,
}
impl ImgFilm{
    fn gamma(x:f32)->f32{ 
        if  x <= 0.0031308  {return 12.92  * x;} 
        1.055  * x.powf(1.0 / 2.4)  - 0.055

    }
    // fn gamma_correction(rgb:Srgb)->Srgb{ 
    //   clamp(Self::gamma(rgb.red)*255.0 + 0.5, 0.0, 255.0)

    // }
    fn toRgb(xyz:Srgb)->Srgb{
        Srgb::new(
         3.240479* xyz.red - 1.537150 * xyz.green - 0.498535 * xyz.blue,
         -0.969256 * xyz.red + 1.875991 * xyz.green + 0.041556 * xyz.blue,
         0.055648 * xyz.red - 0.204043 * xyz.green + 1.057311 * xyz.blue
        )
    }
    fn toXyz(rgb: Srgb)->Srgb{

       
        Srgb::new(
            0.412453 * rgb.red + 0.357580* rgb.green + 0.180423 * rgb.blue,
            0.212671 * rgb.red + 0.715160 * rgb.green + 0.072169 * rgb.blue,
            0.019334 * rgb.red + 0.119193 * rgb.green + 0.950227 * rgb.blue
        )
         

    }
    pub fn updateSample(acc: Srgb , sample :Srgb , weight:f32 )->Srgb{
       Srgb::new( acc.red  + sample.red*weight  ,   acc.green  + sample.green*weight ,   acc.blue  + sample.blue*weight)
    }
    pub fn init_pixelbuffer(pixels : &Vec<Pix>){

    }
    pub fn init_filter_table(table : & mut Vec<f64>, filter : &FilterFilm, filterWidth: usize){
    
        let mut ptr = 0_usize;
        for y in 0..filterWidth{
            for x in 0..filterWidth{
                let psample = (( x as f64 + 0.5  ) * filter.radius().0 / filterWidth as f64,( y as f64 + 0.5  ) * filter.radius().1 / filterWidth as f64);
               let f =  filter.eval(&psample);
               table[ptr]=f;
               ptr+=1;
            }
            
        }

    }
    pub fn from_filter_gauss_std(w : usize, h : usize   )->ImgFilm{
        Self::from_filter_gauss(w, h, (2.0,2.0), 2.0)
    }
    pub fn from_filter_gauss(w : usize, h : usize ,  filterRadius : (f64, f64) , alpha : f64   )->ImgFilm{

        let  filterTableSize = 16;
         
         let  pixels : Vec<Pix> = vec![Pix::value_zero();w*h];
         let mut  filterTable: Vec<f64> = vec![0.0;(filterTableSize*filterTableSize ) as usize];
         let filter  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new(filterRadius, alpha));
           
         Self::init_pixelbuffer(&pixels);
         Self::init_filter_table(& mut filterTable, &filter,filterTableSize as usize);
         
         ImgFilm{pixels , filterTable ,  filterRadius, dims:(Point2f::new(0.0,0.0),Point2f::new(w as f64, h as f64)), w, h ,filter , filterTableSize}

    }
    pub fn from_filter_box(w : usize, h : usize, filterRadius  :(f64,f64) )->ImgFilm{
       let  filterTableSize = 16;
    
        let  pixels : Vec<Pix> = vec![Pix::value_zero();w*h];
        let mut  filterTable: Vec<f64> = vec![0.0;(filterTableSize*filterTableSize ) as usize];
        let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( filterRadius));
          
        Self::init_pixelbuffer(&pixels);
        Self::init_filter_table(& mut filterTable, &filterbox,filterTableSize as usize);
        
        ImgFilm{pixels , filterTable ,  filterRadius, dims:(Point2f::new(0.0,0.0),Point2f::new(w as f64, h as f64)), w, h ,filter : filterbox, filterTableSize}

    }
    pub fn from_filter_triangle(w : usize, h : usize )->ImgFilm{
       let  filterTableSize = 16;
       let  filterRadius = (1.5,1.5);
        let  pixels : Vec<Pix> = vec![Pix::value_zero();w*h];
        let mut  filterTable: Vec<f64> = vec![0.0;(filterTableSize*filterTableSize ) as usize];
        let filter = FilterFilm::FilterFilmTriangleType(FilterFilmTriangle::new(filterRadius));
          
        Self::init_pixelbuffer(&pixels);
        Self::init_filter_table(& mut filterTable, &filter,filterTableSize as usize);
        
        ImgFilm{pixels , filterTable ,  filterRadius, dims:(Point2f::new(0.0,0.0),Point2f::new(w as f64, h as f64)), w, h ,filter , filterTableSize}

    }
    pub fn from_filter_lanczos(w : usize, h : usize )->ImgFilm{
        let  filterTableSize = 16;
        let  filterRadius = (4.0,4.0);
         let  pixels : Vec<Pix> = vec![Pix::value_zero();w*h];
         let mut  filterTable: Vec<f64> = vec![0.0;(filterTableSize*filterTableSize ) as usize];
    
         let filter = FilterFilm::FilterFilmLanczosType(FilterFilmLanczos::new(filterRadius,6.0));
         Self::init_pixelbuffer(&pixels);
         Self::init_filter_table(& mut filterTable, &filter,filterTableSize as usize);
         
         ImgFilm{pixels , filterTable ,  filterRadius, dims:(Point2f::new(0.0,0.0),Point2f::new(w as f64, h as f64)), w, h ,filter , filterTableSize}
 
     }
    pub fn new( w : usize, h : usize ,filter:FilterFilm, filterTableSize:u32)->ImgFilm{
       let  pixels : Vec<Pix> = vec![Pix::value_zero();w*h];
       let mut  filterTable: Vec<f64> = vec![0.0;(filterTableSize*filterTableSize ) as usize];
      
       

       Self::init_pixelbuffer(&pixels);
       Self::init_filter_table(& mut filterTable, &filter,filterTableSize as usize);
       
       ImgFilm{pixels , filterTable , filterRadius: filter.radius(), dims:(Point2f::new(0.0,0.0),Point2f::new(w as f64, h as f64)), w, h ,filter , filterTableSize}
    }                                                                                                            //index dxtable, usize, index dytable, usize
    pub fn offset_filtertable(pixel: f64,p:f64, pmin:f64, filterinv_radius : f64,filterWidth: usize)->(usize,usize ){
        let dtable =      p as i64 - pmin as i64;
        ( dtable as usize, ((p - pixel )* filterinv_radius* (filterWidth as f64)).abs().floor().min((filterWidth-1) as f64) as usize)
      
    }
    pub fn add_sample(&mut self, psample:&Point2f, v : Srgb){
        


      
        let pmed = Point2f::new(0.5, 0.5);
        let px = (psample - pmed);
       let mut pmin =  Point2f::new((px.x- self. filterRadius.0 ).ceil(),(px.y-self. filterRadius.1).ceil()); 
       let  mut pmax =  Point2f::new((px.x+self. filterRadius.0 ).floor()+1.0,(px.y+self. filterRadius.1).floor()+1.0);
       pmin.x = pmin.x.max(self.dims.0.x) ;
        
       pmin.y =  pmin.y.max(self.dims.0.y);
       pmax.x =  pmax.x.min(self.dims.1.x);
      
       pmax.y = pmax.y.min(self.dims.1.y);
       let difxtablesize =  ( pmax.x -pmin.x ) as usize ;
       let difytablesize =  ( pmax.y -pmin.y ) as usize ;
        let mut difxtable = vec![0;difxtablesize];
        let mut difytable = vec![0;difytablesize];
        for ix in pmin.x as usize.. pmax.x as usize{
            let bucketAndValues =  Self::offset_filtertable(px.x, ix as f64, pmin.x,self. filter.inv_radius().0, self.filterTableSize as usize);
            difxtable[bucketAndValues.0] = bucketAndValues.1;
          //  println!("{}, {}", bucketAndValues.0,bucketAndValues.1);
           
         }
         
        for iy in pmin.y as usize.. pmax.y as usize{
            let bucketAndValues =  Self::offset_filtertable(px.y, iy as f64, pmin.y,self. filter.inv_radius().1, self.filterTableSize as usize);
            difytable[bucketAndValues.0] = bucketAndValues.1;
        //    println!("{}, {}", bucketAndValues.0,bucketAndValues.1);
         }

         (&mut self.pixels[psample .y as usize  * self.w + psample .x as usize]).stats_hit +=1;
       for iy in pmin.y as usize.. pmax.y as usize{
        for ix in pmin.x as usize.. pmax.x as usize{ 
                let pix = (&mut self.pixels[iy * self.w + ix]) ; 
                 let idifx = ix - pmin.x as usize;
                 let idify = iy - pmin.y as usize;
                  
                let ioffset  =  difytable[idify] *( self.filterTableSize as usize) + difxtable[idifx];

                let weight = self.filterTable[ioffset];
                pix.p =  Self::updateSample(pix.p, v , weight as f32  );
                pix.w += weight;
        }   
      }

    }
    pub fn commit_and_write(&self, filename : &str )-> std::io::Result<()>{

      

        let bufferres= self.pixels
            .to_owned()
            .into_iter()
            .enumerate()
            .map(|(offset, px)|{
                let rgb = Self::toRgb(Self::toXyz(px.p));
               let p =  vec![(clamp((rgb.red / px.w as f32), 0.0, 1.0)*255.0 ) as u8, (clamp((rgb.green/ px.w as f32), 0.0, 1.0)*255.0) as u8 ,(clamp((rgb.blue/ px.w as f32), 0.0, 1.0)*255.0)as u8];
          //    println!("pixel {} w: {} {} , {} , {}",offset , px.w,  rgb.red / px.w as f32,  rgb.green  / px.w as f32,  rgb.blue / px.w as f32);
                return p;
                }
            )
            .flatten()
            .collect::<Vec<u8>>();
         
       let fd = File::create(filename).unwrap();
        let encoder = PngEncoder::new(fd);
        encoder
            .write_image( bufferres.as_slice(), self.w as u32, self.h as u32, image::ColorType::Rgb8)
            .unwrap();
        println!("render file in :  {:?}",  filename);
        Ok(())


    }
    pub fn log(&self){
        for iy in 0..self.h{
            for ix  in 0..self.w{
                let px = self.pixels[iy*self.w+ix];
              
               // let finalColor  =  vec![px.p.red / px.w as f32 , px.p.green / px.w as f32 , px.p.blue / px.w as f32];
               println!(" pix ({:?},{:?})  raw : {:?} w: {} ", ix, iy,Self::toXyz(px.p), px.w );
            }
        }
    }
}


