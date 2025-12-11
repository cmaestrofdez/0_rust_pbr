use core::fmt;
use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    ops::{Add, SubAssign, AddAssign, MulAssign, DivAssign, RemAssign},
    sync::{
        atomic::{AtomicI64, Ordering},
        Mutex,
    },
    time::Instant, fmt::{Display, Debug},
};

use crate::{
    primitives::{prims::Ray, Bounds3f, PrimitiveType,prims::{HitRecord }},
    sampler::{Sampler, SamplerUniform , SamplerType, SeedInstance},
    samplerhalton::SamplerHalton, Point2i, Cameras::PerspectiveCamera, 
    threapoolexamples1::pararell::{PixelIterator, FilmTile}, 
    threapoolexamples::{init_tiles, init_tiles_withworkers}, 
    imagefilm::FilterFilmGauss, Scene, scene::Scene1, integrator, 
    Lights::{IsAreaLight, IsAmbientLight, GetShape, GetEmission, GetShape1}, raytracerv2::{to_map, lerp, self}, Spheref64, volumentricpathtracer::volumetric::MediumOutsideInside, Point3i
};
use crate::{
    imagefilm::{FilterFilm, FilterFilmBox, ImgFilm, Pix},
    materials,
    primitives::Cylinder,
    raytracerv2::interset_scene,
 
    Bounds2f, Point2f, Point3f, Vector3f,
    Lights::Light
};
use cgmath::{Deg, InnerSpace, Matrix3, Point3, Point2};
use crossbeam::{atomic::AtomicCell, thread};
use itertools::Itertools;
use num::{integer, Integer, complex::ComplexFloat};
use num_traits::{Float, Num, NumCast};
use palette::{Srgb, white_point::B};
use rayon::prelude::*;
use std::sync::Arc;
pub
mod pararell{
    use std::{
        borrow::{Borrow, BorrowMut},
        cell::RefCell,
        ops::Add,
        sync::{
            atomic::{AtomicI64, Ordering},
            Mutex, RwLock,
        },
        time::Instant, fs::{File, self}, path::Path,
    };
    
    use crate::{
        primitives::prims::Ray, 
        sampler::Sampler, 
        samplerhalton::SamplerHalton, Point2i, 
        Cameras::PerspectiveCamera, 
        imagefilm::{FilterFilmGauss, FilterEval}};
    use crate::{
        imagefilm::{FilterFilm, FilterFilmBox, ImgFilm, Pix},
        materials,
        primitives::Cylinder,
        raytracerv2::interset_scene,
        Bounds2f, Point2f, Point3f, Vector3f,
        raytracerv2::Scene
    };
    use cgmath::{Deg, InnerSpace, Matrix3, Point2};
    use crossbeam::atomic::AtomicCell;
    use image::{codecs::png::{FilterType, PngEncoder}, ImageEncoder};
    use itertools::Itertools;
    use palette::Srgb;
    use rayon::prelude::*;
    use std::sync::Arc;
    use crate::raytracerv2::clamp;
 
    use super::{BBds2i, BBds2f};

#[derive(Debug,   Clone, Copy )]
pub struct FilmPx{
   pub  c : Srgb,
    pub w : f64,
    stats_hit : u64,
}
impl FilmPx{
    pub fn from_w(w:f64)-> Self{
        FilmPx { c : Srgb::new(0.,0., 0.), w , stats_hit:0}
    }
}
impl Default for FilmPx{
    fn default() -> Self {
        FilmPx { c : Srgb::new(0.,0., 0.), w: 0.,stats_hit:0 }
    }
}
pub struct FilmTile<'a>{
    pub  filterRadius : (f64, f64),
    pub bounds : BBds2i<i64>,
    pub pxs : Vec<FilmPx>,
    pub width_cropped : i64, 
    pub height_cropped : i64, 
    pub area_cropped:i64,
    pub table_filter : & 'a [f64;16*16],
    filterinv_radius :(f64, f64),
    filterTableSize:i64, // filter table width
}
impl <'a> FilmTile<'a>{
    pub fn new(
        bounds: BBds2i<i64>, 
        table_filter : &'a [f64;16*16],
        filterRadius : (f64, f64),
        filterinv_radius :(f64, f64),
        filterTableSize:i64, // filter table width
    
    )->FilmTile<'a>{
     
        let w =   bounds.pmax.x - bounds.pmin.x ;
        let h =    bounds.pmax.x -  bounds.pmin.x ;
        let area_size = bounds.area() ;
        let mut px_buffer : Vec<FilmPx> = vec![FilmPx::from_w(0.0); area_size as usize];
        // let mut px_buffer : Vec<FilmPx> = Vec::with_capacity(area_size);
        // unsafe {
        //     px_buffer .set_len(area_size);
        // }
        // let tablemock =  [0.0 as f64;16*16];
        FilmTile{
            filterRadius,
            filterinv_radius,
            filterTableSize, 
            bounds,
            pxs:  px_buffer,
            height_cropped:h, width_cropped:w,
            area_cropped:area_size,
            table_filter,
        }
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
    pub fn add_sample_mlt(&mut self,  psample:&(f64, f64), v : Srgb){
       let  praster = &(psample.0.floor() as i64, psample.1.floor()as i64);
       let isinclusive  = ( self.bounds.pmin.x>=praster.0  &&   praster.0 < self.bounds.pmin.x) && (   praster.1 < self.bounds.pmax.x && praster.1 > self.bounds.pmin.x);
       let pxoffset =  self.get_px(praster);
       
       let px = (&mut self.pxs[pxoffset]);

    
       px.c = Srgb::new( px.c.red+v.red,   px.c.green+v.green,     px.c.blue+v.blue);
      
    //    println!("add_sampler  {:?} {:?} ", praster, px.c);
      
    }
    pub fn mult_by_constant(&mut self, normalization_constant : f32) {
       let debug = true;
        for iy in 0..self.height_cropped{
            for ix  in  0..self.width_cropped {
             let pxoffset =    ( self.width_cropped * iy + ix) as usize;
             let px =(&mut self.pxs[pxoffset]);
             if px.c.red>0.0 && px.c.green>0.0 && px.c.blue>0.0 && debug {
                println!("px   {:?} ",(ix, iy));
                println!("     rgb            {:?}  ",  px .c);
             }
             
           
              
             px.c.red*=normalization_constant ; 
             px.c.green*=normalization_constant ; 
              px.c.blue*=normalization_constant ; 
         
              if px.c.red>0.0 && px.c.green>0.0 && px.c.blue>0.0 && debug {
                println!("     normalize  rgb {:?}  ",  px .c);
             }
            //  println!("     normalize  rgb {:?}  ",  px .c);
           
            }

         }
        // self.pxs.par_iter_mut().for_each(|mut px|{
            
        //     px.c =Film1::toRgb(px.c);
        //     px.c.red*=normalization_constant ; 
        //     px.c.green*=normalization_constant ; 
        //     px.c.blue*=normalization_constant ; 
        // });
         
         for iy in 0..self.height_cropped{
            for ix  in  0..self.width_cropped {
             let pxoffset =    ( self.width_cropped * iy + ix) as usize;
                if! (self.pxs[pxoffset].c.red == 0.0 &&  self.pxs[pxoffset].c.green == 0.0 &&  self.pxs[pxoffset].c.blue== 0.0) {
                    //   println!("px {:?} {:?}", (ix, iy),self.pxs[pxoffset].c);
                }
           
            }

         }
    }
    pub fn add_sample(&mut self,  psample:&Point2f, v : Srgb){




        let pmed = Point2f::new(0.5, 0.5);
        let px = (psample - pmed);
       let mut pmin =  Point2f::new((px.x- self. filterRadius.0 ).ceil(),(px.y-self. filterRadius.1).ceil()); 
       let  mut pmax =  Point2f::new((px.x+self. filterRadius.0 ).floor()+1.0,(px.y+self. filterRadius.1).floor()+1.0);
       pmin.x = pmin.x.max(self.bounds.pmin.x as f64) ;
        
       pmin.y =  pmin.y.max(self.bounds.pmin.y as f64 );
       pmax.x =  pmax.x.min(self.bounds.pmax.x as f64); 
       pmax.y = pmax.y.min(self.bounds.pmax.y as f64);





       let difxtablesize =  ( pmax.x -pmin.x ) as usize ;
       let difytablesize =  ( pmax.y -pmin.y ) as usize ;
        let mut difxtable = vec![0;difxtablesize];
        let mut difytable = vec![0;difytablesize];
        for ix in pmin.x as usize.. pmax.x as usize{
            let bucketAndValues =  Self::offset_filtertable(px.x, ix as f64, pmin.x,self. filterinv_radius.0 , self.filterTableSize as usize);
            difxtable[bucketAndValues.0] = bucketAndValues.1;
          
           
         }
         
        for iy in pmin.y as usize.. pmax.y as usize{
            let bucketAndValues =  Self::offset_filtertable(px.y, iy as f64, pmin.y,self. filterinv_radius.1, self.filterTableSize as usize);
            difytable[bucketAndValues.0] = bucketAndValues.1;
       
         }


         for iy in pmin.y as usize.. pmax.y as usize{
            for ix in pmin.x as usize.. pmax.x as usize{ 
                
                let pxoffset =  self.get_px(&(ix as i64,iy as i64));
                 let pix = (&mut self.pxs[pxoffset]) ; 
                     let idifx = ix - pmin.x as usize;
                     let idify = iy - pmin.y as usize;
                      
                   let ioffset  =  difytable[idify] *( self.filterTableSize as usize) + difxtable[idifx];
    
                      let weight = self.table_filter[ioffset];
                      pix.c =  Self::updateSample(pix.c, v , weight as f32  );
                     pix.w += weight;
            }   
          }







        // self.pxs[0].c.blue=1.0;
    }
    // usa coords desde (0,0)->( self.bounds.pmax.x,  self.bounds.pmax.y)
    pub fn get_px( &self, p :&(i64, i64) )->usize{
       
      let dy =  p.1 - self.bounds.pmin.y ;
      let dx =  p.0 - self.bounds.pmin.x ;
      (dy  * self.width_cropped  + dx) as usize
    }
    pub fn get_bounds(&self) -> BBds2i<i64>{
        self.bounds
    }
    pub fn iter(&  self)->PixelIterator{
        PixelIterator::new( self.bounds  )
    }
    pub fn area(&self)->i64{
        self.area_cropped
    }


 






    pub fn offset_filtertable(pixel: f64,p:f64, pmin:f64, filterinv_radius : f64,filterWidth: usize)->(usize,usize ){
        let dtable =      p as i64 - pmin as i64;
        ( dtable as usize, ((p - pixel )* filterinv_radius* (filterWidth as f64)).abs().floor().min((filterWidth-1) as f64) as usize)
      
    }

    pub fn updateSample(acc: Srgb , sample :Srgb , weight:f32 )->Srgb{
        Srgb::new( acc.red  + sample.red*weight  ,   acc.green  + sample.green*weight ,   acc.blue  + sample.blue*weight)
     }



    
}
pub
struct PixelIterator{
    pub count_bounds : BBds2i<i64>, 
    pub current_count : (i64, i64),
    pub flagcondition :bool,
}
impl PixelIterator{
    pub fn new(count_bounds : BBds2i<i64>)->Self{
        
        PixelIterator{count_bounds, current_count :  (count_bounds.pmin.x,count_bounds.pmin.y), flagcondition:false}
    }
}

impl Iterator for PixelIterator {
    type Item = (i64, i64);
    fn next(&mut self) -> Option<Self::Item>{
        if self.current_count.0 == self.count_bounds.pmin.x &&  self.current_count.1 == self.count_bounds.pmin.y {
            let o =    Some(self.current_count ) ;
            self.current_count.0+=1;
            self.flagcondition = true;
            return  o;
        }
        if self.flagcondition == true{
            //total mierda
            self.flagcondition=false;
           return  Some(self.current_count )

        }
        if  self.current_count.0  <  self.count_bounds.pmax.x-1 {
            self.current_count.0+=1;
        }else{
            self.current_count.1+=1;
            self.current_count.0 = self.count_bounds.pmin.x ;
        }
        if self.current_count.0 >= self.count_bounds.pmax.x || self.current_count.1 >= self.count_bounds.pmax.y {return  None}
        Some(self.current_count )
    }
}

pub struct Film1{
    pub  filterRadius : (f64, f64),
   pub bounds : BBds2i<i64>,
   pub width_cropped  :  i64,
   pub height_cropped  :  i64,
//    pub pxs : Vec<FilmPx>,
   pub pixels : RwLock<Vec<FilmPx>>,
   pub filter : FilterFilm,
   pub stats_num_merge_tiles : AtomicI64,
   pub table_filter : [f64; 16*16],
   pub filterTableSize: i64,
}
impl Film1{
    //esto ya esta! tengo que añadir la referecia al filtro a ver si lo puede hacer
    pub fn init_filter_table(table : & mut [f64; 256], filter : &FilterFilm, filterWidth: usize){
    
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
    pub fn init(bounds : BBds2i<i64> , filter : FilterFilm)->Arc<Film1>{
        let w =  bounds.pmax.x - bounds.pmin.x;
        let h =  bounds.pmax.y - bounds.pmin.y;
        let area_size = bounds.area() as usize;
        const  filterTableSize:u32 =16;
        let mut  filterTable: Vec<f64> = vec![0.0;(filterTableSize*filterTableSize ) as usize]; 
         let mut filter_table : [f64; 256 ] = [0.0; (filterTableSize*filterTableSize) as usize];
        Self::init_filter_table(& mut filter_table, &filter, filterTableSize as usize);
        let mut px_buffer : Vec<FilmPx> = vec![FilmPx::from_w(0.0); area_size];
       
        Arc::new(Film1{
            filterRadius:filter.radius(),
            filterTableSize: filterTableSize as i64,
            bounds, 
            width_cropped:w, 
            height_cropped:h  , 
            // pxs:px_buffer,
            pixels: RwLock::new(vec![FilmPx::default(); area_size ]) , 
            filter ,
            stats_num_merge_tiles:AtomicI64::new(0) ,
           table_filter :  filter_table
        }
        )
    }
    pub fn new(bounds : BBds2i<i64> , filter : FilterFilm)->Film1{
        let w =  bounds.pmax.x - bounds.pmin.x;
        let h =  bounds.pmax.y - bounds.pmin.y;
        let area_size = bounds.area() as usize;
        const  filterTableSize:u32 =16;
        let mut  filterTable: Vec<f64> = vec![0.0;(filterTableSize*filterTableSize ) as usize]; 
         let mut filter_table : [f64; 256 ] = [0.0; (filterTableSize*filterTableSize) as usize];
        Self::init_filter_table(& mut filter_table, &filter, filterTableSize as usize);
        let mut px_buffer : Vec<FilmPx> = vec![FilmPx::from_w(0.0); area_size];
        Film1{
            filterRadius:filter.radius(),
            filterTableSize: filterTableSize as i64,
            bounds, 
            width_cropped:w, 
            height_cropped:h  , 
            // pxs:px_buffer,
            pixels: RwLock::new(vec![FilmPx::default(); area_size ]) , 
            filter ,
            stats_num_merge_tiles:AtomicI64::new(0) ,
           table_filter :  filter_table
        }
    }
    
    
    pub fn init_tile(&self ,boundstile: BBds2i<i64>)->FilmTile{
        // check intesections..
         if (self.bounds.pmax.x < boundstile.pmax.x || self.bounds.pmax.y < boundstile.pmax.y ) ||
         (self.bounds.pmin.x > boundstile.pmin.x || self.bounds.pmin.y > boundstile.pmin.y )
           {
            panic!("check dims!")
         }
         // compute new dimensions for new tile
        let b  = self.compute_tile_dimensions(boundstile, self.filterRadius);
        
        // instancite the new tile
        FilmTile ::new(BBds2i::from(b) , &self.table_filter, self.filterRadius, self.filter.inv_radius(), self.filterTableSize)
    }
    /**
     * 
     * es usado en el integrador  metropolis.
     * solo transfiere el contenido de la tile al film
     * con el fin de adquirir los metodos de film.
     * No hay que sincronizar nada
     */
    pub fn merge_splat_tile(& self ,tile : &FilmTile){
        for a in tile.iter(){
            let px_idx  =  self.get_px(&a);
            let px_idx_tile = tile.get_px(&a); 
            // debe tener la misma dimension
            assert!(px_idx==px_idx_tile);
            let px_tile =  &tile.pxs[px_idx_tile];
            let mut px_bufferlockguard =  self.pixels.write().unwrap();
            let mut px  = &mut  px_bufferlockguard[px_idx];
           
            px.c =px_tile.c;
         
        }
    }
    pub fn merge_tile(& self ,tile : &FilmTile){
        self. stats_num_merge_tiles.fetch_add(1, Ordering::AcqRel);
         for a in tile.iter(){
                let px_idx  =  self.get_px(&a);
                 let px_idx_1 = tile.get_px(&a); 
              //   println!("px film {:?} tile  {:?}", px_idx , px_idx_1);
                let px_tile =  &tile.pxs[px_idx_1];
                let mut px_bufferlockguard =  self.pixels.write().unwrap();
                let mut px  = &mut  px_bufferlockguard[px_idx];
                px.c = Self::mergeSample(px.c, Self::toXyz(px_tile.c) );
                px.w+=px_tile.w;
               // Self::updateSample( px.c, px_tile.c, weight)
                // pxfromfilm.w=1.0;
                 
            }
            println!("merge files: {:?}",     self.stats_num_merge_tiles);
    }
    pub fn compute_tile_dimensions (&self, other:BBds2i<i64>,  fradius : (f64, f64) )->BBds2f<f64>{    
        let bf : BBds2f<f64> =  BBds2f::from(  other   );
        let p0 = Point2::new((bf.pmin.x - 0.5 - fradius.0 ).ceil()as i64 ,(bf.pmin.y - 0.5- fradius.1 ).ceil()as i64);
        let p1 = Point2::new((bf.pmax.x - 0.5 + fradius.0 +1.  ).floor()as i64 ,(bf.pmax.y - 0.5 + fradius.1 +1. ).floor() as i64);
        
       let bi =  BBds2i::<i64>::intersect ( BBds2i ::new(p0, p1),  self.bounds);
      BBds2f::from(bi) 
      
     }

     pub fn area(&self)->i64{
        self.bounds.area()
     }
     #[inline]
     pub fn get_px(&self, p :  &(i64, i64))-> usize {
      // return bucket offset 
       (( p.1-  self.bounds.pmin.y) * self.width_cropped +  (p.0 - self.bounds.pmin.x) ) as usize
     }
     pub fn  width(&self)->i64{self.width_cropped}
     pub fn  height(&self)->i64{self.height_cropped}
     #[inline]
     fn updateSample(acc: Srgb , sample :Srgb , weight:f32 )->Srgb{
        Srgb::new( acc.red  + sample.red*weight  ,   acc.green  + sample.green*weight ,   acc.blue  + sample.blue*weight)
     }
     #[inline]
     fn mergeSample(acc: Srgb , sample :Srgb  )->Srgb{
        Srgb::new( acc.red  + sample.red  ,   acc.green  + sample.green  ,   acc.blue  + sample.blue )
     }
     #[inline]
     fn toRgb(xyz:Srgb)->Srgb{
        Srgb::new(
         3.240479* xyz.red - 1.537150 * xyz.green - 0.498535 * xyz.blue,
         -0.969256 * xyz.red + 1.875991 * xyz.green + 0.041556 * xyz.blue,
         0.055648 * xyz.red - 0.204043 * xyz.green + 1.057311 * xyz.blue
        )
    }
   #[inline]
    pub fn toXyz(rgb: Srgb)->Srgb{

       
        Srgb::new(
            0.412453 * rgb.red + 0.357580* rgb.green + 0.180423 * rgb.blue,
            0.212671 * rgb.red + 0.715160 * rgb.green + 0.072169 * rgb.blue,
            0.019334 * rgb.red + 0.119193 * rgb.green + 0.950227 * rgb.blue
        )
         

    }
    pub fn commit_and_write(&self, directory: &str,  filename : &str , trafoToXYZ : bool)-> std::io::Result<()>{
        let dirpath : &Path = Path::new(directory);
        if  fs::create_dir_all(dirpath).is_err() {
            panic!(" fs::create_dir_all has throwned a error trying created file directory : {:?}", dirpath.as_os_str())

        }
       let final_filename =  &dirpath .join(filename);
       

        let bufferres = self.pixels.read().unwrap().to_owned().into_iter().enumerate().map(|(offset, px)|{
            if trafoToXYZ == true{
                let rgb = Self::toRgb(Self::toXyz(px.c));
                let p =  vec![(clamp((rgb.red / px.w as f32), 0.0, 1.0)*255.0 ) as u8, (clamp((rgb.green/ px.w as f32), 0.0, 1.0)*255.0) as u8 ,(clamp((rgb.blue/ px.w as f32), 0.0, 1.0)*255.0)as u8];
                return p;
            }else{
                let rgb = px.c;
                let p =  vec![(clamp((px.c.red   as f32), 0.0, 1.0)*255.0 ) as u8, (clamp((px.c.green as f32), 0.0, 1.0)*255.0) as u8 ,(clamp((px.c.blue  as f32), 0.0, 1.0)*255.0)as u8];
                return p;
            }
          
        })   .flatten()
        .collect::<Vec<u8>>();
        let fd = File::create(final_filename).unwrap();
        let encoder = PngEncoder::new(fd);
        encoder.write_image( bufferres.as_slice(), self.width() as u32, self.height() as u32, image::ColorType::Rgb8).unwrap();
        println!("render in : {:?}",final_filename );
    Ok(())
    }
}


}
 

#[derive(Debug ,  Clone,Copy )]
pub
struct  BBds2i< T : num::Integer  +Copy +Clone +Display +num_traits::NumCast  >{
   pub pmin : Point2<T>,
   pub pmax : Point2<T>
}
impl <T : num::Integer  +Copy +Clone +Display +num_traits::NumCast > BBds2i<T> {
    
    pub fn new( pmin:Point2<T>,pmax:Point2<T>)->BBds2i<T>{
        BBds2i {pmin , pmax  }
    }
    pub fn from_minmax(xmin:T, ymin:T, xmax:T, ymax:T)->BBds2i<T>{
        BBds2i {pmin:Point2::new(xmin, ymin),pmax: Point2::new(xmax, ymax)}
    }
     
    pub fn area(&self)->T{
            (self.pmax.x - self.pmin.x ) *(self.pmax.y  - self.pmin.y)
    }
    pub fn compute_tile (self, other:BBds2i<i64>,  fradius : (f64, f64) )->BBds2f<f64>{    
       let bf : BBds2f<f64> =  BBds2f::from(  self  );
       let p0 = Point2::new((bf.pmin.x - 0.5 - fradius.0 ).ceil()as i64 ,(bf.pmin.y - 0.5- fradius.1 ).ceil()as i64);
       let p1 = Point2::new((bf.pmax.x - 0.5 + fradius.0 +1.  ).floor()as i64 ,(bf.pmax.y - 0.5 + fradius.1 +1. ).floor() as i64);
       
      let bi =  Self::intersect ( BBds2i ::new(p0, p1),  BBds2i::new(p0, p1));
     BBds2f::from(bi) 
     
    }
    
    pub fn intersect (bcurrent :BBds2i<i64>, other:BBds2i<i64>)->BBds2i<i64>{
        BBds2i::new(
            Point2::new(bcurrent.pmin.x.max(other.pmin.x),bcurrent.pmin.y.max(other.pmin.y)),
            Point2::new(bcurrent.pmax.x.min(other.pmax.x),bcurrent.pmax.y.min(other.pmax.y))
        )
       
    }
    pub fn lerp(&self, samplerlerp :& (f64, f64))->(f64, f64){
        let bf : BBds2f<f64> =  BBds2f::from(  *self  );
        ( raytracerv2::lerp(samplerlerp.0, bf.pmin.x, bf.pmax.x),raytracerv2::lerp(samplerlerp.1, bf.pmin.y, bf.pmax.y) )
    }
}

impl < T : num::Integer  +Copy +Clone +Display +num_traits::NumCast +Debug > fmt::Display for  BBds2i <T>{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pmin {:?}", self.pmin);
        write!(f, "pmax {:?}", self.pmax)
    }
}


// TODO: tengo que meter esto es primitives y quitar estos tratos  para simplificar 

#[derive(Debug,    Clone, Copy )]
pub
struct  BBds2f< T : num::Float  +Copy +Clone +Display  +num_traits::NumCast  >{
    pmin : Point2<T>,
    pmax : Point2<T>
}

impl <F : num::Float +Copy +Clone +Display +num_traits::NumCast > BBds2f<F> {
    pub fn new( p0:Point2<F>,p1:Point2<F>)->BBds2f<F>{
        BBds2f {pmin:p0, pmax:p1}
    }
   
  
  
}
  
 impl  <T:  num::Integer  +Copy +Clone +Display +  num_traits::NumCast  , F :num::Float  +Copy +Clone +Display  +num_traits::NumCast >  From<BBds2f<F>> for BBds2i<T> {
     fn from(value: BBds2f<F>) ->  BBds2i<T> {
         
         BBds2i::<T>::new(
            Point2 ::new( 
                num_traits:: cast::<F, T >(value.pmin.x).unwrap(),
                num_traits:: cast::<F, T >(value.pmin.y).unwrap()
            ),
            Point2::new(
                num_traits:: cast::<F, T >(value.pmax.x).unwrap(),
              num_traits:: cast::<F, T >(value.pmax.y).unwrap() 
            ) 
         )
     }
 }
 impl  <I:  num::Integer  +Copy +Clone +Display +  num_traits::NumCast  , F :num::Float  +Copy +Clone +Display  +num_traits::NumCast >  From<BBds2i<I>> for BBds2f<F> {
    fn from(value: BBds2i<I>) ->  BBds2f<F> {
       let z = num_traits:: cast::<u64, F >(0 as u64).unwrap();
        BBds2f::<F>::new(Point2 ::new(
            num_traits:: cast::<I , F>(value.pmin.x).unwrap(),
            num_traits:: cast::<I , F >(value.pmin.y).unwrap()
        ),Point2::new(
            num_traits:: cast::<I , F >(value.pmax.x).unwrap(),
            num_traits:: cast::<I , F >(value.pmax.y).unwrap()  )  )
    }
}

impl < T : num::Float  +Copy +Clone +Display  +num_traits::NumCast  +Debug > fmt::Display for  BBds2f <T>{
     
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pmin {:?}", self.pmin);
        write!(f, "pmax {:?}", self.pmax)
    }
}

 








#[derive(Debug ,  Clone,Copy )]
pub
struct  BBds3i< T : num::Integer  +Copy +Clone +Display +num_traits::NumCast  >{
   pub pmin : Point3<T>,
   pub pmax : Point3<T>
}
impl <T : num::Integer  +Copy +Clone +Display +num_traits::NumCast > BBds3i<T> {
    
    pub fn new( pmin:Point3<T>,pmax:Point3<T>)->BBds3i<T>{
        BBds3i {pmin , pmax  }
    }
    pub fn from_minmax(xmin:T, ymin:T, zmin:T , xmax:T, ymax:T, zmax:T)->BBds3i<T>{
        BBds3i {pmin:Point3::new(xmin, ymin, zmin),pmax: Point3::new(xmax, ymax, zmax)}
    }
     
     pub fn is_inside(&self, p :&Point3<T>)->bool{
        p.x >= self.pmin.x && p.x < self.pmax.x &&
        p.y >= self.pmin.y && p.y < self.pmax.y &&
        p.z >= self.pmin.z && p.z < self.pmax.z 
     
     }
    
    
    
}

impl < T : num::Integer  +Copy +Clone +Display +num_traits::NumCast +Debug > fmt::Display for  BBds3i <T>{
    // This trait requires `fmt` with this exact signature.
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "pmin {:?}", self.pmin);
        write!(f, "pmax {:?}", self.pmax)
    }
}
pub type BBds3i64 = BBds3i<i64>;






























pub
fn new_interface_for_intersect_occlusion(mut ray: &Ray<f64>,
    scene: &Scene1 ,
    target: &Point3<f64>)-> Option<(bool, f64, Point3<f64>, Option<MediumOutsideInside>)>{
        let prims =&scene.prims;
        let lights =&scene.lights;
       // panic!("revisar  new_interface_for_intersect_occlusion usar box dyn y quiero Ligth type en area light");
        for prims in prims{
            if let Some((ishit, _thit, phit, _)) =  prims.intersect_occlusion(ray, target){
                if (phit - ray.origin).magnitude2() < (*target - ray.origin).magnitude2() {
                    let epsi = (phit - ray.origin).magnitude2() - (*target - ray.origin).magnitude2() ;
                 
                    if epsi.abs()<1e-5{
                        continue;
                    }
                    
                    return Some((true, _thit, phit,    prims.get_medium()));
                }
            }
          
        }
        for light in lights{
            if light.is_arealight() {
                let shape =   light.get_shape1();
                //panic!("revisar  new_interface_for_intersect_occlusion usar box dyn y quiero Ligth type en area light");
                if let Some((ishit, _thit, phit, _)) =  shape.intersect_occlusion(ray, target){
                    if (phit - ray.origin).magnitude2() < (*target - ray.origin).magnitude2() {
                        (phit - ray.origin).magnitude2();
                        (*target - ray.origin).magnitude2() ;
                        let epsi = (phit - ray.origin).magnitude2() - (*target - ray.origin).magnitude2() ;

                        // println!("{:?}, {:?}", epsi.abs() , epsi.abs()<1e-5);
                        if epsi.abs()<1e-5{
                            continue;
                        }
                        return Some((true, _thit, phit, None));
                    }
                }

            }
        }
        Some((false, std::f64::MIN, Point3f::new(std::f64::MIN, std::f64::MIN, std::f64::MIN),None))
    }
pub
 fn new_interface_for_intersect_scene( ray: &Ray<f64>,scene: &Scene1  )-> Option<HitRecord<f64>>{
    let mut tmin = f64::MAX;
    let mut hitcurrent = None;
    let prims =&scene.prims;
    let lights =&scene.lights;
    for prims in prims{
       if let Some(hit) =  prims.intersect(ray, 0.0001, std::f64::MAX){
            if  tmin>hit.t{
               
                hitcurrent =Some(hit);
                tmin = hit.t;
            } 
       }
    }
    for light in lights{
        if  light.is_arealight() ||  light.is_ambientlight(){
           
          let shape =   light.get_shape1() ;
          if let Some(mut hit) = shape.intersect(ray, 0.0001, f64::MAX){
            if  tmin>hit.t{
                let Lemit = light.get_emission(Some(hit),ray);
                hit.is_emision = Some(true);
                hit.emission = Some(Lemit);
                
                hitcurrent =Some(hit);

              
                
                tmin = hit.t;
            }
          }
        }
    
    }
    hitcurrent
 }

 pub
 fn new_interface_for_intersect_scene_for_medium( ray: &mut Ray<f64>,scene: &Scene1  )-> Option<HitRecord<f64>>{
    let mut tmin = f64::MAX;
    let mut hitcurrent = None;
    let prims =&scene.prims;
    let lights =&scene.lights;
    for prims in prims{
       if let Some(hit) =  prims.intersect(ray, 0.0001, std::f64::MAX){
            if  tmin>hit.t{
                let is_entering = hit.is_surface_interaction(); 
                // marcamos el ray como intersection entrante en un medium
                if(hit.has_medium().is_some()&&!is_entering ){
                    ray.is_in_medium = true;
                    
                }else if hit.has_medium().is_some(){ 
                    ray.is_in_medium = ray.is_in_medium;
                }
                hitcurrent =Some(hit);
                tmin = hit.t;
            } 
       }
    }
    for light in lights{
        if  light.is_arealight() ||  light.is_ambientlight(){
           
          let shape =   light.get_shape1() ;
          if let Some(mut hit) = shape.intersect(ray, 0.0001, f64::MAX){
            if  tmin>hit.t{
                let Lemit = light.get_emission(Some(hit),ray);
                hit.is_emision = Some(true);
                hit.emission = Some(Lemit);
                
                hitcurrent =Some(hit);

              
                
                tmin = hit.t;
            }
          }
        }
    
    }
    hitcurrent
 }

pub
 fn new_interface_for_intersect_scene_surfaces( ray: &Ray<f64>,scene: &Scene1  )-> Option<HitRecord<f64>>{
    let mut tmin = f64::MAX;
    let mut hitcurrent = None;
    let prims =&scene.prims;
    let lights =&scene.lights;
    for prims in prims{
       if let Some(hit) =  prims.intersect(ray, 0.0001, std::f64::MAX){
            if  tmin>hit.t{
                hitcurrent =Some(hit);
                tmin = hit.t;
            } 
       }
    } 
    hitcurrent
 }

pub fn get_ray_path_1 (
    r: &Ray<f64>,

    lights: &Vec<Light>,
    scene: &Scene1 ,
    depth: i32,
    
     sampler: &mut SamplerType,
    // px:(i64, i64)
) -> bool {
 
    let rhit = Ray::new(Point3f::new(sampler.get1d(),sampler.get1d(),sampler.get1d()), Vector3f::new(0.0,0.0,1.0));
    let hit = new_interface_for_intersect_scene(&r, scene);
     if hit.is_some(){
         true 
     }else{
        false
     }
    
}

#[test]
pub fn debug_par_film_tile_sample_tile_generator_debug_como_lo_hago(){
par_render();
render_brute_force_();
}
pub fn  render_brute_force_(){
    println!("brute force");
    let start = Instant::now();
    let res = (512, 512);
    let spp = 256;
    let num_total_of_hits: AtomicI64 = AtomicI64::new(0);
    let mut sampler2 = Arc::new(SamplerType::UniformType( SamplerUniform::new(((0, 0), (res.0 as u32, res.1 as u32)),spp,false,Some(0)) ));
    let cameraInstance =  PerspectiveCamera::from_lookat(Point3f::new(0.0,-0.500, 0.50), Point3f::new(0.0,0.0,1.00), 1e-2, 100.0, 90.0, res);
    let scene  =  Arc::new (
        Scene1::make_scene(
            res.0 as usize, 
            res.1 as usize, 
            vec![], 
            vec![
                PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.0,0.0,3.0), 2.0, materials::MaterialDescType::NoneType))
                //    PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 0.0,  3.0), Matrix3::from_angle_x(Deg(-70.0)),-1.0, 1.0,1.0, materials::MaterialDescType::NoneType))
                ], 
            1,1));
    let filtergauss  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0,2.0),3.0));
    let mut film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))) , filtergauss); 
    let mut wholetile = film.init_tile(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))));
     let mut sampler = sampler2.seed_instance();
    for iy in 0..res.1{
        for ix in 0..res.0{
            sampler.start_pixel(Point2i::new(ix as i64, iy as i64)); 
            let samplecamera = sampler.get2d();
            let sampleforcamerauv = to_map(&samplecamera, res.0 as i64  , res.1 as i64);
            let ray = cameraInstance.get_ray(&sampleforcamerauv);
            let ishit =  get_ray_path_1(&ray,&scene.lights, &scene, 1, &mut sampler );
            if ishit {num_total_of_hits.fetch_add(1, Ordering::SeqCst);}
            // wholetile.add_sample(samplecamera, v)
            wholetile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), Srgb::new(1.0, 1.0,0.0));
            while sampler.start_next_sample(){
                let samplecamera = sampler.get2d();
                let sampleforcamerauv = to_map(&samplecamera, res.0 as i64  , res.1 as i64);
                let ray = cameraInstance.get_ray(&sampleforcamerauv);
                let ishit =  get_ray_path_1(&ray,&scene.lights, &scene, 1, &mut sampler );
                if ishit {num_total_of_hits.fetch_add(1, Ordering::SeqCst);}
                wholetile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), Srgb::new(1.0, 1.0,0.0));
            }
            
        }
    }
    film.merge_tile(&wholetile);
    film.commit_and_write("filtertest", "miexample_brute_force.png",true).unwrap();
    println!("num_total_of_hits --------------->{:?}",num_total_of_hits ); 
    println!(" time: {} seconds", start.elapsed().as_secs_f64());

}
  
pub fn par_render(){
    let start = Instant::now();
    let res = (512_i64, 512_i64);
    let spp = 256_i64;
    let total_rays:i64 = 1024*1024*2048;
    println!("{}", total_rays);
   
    let filtergauss  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0,2.0),3.0));
    let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( (1.0,1.0)));
    let mut film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))) , filtergauss); 
    let num_workers =  12;// num_cpus::get();
    let total_area : AtomicI64 = AtomicI64::new(0);
    let num_total_of_hits: AtomicI64 = AtomicI64::new(0);
    let num_total_of_pixel: AtomicI64 = AtomicI64::new(0);
  
     let sampler2 = Arc::new(SamplerType::UniformType( SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),spp as u32,false,Some(0)) ));
     let cameraInstance =  PerspectiveCamera::from_lookat(Point3f::new(0.0,-0.500, 0.50), Point3f::new(0.0,0.0,1.00), 1e-2, 100.0, 90.0, (res.0 as u32, res.1 as u32));
   
    let scene  =  Arc::new (
        Scene1::make_scene(
            res.0 as usize, 
            res.1 as usize, 
            vec![], 
            vec![
                PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.0,0.0,3.0), 2.0, materials::MaterialDescType::NoneType))
                //    PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 0.0,  3.0), Matrix3::from_angle_x(Deg(-70.0)),-1.0, 1.0,1.0, materials::MaterialDescType::NoneType))
                ], 
            1,1));
    crossbeam::scope(|scope| {

        let (sendertorender0, receiverfrominitialize0) = crossbeam_channel::bounded(num_workers);
      
        let filmref =  &film;
        let camera = &cameraInstance;
        let sceneshared = &scene;
        
    
        scope.spawn(move|_|{
            let mut area = 0 ;
            let sender0clone = sendertorender0.clone();
            let vlst =    init_tiles_withworkers(num_workers as i64, filmref.width(), filmref.height());
            let mut cnt = 0;
            for tilescoords in vlst.into_iter(){
               
                let tile = filmref.init_tile(BBds2i::new(Point2::new(tilescoords.1.0.0  as  i64 ,tilescoords.1.0.1 as  i64,),Point2::new(tilescoords.1.1.0 as  i64 ,tilescoords.1.1.1 as  i64  )));
                
                println!("init_tiles_withworkers -> thread id {:?}", std::thread::current().id()) ;
                let packedwork = (tilescoords, tile);
                    sender0clone.send(packedwork).unwrap_or_else(|op| panic!("---sender error!!--->{}", op));     
            }
            // println!("{}", area);
            drop(sender0clone)
        });
       
        for _ in 0..num_workers {
            let mut sampler = sampler2.seed_instance();
            let atom_hits_ref =&num_total_of_hits;
            let num_total_of_px_ref = &num_total_of_pixel;
            let rxclone = receiverfrominitialize0.clone();
            scope.spawn(move|_|{
                
             
                for mut packedwork in  rxclone.recv().into_iter() {
                  
                    println!("render worker -> thread id {:?}", std::thread::current().id()) ;
                    println!("       compute tile {:?}",  packedwork.0) ;
                
                    //unfold the stuff send by thread "parent"
                    let mut tile = & mut packedwork.1;
                    let tilescoords =   packedwork.0;

                    let startx = tilescoords.1.0.0;
                    let starty = tilescoords.1.0.1;
                    let endx = tilescoords.1.1.0 ;
                    let endy =tilescoords.1.1.1;
            
                    for px in  itertools::iproduct!(startx..endx,starty..endy ){
                        let x = px.0  as f64;
                        let y = px.1  as f64;
                        let r =   (x - startx as f64) as  f32 / (endx - startx) as  f32;
                        let g=   (y - starty as f64) as  f32 / (endy - starty)as  f32;
                        sampler.start_pixel(Point2i::new(x as i64, y as i64));
                    
                        let samplecamera = sampler.get2d();
                        debug_assert_eq!(samplecamera.0.floor()as usize , x as usize);
                        debug_assert_eq!(samplecamera.1.floor()as usize , y as usize );
                        
                        let sampleforcamerauv = to_map(&samplecamera, endx  , endy as i64);
                        let ray = cameraInstance.get_ray(&sampleforcamerauv);
                        let ishit =  get_ray_path_1(&ray,&sceneshared.lights, &sceneshared, 1, &mut sampler );
                        if ishit {atom_hits_ref.fetch_add(1, Ordering::SeqCst);}
                        num_total_of_px_ref.fetch_add(1, Ordering::Acquire);
                        tile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), Srgb::new(r, g,0.0));
                        while sampler.start_next_sample(){

                                let samplecamera = sampler.get2d();
                                debug_assert_eq!(samplecamera.0.floor()as usize , x as usize);
                                debug_assert_eq!(samplecamera.1.floor()as usize , y as usize );
                                let sampleforcamerauv = to_map(&samplecamera, endx  , endy as i64);
                                let ray = cameraInstance.get_ray(&sampleforcamerauv);
                                let ishit = get_ray_path_1(&ray,&sceneshared.lights, &sceneshared, 1, &mut sampler );
                                if ishit {atom_hits_ref.fetch_add(1, Ordering::SeqCst);}
                                num_total_of_px_ref.fetch_add(1, Ordering::Acquire);
                                tile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), Srgb::new(r, g,0.0));
                        }
                            // println!("num_total_of_px --------------->{:?}", num_total_of_px_ref );
                        
                          

                    } 

                     filmref.merge_tile(&tile);
                 
                 }

                

             });
        }

    }).unwrap();
 
     println!("num_total_of_hits --------------->{:?}",num_total_of_hits );
     println!("num_total_of_px --------------->{:?}",num_total_of_pixel );
     let  mut acc  = 0.0;
    let  mut acc_color  = 0.0;
   if film.commit_and_write("filtertest", "miexample2.png",true).is_ok(){
    println!("" );
   }

     
    println!(" time: {} seconds", start.elapsed().as_secs_f64());
 
}











 
#[test]
 
pub fn par_film_tile_pararell_generator_test(){



    [
         (8,8),
         (8,8),
         (16,8),
         (8,16),
         (32,32),
          (64,64),
         (16,64),
       (32,64),
          (64,64),
       (128,64), 
       (128,128),
      (128,256),
     (128,512),
     (512,128),
     (512,512),
        ].to_vec().iter().for_each(|res|{


            let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( (0.0,0.0)));
            let mut film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))) ,filterbox); 
            let num_workers =  12;// num_cpus::get();
            let total_area : AtomicI64 = AtomicI64::new(0);
            crossbeam::scope(|scope| {
                let filmref =  &film;
                let total_area_ref = &total_area;
                let (sx, rx) = crossbeam_channel::bounded(num_workers);
                // let sxclone = sx.clone();
                scope.spawn(move|_|{
                    let vlst =    init_tiles_withworkers(num_workers as i64, filmref.width(), filmref.height());
                    println!("vlst.len() {:?}", vlst.len());
                    
                    for tilescoords in vlst.iter(){
                       println!("{:?}", tilescoords);
                        
                        let tile = filmref.init_tile(
                            BBds2i::new(
                                Point2::new(tilescoords.1.0.0  as  i64 ,tilescoords.1.0.1 as  i64,),
                                Point2::new(tilescoords.1.1.0 as  i64 ,tilescoords.1.1.1 as  i64  )));
                        total_area_ref. fetch_add(tile.area(), Ordering::Release);
                        sx.send(tile).unwrap_or_else(|_| panic!("panic!"));
                    }
                    // let tile = filmref.init_tile(BBds2i::new(Point2::new(0,0,),Point2::new(8,8)));
                    drop(sx);
        
                });
                for _ in 0..num_workers {
                    let rxclone = rx.clone();
                    
                   
                    scope.spawn(move|_|{
                     
                        for tile in  rxclone.recv().iter() {
                            
                            filmref.merge_tile(tile);
                         
                        }
         
                        //
        
                     });
                }
              
        
            }).unwrap();
            let mut cnt  = 0_f64;
            for px in  PixelIterator::new(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(film.width() as f64 , film.height() as f64)))){
                      let offset = film.get_px(&px);
                  //    println!("{:?}  {}",px,  film.pixels.read().unwrap()[offset].w);
                      cnt= cnt + film.pixels.read().unwrap()[offset].w;
                     
            }
         
         
       println!("{:?}, {:?}",cnt, total_area);
            
//       let mut film0 = film.to_owned() ;
//    let mut bind =  film0.borrow_mut();
//     // bind. add_sample(&Point2::new(0 as f64, 0 as f64), Srgb::new(1.0,1.0,1.0));
//     bind.commit_and_write("raytracer3_ray_path_emission_direct_light__with_arealight.png");

        });
    
   

}
#[test]
pub fn par_film_tile_generator_test(){
    if(true){
        let mut cnt = 0_f64;
        let mut total_area = 0_i64;
        let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( (0.0,0.0)));
        let film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(8. ,8. ))) ,filterbox); 
        
        let mut vtiles:Vec<FilmTile> = vec![];
        for t in init_tiles(2, 2, film.width(), film.height()) {
            println!("{:?}", t);
            let mut tile00   = film.init_tile(BBds2i::new(Point2::new(t.0.0 as i64,t.0.1 as i64,),Point2::new(t.1.0  as i64,t.1.1 as i64)));
            total_area = total_area + tile00.area();
            vtiles.push(tile00);
        }
        for tile_i in vtiles.iter()  {
            film.merge_tile(&tile_i);
        }
    // film.merge_tile(&tile00_c);
       for px in  PixelIterator::new(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(film.width() as f64 , film.height() as f64)))){
          let offset = film.get_px(&px);
          println!("{:?}  {}",px,  film.pixels.read().unwrap()[offset].w);
         cnt = cnt + film.pixels.read().unwrap()[offset].w;

         
       }
       println!("{:?} {} {}", cnt, total_area as f64 , film.area() as f64);
    }


    // for t in init_tiles(4, 4, 9, 9) {
    //     println!("{:?}",  t );
    // }
}
pub fn par_film_test(){
//     let film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(1.0,1.0,),Point2::new(16. ,16. ))) );

//     println!("{:?} {}",     film.bounds,   film.area());
//    // film.init_tile(BBds2i::new(Point2::new(0,0,),Point2::new(16,16 )));
//     let mut tile = film.init_tile(BBds2i::new(Point2::new(1,1,),Point2::new(16,16 )));
    // for a in tile.iter(){
    //     let px_idx  =  film.get_px(&a);
    //     let px_idx_1 = tile.get_px(&a);
       
    //      println!("{:?} {} {}", a, px_idx, px_idx_1);
    //  }
    
    if(false){
        let mut cnt = 0_f64;
        let mut total_area = 0_i64;
        let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( (0.0,0.0)));
        let film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(8. ,8. ))) ,filterbox); 
        let mut tile00   = film.init_tile(BBds2i::new(Point2::new(0,0,),Point2::new(4,4)));
        let mut tile10   = film.init_tile(BBds2i::new(Point2::new(4,0,),Point2::new(8,4)));
        let mut tile01   = film.init_tile(BBds2i::new(Point2::new(0,4,),Point2::new(4,8)));
        let mut tile11   = film.init_tile(BBds2i::new(Point2::new(4,4,),Point2::new(8,8)));
        // let mut tile00_c   = film.init_tile(BBds2i::new(Point2::new(0,0,),Point2::new(8,8)));
        total_area = total_area + tile00.area();
        total_area = total_area + tile10.area();
        total_area = total_area + tile01.area();
        total_area = total_area + tile11.area();
        // let mut tile11 = film.init_tile(BBds2i::new(Point2::new(8,8,),Point2::new(16,16 )));
        film.merge_tile(&tile00);
          film.merge_tile(&tile10);
        film.merge_tile(&tile01);
          film.merge_tile(&tile11);
    // film.merge_tile(&tile00_c);
       for px in  PixelIterator::new(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(8. , 8.)))){
          let offset = film.get_px(&px);
          println!("{:?}  {}",px,  film.pixels.read().unwrap()[offset].w);
         cnt = cnt + film.pixels.read().unwrap()[offset].w;

         
       }
       println!("{:?} {} {}", cnt, total_area as f64 , film.area() as f64);
    }
    // for t in init_tiles(4, 4, 9, 9) {
    //     println!("{:?}", t);
     
    // }



    // assert with tile generator. we use function init_tiles
   

   
   ;
  
  if(false){
    let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( (0.0,0.0)));
    let film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(8. ,8. ))) ,filterbox); 
    let num_cores = num_cpus::get();
    crossbeam::scope(|scope| {
        let filmref =  &film;
        let (sx, rx) = crossbeam_channel::bounded(num_cores);
        // let sxclone = sx.clone();
        scope.spawn(move|_|{
            let vlst = init_tiles(4, 4, 8, 8);
            for tilescoords in vlst.iter(){
                println!("{:?}", tilescoords);
                let tile = filmref.init_tile(BBds2i::new(Point2::new(tilescoords.0.0 as  i64 ,tilescoords.0.1 as  i64,),Point2::new(tilescoords.1.0 as  i64,tilescoords.1.1 as  i64)));
                sx.send(tile).unwrap_or_else(|_| panic!("panic!"));
            }
            // let tile = filmref.init_tile(BBds2i::new(Point2::new(0,0,),Point2::new(8,8)));
            

        });
        for _ in 0..num_cores {
            let rxclone = rx.clone();
            scope.spawn(move|_|{

                for tiles in  rxclone.recv().iter() {
                 filmref.merge_tile(tiles);
                }
             });
        }
       

    }).unwrap();
    for px in  PixelIterator::new(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(8. , 8.)))){
              let offset = film.get_px(&px);
              println!("{:?}  {}",px,  film.pixels.read().unwrap()[offset].w);
             
    }
   
  }
    
  
}

pub fn main_test_intersection_8_test(){
//      let v = BBds2i::<u64>::new(Point2::new(0,0,),Point2::new(0 ,0 ));
//      let bv = BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(0. ,0. ));
//   let c :BBds2i<i64> = BBds2i::from(bv);
//   let cc :BBds2f<f64> = BBds2f::from(c);
// let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( (2.0,2.0)));
//     let film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(255. ,255. ))) ,filterbox);
//     let num_cores = num_cpus::get();
//     crossbeam::scope(|scope| {
//          let filmref =  &film;
//       let (sx, rx) = crossbeam_channel::bounded(num_cores);
//         for _ in 0..num_cores { 
//             let sxclone = sx.clone();
//            let f  =  Arc::clone(filmref);
//             scope.spawn(move|_|{
                
//                 let mut tile =  f.init_tile( BBds2i::<i64>::new(Point2::new(0,0,),Point2::new(255 ,255 )));
//                 // tile.add_sample();
//                 sxclone.send(tile).unwrap_or_else(|_| panic!("panic!"))
//             });
//         }
//         scope.spawn(move|_|{
//            for tiles in  rx.recv().iter() {
//             filmref.merge_tile(tiles);
//            }
//         });




//     }).unwrap();
}






pub fn  with_integrator_api_render_brute_force_(){
    println!("brute force");
    let start = Instant::now();
    let res = (512, 512);
    let spp = 256;
    let num_total_of_hits: AtomicI64 = AtomicI64::new(0);
    let mut sampler2 = Arc::new(SamplerType::UniformType( SamplerUniform::new(((0, 0), (res.0 as u32, res.1 as u32)),spp,false,Some(0)) ));
    let cameraInstance =  PerspectiveCamera::from_lookat(Point3f::new(0.0,-0.500, 0.50), Point3f::new(0.0,0.0,1.00), 1e-2, 100.0, 90.0, res);
    let scene  =  Arc::new (
        Scene1::make_scene(
            res.0 as usize, 
            res.1 as usize, 
            vec![], 
            vec![
                PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.0,0.0,3.0), 2.0, materials::MaterialDescType::NoneType))
                //    PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 0.0,  3.0), Matrix3::from_angle_x(Deg(-70.0)),-1.0, 1.0,1.0, materials::MaterialDescType::NoneType))
                ], 
            1,1));
    let filtergauss  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0,2.0),3.0));
    let mut film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))) , filtergauss); 
    let mut wholetile = film.init_tile(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))));
     let mut sampler = sampler2.seed_instance();
    for iy in 0..res.1{
        for ix in 0..res.0{
            sampler.start_pixel(Point2i::new(ix as i64, iy as i64)); 
            let samplecamera = sampler.get2d();
            let sampleforcamerauv = to_map(&samplecamera, res.0 as i64  , res.1 as i64);
            let ray = cameraInstance.get_ray(&sampleforcamerauv);
            let ishit =  get_ray_path_1(&ray,&scene.lights, &scene, 1, &mut sampler );
            if ishit {num_total_of_hits.fetch_add(1, Ordering::SeqCst);}
            // wholetile.add_sample(samplecamera, v)
            wholetile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), Srgb::new(1.0, 1.0,0.0));
            while sampler.start_next_sample(){
                let samplecamera = sampler.get2d();
                let sampleforcamerauv = to_map(&samplecamera, res.0 as i64  , res.1 as i64);
                let ray = cameraInstance.get_ray(&sampleforcamerauv);
                let ishit =  get_ray_path_1(&ray,&scene.lights, &scene, 1, &mut sampler );
                if ishit {num_total_of_hits.fetch_add(1, Ordering::SeqCst);}
                wholetile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), Srgb::new(1.0, 1.0,0.0));
            }
            
        }
    }
    film.merge_tile(&wholetile);
    film.commit_and_write("filtertest", "miexample_brute_force.png",true).unwrap();
    println!("num_total_of_hits --------------->{:?}",num_total_of_hits ); 
    println!(" time: {} seconds", start.elapsed().as_secs_f64());

}
  
pub fn with_integrator_api_par_render(){

    let start = Instant::now();
    let res = (512_i64, 512_i64);
    let spp = 256_i64;
    let total_rays:i64 = 1024*1024*2048;
    println!("{}", total_rays);
   
    let filtergauss  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0,2.0),3.0));
    let filterbox  =  FilterFilm::FilterFilmBoxType(FilterFilmBox::new( (1.0,1.0)));
    let mut film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))) , filtergauss); 
    let num_workers =  12;// num_cpus::get();
    let total_area : AtomicI64 = AtomicI64::new(0);
    let num_total_of_hits: AtomicI64 = AtomicI64::new(0);
    let num_total_of_pixel: AtomicI64 = AtomicI64::new(0);
  
     let sampler2 = Arc::new(SamplerType::UniformType( SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),spp as u32,false,Some(0)) ));
     let cameraInstance =  PerspectiveCamera::from_lookat(Point3f::new(0.0,-0.500, 0.50), Point3f::new(0.0,0.0,1.00), 1e-2, 100.0, 90.0, (res.0 as u32, res.1 as u32));
   
    let scene  =  Arc::new (
        Scene1::make_scene(
            res.0 as usize, 
            res.1 as usize, 
            vec![], 
            vec![
                PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.0,0.0,3.0), 2.0, materials::MaterialDescType::NoneType))
                //    PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 0.0,  3.0), Matrix3::from_angle_x(Deg(-70.0)),-1.0, 1.0,1.0, materials::MaterialDescType::NoneType))
                ], 
            1,1));
    crossbeam::scope(|scope| {

        let (sendertorender0, receiverfrominitialize0) = crossbeam_channel::bounded(num_workers);
      
        let filmref =  &film;
        let camera = &cameraInstance;
        let sceneshared = &scene;
        
    
        scope.spawn(move|_|{
            let mut area = 0 ;
            let sender0clone = sendertorender0.clone();
            let vlst =    init_tiles_withworkers(num_workers as i64, filmref.width(), filmref.height());
            let mut cnt = 0;
            for tilescoords in vlst.into_iter(){
               
                let tile = filmref.init_tile(BBds2i::new(Point2::new(tilescoords.1.0.0  as  i64 ,tilescoords.1.0.1 as  i64,),Point2::new(tilescoords.1.1.0 as  i64 ,tilescoords.1.1.1 as  i64  )));
                
                println!("init_tiles_withworkers -> thread id {:?}", std::thread::current().id()) ;
                let packedwork = (tilescoords, tile);
                    sender0clone.send(packedwork).unwrap_or_else(|op| panic!("---sender error!!--->{}", op));     
            }
            // println!("{}", area);
            drop(sender0clone)
        });
       
        for _ in 0..num_workers {
            let mut sampler = sampler2.seed_instance();
            let atom_hits_ref =&num_total_of_hits;
            let num_total_of_px_ref = &num_total_of_pixel;
            let rxclone = receiverfrominitialize0.clone();
            scope.spawn(move|_|{
                
             
                for mut packedwork in  rxclone.recv().into_iter() {
                  
                    println!("render worker -> thread id {:?}", std::thread::current().id()) ;
                    println!("       compute tile {:?}",  packedwork.0) ;
                
                    //unfold the stuff send by thread "parent"
                    let mut tile = & mut packedwork.1;
                    let tilescoords =   packedwork.0;

                    let startx = tilescoords.1.0.0;
                    let starty = tilescoords.1.0.1;
                    let endx = tilescoords.1.1.0 ;
                    let endy =tilescoords.1.1.1;
            
                    for px in  itertools::iproduct!(startx..endx,starty..endy ){
                        let x = px.0  as f64;
                        let y = px.1  as f64;
                        let r =   (x - startx as f64) as  f32 / (endx - startx) as  f32;
                        let g=   (y - starty as f64) as  f32 / (endy - starty)as  f32;
                        sampler.start_pixel(Point2i::new(x as i64, y as i64));
                    
                        let samplecamera = sampler.get2d();
                        debug_assert_eq!(samplecamera.0.floor()as usize , x as usize);
                        debug_assert_eq!(samplecamera.1.floor()as usize , y as usize );
                        
                        let sampleforcamerauv = to_map(&samplecamera, endx  , endy as i64);
                        let ray = cameraInstance.get_ray(&sampleforcamerauv);
                        let ishit =  get_ray_path_1(&ray,&sceneshared.lights, &sceneshared, 1, &mut sampler );
                        if ishit {atom_hits_ref.fetch_add(1, Ordering::SeqCst);}
                        num_total_of_px_ref.fetch_add(1, Ordering::Acquire);
                        tile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), Srgb::new(r, g,0.0));
                        while sampler.start_next_sample(){

                                let samplecamera = sampler.get2d();
                                debug_assert_eq!(samplecamera.0.floor()as usize , x as usize);
                                debug_assert_eq!(samplecamera.1.floor()as usize , y as usize );
                                let sampleforcamerauv = to_map(&samplecamera, endx  , endy as i64);
                                let ray = cameraInstance.get_ray(&sampleforcamerauv);
                                let ishit = get_ray_path_1(&ray,&sceneshared.lights, &sceneshared, 1, &mut sampler );
                                if ishit {atom_hits_ref.fetch_add(1, Ordering::SeqCst);}
                                num_total_of_px_ref.fetch_add(1, Ordering::Acquire);
                                tile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), Srgb::new(r, g,0.0));
                        }
                            // println!("num_total_of_px --------------->{:?}", num_total_of_px_ref );
                        
                          

                    } 

                     filmref.merge_tile(&tile);
                 
                 }

                

             });
        }

    }).unwrap();
 
     println!("num_total_of_hits --------------->{:?}",num_total_of_hits );
     println!("num_total_of_px --------------->{:?}",num_total_of_pixel );
     let  mut acc  = 0.0;
    let  mut acc_color  = 0.0;
   if film.commit_and_write("filtertest", "miexample2.png",true).is_ok(){
    println!("" );
   }

     
    println!(" time: {} seconds", start.elapsed().as_secs_f64());
 
}






