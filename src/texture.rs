#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unused_parens)]
use crate::raytracerv2::{clamp, lerp, remainder};
use cgmath::Angle;
use cgmath::*;
use float::Trig;
use image::codecs::png::{PngDecoder, PngEncoder};
use image::{GenericImageView, Pixel, ImageBuffer};
use num_traits::Float;
use palette::rgb::Rgb;
use palette::Srgb;
use std::f64;
use std::fmt::Debug;
use std::fs::File;
use std::marker::Sized;
use std::{f32, string};

use image::io::Reader as ImageReader;
use std::io::Cursor;
use std::path::Path;

pub trait EvalTexture {
    fn eval(&self, p: &Point3<f64>) -> Srgb;
}
pub trait Mapping2d: std::fmt::Debug + Sized {
    fn to_map(self: &Self, p: &Point3<f64>) -> (f64, f64);
    // fn to_cylindrical(&self)->();
    // fn to_uv(&self)->();
    // fn to_plane(&self)->();
}
#[derive(Debug, Clone)]
pub struct Texture2D<R, M: Mapping2d>
where
    R: Copy,
{
    pub data: Option<Vec<R>>,
    pub w: usize,
    pub h: usize,
    pub ch: usize,
    pub local: Option<Matrix4<f64>>,

    pub mapping: M,
}
impl<R, M> Texture2D<R, M>
where
    M: Mapping2d,
    R: Copy + Debug,
{
    pub fn new(r: R, data: Option<Vec<R>>, w: usize, h: usize, ch: usize, m: M) -> Texture2D<R, M> {
        Texture2D {
            data: Some(vec![r]),
            w,
            h,
            ch,
            local: None,
            mapping: m,
        }
    }
    pub fn createConstantText(c: &Srgb) -> Texture2DSRgbMapConstant {
        Texture2DSRgbMapConstant {
            data: Some(vec![*c]),
            w: 1,
            h: 1,
            ch: 1,
            local: None,
            mapping: MapConstant {},
        }
    }
    pub fn createTexture2DSRgbShpericalImg() -> Texture2DSRgbSpherical {
        // println!("createTexture2DSRgbShpericalImg");
        let mut bytes: Vec<u8> = Vec::new();
        
        let i = ImageReader::open("grid.png").unwrap();
        let mut ii = i.with_guessed_format().unwrap().decode().unwrap();
        let h = ii.height();
         let w = ii.width();
         println!("grid.png {} {}", w, h);
        // ii.fliph()
        //     .write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png)
        //     .unwrap();
        let mut vv: Vec<Srgb> = vec![Srgb::new(0.0,0.0,0.0); ( h * w )as usize];
        for p in ii.pixels() {
            let x = p.0;
               let y = p.1;
            let co = p.2.to_rgba();
            let channels = co.channels();
            let c = channels[0] as f32 / 255.0;
            let c1 = channels[1] as f32 / 255.0;
            let c2 = channels[2] as f32 / 255.0;
            vv[ (w*(h-1-y)+x ) as usize ] = Srgb::new(c, c1, c2);
             
        }
    //     let h = 32;
    //     let w = 32;
    //     let mut ii = ImageBuffer::from_fn(w, h, |x, y| {
    //        let ny = ( (y as f32 / h as f32 )*255.0 )as u32;
    //         image::Rgb([0, 0, ny])
    //     });
     
    //     let mut vv: Vec<Srgb> = vec![Srgb::new(0.0,0.0,0.0); (h * w )as usize];
    //    for e in ii.enumerate_pixels(){
    //         let x = e.0;
    //         let y = e.1;
    //         let co = e.2.to_rgba();
    //        let channels =  co.channels();
    //        let c = channels[0] as f32 / 255.0;
    //      let c1 = channels[1] as f32 / 255.0;
    //          let c2 = channels[2] as f32 / 255.0;
    //         //  println!("{},{}, {:?}",x, y ,Srgb::new(c, c1, c2));
    //          vv[ (w*y +x ) as usize ] = Srgb::new(c, c1, c2);
         

            
    //    }
 //       println!("{:?}", vv);
        Texture2DSRgbSpherical {
            data: Some(vv),
            w: ii.width() as usize,
            h: ii.height() as usize,
            ch: 1,
            local: None,
            mapping:MapSpherical{}
        }
    }
    pub fn createTexture2Df64MapConstant(c: &f64) -> Texture2Df64MapConstant {
        Texture2Df64MapConstant {
            data: Some(vec![*c]),
            w: 1,
            h: 1,
            ch: 1,
            local: None,
            mapping: MapConstant {},
        }
    }
    pub fn createTexture2DSRgbMapUV() -> Texture2DSRgbMapUV {
        let bytes: Vec<Srgb> = (0..100)
            .into_iter()
        .map(|(i)| Srgb::new(i as f32 / 10.0, i as f32/10.0, 1.0))
          
            .collect();
        Texture2DSRgbMapUV {
            data: Some(bytes),
            w: 10,
            h: 10,
            ch: 1,
            local: None,
            mapping: MapUV {
                scale: 1.0,
                translate: 0.0,
            },
        }
    }
    pub fn createTexture2DSRgbMapUVImg() -> Texture2DSRgbMapUV {
        println!("img!");
     //   let mut bytes: Vec<u8> = Vec::new();
         
        let i = ImageReader::open("grid.png").unwrap();
        let mut ii = i.with_guessed_format().unwrap().decode().unwrap();

        

       let mut  vv :Vec<Srgb> =vec![Srgb::new(0.0,0.0,0.0);(ii.height()*ii.width()) as usize] ;
        
        for p in ii.pixels() {
            let x = p.0;
            let y = p.1;
            let co = p.2.to_rgba();
            let channels = co.channels();
         
            let c = channels[0] as f32 / 255.0;
            let c1 = channels[1] as f32 / 255.0;
            let c2 = channels[2] as f32 / 255.0;

            vv[ (ii.width()*(y)+x ) as usize ] = Srgb::new(c, c1, c2);
        }
        println!("img end! {}", vv.len());
        Texture2DSRgbMapUV {
            data: Some(vv),
            w: ii.width() as usize,
            h: ii.height() as usize,
            ch: 1,
            local: None,
            mapping: MapUV {
                scale: 1.0,
                translate: 0.0,
            
            },
        }
    }
    pub fn get(&self, i: usize, j: usize) -> R {
        //  let ii =   remainder(i as i32, ((self.w-1) as i32)) as usize;
        //   let jj=   remainder(j as i32, ((self.h-1) as i32)) as  usize;
        let ii = clamp(i, 0, self.w - 1);
        let jj = clamp(j, 0, self.h - 1);
        
        let p = jj * (self.w * self.ch) + ii;
        //   println!("pix {:?},{:?}", ii, jj);
    //    println!("p ,{} ,x->{} y->{}", p, p % self.w, p /self.h);
     //  println!("  ,x->{} y->{}", ii, jj);
        self.data.as_ref().unwrap()[p]
    }
    pub fn eval(&self, p: &Point3<f64>) -> R {
        let st = self.mapping.to_map(p);
     //   println!("st {:?}", st);
        self.get(
            (st.0 * (self.w) as f64) as usize,
            (st.1 * (self.h) as f64) as usize,
        )
    }
}

#[derive(Debug, Clone)]
pub struct MapSpherical {}
impl Mapping2d for MapSpherical {
    fn to_map(self: &Self, p: &Point3<f64>) -> (f64, f64) {
        let v = (p - Point3::new(0.0, 0.0, 0.0)) ;
         
        let mut s = v.y.atan2(v.x);
 

        if (s < 0.0) {
            s = s + f64::consts::PI * 2.0;
        };
        s = s / (f64::consts::PI * 2.0); 
        let mut t = clamp(v.z, -1.0, 1.0).acos() *  f64::consts::FRAC_1_PI;
          
        (s, t)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MapCylindrical {}
impl Mapping2d for MapCylindrical {
    fn to_map(self: &Self, p: &Point3<f64>) -> (f64, f64) {
        let v = (p - Point3::new(0.0, 0.0, 0.0)) ;
        let s = v.z;
        let t = (f64::consts::PI + v.y.atan2(v.x)) * f64::consts::FRAC_2_PI;
        (s, t)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MapUV {
    scale: f64,
    translate: f64,
}
impl Mapping2d for MapUV {
    fn to_map(self: &Self, p: &Point3<f64>) -> (f64, f64) {
        let s = self.translate + p.x * self.scale;
        let t = self.translate + p.y * self.scale;

        (s, t)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MapConstant {}
impl Mapping2d for MapConstant {
    fn to_map(self: &Self, p: &Point3<f64>) -> (f64, f64) {
        (0.0, 0.0)
    }
}
pub type Texture2Df64Spherical = Texture2D<f64, MapSpherical>;
pub type Texture2Du8Spherical = Texture2D<u8, MapSpherical>;
pub type Texture2Du32Spherical = Texture2D<u32, MapSpherical>;
pub type Texture2DSRgbSpherical = Texture2D<Srgb, MapSpherical>;

pub type Texture2DMapCylindrical = Texture2D<f64, MapCylindrical>;
pub type Texture2DMapUV = Texture2D<f64, MapUV>;
pub type Texture2Df64f64MapUV = Texture2D<(f64, f64), MapUV>;
pub type Texture2DSRgbMapUV = Texture2D<Srgb, MapUV>;
pub type Texture2Du8MapUV = Texture2D<u8, MapUV>;
pub type Texture2DSRgbMapConstant = Texture2D<Srgb, MapConstant>;
pub type Texture2DRbMap = Texture2D<u8, MapConstant>;
pub type Texture2Df64MapConstant = Texture2D<f64, MapConstant>;

#[derive(Debug, Clone)]
pub enum TexType {
    None,
    Texture2DSRgbSpherical(Texture2DSRgbSpherical),
    // Texture2Du8MapUV(Texture2Du8MapUV),
    Texture2DSRgbMapConstant(Texture2DSRgbMapConstant),
    Texture2DSRgbMapUV(Texture2DSRgbMapUV),
    Texture2Df64MapConstant(Texture2Df64MapConstant),
}

impl EvalTexture for TexType {
    fn eval(&self, p: &Point3<f64>) -> Srgb {
        match self {
            Self::None => Srgb::new(0.0, 0.0, 0.0),
            Self::Texture2DSRgbMapConstant(tx) => tx.eval(p),
            Self::Texture2DSRgbMapUV(tx) => {
              //    println!("{:?}", p);
                tx.eval(p)
            }
            Self::Texture2Df64MapConstant(tx) => {
                // println!("{:?}", p);
                let f = tx.eval(p) as f32;
                Srgb::new(f, f, f)
            }
            Self::Texture2DSRgbSpherical(tx) => {
                // println!("{:?}", p);
                tx.eval(p)
            }
        }
    }
}

#[test]
fn test_texture2du8_spherical() {
    let w = 32u32;
    let h = 32u32;
    let bytes: Vec<u32> = (0..w * h).collect();

    let t = Texture2Du32Spherical {
        data: Some(bytes),
        w: w as usize,
        h: h as usize,
        ch: 1,
        local: None,
        mapping: MapSpherical {},
    };
    t.eval(&Point3::new(0.0,0.0,1.0));
    t.eval(&Point3::new(0.0,0.0,0.00001));
    t.eval(&Point3::new(0.0,0.0,-1.0));
    for i in 0..100 {
        let xx = (i as f64) / 100.0;
        //  println!("{}",  t.eval(&Point3::new(0.0,1.0,xx)));
    }
    for i in 0..100 {
        let xx = (((i as f64) / 100.0) * f64::consts::PI * 2.0).sin_cos();
        t.eval(&Point3::new(xx.0, xx.1, 0.0));
    }
}

fn test_texture2du8Mapuv() {
    let w = 10;
    let h = 10;
    let bytes: Vec<u8> = (0..(w) * (h)).collect();
    println!("{:?}", bytes);
    let t = Texture2Du8MapUV {
        data: Some(bytes),
        w: w as usize,
        h: h as usize,
        ch: 1,
        local: None,
        mapping: MapUV {
            scale: 1.0,
            translate: 0.0,
        },
    };
    for (ii, jj) in std::iter::zip((0..11).into_iter(), (0..11).into_iter()) {
        let xx = ii as f64 / 10.0;
        let yy = jj as f64 / 10.0;
        println!(
            "x: {}, y: {}, res:{}",
            xx,
            yy,
            t.eval(&Point3::new(xx, yy, 0.0))
        );
    }
    for (ii, jj) in std::iter::zip((0..20).into_iter(), (0..13).into_iter()) {
        let xx = ii as f64 / 10.0;
        let yy = jj as f64 / 10.0;
        let resscale = xx * 100.0 - 50.0;
        println!(
            "x: {}, y: {}, res:{}",
            resscale,
            resscale,
            t.eval(&Point3::new(resscale, resscale, 0.0))
        );
    }
}
#[test]
fn test_Texture2DSRgbMapUV() {
    //  let mut bytes: Vec<u8> = Vec::new();
    //  let i =  ImageReader::open("grid.png").unwrap();
    //  let ii =  i.with_guessed_format(). unwrap(). decode().unwrap();
    // ii.write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png).unwrap();
    // let t = Texture2Du8MapUV{data:Some(bytes),w:ii.width()as usize,h:ii.height()as usize,ch:ii.color().channel_count()as usize,local:None,mapping:MapUV{scale:1.0,translate:0.0}  };

    //    let  bytes: Vec<Srgb> = (0..(w)*(h)).map(|f|)
    let bytes: Vec<Srgb> = (0..100)
        .into_iter()
        .map(|(i)| Srgb::new(i as f32, i as f32, i as f32))
        .collect();
    let t = Texture2DSRgbMapUV {
        data: Some(bytes),
        w: 10,
        h: 10,
        ch: 1,
        local: None,
        mapping: MapUV {
            scale: 1.0,
            translate: 0.0,
        },
    };
    for (ii, jj) in std::iter::zip((0..11).into_iter(), (0..11).into_iter()) {
        let xx = ii as f64 / 10.0;
        let yy = jj as f64 / 10.0;
        println!(
            "x: {}, y: {}, res:{:?}",
            xx,
            yy,
            t.eval(&Point3::new(xx, yy, 0.0))
        );
    }
    println!("fin");
    // Texture2DSRgbMapUV{}
    //    DataB{data:bytes.clone()};

    //  //  println!("{:?}", i.with_guessed_format().unwrap().format().unwrap());
    //    let ii =  i.with_guessed_format(). unwrap(). decode().unwrap();
    //    println!("{:?} , {:?}",ii.color().channel_count(),ii.color().bits_per_pixel());
    //   ;

    //   ii.write_to(&mut Cursor::new(&mut bytes), image::ImageOutputFormat::Png).unwrap();

    //     let t = Texture2DSRgbMap{
    //          data:Some(bytes),w:ii.width()as usize,h:ii.height()as usize,local:None,mapping:MapConstant{}
    //      };
    //     println!("{:?}",t);
    //  Texture2D::createConstantText(&Srgb::new(1.0, 1.0, 1.0));
    //     let map  =   MapCylindrical{};
    //     let mapUV  =   MapUV{scale:1.0, translate:0.0};
    //   let text : Texture2DMapCylindrical = Texture2DMapCylindrical::new(None,  1,1,map);
    //   text.eval(&Point3::new(0.0,0.0,0.0));
}
