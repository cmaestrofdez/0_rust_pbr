 
use std::ffi::OsString;
use std::fmt::Debug;
use std::fs::File;
use std::fs::ReadDir;
use std::fs::*;
use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Rem;
use std::path::Path;
use std::process::Output;
use std::*;
use image::RgbImage;
 
use image :: {GenericImage , ImageBuffer};
extern crate float;

// cargo watch -c -x  run clean

fn dot<N>(v0: &[N], v1: &[N]) -> N
where
    N: Default + Add<Output = N> + Mul<Output = N> + Copy,
{
    let mut res: N = N::default();
    for i in 0..v0.len() {
        res = res + v0[i] + v1[1];
    }
    res
}
#[derive(Clone, Copy, Debug)]
struct Point<N> {
    x: N,
    y: N,
}

impl<N> Point<N> {
    fn new(x: N, y: N) -> Point<N>
    where
        N: Default,
    {
        Point { x: x, y: y }
    }
}
impl<N> Default for Point<N>
where
    N: Default,
{
    fn default() -> Self {
        Point {
            x: N::default(),
            y: N::default(),
        }
    }
}

impl<N> Add for Point<N>
where
    N: Add<Output = N> + Default,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Point::default()
    }
}
// Point + Point

impl<T: Mul<S>, S: Copy> Mul<S> for Point<T> {
    type Output = Point<<T as Mul<S>>::Output>;

    fn mul(self, scalar: S) -> Self::Output {
        todo!()
    }
}

// -x
impl<N> Neg for Point<N>
where
    N: Default + Neg<Output = N>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        // Point{self.x,self.y};
        Point::new(-self.x, -self.y)
    }
}

// not !x
impl<N> Rem for Point<N>
where
    N: Rem<Output = N> + Default + Copy,
{
    type Output = Self;
    fn rem(self, rhs: Self) -> Self::Output {
        Point::new(self.x % rhs.x, self.x % rhs.y)
    }
}
use float::Float;
use float :: Sqrt;
impl<T:Copy + Mul<Output=T> + Add<Output=T> +  Add<Output=T> +Sqrt> Point<T>{
    fn sqrt(self)->T{
        (self.x *  self.x )+(self.y *  self.y).sqrt()
    }
}
impl <T : Copy + Add<u32,Output=T> + FromPrimitive +Default> Point<T>{
    fn normal(&self, from:(u32, u32))->Point<<T as Add<u32>>::Output>{
       let a =  self.x + from.0;
       let b  =  self.x + from.0;
     
        Point ::new(a,b)
    }
}

use num_traits::Bounded;
use num_traits::cast::*;

use  std::ops::Div;
impl <T: Copy+ Default+Div<Output=T>> Div for Point<T> {
    type Output = Point<T>;
    fn div(self, rhs: Self) -> Self::Output {
        Point::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl<T : Default + FromPrimitive  + num_traits::Float , S : Div<T,Output=T>+Copy > Div<(S, S)> for Point<T>{
    type Output = Point<T>;

    fn div(self, rhs: (S, S)) -> Self::Output {
    

       Point::new(rhs.0 / self.x , rhs.0 / self.x )

    }
    
}

impl <T : Copy +Default + Mul<Output=T> +Div<Output=T> + Add<Output=T>+Sqrt> Point<T> {
    fn normalize(self)->Point<T>{
       self/ Point::new(  self.sqrt(),self.sqrt())
           
    }
}















#[derive(Clone, Copy, Debug)]
struct  Point3<T>{
    x : T, 
    y : T, 
    z : T
}
impl  <T> Point3<T> {
    pub fn new(x:T, y:T, z:T)->Point3<T>{
        Point3{x:x, y:y, z:z}
    }
  
}
impl  <T : Copy+ Default +Clone + Sqrt + Add<Output = T > + Mul<Output = T >> Point3<T> {
    fn length(self)-> T {
        (self.x + self.x + self.y *self.y + self.y * self.z).sqrt()
    }
}
impl  <T :Copy + Clone + std::ops::Div<Output=T>> Div for Point3<T>{
    type Output =  Self;
    fn div(self, rhs: Self) -> Self::Output {
        Point3::new( self.x / rhs.x,
            self.y/ rhs.y,
            self.z / rhs.z)
    }
    
}

impl <T : Copy + Default + Clone +Sqrt + Div<Output = T> + Add<Output=T> + Mul<Output = T>> Point3<T>{
     fn normalize(self) -> Point3<T>{
        let t = self.length();

        Point3::new(  self.x / t,self.x / t ,self.x / t )
        
    }

}
impl  <T : Default> Default for Point3<T> {
    fn default() -> Self {
        Point3{x:T::default(), y:T::default(), z:T::default()}
    }
}

impl <T : Copy + Clone + Add<Output=T> +Default > Add for Point3<T> {
    type Output =  Self;
    fn add(self, rhs: Self) -> Self::Output {
        Point3::new(self.x + rhs.x ,self.y + rhs.y ,self.z+ rhs.z )
    }
}
use std::ops::Sub;
impl <T : Copy + Clone + std::ops::Sub<Output=T>> Sub for Point3<T>{
    type Output =  Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Point3::new(self.x - rhs.x, self.y - rhs.y ,  self.z- rhs.z )
    }
}
impl <T:Clone +Copy + Mul<Output=T> +Add<Output=T>>  Point3<T>{
    fn dot(a : Point3<T>, b:Point3<T>)->T{
        a.x * b.x + a.y * b.y + a.z * b.z
    }
}
impl <T : Copy + Clone + Add<Output=T>> AddAssign for Point3<T> {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self{
            x : self.x+rhs.x,
            y : self.y+rhs.y,
            z : self.z+rhs.z
        }
    }
}
pub trait  Cross <Rhs=Self>{
    type  Output;
    fn cross(self, rhs:Self)->Self;
}
impl <T: Copy+Clone+Sub<Output = T>+ Mul<Output=T>> Cross<Point3<T>> for Point3<T> {
    type Output = Point3<T>;
    fn cross(self, other:Self) ->Self::Output {
        Point3{
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
       
    }    
}
// impl <T : Copy + Clone + Add<Output=T> +Default > Add for Point3<T> {
//     type Output =  Point3<T>;
//     fn add(self, rhs: T) -> Self::Output {
//         Point3::new(self.x + rhs ,self.y + rhs ,self.z )
//     }
// }
// https://canmom.art/programming/graphics/raytracer/rust-raytracer-part-3
//  https://canmom.art/programming/graphics/raytracer/rust-raytracer-part-3

#[derive(Clone, Copy, Debug)]

struct  Triangle<T>{
    p0 : Point3<T>,
    p1: Point3<T> ,
    p2 : Point3<T>,
    centroid : Point3<T>
}

impl <T : Default + Copy + Add> Default for Triangle<T>{
    fn default() -> Self {
         Triangle::new(Point3::default(),Point3::default(),Point3::default())
    }
}
impl <T : Default + Copy + Add> Triangle<T>{
    fn new(p0:Point3<T>,p1:Point3<T>,p2:Point3<T>)->Triangle<T>{
        Triangle::default() 
    }
    
} 

trait  Primitive : NumCast {
    
}

impl  Primitive for u8 {}
impl  Primitive  for u32 {}
impl  Primitive  for f32 {}
trait Lerrp : NumCast + Bounded +Add {
    type Ratio : Primitive  ;
    fn lerp(a:Self, b :Self, ratio : Self::Ratio){
       let a =  <Self::Ratio as NumCast>::from(a).unwrap();
       let b =  <Self::Ratio as NumCast>::from(b).unwrap();
   //    a+b;
    }
}
impl Lerrp for u32 {
    type Ratio = u32;
}
impl Lerrp for u8 {
    type Ratio = u8;
}
fn mi_lerp<T : Lerrp>(a:T, b :T, t : T::Ratio){
    Lerrp::lerp(a, b, t)
}
pub trait Px : Clone + Copy +Debug{
    type PxType : Primitive;
    fn channels(&self)->&[Self::PxType];
    //  fn channels_mut(&self)->&mut[Self::PxType];
    //  fn channel4(&self)->(Self::PxType, Self::PxType ,Self::PxType, Self::PxType);
    //  fn to_luma(&self)-> Luma<Self::PxType>;
    //  fn apply <F>(&mut self, f : F)->Self where F : FnMut(Self::PxType)->Self::PxType;
}

#[derive(Debug , Clone, Copy  )]
struct Luma<Primitive>{
    y : Primitive
}

#[derive(Debug , Clone, Copy  )]
struct Rgba<  Primitive >{
    
    r  :Primitive,
    g  :Primitive,
    b  :Primitive ,
    a  : Primitive , 
}
impl Px for Rgba<u32> {
    type PxType  = u32;
    fn channels(&self)->&[Self::PxType] {
         &[self.r]
    }

    // fn channels_mut(&self)->&mut[Self::PxType] {
    //     &mut[self.r]
    // }

    // fn channel4(&self)->(Self::PxType, Self::PxType ,Self::PxType, Self::PxType) {
    //      (self.r,self.r,self.r,self.r)
    // }
    // fn to_luma(&self) -> Luma<Self::PxType> {
    //    let r =  self.r * 0.21 +self.g * 0.71 +self.g * 0.071;
    //     Luma{y:self.r * 0.21 +self.g * 0.71 +self.g * 0.071}
    // }
    // fn apply<F>(&mut self, f : F)->Self
    // where F : FnMut(Self::PxType)->Self::PxType {
    //     todo!()
    // }
  
    
}
// hacer un simple ray tracer!
//  https://github.com/dps/rust-raytracer

fn main() -> Result<()> {
    let yy : u8 = 0;
    let yy1 : u8 = 0;
    let t : u8 = 0;
    mi_lerp(yy, yy1, t);
    
    // const w : u32 = 5;
    // const h : u32 = 5;
    // let im= ImageBuffer::from_pixel(5, 5, image::Rgb([0.,0.,0.]));
    // for p  in im.enumerate_pixels(){
    //         // let pnormal : (f32,f32) = ( p.0 as f32 / w as f32,p.1 as f32 / h as f32);
        
    //          }
    // let rgb: RgbImage = RgbImage::new(10, 10);
    // let imbuf = RgbImage::from_fn(10, 10, |x,y |{
    //   let rg =   (( (y as f32 ) / 10.) * 255. ) as u8;
    //     image::Rgb([rg,rg, 255])
    // });
    // image::DynamicImage::ImageRgb8(imbuf).to_rgb8().save("mipng.png").unwrap();
    // im.save("mipng.png").unwrap();
//    let mut ima =  image::open("tar.jpg").unwrap().into_rgb8();
//    println!("{:?}", ima.height());
//    println!("{:?}", ima.width());
//   let mut hh =  ima.height();
//   hh /=3;
//    image::imageops::crop( &mut ima, hh,hh, 320, 320).to_image().save("tar_png.png");
 for o in RgbImage::new(10,10).enumerate_pixels(){
   let p = Point::new( o.0 as f32 / 10.0,o.1 as f32 /10.0 * 2.0 -1.0 ) ;
   
   println!("{:?}", p);
 }
    return Ok(())
//    


//     let p : Point3<f32> = Point3::default();
//     let p1 : Point3<f32> = Point3::default();
//     let p  =p + p1;
//     let p  =p + p;
//     {
        

//     let p : Point3<f32> = Point3::default();
//     let p1 : Point3<f32> = Point3::default();
//     p -p1;
//     let p1 = p.clone();
//     let p2 = p.clone();
//     let p1 = p1 / p2;
//     let p3=p1.normalize();
    
//     }
    

// //     let p: Point<f32> = Point::new(0., 0.);
// //     let p1: Point<f32> = Point::new(0., 0.);
    
// //    let p2  =  p / p1;
// //    let p3: Point<f32> = Point::new(0., 0.);
// //   let p3 = p3.normalize();
  
    
//     Ok(())
   
}
