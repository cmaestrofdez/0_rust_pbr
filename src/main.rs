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
use std::ops::Div;
use std::ops::Index;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Rem;
use std::ops::Sub;
use std::path::Path;
use float::Sqrt;
// use float::Sqrt;
use image::codecs::png::PngEncoder;
use image::ImageEncoder;

use palette::rgb::Rgb;
// use num_traits::Float;
use rand::Rng;
// use std::process::Output;
use image::RgbImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::*;

use palette::{Pixel, Srgb};
use std::time::Instant;

#[derive(
    Debug,
    Clone,
    // Deserialize, Serialize
)]
pub struct Scene {
    pub num_samples: usize,
    pub primitives: Vec<Sphere>,
    pub max_depth: usize,
    pub width: usize,
    pub height: usize,
    pub lights: Vec<Light>,
    pub sky: Option<Sphere>,
}

impl Scene {
    pub fn make_scene(width: usize, height: usize, lights :Vec<Light>, primitives :  Vec<Sphere>, num_samples: usize, max_depth:usize) -> Scene {
        Scene {
            num_samples  ,
            max_depth ,
            width,
            height,
            lights,
            primitives,
            sky: None,
        }
    }
}

pub fn clamp(t: f32) -> f32 {
    if t < 0.0 {
        0.0
    } else if t > 1.0 {
        1.0
    } else {
        t
    }
}

//todo: hacer que esto sea un point light
// hacer un interface generico para Light.
#[derive(Debug, Clone, Copy)]
pub struct Light {
    center: Point3d,
    color: Srgb,
}

impl Light {
    fn new(center: Point3d, color: Srgb) -> Light {
        Light { center, color }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Material {
    Lamber(Lambertian),
    Metal(Metal),
}

#[derive(Debug, Clone, Copy)]
pub struct Lambertian {
    pub albedo: Srgb,
}
impl Lambertian {
    pub fn new(albedo: Srgb) -> Lambertian {
        Lambertian { albedo }
    }
}
impl Scatterable for Lambertian {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<(Option<Ray>, Srgb)> {
        let dir =     hit.normal  +  Point3d::random_unit_vector();
        let target  = hit.point + dir ;
        Ray::new(hit.point , target);
        Some((Some(Ray::new(hit.point , target)), self.albedo))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Metal {
    pub albedo: Srgb,
    pub fuzz: f64,
}
impl Metal {
    pub fn new(albedo: Srgb, fuzz: f64) -> Metal {
        Metal { albedo, fuzz }
    }
}

mod materials {
    use super::*;
    pub fn reflect(v :&Point3d, n:&Point3d)->Point3d{
        *v - *n *(2.0 * v.dot(n))
    }
}
impl Scatterable for Metal {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<(Option<Ray>, Srgb)> {
       let reflect =  materials:: reflect(&ray.direction, &hit.normal);
    //    
       let scattered  = Ray::new(hit.point, reflect + Point3d::random_unit_vector() *self.fuzz);
       let albedo =  self.albedo;
       if scattered.direction.dot(&hit.normal)>0.0{
          Some((Some(scattered), albedo))

       }else{
        None
       }
        
    }
}
#[test]
pub fn test_metal(){
   let metal = Metal::new(Rgb::new(1.0, 1.0, 1.0), 1.0);

   for y in 0..100{
    let mut  yy =  y as f64 / 100.0;
    let orig = Point3d::new(0.0, yy *2.0 -1.0, 2.0);
    let r =  Ray::new(orig,  Point3d::new(0.0,yy *2.0 -1.0, -100.0) - orig);
    let sphere = Sphere::new(Point3d::new(0.0, 0.0, 0.0), 1.0, Material::Metal(metal));
    if let Some(hit) = sphere.hittable(&r, 0.001, f64::MAX){
        if let Some(sample)=sphere.material.scatter(&r, &hit){
            println!("{}",  hit.normal.dot(&sample.0.unwrap().direction.unit_vector()).acos().to_degrees());
            
            
            println!("{:?}", sample.0.unwrap().direction.unit_vector());
        }
       
    }


   }

   


}

impl Scatterable for Material {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<(Option<Ray>, Srgb)> {
        match self {
            Material::Lamber(l) => {
                l.scatter(ray, hit)
                
            }
            Material::Metal(m) => {
                m.scatter(ray, hit)
                
            }
        }
    }
}

pub trait Scatterable {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<(Option<Ray>, Srgb)>;
}

impl Scatterable for Light {
    fn scatter(&self, ray: &Ray, hit: &HitRecord) -> Option<(Option<Ray>, Srgb)> {
        None
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Point3d {
    x: f64,
    y: f64,
    z: f64,
}

impl Point3d {
    pub fn new(x: f64, y: f64, z: f64) -> Point3d {
        Point3d { x, y, z }
    }
    pub fn random(min: f64, max: f64) -> Point3d {
        let mut rng = rand::thread_rng();
        Point3d::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }
    pub fn random_unit_vector() -> Point3d{
        Point3d::random(-1.0, 1.0) 
    }
    pub fn distance(&self, other: &Point3d) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    pub fn length(&self) -> f64 {
        self.distance(&Point3d::new(0., 0., 0.))
    }
    pub fn unit_vector(&self) -> Point3d {
        let inv = 1. / self.length();
        Point3d::new(self.x * inv, self.y * inv, self.z * inv)
    }
    pub fn dot(&self, other: &Point3d) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    pub fn cross(&self, other: &Point3d) -> Point3d {
        Point3d::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
}
impl  PartialEq for Point3d {
   
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x  &&  self.y == other.y &&  self.z == other. z
    }
   
}
impl Add for Point3d {
    type Output = Point3d;
    fn add(self, rhs: Self) -> Self::Output {
        Point3d {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}
impl Sub for Point3d {
    type Output = Point3d;
    fn sub(self, rhs: Self) -> Self::Output {
        Point3d {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl Neg for Point3d {
    type Output = Point3d;
    fn neg(self) -> Self::Output {
        Point3d {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Mul<Point3d> for Point3d {
    type Output = Point3d;
    fn mul(self, rhs: Point3d) -> Self::Output {
        Point3d {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl Mul<f64> for Point3d {
    type Output = Point3d;
    fn mul(self, rhs: f64) -> Self::Output {
        Point3d {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Div<Point3d> for Point3d {
    type Output = Point3d;
    fn div(self, rhs: Point3d) -> Self::Output {
        Point3d {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl Div<f64> for Point3d {
    type Output = Point3d;
    fn div(self, rhs: f64) -> Self::Output {
        Point3d {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Ray {
    pub origen: Point3d,
    pub direction: Point3d,
}
impl Ray {
    pub fn new(origen: Point3d, direction: Point3d) -> Ray {
        Ray { origen, direction }
    }
    pub fn at(&self, t: f64) -> Point3d {
        self.origen + self.direction * t
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    // Deserialize, Serialize
)]
pub struct HitRecord {
    pub t: f64,
    pub point: Point3d,
    pub normal: Point3d,
    pub front_face: bool,
    pub material: Material,

    pub u: f64,
    pub v: f64,
}
pub trait Hit {
    fn hittable(&self, ray: &Ray, tmin: f64, tmax: f64) -> Option<HitRecord>;
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
pub struct Camera {
    pub origin: Point3d,
    pub lower_left_corner: Point3d,
    pub focal_length: f64,
    pub horizontal: Point3d,
    pub vertical: Point3d,
    pub look_from: Point3d,
    pub look_at: Point3d,
    pub vup: Point3d,
    pub vfov: f64,
}
impl Camera {
    pub fn new(
        look_from: Point3d,
        look_at: Point3d,
        vup: Point3d,
        vfov: f64,
        aspect: f64,
    ) -> Camera {
        let theta = vfov.to_radians();
        let half_height = (theta / 2.0).tan();
        let half_width = half_height * aspect;

        let w = (look_from - look_at).unit_vector();
        let u = vup.cross(&w).unit_vector();
        let v = w.cross(&u);
        //      println!(" w {:?},  u {:?}, v {:?}", w, u, v);
        let origin = look_from;
        let lower_left_corner = origin - (u * half_width) - (v * half_height) - w;
        let vertical = v * (2.0 * half_height);
        let horizontal = u * (2.0 * half_width);

        let focal_length = (look_from - look_at).length();
        //       println!("lower_left_corner {:?}", lower_left_corner);
        Camera {
            origin,
            lower_left_corner,
            focal_length,
            horizontal,
            vertical,
            look_from,
            look_at,
            vup,
            vfov,
        }
    }

    pub fn get_ray(&self, u: f64, v: f64) -> Ray {
        let dir =
            self.lower_left_corner + (self.horizontal * u) + (self.vertical * v) - self.origin;
        Ray::new(self.origin, dir)
    }
}

// TODO : add triangle, cambiar Hit name for Intersection
#[derive(
    Debug,
    Clone,
    Copy,
    // Deserialize, Serialize
)]

pub struct Sphere {
    pub center: Point3d,
    pub radius: f64,
    pub material: Material,
}
impl Sphere {
    pub fn new(center: Point3d, radius: f64, material:Material) -> Sphere {
        Sphere {
            center,
            radius,
            material
        }
    }
}
impl Hit for Sphere {
    fn hittable(&self, ray: &Ray, tmin: f64, tmax: f64) -> Option<HitRecord> {
        let oc = ray.origen - self.center;

        //  println!("{:?}", oc);
        let A = ray.direction.dot(&ray.direction);
        let B = oc.dot(&ray.direction);

        let C = oc.dot(&oc) - self.radius * self.radius;

        let disc = B * B - A * C;
        //       println!("{:?}, A {} B {} C {} D {}", oc, A, B,C, disc);
        if disc >= 0.0 {
            let sqrtd = disc.sqrt();
            let root_a = ((-B) - sqrtd) / A;
            let root_b = ((-B) + sqrtd) / A;
            for root in [root_a, root_b].iter() {
                if *root > tmin && *root < tmax {
                    let p = ray.at(*root);
                    let n = (p - self.center) / self.radius; //normalizate
                    let is_face = ray.direction.dot(&n) < 0.0;
                    let (u, v) = (1.0, 1.0);
                    // println!("un hit! {} p :  {:?}", *root , p );
                    return Some(HitRecord {
                        t: *root,
                        point: p,
                        normal: if is_face {n} else {-n},
                        front_face: is_face,
                        
                        material: self.material,
                        u,
                        v,
                    });
                }
            }
        }
        None
    }
}
// #[test]
pub fn test_hitsphere(){
//     |y
//     |
//     |
//    / \
//+ z /   \ x

    let sphere = Sphere::new(Point3d::new(0.0, 0.0, 0.0), 2.0, Material::Lamber(Lambertian::new(Rgb::new(1.0,1.0,1.0))) );
   
   
    fn sample_test(p : &Point3d, dir : &Point3d , sphere : &Sphere)->i32{
        let nsamples = (10,10);
        let mut cnt  = 0;
        for x in 0..nsamples.0{
            for y in 0..nsamples.1{
               let mut xx =  x as f64 / 10.0 as f64;
               let mut yy =  x as f64 / 10.0 as f64;
               xx = xx * 2.0 -1.0;
                yy = yy * 2.0 -1.0;
               let ray = Ray::new(
                // Point3d::new(xx*0.7, yy*0.7, 12.0),
                Point3d::new(0.0,0.0,10.0),
                Point3d::new(0.0, 0.0, -1.0),
                );
               
                if let Some(hit) =sphere.hittable(&ray, 0.001, f64::MAX){
                    cnt+=1;
                    println!("hit {}, {}, {:?}, {:?}",cnt,hit.t, hit.point, hit.normal);
                   let ray1 =  Ray::new(hit.point, -hit.normal);
                   if let Some(hit) = sphere.hittable(&ray1, 0.001, f64::MAX){
                    println!("  hit {}, {}, {:?}, {:?}",cnt,hit.t, hit.point, hit.normal);
                   }
                }
            }
    
        }
        cnt


    }
    fn sample_test_occlusion()->i32{
        
        let sphere = Sphere::new(Point3d::new(0.0, 0.0, -10.0), 0.3, Material::Lamber(Lambertian::new(Rgb::new(1.0,1.0,1.0))) );
        let sphere1 = Sphere::new(Point3d::new(0.0, 0.0, 0.0), 0.2, Material::Lamber(Lambertian::new(Rgb::new(1.0,1.0,1.0))) );
        let sphere2 = Sphere::new(Point3d::new(0.0, 0.0, 1.0), 0.1, Material::Lamber(Lambertian::new(Rgb::new(1.0,1.0,1.0))) );
       let spheres  =vec![sphere, sphere1, sphere2];
        let ray = Ray::new(Point3d::new(0.0,0.25,10.0),Point3d::new(0.0, 0.25, 0.0) - Point3d::new(0.0,0.25,10.0));

        for shps in spheres{

            if let Some(hit) =shps.hittable(&ray, 0.001, f64::MAX){
                println!("hay una hit !, {} en {:?}",  hit.t,  ray.at(hit.t));
                if   hit.t<1.0{
                    println!("hay una geometria que oclude! en {:?}", ray.at(hit.t));

                }
                // if ray.at(hit.t) == Point3d::new(0.0, 0.0, 0.0){
                //     println!("hit no hay oclussion {}, {:?}, {:?}",hit.t, hit.point, hit.normal);
                // }
    
               // println!("hit  {}, {:?}, {:?}",hit.t, hit.point, hit.normal);
            }

        }
        

        0
    }
   //  let cnt = sample_test_occlusion();
  //   let cnt = sample_test(&Point3d::new(0.0, 0.0,0.0),&Point3d::new(0.0, 0.0,1.0),  &sphere );

     fn sample_internal_sphere(){
        let sphere1 = Sphere::new(Point3d::new(0.0, 0.0, 0.0), 1000.0, Material::Metal(Metal::new(Rgb::new(1.0,1.0,1.0),1.0)) );
        let spherelist  =vec![sphere1];
        let mut ray = Ray::new(Point3d::new(0.0,0.0,0.0),Point3d::new(0.0, 0.0, -1.0)  );
        let depth = 32;
        for shps in spherelist {
            for iterationdepth in 0..depth{
                if let Some(hit) =shps.hittable(&ray, 0.001, f64::MAX){
                
                    println!(" {:?}", hit.point);
               
                   if let Some((r))  =  hit.material.scatter(&ray, &hit){
                        if let Some(raynew) = r.0 {
                            println!("{:?}", raynew);

                            ray = raynew;
                        }
                   }

               

                }
            }

            
        }

     }
    //  sample_internal_sphere()
     
    
}











#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
struct Triangle {}
impl Hit for Triangle {
    fn hittable(&self, ray: &Ray, tmin: f64, tmax: f64) -> Option<HitRecord> {
        None
    }
}

// scene intersection

pub fn hit_world(spheres: &Vec<Sphere>, ray: &Ray, tmin: f64, tmax: f64) -> Option<HitRecord> {
    for sphere in spheres {
        if let Some(hit) = sphere.hittable(ray, tmin, tmax) {
            return Some(HitRecord {
                front_face: true,
                 material: hit.material,

                normal: (hit.point - sphere.center).unit_vector(),
                point: ray.at(hit.t),
                t: hit.t,
                u: 0.0,
                v: 0.0,
            });
        }
    }
    None
}

pub fn render_write_image(
    filename: &str,
    pixels: &[u8],
    bounds: (u32, u32),
) -> std::io::Result<()> {
    let fd = File::create(filename).unwrap();
    let encoder = PngEncoder::new(fd);
    encoder
        .write_image(pixels, bounds.0, bounds.1, image::ColorType::Rgb8)
        .unwrap();

    Ok(())
}
// https://github.com/dps/rust-raytracer/blob/main/raytracer/src/raytracer.rs
//todo: hay que hacer lo dl shadow ray.
// 1.tengo que saber como hacer que el rayo comprube que efectivamente no hay shadows con la luz. 
// 2.tengo que hacer los materiales, scattering... hacer el metal y el lambert son añadidos en el hit record con
pub fn get_shadow_ray(r: &Ray, prims: &Vec<Sphere>, light: &Light ) -> Option<Srgb> {
  //  aqui hay que tirar un rayo , comprobamos que la distancia es la requerida, es decir no hay oclusion
    Some(light.color)
}
pub fn get_ray_color(r: &Ray, prims: &Vec<Sphere>, lights: &Vec<Light>, scene: &Scene, depth : i32) -> Srgb {
    if depth <=0{
        return Srgb::new(0.0, 0.0, 0.0)
    }
  //   println!("solo una vez, depth : {}", depth);
    let hit = hit_world(prims, r, 0.001, f64::MAX);

    // hay un hit
    match hit {

        Some(hit_record) => {
            // sampleamos la superficie
            let scatter = hit_record.material.scatter(r, &hit_record);

            match scatter {
                Some((scatter_ray, albedo)) => {
                    let mut colorRes: Vec<f32> = vec![0.0; 3];
                    
                    // sampleamos la luz
                    if lights.len() > 0 {
                        lights.into_iter().for_each(|light| {
                            let shadow_ray = Ray::new(hit_record.point, light.center - hit_record.point);
                            if let Some(color) = get_shadow_ray(&shadow_ray, prims, light) {
                                // println!("color : {:?}", color );
                                //  println!("albedo : {:?}", albedo );
                                colorRes[0] += albedo.red * color.red;
                                colorRes[1] += albedo.green * color.green;
                                colorRes[2] += albedo.blue * color.blue;
                            }
                        });

                        let inv = 1.0 / lights.len() as f32;
                        // let mut colorLight =
                        //     Srgb::new();
                        // return Srgb::new(0.0, 0.0, 0.0);
                        colorRes[0]  =   colorRes[0] * inv; 
                        colorRes[1] =  colorRes[1] * inv;
                        colorRes[2] = colorRes[2] * inv;
               //         println!("color : {:?}", colorRes );
                    };
                    match scatter_ray{
                        Some(rayscat)=>{
                            // volvemos a tirar un nuevo  rayo. depth -1 
                          // let color_ret =  get_ray_color(&rayscat, prims, lights, scene,depth-1);
                           // return Srgb::new(clamp(color_ret.red),clamp(color_ret.green), clamp(color_ret.blue));
                        //    return Srgb::new(1.0,1.0, 0.0);
                     let target =   get_ray_color(&rayscat,prims,lights,scene, depth-1);
                     let res =Srgb::new(clamp(colorRes[0] +  albedo.red * target.red),clamp(colorRes[1] + albedo.green *target.green), clamp(colorRes[2] +albedo.blue* target.blue));
                //    println!("target : {:?}", colorRes);
                //   println!("target : {:?}", target);
                //    println!("color : {:?}", res);
                     //  return Srgb::new(clamp(colorRes[0] +  albedo.red * target.red),clamp(colorRes[1] + albedo.green *target.green), clamp(colorRes[2] +albedo.blue* target.blue));
                     return res;
                        }
                        None =>{
                            // si  hemos llegado hasta aqui es que tenemos un sampleo de light sampleado
                            return  albedo;
                        }


                    }
                }
                None => {
                    return Srgb::new(0.0, 0.0, 0.0);
                }
            }

          

        }

        None => {
            let uu = clamp((r.direction.unit_vector().x as f32 + 1.0) * 0.5);
            let vv = clamp((r.direction.unit_vector().y as f32 + 1.0) * 0.5);

            match scene.sky {
                Some(sky) => {
                    return Srgb::new(1.0, 0.0, 0.0);
                }
                None => {
                    let rr = Srgb::new(
                        (1.0 - vv) * 1.0 + vv * 0.5,
                        (1.0 - vv) * 1.0 + vv * 0.7,
                        (1.0 - vv) * 1.0 + vv * 1.0,
                    );
                  //   println!("{:?}",rr );
                    return rr;
                }
            }
        }
    };
}
pub fn render_line(pixels: &mut [u8], y: u32, bounds: (u32, u32), camera: &Camera, scene: &Scene) {
    let v: f64 = (((scene.height as f64 - y as f64 - 1.0) as f64) / (scene.height as f64 - 1.0));
    let n_samples =  scene.num_samples;
    let normalization: f32 = 1.0 / scene.num_samples as f32;
    for x in 0..scene.width {
        let mut pixelColor: Vec<f32> = vec![0.0; 3];

        for _sample in 0..scene.num_samples {
            let u: f64 = (x as f64 / (scene.width as f64 - 1.0));
            //   println!("{},{} ->", u, v);
            let color = get_ray_color(
                &camera.get_ray(u, v),
                &scene.primitives,
                &scene.lights,
                scene,
                scene.max_depth as i32
            );
            //    println!("{:?}", color);
            pixelColor[0] += color.red;
            pixelColor[1] += color.green;
            pixelColor[2] += color.blue
        }

        let color = Srgb::new(
           ( pixelColor[0] * normalization).sqrt(),
           ( pixelColor[1] * normalization).sqrt(),
           ( pixelColor[2] * normalization).sqrt(),
        );

        let pixel: [u8; 3] = color.into_format().into_raw();

        pixels[x as usize * 3] = pixel[0];
        pixels[x as usize * 3 + 1] = pixel[1];
        pixels[x as usize * 3 + 2] = pixel[2];
    }
}
pub fn render(scene: &Scene) {
    let camera = Camera::new(
        Point3d::new(0.0, 1.0, 6.0),
        Point3d::new(0.0, 0.0, -1.0),
        Point3d::new(0.0, 1.0, 0.0),
        90.0,
        scene.width as f64 / scene.height as f64,
    );

    // scene.width * scene.height

    let mut pixels = vec![0; scene.width * scene.height * 3];
    let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(scene.width * 3).enumerate().collect();
    // bands.into_iter().for_each(|(i, band)| {
    //     render_line(
    //         band,
    //         i as u32,
    //         (scene.width as u32, scene.height as u32),
    //         &camera,
    //         scene,
    //     );
    // });
      bands.into_par_iter().for_each(|(i, band)|{
     
 render_line(
    band,
    i as u32,
    (scene.width as u32, scene.height as u32),
    &camera,
    scene,
 );
    
   });

    render_write_image(
        "mipng1.png",
        &pixels,
        (scene.width as u32, scene.height as u32),
    )
    .unwrap();
}
 
mod primitives;
fn main() -> Result<()> {
    primitives::main_prim();
    
    return Ok(());
    // test_hitsphere();
    // return Ok(());
    // //  render();
    // let center = Point3d::new(0.0, 0.0, 0.0);
    // let ray: Ray = Ray::new(
    //     Point3d::new(0.0, 0.99999, 2.0),
    //     Point3d::new(0.0, 0.0, -1.0),
    // );
   
    // let sphere = Sphere::new(Point3d::new(0.0, 0.0, 0.0), 1.0, Material::Lamber(Lambertian::new(Rgb::new(1.0,1.0,1.0))) );
    // let hit =  sphere.hittable(&ray , 0.001, f64::MAX).unwrap();
    // tengo que añadir textures , glass y demas
    // el shadow ray  para hacer las luces.
    // tengo que añadir la forma recursiva iterative del method usar algo asi como render_recursive() | render_iterative()
    //  tengo que añadir el fuzz para el metal y poca cosa mas
    // tengo que  mirar como hacer la matrix
    


   let primitives =  vec![
               //Sphere::new(Point3d::new(0.0, 1.0, 0.0),1.0, Material::Metal(Metal::new(Rgb::new(0.911, 0.10, 0.10),1.0))),
                Sphere::new(Point3d::new(0.0, 0.5, 1.0), 0.5, Material::Lamber(Lambertian::new(Rgb::new(0.10,0.80,0.10)))),
                Sphere::new(Point3d::new(2.0, 1.0, 1.0), 1.0, Material::Lamber(Lambertian::new(Rgb::new(0.10,0.80,0.10)))),
              Sphere::new(Point3d::new(0.0, -200.0, 0.0), 200.0, Material::Lamber(Lambertian::new(Rgb::new(0.10,0.10,0.80)))),
            ];
    let lights  = vec![Light::new(  Point3d::new(0.0, 2.0, 0.0),   Srgb::new(0.11, 0.10, 0.10))];
    let scene = Scene::make_scene(600, 600,lights, primitives, 128,3);
    let start = Instant::now();
      render(&scene);
    println!("Frame time: {}ms", start.elapsed().as_millis());



    // let camera = Camera::new(
    //     Point3d::new(0.0, 0.0, 2.0),
    //     Point3d::new(0.0, 0.0, -1.0),
    //     Point3d::new(0.0, 1.0, 0.0),
    //     90.0,
    //     scene.width as f64 / scene.height as f64,
    // );

    // let color = get_ray_color(
    //     &camera.get_ray(0.5, 0.7),
    //     &scene.primitives,
    //     &scene.lights,
    //     &scene,
    //     1
    // );
    

    Ok(())
}
