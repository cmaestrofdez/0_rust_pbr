#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
use cgmath::BaseFloat;
use cgmath::Point3;
use cgmath::Vector3;
use num_traits::Euclid;
use rand::prelude::Distribution;
use rayon::iter::IntoParallelRefMutIterator;

use std::borrow::Borrow;
use std::borrow::BorrowMut;
use std::borrow::Cow;
use std::cell::RefMut;
use std::fmt::Debug;
use std::fs::File;

use std::io::Result;
use std::path::Path;
use std::process::Command;
use std::rc::Rc;
use std::sync::Arc;

// use float::Sqrt;
use image::codecs::png::PngEncoder;
use image::ImageEncoder;
use palette::rgb::Rgb;
// use num_traits::Float;
use rand::Rng;
// use std::process::Output;

use std::*;

use num_traits::cast::cast;
use palette::{Pixel, Srgb};
use std::time::Instant;

extern crate cgmath;

use crate::Lights::AmbientLight;
use crate::Lights::AreaLight;
use crate::Lights::AreaLightDisk;
use crate::Lights::AreaLightSphere;
use crate::Lights::BackgroundAreaLight;
use crate::Lights::GetEmission;
use crate::Lights::GetShape;
use crate::Lights::IsAreaLight;
use crate::Lights::IsBackgroundAreaLight;
use crate::Lights::RecordSampleLightIlumnation;
use crate::Lights::SpotLight;
use crate::Point2i;
use crate::assert_delta;
use crate::imagefilm::FilterFilm;
use crate::imagefilm::ImgFilm;
 
use crate::integrator;
use crate::materials;
use crate::materials::Fr;
use crate::materials::MaterialDescType;
use crate::materials::Plastic;
use crate::materials::RecordSampleIn;
use crate::materials::RecordSampleOut;
use crate::materials::SampleIllumination;
use crate::primitives::Disk;
use crate::primitives::Plane;
use crate::primitives::PrimitiveIntersection;
use crate::primitives::prims::Ray;
use crate::primitives::prims::*;
use crate::sampler::Sampler;
use crate::sampler::SamplerLd2;
use crate::sampler::SamplerUniform;
use crate::samplerhalton::SamplerHalton;
use crate::texture::MapConstant;
use crate::texture::MapSpherical;
use crate::texture::MapUV;
use crate::texture::TexType;
use crate::texture::Texture2D;
use crate::texture::Texture2DSRgbMapConstant;
use crate::Lights::Light;
use crate::Lights::PointLight;
use crate::Lights::SampleLightIllumination;
use std::cell::RefCell;
use cgmath::*;
















trait AbsDot<Scalar> {
    fn abs_dot(&self, other: Self) -> Scalar;
}
impl AbsDot<f64> for Vector3<f64> {
    fn abs_dot(&self, other: Self) -> f64 {
        self.dot(other).abs()
    }
}

pub fn lerp<Scalar: BaseFloat>(t: Scalar, low: Scalar, hight: Scalar) -> Scalar {
    low + (hight - low) * t
}

pub fn clamp<Scalar: std::cmp::PartialOrd>(t: Scalar, low: Scalar, hight: Scalar) -> Scalar {
    if t <= low {
        low
    } else if t >= hight {
        hight
    } else {
        t
    }
}
pub fn remainder(t: i32, r: i32) -> i32 {
    t.rem_euclid(r)
}
fn map_to_spherical_coords(v3: &Vector3<f64>) -> (f64, f64) {
    let mut v = clamp(v3.y, -1.0, 1.0);
    v = v * 0.5 + 0.5;
    let mut phi = v3.x.atan2(v3.z);
    phi += f64::consts::FRAC_2_PI * 2.0;
    if phi < 0.0 {
        phi += 2.0f64 * std::f64::consts::PI;
    };
    let u = phi / (2.0f64 * std::f64::consts::PI);
    (u, v)
}
pub fn powerHeuristic(f:f64, pdfb:f64)->f64{
   ( f * f )/(f*f+pdfb*pdfb)
}

#[derive( Clone )]
pub struct Scene<'a, Scalar> {
    pub num_samples: usize,
//     pub primitives: Vec<&'a Sphere<Scalar>>,
//    pub primitives1: Vec<&'a  Box<dyn Intersection<Scalar, Output = HitRecord<Scalar>>>>,
   pub primitives2:  Vec<& 'a Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>>,
    pub max_depth: usize,
    pub width: usize,
    pub height: usize,
    pub lights: Vec<Light>,
    pub sky: Option<Sphere<Scalar>>,
    pub background: Option<TexType>,
    
}
impl<'a, Scalar: BaseFloat> Scene<'a, Scalar> {
    pub fn make_scene(
        width: usize,
        height: usize,
        lights: Vec<Light>,
        primitives: Vec<&'a Sphere<Scalar>>,
        // primitives1: Vec<&'a  Box<dyn Intersection<Scalar, Output = HitRecord<Scalar>>>>,
        num_samples: usize,
        max_depth: usize,
    ) -> Scene<Scalar> {

        Scene {
            num_samples,
            max_depth,
            width,
            height,
            lights,
            // primitives,
            sky: None,
            background: None,
            // primitives1:vec![],
            primitives2:vec![],
        }
    }
    pub fn make_scene1(
        width: usize,
        height: usize,
        lights: Vec<Light>,
        primitives: Vec<&'a Sphere<Scalar>>,
       
        prims2 :  Vec<& 'a Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>>,
        num_samples: usize,
        max_depth: usize,
        bck: Option<TexType>,
    ) -> Scene< 'a ,Scalar> {
        Scene {
            num_samples,
            max_depth,
            width,
            height,
            lights,
            // primitives,
            sky: None,
            background: bck,
        //    primitives1:vec![],
           primitives2:prims2,
        }
    }

    pub fn make_scene_with_film(
        width: usize,
        height: usize,
        lights: Vec<Light>,
        primitives: Vec<&'a Sphere<Scalar>>,
       
        prims2 :  Vec<& 'a Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>>,
        num_samples: usize,
        max_depth: usize,
        bck: Option<TexType>,
        film:Option<RefCell<ImgFilm>>
    ) -> Scene< 'a ,Scalar> {
        Scene {
            num_samples,
            max_depth,
            width,
            height,
            lights,
            // primitives,
            sky: None,
            background: bck,
        //    primitives1:vec![],
           primitives2:prims2,
        }
    }
}
use crate::primitives::prims::Intersection;
use crate::primitives::prims::IntersectionOclussion;

// returna true hay oclussion
impl 
    IntersectionOclussion<f64> for Scene<'_, f64>
{
    
    fn intersect_occlusion(
        &self,
        ray: &Ray<f64>,
        target: &Point3<f64>,
    ) -> Option<(bool, f64, Point3<f64>)> {
        let mut _tcurrent = cast::<f64, f64>(f64::MAX).unwrap();
        let mut _currentpoint: Option<Point3<f64>> = None;
        for iprim in &self.primitives2 {
            if let Some((ishit, _thit, phit)) = iprim.intersect_occlusion(ray, target) {
                if (phit - ray.origin).magnitude2() < (*target - ray.origin).magnitude2() {
                    return Some((true, _thit, phit));
                }
            }
        }
        Some((false, cast::<f64, f64>(f64::MAX).unwrap(), Point3::new(cast::<f64,f64>(f64::MAX).unwrap(),cast::<f64, f64>(f64::MAX).unwrap(),cast::<f64, f64>(f64::MAX).unwrap())))
    }
}
#[test]
fn test_scene_oclussion_ray_1() {
    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default()),
    );

    let sphere1 = Sphere::new(
        Vector3::new(0.0, 0.0, 1110.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default()),
    );
    let spheresuper = Sphere::new(
        Vector3::new(0.0, 0.0, 1110.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default()),
    );
    let scene = Scene::<f64>::make_scene(0, 0, vec![], vec![&sphere, &sphere1], 0, 0);
    let mut cnt = 0;
    for i in 0..100 {
        let fi = 2.0 * (i as f64) - 1.0;
        let ptarget = Point3::new(0.0, fi, -11.0);
        let porigin = Point3::new(0.0, fi, 10.0);
        let direction = (ptarget - porigin).normalize();
        let ray = Ray::new(porigin, direction);
        if let Some(hayoclu) = scene.intersect_occlusion(&ray, &ptarget) {
            cnt += 1;
        }
    }
    assert_eq!(cnt, 100);
    println!("num de oclussions {}", cnt);
    cnt = 0;

    for i in 0..100 {
        let fi = 2.0 * (i as f64) - 1.0;
        let ptarget = Point3::new(0.0, 10.0, fi);
        let porigin = Point3::new(0.0, -10.0, fi);
        let direction = (ptarget - porigin).normalize();
        let ray = Ray::new(porigin, direction);
        if let Some(hayoclu) = scene.intersect_occlusion(&ray, &ptarget) {
            cnt += 1;
        }
    }
    assert_eq!(cnt, 100);
    println!("num de oclussions {}", cnt);
 
    // oclusion dentro de la espefera

    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default()),
    );
    // origen dentro de la sphere, target fuera

    let ptarget = Point3::new(0.0, 0.0, 10.0);
    let porigin = Point3::new(0.0, 0.0, 0.0);
    let direction = (ptarget - porigin).normalize();
    let ray = Ray::new(porigin, direction);
    if let Some(hayoclu) = scene.intersect_occlusion(&ray, &ptarget) {
        println!("hay oclusion dentro de la esfera{:?}", hayoclu);
        assert_eq!(hayoclu.0, true);
    }

    let ptarget = Point3::new(0.0, 0.0, 0.5);
    let porigin = Point3::new(0.0, 0.0, 0.0);
    let direction = (ptarget - porigin).normalize();
    let ray = Ray::new(porigin, direction);
    if let Some(hayoclu) = scene.intersect_occlusion(&ray, &ptarget) {
        println!(
            "no hay oclusion dentro de la esfera, el target esta dentro{:?}",
            hayoclu
        );
        assert_eq!(hayoclu.0, false);
    }

    let spheresuper = Sphere::new(
        Vector3::new(0.0, 10.0, 0.0),
        100.0,
        MaterialDescType::PlasticType(Plastic::default()),
    );
    let scene = Scene::<f64>::make_scene(0, 0, vec![], vec![&spheresuper], 0, 0);

    let ptarget = Point3::new(0.0, 10.0, 110.5);
    let porigin = Point3::new(0.0, 10.0, 0.0);
    let direction = (ptarget - porigin).normalize();
    let ray = Ray::new(porigin, direction);
    if let Some(hayoclu) = scene.intersect_occlusion(&ray, &ptarget) {
        println!(
            "si! hay oclusion dentro de la esfera, el target esta dentro{:?}",
            hayoclu
        );
        assert_eq!(hayoclu.0, true);
    }
    println!("");
    // una esfera dentro de otra despplazada
    let spheremini = Sphere::new(
        Vector3::new(0.0, 10.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default()),
    );
    let ptarget = Point3::new(0.0, 10.0, 11.5);
    let porigin = Point3::new(0.0, 10.0, 2.0);
    let direction = (ptarget - porigin).normalize();
    let ray = Ray::new(porigin, direction);
    let scene = Scene::<f64>::make_scene(0, 0, vec![], vec![&spheresuper, &spheremini], 0, 0);
    if let Some(hayoclu) = scene.intersect_occlusion(&ray, &ptarget) {
        println!(
            "no hay oclusion dentro de la esfera, el target esta dentro{:?}",
            hayoclu
        );
        assert_eq!(hayoclu.0, false);
    }
    println!("");
}

impl 
    Intersection<f64> for Scene<'static, f64>
{
    type Output = HitRecord<f64>;
    fn intersect(&self, ray: &Ray<f64>, tmin: f64, tmax: f64) -> Option<Self::Output> {
        let mut tcurrent = cast::<f64, f64>(f64::MAX).unwrap();
        let mut _current_point: Option<Point3<f64>> = None;
        let mut hitopt = None;
        for iprim in &self.primitives2 {
            if let Some(hit) = iprim.intersect(ray, tmin, tmax) {
                if (tcurrent > hit.t) {
                    tcurrent = hit.t;
                    hitopt = Some(hit);
                }
                // if (phit - ray.origin).magnitude2() < (*target - ray.origin).magnitude2() {
                //     return Some(true);
                // }
            }
        }
        hitopt
    }
}

#[test]
fn test_scene_intersection_ray() {
    // let sphere = Sphere::new(
    //     Vector3::new(0.0, 0.0, 0.0),
    //     1.0,
    //     MaterialDescType::PlasticType(Plastic{
    //         R : Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),
    //         ft :Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),

    //      })
    // );
    // let sphere1 = Sphere::new(
    //     Vector3::new(0.0, 0.0, 2.0),
    //     1.0,
    //     MaterialDescType::PlasticType(Plastic{
    //         R : Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),
    //         ft :Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),

    //      })
    // );
    // let sphere2 = Sphere::new(
    //     Vector3::new(0.0, 0.0, 3.0),
    //     1.0,
    //     MaterialDescType::PlasticType(Plastic{
    //         R : Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),
    //         ft :Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),

    //      })
    // );
    // let sphere3 = Sphere::new(
    //     Vector3::new(0.0, 0.0, 13.0),
    //     2.0,
    //     MaterialDescType::PlasticType(Plastic{
    //         R : Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),
    //         ft :Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),

    //      })
    // );
    // let sphere4 = Sphere::new(
    //     Vector3::new(10.0, 10.0, 13.0),
    //     2.0,
    //     MaterialDescType::PlasticType(Plastic{
    //         R : Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),
    //         ft :Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),

    //      })
    // );
    // let scene =
    //     Scene::<f64>::make_scene(0, 0, vec![], vec![&sphere1, &sphere2, &sphere3, &sphere4], 0, 0);
    // let ray = Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
    // if let Some(hit) = scene.intersect(&ray, 0.0001, f64::MAX) {
    //     println!("{:?},{:?}", hit.t, hit.point);
    // }
}

#[derive(Debug, Clone, Copy)]
pub struct Camera<Scalar> {
    pub origin: Point3<Scalar>,
    pub lower_left_corner: Point3<Scalar>,
    pub focal_length: Scalar,
    pub horizontal: Vector3<Scalar>,
    pub vertical: Vector3<Scalar>,
    pub look_from: Point3<Scalar>,
    pub look_at: Point3<Scalar>,
    pub vup: Vector3<Scalar>,
    pub vfov: Scalar,
}
impl<Scalar: BaseFloat> Camera<Scalar> {
    pub fn new(
        look_from: Point3<Scalar>,
        look_at: Point3<Scalar>,
        vup: Vector3<Scalar>,
        vfov: Scalar,
        aspect: Scalar,
    ) -> Camera<Scalar> {
        let cast_2 = cast::<f64, Scalar>(2.0).unwrap();
        let theta = vfov.to_radians();
        let half_height = (theta / cast_2).tan();
        let half_width = half_height * aspect;

        let w = (look_from - look_at).normalize();
        let u = vup.cross(w).normalize();
        let v = w.cross(u);
        //      println!(" w {:?},  u {:?}, v {:?}", w, u, v);
        let origin = Point3::new(look_from.x, look_from.y, look_from.z);

        let lower_left_corner = origin - (u * half_width) - (v * half_height) - w;
        let vertical = v * (cast_2 * half_height);
        let horizontal = u * (cast_2 * half_width);

        let focal_length = (look_from - look_at).magnitude();

        // use crate pri
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

    pub fn get_ray(&self, u: Scalar, v: Scalar) -> Ray<Scalar> {
        let dir =
            self.lower_left_corner + (self.horizontal * u) + (self.vertical * v) - self.origin;
        Ray::new(self.origin, dir.normalize())
    }
    pub fn get_normalize_coord_ray(&self, u: Scalar, v: Scalar) -> Ray<Scalar> {
        let cast_05 = cast::<f64, Scalar>(0.5).unwrap();
        let uu = -cast_05 * u + cast_05;
        let vv = -cast_05 * v + cast_05;
        let dir =
            self.lower_left_corner + (self.horizontal * uu) + (self.vertical * vv) - self.origin;
        Ray::new(self.origin, dir)
    }
}

pub fn render_write_film(filename: &str, pixels: &[u8], bounds: (u32, u32)) -> std::io::Result<()> {
    let fd = File::create(filename).unwrap();
    let encoder = PngEncoder::new(fd);
    encoder
        .write_image(pixels, bounds.0, bounds.1, image::ColorType::Rgb8)
        .unwrap();

    Ok(())
}
 

pub fn interset_scene(r: &Ray<f64>, scene: &Scene<f64>) -> Option<HitRecord<f64>> {
 //  println!("{:?} , {:?}", r.origin, r.direction);
    let mut tmin = f64::MAX;
    let mut hitcurrent = None;
    for  primitive_i in &scene.primitives2 {
        if let Some(hit) = primitive_i.intersect(r, 0.0001, f64::MAX) {
           // println!("t:{}, p:{:?} , n :{:?}", hit.t, hit.point, hit.normal);
            if  tmin>hit.t{
                hitcurrent =Some(hit);
                tmin = hit.t;
            }
             
            
        }
     
    }
    for lights in &scene.lights{
        if  lights.is_arealight() && !lights.is_background_area_light() {
           let prim =   lights .get_shape().unwrap();
           if let Some(mut hit) = prim.intersect(r, 0.0001, f64::MAX){
                if  tmin>hit.t{
                    let Lemit = lights.get_emission(Some(hit),r);
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

pub fn get_ray_color_iteration(
    r: &mut Ray<f64>,
    prims: &Vec<Sphere<f64>>,
    lights: &Vec<Light>,
    scene: &Scene<f64>,
    depth: i32,
) {
    let mut hitop = interset_scene(r, scene);
    for _ith in 1..1 {
        let paththrought: Srgb = Srgb::new(1.0, 1.0, 1.0);
        //   let L : Srgb = Srgb::new(0.0,0.0,0.0);
        let mut L: Vec<f32> = vec![0.0; 3];
        match hitop {
            Some(hit) => {
                let mut colorRes: Vec<f32> = vec![0.0; 3];

                if lights.len() > 0 {
                    lights.into_iter().for_each(|light| {
                        let sample_light = light.sampleIllumination(&RecordSampleLightIlumnation::from_hit(hit)  );

                        let Ld = sample_light.1;
                        let pdflight = sample_light.2;
                        let lightdir = sample_light.0;
                        let occlray = sample_light.3.unwrap();
                        if let Some(hayoclusion) =
                            scene.intersect_occlusion(&occlray, &light.getPositionws())
                        {
                            if !hayoclusion.0 {
                                let albedo = hit.material.fr(hit.prev.unwrap(), lightdir);

                                let fa = (hit.normal.dot(lightdir).abs() / pdflight) as f32;
                                colorRes[0] += Ld.red * fa * albedo.red;
                                colorRes[1] += Ld.green * fa * albedo.green;
                                colorRes[2] += Ld.blue * fa * albedo.blue;
                            }
                        }
                    });
                    let inv = 1.0 / lights.len() as f32;

                    colorRes[0] = colorRes[0] * inv;
                    colorRes[1] = colorRes[1] * inv;
                    colorRes[2] = colorRes[2] * inv;
                }
                let ldirect = Srgb::new(
                    clamp(colorRes[0], 0.0, 1.0),
                    clamp(colorRes[1], 0.0, 1.0),
                    clamp(colorRes[2], 0.0, 1.0),
                );
                //  L +=Ldirect;

                let mut rec = RecordSampleIn {
                    prevW: hit.prev.unwrap(),
                    pointW: hit.point,
                    sample:Some((0.0,0.0))
                };
                let srec = hit.material.sample(rec);
                *r = srec.newray.unwrap()
            }
            None => {}
        }
        hitop = interset_scene(r, scene);
    }
}

pub fn get_ray_color(
    r: &Ray<f64>,
    _prims: &Vec<Sphere<f64>>,
    lights: &Vec<Light>,
    scene: &Scene<f64>,
    depth: i32,
) -> Srgb {
    if depth <= 0 {
        return Srgb::new(0.0, 0.0, 0.0);
    }

    let hitop = interset_scene(r, scene);

    match hitop {
        Some(hit) => {
            let mut rec = RecordSampleIn {
                prevW: hit.prev.unwrap(),
                pointW: hit.point,
                sample:Some((0.0,0.0))
            };
            let srec = hit.material.sample(rec);
            let albedo = srec.f;

            let mut color_res: Vec<f32> = vec![0.0; 3];

            // lightdirect sample
            if lights.len() > 0 {
                lights.into_iter().for_each(|light| {
                    let sample_light = light.sampleIllumination(&RecordSampleLightIlumnation::from_hit(hit)  );
                    let ld = sample_light.1;
                    let pdflight = sample_light.2;
                    let nextdir = sample_light.0;
                    let occlray = sample_light.3.unwrap();
                    if let Some(hayoclusion) =
                        scene.intersect_occlusion(&occlray, &light.getPositionws())
                    {
                        if !hayoclusion.0 {
                            let fa = (hit.normal.dot(nextdir).abs() / pdflight) as f32;
                            color_res[0] += ld.red * fa * albedo.red;
                            color_res[1] += ld.green * fa * albedo.green;
                            color_res[2] += ld.blue * fa * albedo.blue;
                        }
                    }
                });
                let inv = 1.0 / lights.len() as f32;

                color_res[0] = color_res[0] * inv;
                color_res[1] = color_res[1] * inv;
                color_res[2] = color_res[2] * inv;
            }

            let res = Srgb::new(
                clamp(color_res[0], 0.0, 1.0),
                clamp(color_res[1], 0.0, 1.0),
                clamp(color_res[2], 0.0, 1.0),
            );
            return res;
        }
        None => {
            match &scene.sky {
                Some(sky) => {
                    return Srgb::new(1.0, 0.0, 0.0);
                }
                None => {
                    let uu = clamp((r.direction.normalize().x as f32 + 1.0) * 0.5, 0.0, 1.0);
                    let vv = clamp((r.direction.normalize().y as f32 + 1.0) * 0.5, 0.0, 1.0);
                    let rr = Srgb::new(
                        (1.0 - vv) * 1.0 + vv * 0.5,
                        (1.0 - vv) * 1.0 + vv * 0.7,
                        (1.0 - vv) * 1.0 + vv * 1.0,
                    );
                    //   println!("{:?}",rr );
                    //     return   Srgb::new(0.0,0.0, 0.0);
                    return rr;
                }
            }
        }
    }
}

pub fn render_line(
    pixels: &mut [u8],
    y: u32,
    _bounds: (u32, u32),
    camera: &Camera<f64>,
    scene: &  Scene<f64>,
    film : Option<RefCell<ImgFilm>>,
    sampler:&mut Box<dyn Sampler>,
    
) {
    
      let   filmbinding =   film. unwrap();
   let mut film = filmbinding.borrow_mut();
  
    let normalization: f32 = 1.0 / scene.num_samples as f32;
    for x in 0..scene.width {
        let mut pixel_color: Vec<f32> = vec![0.0; 3];

        for _sample in 0..scene.num_samples {
            fn get_ray_xy(xx: i64, yy: i64, scene: &  Scene<f64>, camera: &Camera<f64>,sampler: &mut Box<dyn Sampler> ) -> Srgb {

       
              let uvs =  to_map(&(xx as f64, yy as f64), 1024, 1024);
               

                //  let color = integrator::get_ray_ao(&camera.get_ray(u,v),&scene.primitives,&scene.lights,scene,scene.max_depth as i32,);
                   integrator::get_ray_direct_light (&camera.get_ray(uvs.0,uvs.1),&scene.lights,scene,scene.max_depth as i32,sampler) 

                  
                // integrator::get_ray_path( &mut camera.get_ray(u, v),    &scene.lights,scene,     scene.max_depth as i32, )
            }
            if x / 128 == 0 &&  y %128  == 0 {
               println!("pixel: {} , {} ", x, y);
            }
            let color = get_ray_xy(x as i64, y  as i64, scene, camera,  sampler );
        
            pixel_color[0] += color.red;
            pixel_color[1] += color.green;
            pixel_color[2] += color.blue
        }

        let color = Srgb::new(
            (pixel_color[0] * normalization),
            (pixel_color[1] * normalization) ,
            (pixel_color[2] * normalization) 
        );

        let pixel: [u8; 3] = color.into_format().into_raw();

        pixels[x as usize * 3] = pixel[0];
        pixels[x as usize * 3 + 1] = pixel[1];
        pixels[x as usize * 3 + 2] = pixel[2];
    }
}



fn to_map(cam:&(f64, f64), w:i64, h:i64)->(f64, f64){
    let v: f64 = (((h as f64 - cam.1 as f64 - 1.0) as f64)/ (h  as f64 - 1.0));
    let u: f64 = (cam.0 as f64 / (w as  f64 - 1.0));
    
    (clamp(u,0.0,1.0), clamp(v,0.0,1.0))
    // (u, v)
   }
pub fn render_line_film( 
     pixels: &mut [u8],
    y: u32,
    bounds: (u32, u32),
    camera: &Camera<f64>,
    scene: &  Scene<f64>,
  mut reffilm:  RefMut<ImgFilm>,
   //  mut reffilm: Rc<RefCell<ImgFilm>>,
    sampler:&mut Box<dyn Sampler>,

    ){
        
        println!("pix {}% ",   (y as f32  ) / reffilm.h  as f32);
        // https://github.com/Twinklebear/tray_rust/blob/master/src/film/filter/mitchell_netravali.rs
        // para ver como puede ser... 
        // https://github.com/Twinklebear/tray_rust/blob/master/src/sampler/ld.rs
        // https://www.willusher.io/tray_rust/tray_rust/sampler/ld/fn.van_der_corput.html
       
        // Cow::Borrowed(reffilm);
        
    for x in 0..scene.width {
         
          
        sampler.start_pixel(Point2i::new(x as i64, y as i64));
        let samplecamera = sampler.get2d();
   
         debug_assert_eq!(samplecamera.0.floor()as usize , x );
        debug_assert_eq!(samplecamera.1.floor()as usize , y as usize );
       
       
        
        
        let sampleforcamerauv = to_map(&samplecamera, reffilm.w as i64, reffilm.h as i64);
        // println!("{:?},{:?}", samplecamera, sampleforcamerauv);
      //   let color = integrator::get_ray_ao(&camera.get_ray(sampleforcamerauv.0,sampleforcamerauv.1), &scene, scene.max_depth as i32, sampler);
 //      let color =   integrator::get_ray_direct_light (&camera.get_ray(sampleforcamerauv.0,sampleforcamerauv.1),&scene.lights,scene,scene.max_depth as i32,sampler)  ;
        //  let color = integrator::get_ray_ao(&camera.get_ray(u,v),&scene.primitives,&scene.lights,scene,scene.max_depth as i32,);
     let color =integrator::get_ray_path_1( &camera.get_ray(sampleforcamerauv.0,sampleforcamerauv.1),    &scene.lights,scene,     scene.max_depth as i32, sampler , (x as i64, y as i64));
   //   let color =integrator::get_ray_path_1( &camera.get_ray(0.50, 0.50),    &scene.lights,scene,     scene.max_depth as i32, sampler , (x as i64, y as i64));
     //    reffilm.add_sample(&Point2::new(x as f64,y as f64 ), color);
         reffilm.add_sample(&Point2::new(samplecamera.0, samplecamera.1), color);
     //    println!("  fist element ------------------->{:?} {:?}", samplecamera, color);
        
         while sampler.start_next_sample(){
             
             let samplecamera = sampler.get2d();
             debug_assert_eq!(samplecamera.0.floor()as usize , x );
             debug_assert_eq!(samplecamera.1.floor()as usize , y as usize );
         
           let sampleforcamerauv = to_map(&samplecamera, reffilm.w as i64, reffilm.h as i64);
          let color1 =integrator::get_ray_path_1( &camera.get_ray(sampleforcamerauv.0,sampleforcamerauv.1),    &scene.lights,scene,     scene.max_depth as i32, sampler, (x as i64, y  as i64) );
           //   println!("  second element ------------------->{:?} {:?}", samplecamera, color1);
           //   let color1 =   integrator::get_ray_direct_light (&camera.get_ray(sampleforcamerauv.0,sampleforcamerauv.1),&scene.lights,scene,scene.max_depth as i32,sampler)  ;
        //  println!("color {:?}", color1);
        // // for _nextdims in 0..10{
        // //     let nextdimssamples  = sampler.get2d();
        // //     println!("  ------------------->std samples {:?},", nextdimssamples);
        // // }
         
           
       //   let color = integrator::get_ray_ao(&camera.get_ray(sampleforcamerauv.0,sampleforcamerauv.1), &scene, scene.max_depth as i32, sampler);
                reffilm.add_sample(&Point2::new(samplecamera.0, samplecamera.1), color1);
         }
       
        
       
        

      
    
        

          
      

        

       
    }

}
pub fn render_film(scene: &  Scene<f64>, film : RefCell<  ImgFilm>, sampler: &mut Box<dyn Sampler> , camera :&Camera<f64> )-> std::io::Result<()>{

   
    let mut pixels = vec![0; scene.width * scene.height * 3];
    let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(scene.width * 3).enumerate().collect();

    
    bands.into_iter().for_each(|(i, band)| {
        render_line_film(
            band,
            i as u32,
            (scene.width as u32, scene.height as u32),
            camera,
            scene,
            film.borrow_mut() ,
            sampler 
        );
    });

    
  

    let dirpath : &Path = Path::new("render4test");
    fs::create_dir_all(dirpath)?;
 
    let bind = film.borrow_mut();
    // raytracer_sampler1
    let spath = & format!("raytracer_path_to_raytracer_pathtracer_depth_3_ambientlight_test{}.png", 1);
    bind.commit_and_write(dirpath .join(spath).as_os_str().to_str().unwrap())

     //  bind.commit_and_write("raytracer3_samplers_samper_ld_4spp_spot_light.png") ;
      
    // let path =  fs::canonicalize("C:\\Program Files\\GIMP 2\\bin\\gimp-console-2.10.exe ").unwrap();
    // Command::new(path.as_os_str() )  .arg("raytracer3_samplers.png"). spawn(). expect("gimp command failed to start");

     
    
     
}
 












pub fn render(scene: &  Scene<f64>,    film : RefCell<ImgFilm>)-> std::io::Result<()> {

    
    let camera = Camera::new(


        Point3::new(0.30, 1.31010, 2.5), // puede ser que a  medida que me acerco a y = 1 no funciona bien...esta bien la camara...no saldra allgun error con el vector up
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        1.0f64 / 1.0f64,
    );

 
    // let mut pixels = vec![0; scene.width * scene.height * 3];
    // let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(scene.width * 3).enumerate().collect();
    // let mut samplerhalton  :  Box<dyn Sampler> = Box::new(SamplerHalton::new(&Point2i::new(0, 0),&Point2i::new(1 as i64, 1 as i64),32,false,));
    // bands.into_iter().for_each(|(i, band)| {
    //     render_line(
    //         band,
    //         i as u32,
    //         (scene.width as u32, scene.height as u32),
    //         &camera,
    //         scene,
    //         Some(film),
    //         &mut  samplerhalton
    //     );
    // });
     
    let film0 =  film.to_owned() ;
   let mut bind =  film0.borrow_mut();
   bind. add_sample(&Point2::new(0 as f64, 0 as f64), Srgb::new(1.0,1.0,1.0));
   bind.commit_and_write("raytracer3_ray_path_emission_direct_light__with_arealight.png");
 
 Ok(())
  
}





#[test]
fn main_render_debug() {
    // tengo que hacer glass material . saber como funciona el path tracer.
    // tengo ademas que hacer que meter microfacet transmision +  microfacet reflection
    // mira el cuaderno de notas
    main_render_3();
}

pub fn main_render_3() -> Result<()> {

    //   let w = 1920 ;
    //  let h = 1180 ;
    
    // con cuidado puedo variar el filtro, quizas suavice todo , tenemos ademas que añadir dos luces
     let w = 512;
     let h = 512 ;
    let  film = RefCell::new( ImgFilm::from_filter_gauss(w, h,(3.0,3.0),1.0));
    let mut samplerhalton  :  Box<dyn Sampler> = Box::new(SamplerHalton::new(&Point2i::new(0, 0),&Point2i::new(w as i64, h as i64),128,false,));
    let camera = Camera::new(


        Point3::new(0.30, 1.31010, 2.5), // puede ser que a  medida que me acerco a y = 1 no funciona bien...esta bien la camara...no saldra allgun error con el vector up
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        w as f64 /h as f64 ,
    );

    use crate::materials::*;
    use crate::primitives::prims::*;
   
    let glass2fresneltrasnmision =MaterialDescType::Glass2Type(Glass2::new());
    let glass =MaterialDescType::GlassType(Glass::default());
    let metal = MaterialDescType::MetaType(Metal::from_constant_rought(0.01, 0.01));
    let plastic = MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,1.0));
    let mirror = MaterialDescType::MirrorType(Mirror {..Default::default()});
    let arealightsphere = Light::AreaLightSphere(AreaLightSphere::new(Vector3::new(0.0, 0.40,0.0),0.20, Srgb::new(5.0,5.0,5.0)));
    
    let sphererightsmallblue = &(
        Box::new(Sphere::new(Vector3::new(0.0, 0.30, 0.0), 0.30, MaterialDescType::PlasticType(Plastic::from_albedo(0.60,0.60,1.950)))
        ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );
    let sphere2smallptr_metal = &(
        Box::new(Sphere::new(Vector3::new(0.0, 0.30, 0.0), 0.30, metal)
        ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );
    let sphere2smallptr_mirror = &(
        Box::new(Sphere::new(Vector3::new(0.0, 0.30, 0.0), 0.30, mirror)
        ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );
    let sphere2smallptr_glass = &(
        Box::new(Sphere::new(Vector3::new(0.0, 0.50, 0.20), 0.50, glass2fresneltrasnmision)
        ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );
    let spheresmallptrgreen = &(
        Box::new(Sphere::new(Vector3::new(-0.7, 0.30, 0.0), 0.30, MaterialDescType::PlasticType(Plastic::from_albedo(0.0,0.950,0.0)))
        ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );
    let sphereleftsmallptr = &(
        Box::new(Sphere::new(Vector3::new(0.0, 0.30, 0.0), 0.30, MaterialDescType::PlasticType(Plastic::from_albedo(0.50,0.0,1.0)))
        ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );
    let diskescene = Disk::new(Vector3::new( 0.0000,0.0000,0.0000),Vector3::new(0.0, 1.0, 0.0),0.0, 100.0, MaterialDescType::PlasticType(Plastic::from_albedo(4.410,0.5050,0.9594100)));
    let diskescene = &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);

   
    // let sphere2smallptrbox = &(
    //    Sphere::with_box(Vector3::new(1.0, 0.30, 0.0), 0.30, MaterialDescType::PlasticType(Plastic::from_albedo(0.0, 1.0, 0.0)))  as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >
    // );

    let ambientlight = Light::AmbientLightType(AmbientLight::new(Vector3::new(0.0,1.0, 0.0),  Srgb::new(0.190,0.190, 0.190)));
     let pointlight = Light::PointLight(PointLight {iu: 4.0,positionws: Point3::new(0.00004,2.9630,0.500005),color: Srgb::new(1.190,1.190, 1.190),});
    //  let pointlight1 = Light::PointLight(PointLight {iu: 4.0,positionws: Point3::new(0.0015,2.5,0.5015),color: Srgb::new(4.920,4.920,0.9420),});
    //  let pointlight2 = Light::PointLight(PointLight {iu: 4.0,positionws: Point3::new(1.0015,2.5,0.5015),color: Srgb::new(4.920,4.920,0.9420),});
     let bcklight =  Light::BackgroundAreaLightType(BackgroundAreaLight::new(vec![Srgb::new(1.00,1.00,1.00); 8_usize * 8_usize ],8_usize,8_usize));
     let bcklightsky =  Light::BackgroundAreaLightType(BackgroundAreaLight::from_file("sky_128.png", Srgb::new(2.0,2.0,2.0)));
     let arealightdisk = Light::AreaLightDiskType(AreaLightDisk::new( Vector3::new(0.0001,1.90,0.0001 ), Vector3::new(0.0,-1.0, 0.0).normalize(),1.000,Srgb::new(8.950, 8.950, 8.950)));
    let primitivesIntersections  :Vec<&Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>> =vec![ 
    sphererightsmallblue,  //, sphere2smallptr,spherebluesmallptr
       // sphere2smallptr_metal,
       // sphere2smallptr_mirror,
    //    sphererightsmallblue,
    //    sphere2smallptr_glass,
     diskescene,

     //  sphereleftsmallptr,
    spheresmallptrgreen
        ];

      
        let start = Instant::now();

   //      el problema esta en la  estimacion del estimate_area_light no se cual mira el dibujo!
    render_film(& mut Scene::make_scene_with_film(
        w,
        h,
        vec![
      pointlight,
           ambientlight
                  //  pointlight //,pointlight1, pointlight2
            //   arealightdisk
            // arealightsphere
            //  arealightplane
            // spotlight,
            //  bcklight
            // arealightdisk,
        //    bcklightsky
        // bcklightsky
           
        ],
        vec![],
        primitivesIntersections ,
        1,
        1,
        None,
        None,
    ), film, &mut samplerhalton , &camera).unwrap_or_else(|err|{println!("{:?}", err)});
    println!("Frame time: {} ms", start.elapsed().as_millis());

    
    
    Ok(())
}
pub fn main_render_2() -> Result<()> {
   
    

    use crate::materials::*;
    use crate::primitives::prims::*;

    

    let camera = Camera::new(


        Point3::new(0.30, 1.31010, 2.5), // puede ser que a  medida que me acerco a y = 1 no funciona bien...esta bien la camara...no saldra allgun error con el vector up
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        1.0f64 / 1.0f64,
    );

    let spheresmall =  Sphere::new(Vector3::new(0.0, 0.30, 0.0), 0.30, MaterialDescType::PlasticType(Plastic::from_albedo(0.99060,  0.0, 0.0)));
    let spheresmallptr = &(Box::new(spheresmall) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
let sphere2smallptr = &(
        Box::new(Sphere::new(Vector3::new(1.0, 0.30, 0.0), 0.30, MaterialDescType::PlasticType(Plastic::from_albedo(0.0, 1.0, 0.0)))) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >
    );
let spherebluesmallptr = &(
        Box::new(Sphere::new(Vector3::new(-1.0, 0.30, 0.0), 0.30, MaterialDescType::PlasticType(Plastic::from_albedo(0.0, 0.0, 1.0)))) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >
    );
    // let disk0 = Disk::new(Vector3::new(0.0,0.5,0.0),Vector3::new(0.0, 1.0, 0.0),0.0, 0.10, MaterialDescType::PlasticType(Plastic::from_albedo(0.060, 0.96, 0.96)));
    // let disk0ptr = &(Box::new(disk0) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);


    
    let diskescene = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0, 1.0, 0.0),0.0, 100.10, MaterialDescType::PlasticType(Plastic::from_albedo(0.560, 0.96, 0.96)));
    let diskescene = &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);


    let primitivesIntersections  :Vec<&Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>> =vec![ 
       
        diskescene    , spheresmallptr  //, sphere2smallptr,spherebluesmallptr
        ];
        
     let arealightplane = Light::AreaLightType(
       //  AreaLight::new( Vector3::new(0.0,1.0,0.0 ), Vector3::new(0.0,-1.0,0.0),1.10,1.1)
        AreaLight::new_emission( Vector3::new(0.0,1.0,0.0 ), Vector3::new(0.0,-1.0,0.0),1.0,1.0, Srgb::new(1.0,1.0,1.0))
    );
    let arealightdisk = Light::AreaLightDiskType(
        AreaLightDisk::new( Vector3::new(0.0,1.60,0.0 ), Vector3::new(0.0,-1.0, 0.0).normalize(),0.7400,Srgb::new(2.5, 2.5, 2.5)));
    // let bkg : Light::BackgroundAreaLightType();
    let bcklight =  Light::BackgroundAreaLightType(BackgroundAreaLight::new(vec![Srgb::new(12.00,12.00,12.00); 8_usize * 8_usize ],8_usize,8_usize));
     
    let vdirspot = (Point3::new(1.10,0.0,0.0) - Point3::new(0.0,1.0,0.0)   ).normalize();
    let spotlight = SpotLight::from_light(Vector3::new(-1.0,1.20,0.0 ), vdirspot, Srgb::new(4.0,4.0,2.0), 90.0, 30.0);
     let w = 512;
     let h =512;
    

     // let film  =  RefCell::new(  ImgFilm::from_filter_box(w,h, (1.0, 1.0)));
  

      let  film = RefCell::new( ImgFilm::from_filter_gauss(w, h,(2.0,2.0),1.0));
    ;
 //   let film  = Box::new(  ImgFilm::from_filter_gauss(w, h,(2.0,2.0),1.0));
     let mut sampleruniform :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (w as u32, h as u32)),32, false, Some(0)));
     let mut sampler :  Box<dyn Sampler> =  Box::new(SamplerLd2::new(((0, 0), (w as u32, h as u32)), 64, false));
     let mut samplerhalton  :  Box<dyn Sampler> = Box::new(SamplerHalton::new(&Point2i::new(0, 0),&Point2i::new(w as i64, h as i64),128,false,));
    let start = Instant::now();

    render_film(& mut Scene::make_scene_with_film(
        w,
        h,
        vec![
             //  Light::PointLight(PointLight {iu: 4.0,positionws: Point3::new(0.0015,1.515,0.0015),color: Srgb::new(1.20,1.20,23.9420),}),
             // arealightdisk
            //  arealightplane
            spotlight
           
        ],
        vec![],
        primitivesIntersections ,
        1,
        1,
        None,
        None,
    ), film, &mut samplerhalton , &camera).unwrap_or_else(|err|{println!("{:?}", err)});
    println!("Frame time: {} ms", start.elapsed().as_millis());
  
    Ok(())
    // https://rust-lang-nursery.github.io/rust-cookbook/os/external.html
    // C:\Program Files\GIMP 2.9\bin\gimp-2.9 .exe  --as-new mipng.pn
}

pub fn main_render() -> Result<()> {

  
    use crate::materials::*;
    use crate::primitives::prims::*;









    let micro_bck = BsdfType::MicroFacetReflection(MicroFacetReflection {
        frame: None,
        distri: MicroFaceDistribution::BeckmannDistribution(BeckmannDistribution::new(0.1, 0.1)),
        fresnel: Fresnel::FresnelNop(FresnelNop {}),
    });
    let micro_trosky = BsdfType::MicroFacetReflection(MicroFacetReflection {
        frame: None,
        distri: MicroFaceDistribution::TrowbridgeReitzDistribution(
            TrowbridgeReitzDistribution::new(0.01, 0.01),
        ),
        fresnel: Fresnel::FresnelCondutor(FresnelConductor::default()),
    });

    let mirror = MaterialDescType::MirrorType(Mirror {
        ..Default::default()
    });
    // let metal = MaterialDescType::MetaType(Metal::default());
     let plastic0 = MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 0.1, 0.1));
     let plastic1 = MaterialDescType::PlasticType(Plastic::from_albedo(0.060, 0.96, 0.96));
  //  let plastic3green = MaterialDescType::PlasticType(Plastic::from_albedo(0.0, 1.1, 0.1));
 
   
   let sphere0 = &Sphere::new(Vector3::new(0.0, 1.0, 0.0), 1.0, mirror);
    let sphere1 = &Sphere::new(Vector3::new(0.0, -1000.0, 0.0), 1000.0, plastic1);

   
  //   let sphere2 = &Sphere::new(Vector3::new(3.0, 1.0, 0.0), 0.50, plastic3green);
    let primitives = vec![      sphere0 ,sphere1 ]; 
  //   let primitives = vec![      plane  ]; 
     

    let start = Instant::now();

    let txtbck = Some(TexType::Texture2DSRgbSpherical(
        Texture2D::<Srgb, MapSpherical>::createTexture2DSRgbShpericalImg(), 
    ));
    
   let plane = Plane::new(Vector3::new(0.0,1.0,-2.0), Vector3::new(0.0,0.0,1.0), 2.0,1.0, plastic0);
   let plane2 = Plane::new(Vector3::new(0.0,0.0,0.0), Vector3::new(0.0,1.0,0.0), 2.0,2.0,  MaterialDescType::PlasticType(Plastic::from_albedo(0.060, 0.96, 0.96)));
   let plane3 = Plane::new(Vector3::new(0.0,1.0,0.0), Vector3::new(0.0,1.0,-1.0),  0.20,0.20,  MaterialDescType::PlasticType(Plastic::from_albedo(0.060, 0.96, 0.96)));
   let plane4 = Plane::new(Vector3::new(0.0,1.0,0.0), Vector3::new(0.0,-1.0,1.0),  0.20,0.20,  MaterialDescType::PlasticType(Plastic::from_albedo(0.060, 0.96, 0.96)));
   let disk0 = Disk::new(Vector3::new(0.0,0.5,0.0),Vector3::new(0.0, 1.0, 0.0),0.0, 0.10, MaterialDescType::PlasticType(Plastic::from_albedo(0.060, 0.96, 0.96)));
   let diskescene = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0, 1.0, 0.0),0.0, 100.10, MaterialDescType::PlasticType(Plastic::from_albedo(0.060, 0.96, 0.96)));
   let spheresmall =  Sphere::new(Vector3::new(0.0, 0.30, 0.0), 0.30, MaterialDescType::PlasticType(Plastic::from_albedo(0.99060, 0.0, 0.0)));
    let planeptr = &(Box::new(plane) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
   let planeptr2 = &(Box::new(plane2) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
   let planeptr3 = &(Box::new(plane3) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
   let planeptr4 = &(Box::new(plane4) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
   let spheresmallptr = &(Box::new(spheresmall) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
   let disk0ptr = &(Box::new(disk0) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
   let diskescene = &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
    // let planeptr3 = &(Box::new(plane3)  as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);

    
     let primitivesIntersections  :Vec<&Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>> =vec![ 
       
        diskescene     , spheresmallptr
        ];
        let w = 512 ;
        let h =512;
       
   //  let arealight = Light::AreaLightType(AreaLight::new( Vector3::new(0.0,1.0,0.0 ), Vector3::new(0.0,-1.0,1.0),0.10,0.1));
    // let arealightdisk = Light::AreaLightDiskType(AreaLightDisk::new( Vector3::new(0.0,1.60,0.0 ), Vector3::new(0.0,-1.0, 0.0).normalize(),0.7400,Srgb::new(2.5, 2.5, 2.5)));
    // let bkg : Light::BackgroundAreaLightType();
   let bcklight =  Light::BackgroundAreaLightType(BackgroundAreaLight::new(vec![Srgb::new(1.00,1.00,1.00); 8_usize * 8_usize ],8_usize,8_usize));
   let  film = RefCell::new( ImgFilm::from_filter_gauss(w, h,(2.0,2.0),1.0));
    render(& mut Scene::make_scene1(
        w,
        h,
        vec![
            Light::PointLight(PointLight {
            iu: 4.0,
            positionws: Point3::new(0.0015,1.515,0.0015),
            color: Srgb::new(1.20,1.20,23.9420),
            }),
            //bcklight
            // arealightdisk,
        ],
        primitives,
        primitivesIntersections ,
        1,
        1,
        txtbck,
        
    ), film).unwrap_or_else(|err|{println!("{:?}", err)});
    println!("Frame time: {} ms", start.elapsed().as_millis());
    Ok(())
}

#[test]
fn test_camera_rays() {
    use crate::primitives::prims::*;

    use crate::materials::*;

    let camera = Camera::new(
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.0, 0.0, -1.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        1.0f64 / 1.0f64,
    );
    let rup = camera.get_ray(0.5, 0.0);
    let rdown = camera.get_ray(0.5, 1.0);
    let rleft = camera.get_ray(0.0, 0.5);
    let rright = camera.get_ray(1.0, 0.5);

    rup.at(0.0);
    //  println!(" deg top-bottom : {:?} left right {:?}",    Deg::from(rup.direction.angle(rdown.direction)) , Deg::from(rleft.direction.angle(rright.direction)) );

    assert_eq!(Deg::from(rup.direction.angle(rdown.direction)), Deg(90.0));
    assert_eq!(
        Deg::from(rleft.direction.angle(rright.direction)),
        Deg(90.0)
    );

    if false {
        // en algun momento esto tiene que ser un hit
        let mut isHit: bool = false;
        let sphere: Sphere<f64> = Sphere::new(
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
            MaterialDescType::PlasticType(Plastic::default()),
        );
        for i in (10..90).rev() {
            let vfov = i as f64;
            let camera = Camera::new(
                Point3::new(0.0, 0.0, 1.5),
                Point3::new(0.0, 0.0, -1.0),
                Vector3::new(0.0, 1.0, 0.0),
                vfov,
                1.0f64 / 1.0f64,
            );
            //  println!("point to hit {:?}", vfov);

            let r = camera.get_ray(0.0, 0.0);

            if let Some(hit) = sphere.intersect(&r, 0.0001, f64::MAX) {
                println!("point to hit {:?}", hit.point);
                isHit = true;
            }
        }

        assert_eq!(isHit, true)
    }

    if true {
        // from  o ->
        let sphere: Sphere<f64> = Sphere::new(
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
            MaterialDescType::PlasticType(Plastic::default()),
        );
        let vtups = vec![
            (-0.12, 0.0),
            (0.12, 0.0),
            (-0.13, 0.0),
            (0.13, 0.0),
            (-0.1, 0.0),
            (0.1, 0.0),
            (0.0, -0.1),
            (0.0, 0.1),
            (0.0, -0.12),
            (0.0, 0.12),
            (0.0, -0.13),
            (0.0, 0.13),
            (0.0, -0.14),
            (0.0, 0.14),
            (0.0, -0.2),
            (0.0, 0.2),
        ];
        let mut res: Vec<Vector3<f64>> = vec![];
        let camera = Camera::new(
            Point3::new(0.0, 0.2, 0.0),
            Point3::new(0.0001, 0.0, 0.0001),
            Vector3::new(0.0, 1.0, 0.0),
            90.0,
            1.0f64 / 1.0f64,
        );
        for tup in vtups {
            let r = camera.get_normalize_coord_ray(tup.0, tup.1);

            if let Some(hit) = sphere.intersect(&r, 0.0001, f64::MAX) {
                println!("point to hit {:?} {:?}", hit.point, hit.tn);
                res.push(hit.tn);
            }
        }
        let mut iterres = res.iter().peekable().step_by(1);
        while let Some(element) = iterres.next() {
            println!("{:?} ", element);
        }

        println!("point to hit");
    }

    if false {
        // from top to bottom
        let sphere: Sphere<f64> = Sphere::new(
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
            MaterialDescType::PlasticType(Plastic::default()),
        );
        let camera = Camera::new(
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(0.00001, 0.0, 0.0001),
            Vector3::new(0.0, 1.0, 0.0),
            90.0,
            1.0f64 / 1.0f64,
        );
        let r = camera.get_ray(0.5, 0.5);
        if let Some(hit) = sphere.intersect(&r, 0.0001, f64::MAX) {
            println!("point to hit {:?}", hit.point);
        }
    }

    if false {
        // from
        let sphere: Sphere<f64> = Sphere::new(
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
            MaterialDescType::PlasticType(Plastic::default()),
        );
        let camera = Camera::new(
            Point3::new(0.0, -2.0, 0.0),
            Point3::new(0.00001, 0.0, 0.0001),
            Vector3::new(0.0, 1.0, 0.0),
            90.0,
            1.0f64 / 1.0f64,
        );
        let r = camera.get_ray(0.5, 0.5);
        if let Some(hit) = sphere.intersect(&r, 0.0001, f64::MAX) {
            println!("point to hit {:?}, hit: {:?}", hit.point, hit);
            assert_delta!(hit.point.y, -1.0, 0.0001);
        }
    }
}
