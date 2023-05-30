#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unused_parens)]
use crate::Lights::IsAmbientLight;
use crate::assert_delta;
use crate::materials;
use crate::materials::BsdfType;
use crate::materials::Eval;
use crate::materials::Fr;
use crate::materials::Fresnel;
use crate::materials::FresnelSchlickApproximation;
use crate::materials::IsSpecular;
use crate::materials::RecordSampleIn;
use crate::materials::RecordSampleOut;
use crate::materials::SampleIllumination;
use crate::primitives;
use crate::primitives::prims::Ray;
use crate::primitives::prims::*;
use crate::raytracerv2::clamp;
use crate::raytracerv2::interset_scene;
use crate::raytracerv2::Scene;
use crate::sampler::Sampler;
use crate::texture::EvalTexture;
use crate::Lights::generateSamplesDeterministic;
use crate::Lights::generateSamplesRandomVector;
use crate::Lights::generateSamplesWithSamplerGather;
use crate::Lights::GetEmission;
use crate::Lights::IsAreaLight;
use crate::Lights::IsBackgroundAreaLight;
use crate::Lights::LigtImportanceSampling;
use crate::Lights::RecordSampleLightIlumnation;
use cgmath::*;
use num_traits::Float;
use std::borrow::Cow;
use std::ops;
// use crate::texture::MapSpherical;
// use crate::texture::Mapping2d;
// use crate::HitRecord;
use crate::Lights::Light;
use crate::Lights::PointLight;
use crate::Lights::SampleLightIllumination;
use crate::Srgb;
pub fn get_ray_ao(
    r: &Ray<f64>,

    scene: &Scene<f64>,
    depth: i32,
    sampler: &mut Box<dyn Sampler>,
) -> Srgb {
    let mut hitop = interset_scene(r, scene);

    let mut L: Vec<f32> = vec![0.0; 3];
    match hitop {
        Some(hit) => {
            let nsamples = 32;
            for isample in 0..nsamples {
                let psample = sampler.get2d();
                let recout = hit.material.sample(RecordSampleIn {
                    pointW: hit.point,
                    prevW: -r.direction,
                    sample: Some(psample),
                });
                let fnsamples = nsamples as f32;
                //             // println!("{:?}", hit.normal);

                if let Some(h) = interset_scene(&recout.newray.unwrap(), scene) {
                    // println!("sample! {:?}",recout.newray.unwrap());
                    let fnsamples = nsamples as f32;
                    // println!("{}",1.0/fnsamples);
                    L[0] += (1.0 / fnsamples);
                    L[1] += (1.0 / fnsamples);
                    L[2] += (1.0 / fnsamples);
                }
            } // n samples
        }
        None => {}
    }

    //    Srgb::new(clamp( L[0], 0.0, 1.0),clamp( L[1], 0.0, 1.0),clamp( L[2], 0.0, 1.0))
    Srgb::new(
        clamp(1.0 - L[0], 0.0, 1.0),
        clamp(1.0 - L[1], 0.0, 1.0),
        clamp(1.0 - L[2], 0.0, 1.0),
    )
}

pub fn estimatelights<'a>(
    scene: &Scene<'a, f64>,
    hit: primitives::prims::HitRecord<f64>,

    r: &Ray<f64>,
    sampler: &mut Box<dyn Sampler>,
) -> Srgb {
    let mut colorRes: Vec<f32> = vec![0.0; 3];

    if scene.lights.len() > 0 {
        let lights = &scene.lights;
        lights.into_iter().for_each(|light| {
            if !light.is_arealight() && !light.is_background_area_light()&&!light.is_ambientlight() {
                let sampleLight =
                    light.sampleIllumination(&RecordSampleLightIlumnation::from_hit(hit));
                let Ld = sampleLight.1;
                let pdflight = sampleLight.2;
                let nextdir = sampleLight.0;
                let occlray = sampleLight.3.unwrap();
                
              
                if let Some(hayoclusion) =scene.intersect_occlusion(&occlray, &light.getPositionws())
                {
                    if !hayoclusion.0 {
                        let prev = -r.direction;
                        let next = (light.getPositionws() - hit.point).normalize();
                        //  println!("{:?}", prev);
                        let fren = hit.material.fr(prev, next);
                        let absd = hit.normal.dot(next).abs() as f32;

                        // esto sobre. lo quita el compilador, pero hay que ponerlo dentro de samplerIllumination?
                        let d2 = 1.0 / (light.getPositionws()).distance2(hit.point) as f32;
                        //IMPORTANTE: el termino quadratico de la distancia no lo aplico!
                        colorRes[0] += fren.red * Ld.red * absd;
                        colorRes[1] += fren.green * Ld.green * absd;
                        colorRes[2] += fren.blue * Ld.blue * absd;
                       
                         
                        // colorRes[0] += fren.red * Ld.red * d2 *absd;
                        // colorRes[1] += fren.green * Ld.green * d2*absd;
                        // colorRes[2] += fren.blue * Ld.blue * d2*absd;
                    }
                }
            }
             else if light.is_ambientlight(){
                let sampleLight =
                light.sampleIllumination(&RecordSampleLightIlumnation::from_hit(hit));
                let Ld = sampleLight.1;
                let pdflight = sampleLight.2;
                let nextdir = sampleLight.0;
                let occlray = sampleLight.3.unwrap();
                let prev = -r.direction;
                let next = occlray.direction .normalize();
                //  println!("{:?}", prev);
                let fren = hit.material.fr(prev, next);
                let absd = hit.normal.dot(next).abs() as f32;

      
                colorRes[0] += fren.red * Ld.red * absd;
                colorRes[1] += fren.green * Ld.green * absd;
                colorRes[2] += fren.blue * Ld.blue * absd;
              
                // if let Some(hayoclusion) =scene.intersect_occlusion(&occlray, &  sampleLight.4.unwrap()){
                //     if !hayoclusion.0 {
                //         let prev = -r.direction;
                //         let next = occlray.direction .normalize();
                //         //  println!("{:?}", prev);
                //         let fren = hit.material.fr(prev, next);
                //         let absd = hit.normal.dot(next).abs() as f32;

              
                //         colorRes[0] += fren.red * Ld.red * absd;
                //         colorRes[1] += fren.green * Ld.green * absd;
                //         colorRes[2] += fren.blue * Ld.blue * absd;

                        
                //     }
                // }

            } 
            else if light.is_arealight() && !light.is_background_area_light() {
                // estamos en un area light std
                let nsamples = 8;

                let vecsamples = generateSamplesWithSamplerGather(sampler, nsamples);

                let mut newLd: Vec<f32> = vec![0.0; 3];
                for isample in vecsamples.into_iter() {
                    let p = isample;

                    let resnewelight =
                        LigtImportanceSampling::samplelight(&r, &scene, &hit, light.clone(), &p.0);
                    
                    if ! isBlack(resnewelight){
                        let bsdfsample =
                        LigtImportanceSampling::samplebsdf(&r, &scene, &hit, light.clone(), &p.1);
                        updateL(
                            &mut newLd,
                             LigtImportanceSampling::addSrgb(resnewelight, bsdfsample),
                        );
                    }
                   

                    newLd[0] = newLd[0] / (nsamples as f32);
                    newLd[1] = newLd[1] / (nsamples as f32);
                    newLd[2] = newLd[2] / (nsamples as f32);
                    colorRes[0] += newLd[0];
                    colorRes[1] += newLd[1];
                    colorRes[2] += newLd[2];
                }
            } else if !light.is_arealight() && light.is_background_area_light() {
                let nsamples = 8;
                let Ld = LigtImportanceSampling::estimate_bck_area_light(
                    r, light, scene, hit, sampler, nsamples,
                );
                colorRes[0] += Ld.red;
                colorRes[1] += Ld.green;
                colorRes[2] += Ld.blue;
            }
        });
        let L = Srgb::new(colorRes[0], colorRes[1], colorRes[2]); 
        L
    } else {
        Srgb::new(0.0, 0.0, 0.0)
    }
}

pub fn get_ray_direct_light<'a>(
    r: &Ray<f64>,

    lights: &Vec<Light>,
    scene: &Scene<'a, f64>,
    depth: i32,
    sampler: &mut Box<dyn Sampler>,
) -> Srgb {
    let mut hitop = interset_scene(r, scene);
    let mut L: Vec<f32> = vec![0.0; 3];

    match hitop {
        Some(hit) => {
            let mut colorRes: Vec<f32> = vec![0.0; 3];
            if hit.is_emision.unwrap() {
                let Lemit = hit.emission.unwrap();
                //    updateL(&mut L, Lemit);
                // aqui el path continua ... esto lo hago por el test
                return Srgb::new(
                    clamp(L[0], 0.0, 1.0),
                    clamp(L[1], 0.0, 1.0),
                    clamp(L[2], 0.0, 1.0),
                );
            }
            let Ld = estimatelights(scene, hit, r, sampler);
            L[0] = Ld.red;
            L[1] = Ld.green;
            L[2] = Ld.blue;
        }
        None => {
            let ls = &scene.lights;
            ls.into_iter().for_each(|light| {
                if light.is_background_area_light() {
                    let Lemit = light.get_emission(None, r);

                    //    updateL(&mut L, Lemit);
                }
            })
        }
    }
    Srgb::new(
        clamp(L[0], 0.0, 1.0),
        clamp(L[1], 0.0, 1.0),
        clamp(L[2], 0.0, 1.0),
    )
}
fn backcolor<'a>(direction: Vector3<f64>, scene: &Scene<'a, f64>) -> Srgb {
    if let Some(txt) = &scene.background {
        return txt.eval(&Point3::new(-direction.x, -direction.z, -direction.y));
    }
    let uu = clamp((direction.normalize().x as f32 + 1.0) * 0.5, 0.0, 1.0);
    let vv = clamp((direction.normalize().y as f32 + 1.0) * 0.5, 0.0, 1.0);
    let rr = Srgb::new(
        (1.0 - vv) * 1.0 + vv * 0.5,
        (1.0 - vv) * 1.0 + vv * 0.7,
        (1.0 - vv) * 1.0 + vv * 1.0,
    );
    //   println!("{:?}",rr );
    //     return   Srgb::new(0.0,0.0, 0.0);
    return rr;
}
pub fn updateL(L: &mut Vec<f32>, Ld: Srgb) {
    L[0] += Ld.red;
    L[1] += Ld.green;
    L[2] += Ld.blue;
}
pub fn get_ray_path<'a>(
    r: &Ray<f64>,

    lights: &Vec<Light>,
    scene: &Scene<'a, f64>,
    depth: i32,
    sampler: &mut Box<dyn Sampler>,
) -> Srgb {
    let mut hitop = interset_scene(&r.clone(), scene);

    let mut rcow = Cow::Borrowed(r);
    let mut L: Vec<f32> = vec![0.0; 3];
    let mut isthehit = false;

    let mut isPathSpec = false;
    for _idepth in 0..2 {
        if let None = hitop {
            if _idepth == 0 {
                let skycolor = backcolor(rcow.direction, scene);
                return Srgb::new(
                    clamp(skycolor.red + L[0], 0.0, 1.0),
                    clamp(skycolor.green + L[1], 0.0, 1.0),
                    clamp(skycolor.blue + L[2], 0.0, 1.0),
                );
            }
            if isPathSpec {
                let skycolor = backcolor(rcow.direction, scene);
                return Srgb::new(
                    clamp(skycolor.red + L[0], 0.0, 1.0),
                    clamp(skycolor.green + L[1], 0.0, 1.0),
                    clamp(skycolor.blue + L[2], 0.0, 1.0),
                );
            } else {
                return Srgb::new(0.0, 0.0, 0.0);
            }
        }

        if let Some(hit) = hitop {
            let Ld = estimatelights(scene, hit, &rcow, sampler);
            updateL(&mut L, Ld);
            match hit.material {
                BsdfType::LambertianDisneyBsdf(l) => {
                    break;
                }
                BsdfType::Lambertian(l) => {
                    break;
                }
                BsdfType::SpecReflection(l) => {
                    isPathSpec = true;
                }

                _ => {}
            };

            let hit = hitop.unwrap();
            let rec = hit
                .material
                .sample(RecordSampleIn::from_hitrecord(hit, &rcow, (0.0, 0.0)));
            rcow = Cow::Owned(rec.newray.unwrap());

            hitop = interset_scene(&rcow, scene);
        }
    }
    Srgb::new(
        clamp(L[0], 0.0, 1.0),
        clamp(L[1], 0.0, 1.0),
        clamp(L[2], 0.0, 1.0),
    )
}
 
// hit, next dir, f, pdf


pub fn isBlack(c: Srgb) -> bool {
    c.red == 0.0 && c.green == 0.0 && c.blue == 0.0
}
pub fn get_ray_path_1<'a>(
    r: &Ray<f64>,

    lights: &Vec<Light>,
    scene: &Scene<'a, f64>,
    depth: i32,
    
    sampler: &mut Box<dyn Sampler>,
    px:(i64, i64)
) -> Srgb {
   
     let mut hitop = interset_scene(&r.clone(), scene);
      

    let mut L = Srgb::new(0.0,0.0,0.0);

    if let Some(hitt) = hitop {
        let depth = 3;
        let mut hitt = hitop.unwrap();
        let mut rcow = Cow::Borrowed(r);
        let mut hitcow = Cow::Borrowed(&hitt);
        let mut paththrought: Srgb = Srgb::new(1.0, 1.0, 1.0);
    
        for idepth in 0..depth {
            
            if hitcow.is_emision.unwrap() {
                let Lemit = hitcow.emission.unwrap();
                //    updateL(&mut L, Lemit);
                // aqui el path continua ... esto lo hago por el test
                return LigtImportanceSampling::mulSrgb(Lemit, paththrought); 
            }
        //     aqui esta el truco. el rayo cuando hit una superficie specular no añade..solo transporta
        if ! hitcow.material.is_specular(){
            let Ld = estimatelights(scene, *hitcow, &rcow, sampler);
            
            let L0 = LigtImportanceSampling::mulSrgb(Ld, paththrought);

       L = LigtImportanceSampling::addSrgb( L , L0);
            if idepth==1  {
 
              // let L0 = Srgb::new(0.41,0.41, 0.41);
             //   L = LigtImportanceSampling::addSrgb( L , L0);
            }
          
          }
            
     
            let psample = sampler.get2d();
            let rec = hitcow.material.sample(RecordSampleIn::from_hitrecord(*hitcow, &rcow, psample));
            if isBlack(rec.f) || rec.pdf == 0.0 {
                break;
            }
            let absdot = hitcow.normal.dot(rec.next).abs();
            let pdf = rec.pdf;
            let a = absdot / pdf;
            let fr = rec.f;
            let current = LigtImportanceSampling::mulScalarSrgb(fr, a as f32);
            paththrought = LigtImportanceSampling::mulSrgb(paththrought, current);
            rcow = Cow::Owned(Ray::<f64>::new(hitcow.point, rec.next));
            if let Some(newhit) = interset_scene(&rcow, scene) {
// println!("newhit");
                hitcow = Cow::Owned(newhit);
                 
            } else {
                break;
            }
        }
         

       
       
        L
    } else{
       L
    }
   
    
}
