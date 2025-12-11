#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unused_parens)]
use crate::Cameras::PerspectiveCamera;
use crate::Config;
use crate::Lights::Light;
use crate::Lights::IsAmbientLight;
use crate::Lights::New_Interface_AreaLight;
use crate::Lights::NumSamples;
use crate::Lights::generateSamplesWithSamplerGather1;
use crate::Point2f;
use crate::Point2i;
use crate::Point3f;
use crate::Spheref64;
use crate::Vector3f;
use crate::assert_delta;
use crate::bdpt2;

use crate::imagefilm::FilterFilm;
use crate::imagefilm::FilterFilmGauss;
use crate::materials;
use crate::materials::BsdfType;
use crate::materials::Eval;
use crate::materials::Fr;
use crate::materials::Fresnel;
use crate::materials::FresnelSchlickApproximation;
use crate::materials::IsSpecular;
use crate::materials::MaterialDescType;
use crate::materials::Metal;
use crate::materials::Mirror;
use crate::materials::Plastic;
use crate::materials::RecordSampleIn;
use crate::materials::RecordSampleOut;
use crate::materials::SampleIllumination;
use crate::metropolis;
use crate::primitives;
use crate::primitives::Cylinder;
use crate::primitives::Disk;
use crate::primitives::PrimitiveType;
use crate::primitives::prims::Ray;
use crate::primitives::prims::*;
use crate::raytracerv2::clamp;
use crate::raytracerv2::interset_scene;
use crate::raytracerv2::Scene;
use crate::raytracerv2::to_map;
use crate::sampler::Sampler;
use crate::sampler::SamplerType;
use crate::sampler::SamplerUniform;
use crate::sampler::SeedInstance;
use crate::texture::EvalTexture;
use crate::Lights::generateSamplesDeterministic;
use crate::Lights::generateSamplesRandomVector;
use crate::Lights::generateSamplesWithSamplerGather;
use crate::Lights::GetEmission;
use crate::Lights::IsAreaLight;
use crate::Lights::IsBackgroundAreaLight;
use crate::Lights::LigtImportanceSampling;
use crate::Lights::RecordSampleLightIlumnation;
use crate::threapoolexamples::init_tiles_withworkers;
use crate::threapoolexamples1::BBds2f;
use crate::threapoolexamples1::BBds2i;
use crate::threapoolexamples1::new_interface_for_intersect_occlusion;
use crate::threapoolexamples1::new_interface_for_intersect_scene;
use crate::threapoolexamples1::pararell;
use crate::volumentricpathtracer::volumetric::MediumType;
use crate::volumentricpathtracer::volumetric::RecordSampleOutMedium;
// use actix_rt::time::Instant;
use  std::time::Instant;
use cgmath::*;
use num_traits::Float;
use std::borrow::Cow;
use std::ops;
use std::sync::Arc;
// use crate::texture::MapSpherical;
// use crate::texture::Mapping2d;
// use crate::HitRecord;
 
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

//TODO: hay que cambierlo por Colorf64
pub fn  new_interface_estimatelights (
    scene: &Scene1,
    hit: primitives::prims::HitRecord<f64>,
    r: &Ray<f64>,
    sampler:& mut SamplerType,
    recordSampledOutpumedium :Option<&RecordSampleOutMedium>, // for sample media interations. recover pintersectoin in media // TODO esto lo tengo que quitar.
    handleMediaOrhandleShadowrayTransmission : bool, // if true will be use handle transmision. if not handle shadow ray will be use
)-> Srgb{
    let mut colorRes: Vec<f32> = vec![0.0; 3];
    if scene.lights.len() > 0 {
        let lights = &scene.lights;
        lights.into_iter().for_each(|light| {
            if !light.is_arealight() && !light.is_background_area_light()&&!light.is_ambientlight() {
                
                let (nextdir, Ld, pdflight, occlray, _) = light.sampleIllumination(&RecordSampleLightIlumnation::from_hit(hit));
                // falla aqui no sabe que ha cambiado la normal
           
            
                // handle media();
               if(recordSampledOutpumedium.is_some()){
                    if(recordSampledOutpumedium.unwrap().sampleinmedium){
                    //    la idea es haver primero el fr ya sea de un medio o de una superficie y depues hacer occlusion o transmitance
                    
                        let recordmedia  = recordSampledOutpumedium. unwrap(); 
                        let f=  MediumType::phase_medium((-r.direction).dot(nextdir), hit.get_medium().get_g_asymetrical_parameter());
                        let mut tr= hit.get_medium().transmission(scene,sampler,&recordmedia.pinteraction, &light.getPositionws());
                        tr  *=  f;  
                        let l =    LigtImportanceSampling::mulLinearSrgb(tr, Ld);
                        //   println!("{:?}", l);
                        colorRes[0] += l.red ; 
                        colorRes[1] += l.green ; 
                        colorRes[2] += l.blue ;
                    }
              
 
                }else{

                            //  compute fr();
                            let prev = -r.direction;
                            let next = (light.getPositionws() - hit.point).normalize();
                            //  println!("{:?}", prev);
                            let fr = hit.material.fr(prev, next);
                            let absd = hit.normal.dot(next).abs() as f32;
                    if handleMediaOrhandleShadowrayTransmission {
                        let mut tr= hit.get_medium().transmission(scene,sampler,&hit.point, &light.getPositionws());
                        colorRes[0] += fr.red * Ld.red * absd *( tr.red as f32);
                        colorRes[1] += fr.green * Ld.green * absd *( tr.green as f32);
                        colorRes[2] += fr.blue * Ld.blue * absd *( tr.blue as f32);
                        
                    }else{
                        // handle_transmission() or handle_shadowray();
                                                    
                        if let Some(hayoclusion) =new_interface_for_intersect_occlusion(&occlray.unwrap(), scene,&light.getPositionws()){
                            if !hayoclusion.0 {
                                colorRes[0] += fr.red * Ld.red * absd;
                                colorRes[1] += fr.green * Ld.green * absd;
                                colorRes[2] += fr.blue * Ld.blue * absd;

                            }
                        }
                    } 
                }

              

            }else if light.is_ambientlight(){
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
                let nsamples = <Light as  NumSamples >::num_samples(light);
                // testing function : Lights::testing_area_light_new_interface
                // estimo constribucion de luz de area
                let mut newLd: Vec<f32> = vec![0.0; 3];
                for i in 0..nsamples{
                    let plightsample  = sampler.get2d();

                    let resnewe = LigtImportanceSampling::samplelight1(&r, &scene, &hit, scene.lights[0].clone(), &plightsample);
                    updateL(&mut newLd, resnewe); 
                    // let mut accpartial:Srgb=Srgb::new(0.,0.,0.);
                    // accpartial = LigtImportanceSampling::addSrgb(accpartial, resnewe);
                  
                   

                   // estimo constribucion del bsdf
                   let pbsdfsample  = sampler.get2d();
                    // println!("plight {:?}, pbsdf {:?}", plightsample, pbsdfsample);
                   let resnewe  = LigtImportanceSampling::samplebsdf1(&r, &scene, &hit,scene.lights[0].clone(),& pbsdfsample);
                   if !isBlack(resnewe){
                       updateL(&mut newLd,resnewe); 
                   } 
                }
              

               
               newLd[0] = newLd[0] / (nsamples as f32);
               newLd[1] = newLd[1] / (nsamples as f32);
               newLd[2] = newLd[2] / (nsamples as f32);
               colorRes[0] += newLd[0];
               colorRes[1] += newLd[1];
               colorRes[2] += newLd[2];
           
                
            } else if !light.is_arealight() && light.is_background_area_light() {
                panic!("estimate lights !light.is_arealight() && light.is_background_area_light---------");
                // let nsamples = 8;
                // let Ld = LigtImportanceSampling::estimate_bck_area_light(
                //     r, light, scene, hit, sampler, nsamples,
                // );
                // colorRes[0] += Ld.red;
                // colorRes[1] += Ld.green;
                // colorRes[2] += Ld.blue;
            }
        });
        let L = Srgb::new(colorRes[0], colorRes[1], colorRes[2]); 
        L
    }else {
        Srgb::new(0.0, 0.0, 0.0)
    }
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
                // estamos en un area light std y esto es super trampa :-D
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
use crate::scene::Scene1;
use crate::threapoolexamples1::pararell::{*};
pub
struct DirectIntegrator{
    scene : Scene1 ,
    sampler : SamplerType,
    film:pararell::Film1,
    camera : PerspectiveCamera,
    config : Config,
}
impl DirectIntegrator {
    pub fn new(scene : Scene1,   sampler : SamplerType, film: pararell::Film1, camera : PerspectiveCamera, config: Config)->Self{
        DirectIntegrator {scene :  scene  , sampler, film, camera, config}
    }
    pub fn get_config(&self)->&Config{
        &self.config
    }
    pub fn preprocess(& self ){}
    pub fn integrate(&self,pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, sampler: &mut SamplerType)->Srgb{ 
        
        // println!("{:?} {:?}" ,r.origin, r.direction);
        let camera = self.get_camera();
        let r = camera.get_ray(pfilm);
        let mut  hitop = new_interface_for_intersect_scene(&r, scene);
        let mut L: Vec<f32> = vec![0.0; 3];
        if let Some(hit) = hitop {
           
            let mut colorRes: Vec<f32> = vec![0.0; 3];
            if hit.is_emision.unwrap() {
                let Lemit = hit.emission.unwrap();
                    updateL(&mut L, Lemit);
                // aqui el path continua ... esto lo hago por el test
                return Srgb::new(
                    clamp(L[0], 0.0, 1.0),
                    clamp(L[1], 0.0, 1.0),
                    clamp(L[2], 0.0, 1.0),
                );
            }
            let Ld = new_interface_estimatelights(scene, hit, &r, sampler, None, false);
            
            L[0] = Ld.red;
            L[1] = Ld.green;
            L[2] = Ld.blue;
          // return  Srgb::new(1., 1., 1.);
        }else{
            let ls = &scene.lights;
            ls.into_iter().for_each(|light| {
                if light.is_background_area_light() {
                    let Lemit = light.get_emission(None, &r);

                    //    updateL(&mut L, Lemit);
                }
            })
        }
        Srgb::new(
            clamp(L[0], 0.0, 1.0),
            clamp(L[1], 0.0, 1.0),
            clamp(L[2], 0.0, 1.0),
        )

         
     
    }
    pub fn get_res(&self){}
    pub fn get_scene(&self)-> &Scene1 { &self.scene }
    pub fn get_spp(&self)->u32{
        self.sampler.get_spp()
     }
   pub fn get_film(&self)->&pararell::Film1{ &self.film}
   pub fn get_camera(&self)->&PerspectiveCamera{ &self.camera}
   pub fn get_sampler(&self)->&SamplerType{ &self.sampler}
   
}
//    integrator api! 


pub
struct AOIntegrator{
    scene : Scene1 ,
    sampler : SamplerType,
    film:pararell::Film1,
    camera : PerspectiveCamera,
    config : Config,
}


impl AOIntegrator {
    pub fn get_config(&self)->&Config{
        &self.config
    }
    pub fn new(scene : Scene1,   sampler : SamplerType, film: pararell::Film1, camera : PerspectiveCamera ,config : Config)->Self{
        AOIntegrator {scene :  scene  , sampler, film, camera, config}
    }
    pub fn preprocess(& self ){}
    pub fn integrate(&self,pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, sampler: &mut SamplerType)->Srgb{ 
        let camera = self.get_camera();
        let r = camera.get_ray(pfilm);
        let mut  hitop = new_interface_for_intersect_scene(&r, scene);
        let mut L: Vec<f32> = vec![0.0; 3];
        if let Some(hit) = hitop {
            let nsamples = 32;
            let fnsamples = nsamples as f32;
            for isample in 0..nsamples{
                let psample = sampler.get2d();
            
                let recout = hit.material.sample(RecordSampleIn ::from_hitrecord(hit, &r, psample));
                if let Some(h) = new_interface_for_intersect_scene(&recout.newray.unwrap(), scene){
                    L[0] += (1.0 / fnsamples);
                    L[1] += (1.0 / fnsamples);
                    L[2] += (1.0 / fnsamples);
                }
            }
          
            
           return  Srgb::new(clamp(1.0 - L[0], 0.0, 1.0),clamp(1.0 - L[1], 0.0, 1.0),clamp(1.0 - L[2], 0.0, 1.0),);
          // return  Srgb::new(clamp(  L[0], 0.0, 1.0),clamp(  L[1], 0.0, 1.0),clamp(L[2], 0.0, 1.0),);
        };
        Srgb::new(0., 0., 0.)
    }
    pub fn get_res(&self){}
    pub fn get_scene(&self)-> &Scene1 { &self.scene }
    pub fn get_spp(&self)->u32{
        self.sampler.get_spp()
     }
   pub fn get_film(&self)->&pararell::Film1{ &self.film}
   pub fn get_camera(&self)->&PerspectiveCamera{ &self.camera}
   pub fn get_sampler(&self)->&SamplerType{ &self.sampler}
}







pub
struct PathIntegrator{
    scene : Scene1 ,
    sampler : SamplerType,
    film:pararell::Film1,
    camera : PerspectiveCamera,
    depth : u32,
    config : Config
}


impl PathIntegrator {
    pub fn get_config(&self)->&Config{
        &self.config
    }
    pub fn new(scene : Scene1,   sampler : SamplerType, film: pararell::Film1, camera : PerspectiveCamera, depth : u32,config : Config)->Self{
        PathIntegrator {scene :  scene  , sampler, film, camera, depth, config}
    }
    pub fn preprocess(& self ){}
    pub fn integrate(&self,pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, sampler: &mut SamplerType)->Srgb{ 
        let camera = self.get_camera();
        let r = camera.get_ray(pfilm);
        let mut hitop = new_interface_for_intersect_scene(&r.clone(), scene); 

        let mut L = Srgb::new(0.0,0.0,0.0);

        if let Some(hitt) = hitop {

            let depth = self.depth as i32;
            let mut hitt = hitop.unwrap();
            let mut rcow = Cow::Borrowed(&r);
            let mut hitcow = Cow::Borrowed(&hitt);
            let mut paththrought: Srgb = Srgb::new(1.0, 1.0, 1.0);
            for idepth in 0..depth {

                if hitcow.is_emision.unwrap() {
                    let Lemit = hitcow.emission.unwrap();
                    //   updateL(&mut L, Lemit);
                    // aqui el path continua ... esto lo hago por el test
                    return LigtImportanceSampling::mulSrgb(Lemit, paththrought); 
                }

                if ! hitcow.material.is_specular(){
                    let Ld = new_interface_estimatelights(scene, *hitcow, &rcow, sampler, None, false);
                    
                    let L0 = LigtImportanceSampling::mulSrgb(Ld, paththrought);
        
                    L = LigtImportanceSampling::addSrgb( L , L0); 
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


                if let Some(newhit) = new_interface_for_intersect_scene(&rcow, scene) {
                    // println!("newhit");
                     hitcow = Cow::Owned(newhit);
                } else {
                    break;
                }
                
            }
            L
        }else {
            L
        }
    }
    pub fn get_res(&self){}
    pub fn get_scene(&self)-> &Scene1 { &self.scene }
    pub fn get_spp(&self)->u32{
        self.sampler.get_spp()
     }
   pub fn get_film(&self)->&pararell::Film1{ &self.film}
   pub fn get_camera(&self)->&PerspectiveCamera{ &self.camera}
   pub fn get_sampler(&self)->&SamplerType{ &self.sampler}
}






 






pub enum IntergratorBaseType{
    IntergratorType(Intergrator),
    BdptIntegratorType(bdpt2::bdpt2::BdptIntegrator),
    // MetropolisTransportIntegratorType(metropolis::metropolistransport::MetropolisIntegrator),
}
impl IntergratorBaseType {
    pub fn get_config(&self)->&Config{
        match self {
            IntergratorBaseType::IntergratorType(i)=> i.get_config(),
            IntergratorBaseType::BdptIntegratorType(i)=> i.get_config(),
     //       IntergratorBaseType::MetropolisTransportIntegratorType(i)=> i.get_config(),
            
        }
    }
    pub fn preprocess(&self){
        match self {
            IntergratorBaseType::IntergratorType(i)=> i.preprocess(),
            IntergratorBaseType::BdptIntegratorType(i)=> i.preprocess(),
       //      IntergratorBaseType::MetropolisTransportIntegratorType(i)=> i.preprocess(),

        }
   

    }
    pub fn get_scene(&self)->&Scene1{
        match self {
            IntergratorBaseType::IntergratorType(i)=> i.get_scene(),
            IntergratorBaseType::BdptIntegratorType(i)=> i.get_scene(),
      //       IntergratorBaseType::MetropolisTransportIntegratorType(i)=> i.get_scene(),
        }

    }
    pub fn get_spp(&self)->u32{
        match self {
            IntergratorBaseType::IntergratorType(i)=> i.get_spp(),
            IntergratorBaseType::BdptIntegratorType(i)=> i.get_spp(),
        //     IntergratorBaseType::MetropolisTransportIntegratorType(i)=> i.get_spp(),
        }

    }

    pub fn get_film(&self)->&pararell::Film1{
        match self {
            IntergratorBaseType::IntergratorType(i)=> i.get_film(),
            IntergratorBaseType::BdptIntegratorType(i)=> i.get_film(),
       //      IntergratorBaseType::MetropolisTransportIntegratorType(i)=> i.get_film(),
        } 
    }
    pub fn get_camera(&self)->&PerspectiveCamera{
        match self {
            IntergratorBaseType::IntergratorType(i)=> i.get_camera(),
            IntergratorBaseType::BdptIntegratorType(i)=> i.get_camera(),
      //       IntergratorBaseType::MetropolisTransportIntegratorType(i)=> i.get_camera(),
        } 
    }
    pub fn get_sampler(&self)->&SamplerType{
        match self {
            IntergratorBaseType::IntergratorType(i)=> i.get_sampler(),
            IntergratorBaseType::BdptIntegratorType(i)=> i.get_sampler(),
       //      IntergratorBaseType::MetropolisTransportIntegratorType(i)=> i.get_sampler(),

        } 
    }
    pub fn integrate(& self, pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, sampler: &mut SamplerType)->Srgb {
        match self {
            IntergratorBaseType::IntergratorType(i)=> i.integrate(pfilm,lights, scene, depth, sampler),
            IntergratorBaseType::BdptIntegratorType(i)=> i.integrate(pfilm,lights, scene, depth, sampler),
     //        IntergratorBaseType::MetropolisTransportIntegratorType(i)=>i.integrate(pfilm,lights, scene, depth, sampler),

        } 
    }
    
    pub fn integrate_serial(&mut self ){

        let start = Instant::now();
        let scene = self.get_scene();
        let res = (scene.width as i64, scene.height as i64);
        let spp = self.get_spp();
        let film = self.get_film();
         let film_arc = Arc::new(film);
        let cameraInstance = self.get_camera(); 
        let samplertype =  self.get_sampler();
        let filtergauss  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0,2.0),3.0));
   
       let integrator_impl = &self;
       let mut wholetile = film_arc.init_tile(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))));
       let mut sampler = samplertype.seed_instance();
       println!("integrate_serial!");
 
//  tengo que hacer algo con lo de l film que no se...
       for iy in 0..res.1{
    //    println!(" {}", iy);
 
           for ix in 0..res.0{
           
               sampler.start_pixel(Point2i::new(ix as i64, iy as i64)); 
               let samplecamerafilm = sampler.get2d();
            //    println!("{:?}", samplecamerafilm);
              let color =  integrator_impl.integrate( &samplecamerafilm,&scene.lights,scene , 3,&mut sampler);
              wholetile.add_sample(&Point2f::new(samplecamerafilm.0 , samplecamerafilm.1), color);

               while sampler.start_next_sample(){
                   let samplecamera = sampler.get2d();
                //    println!("{:?}", samplecamera);
                    let color =  integrator_impl.integrate( &samplecamera,&scene.lights,scene , 3,&mut sampler);
                   
                  wholetile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), color);
               }
           
           }
       }
    
         film_arc.merge_tile(&wholetile);
    let conf = self.get_config();
    let directory =  conf.directory.as_ref().unwrap();
    let filename : Option<String>  =  conf.filename.clone();
    if filename.is_some(){
        film_arc.commit_and_write(&directory,&filename.unwrap(), true).unwrap();
        let configtoprint = Config::from_config(self. get_config(),  start.clone());
          configtoprint.save();
          println!("{}",    serde_json::to_string(&configtoprint).unwrap() );
    }

   
    



        // self.integrate(pfilm, lights, scene, depth, sampler);
    }
    pub fn integrate_par(&mut self ){
         let start = Instant::now();
        let scene = self.get_scene();
        let res = (scene.width as i64, scene.height as i64);
        let spp = self.get_spp();
        let film = self.get_film();
        let film_arc = Arc::new(film);
        let cameraInstance = self.get_camera(); 
        let sampler = Arc::new( self.get_sampler());
       let integrator_impl = &self;
        let num_workers =  12;// num_cpus::get();
        crossbeam::scope(|scope| {
            let (sendertorender0, receiverfrominitialize0) = crossbeam_channel::bounded(num_workers);
      
            let filmref =  film;
            let camera = cameraInstance;
            let sceneshared = scene;


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
                let mut sampler = sampler.seed_instance();
                // let atom_hits_ref =&num_total_of_hits;
                // let num_total_of_px_ref = &num_total_of_pixel;
                let rxclone = receiverfrominitialize0.clone();
                scope.spawn(move|_|{
                    
                    let timestarttile =  Instant::now();
                 
                    for mut packedwork in  rxclone.recv().into_iter() {
                      
                        println!("render worker -> thread id {:?}", std::thread::current().id()) ;
                        println!("       compute tile {:?}",  packedwork.0) ;
                    
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

                             let  samplecamerafilm = sampler.get2d();
                          
                             debug_assert_eq!( samplecamerafilm.0.floor()as usize , x as usize);
                             debug_assert_eq!( samplecamerafilm.1.floor()as usize , y as usize );
                         
                            //  let ray = cameraInstance.get_ray(& samplecamerafilm);
                            
                             let color = integrator_impl.integrate(&samplecamerafilm,&scene.lights,scene , 1,&mut sampler);
                     
                             tile.add_sample(&Point2f::new(samplecamerafilm.0 , samplecamerafilm.1), color);
                             while sampler.start_next_sample(){
    
                                   let samplecamerafilm = sampler.get2d();
                                   debug_assert_eq!(samplecamerafilm.0.floor()as usize , x as usize);
                                   debug_assert_eq!(samplecamerafilm.1.floor()as usize , y as usize );
                        
                                //    let ray = cameraInstance.get_ray(& samplecamerafilm);
                                   let color =  integrator_impl.integrate(&samplecamerafilm,&scene.lights,scene , 1,&mut sampler); 
                                   tile.add_sample(&Point2f::new(samplecamerafilm.0 , samplecamerafilm.1), color);
                              }
                //                 // println!("num_total_of_px --------------->{:?}", num_total_of_px_ref );
                            
                              
    
  
                        } 
                         
                        filmref.merge_tile(&tile);
                       
                        println!("  integrate_par time: {} seconds", timestarttile.elapsed().as_secs_f64());
                     
                      }
    
                    
    
                  } //scope.spawn 
                );
            } //  for _ in 0..num_workers {
        
        }).unwrap();

       
    
       
        
    

        let conf = self.get_config();
        
    let filename : Option<String>  =  conf.filename.clone();
    if filename.is_some(){
        let directory =  conf.directory.as_ref().unwrap();
        film_arc.commit_and_write(&directory,&filename.unwrap(),true).unwrap();
        let configtoprint = Config::from_config(self. get_config(),  start.clone());
          configtoprint.save();
          println!("{}",    serde_json::to_string(&configtoprint).unwrap() );
    }

   
    

    }
    
}



// struct BdptIntegrator{
//     scene : Scene1 ,
//     sampler : SamplerType,
//     film:pararell::Film1,
//     camera : PerspectiveCamera,
//     config: Config,
  
// }
// impl BdptIntegrator {
//     pub fn get_config(&self)->&Config{
//         &self.config
//     }
//     pub fn new(scene : Scene1,   sampler : SamplerType, film: pararell::Film1, camera : PerspectiveCamera, config:Config)->Self{
//         BdptIntegrator {scene :  scene  , sampler, film, camera, config}
//     }
//     pub fn preprocess(& self ){}
//    pub fn integrate(&self, pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, sampler: &mut SamplerType)->Srgb{ 
        
       
// Srgb::default()
         
     
//     }
//     pub fn get_res(&self){}
//     pub fn get_scene(&self)-> &Scene1 { &self.scene }
//     pub fn get_spp(&self)->u32{self.sampler.get_spp()}
//    pub fn get_film(&self)->&pararell::Film1{ &self.film}
//    pub fn get_camera(&self)->&PerspectiveCamera{ &self.camera}
//    pub fn get_sampler(&self)->&SamplerType{ &self.sampler}
// }











pub enum Intergrator{
    DirectIntegratorType(DirectIntegrator),
    AOIntegratorType(AOIntegrator),
    PathIntegratorType(PathIntegrator)
}
impl Intergrator{
    pub fn get_config(&self)->&Config{
        match self {
            Intergrator::DirectIntegratorType(i)=> i.get_config(),
            Intergrator::AOIntegratorType(i)=>i.get_config(),
            Intergrator::PathIntegratorType(i)=>i.get_config(),
        }

    }

    pub fn preprocess(&self){
        match self {
            Intergrator::DirectIntegratorType(i)=> i.preprocess(),
            Intergrator::AOIntegratorType(i)=>i.preprocess(),
            Intergrator::PathIntegratorType(i)=>i.preprocess(),
        }

    }
    pub fn get_scene(&self)->&Scene1{
        match self {
            Intergrator::DirectIntegratorType(i)=> i.get_scene(),
            Intergrator::AOIntegratorType(i)=>i.get_scene(),
            Intergrator::PathIntegratorType(i)=>i.get_scene(),
        }

    }
    pub fn get_spp(&self)->u32{
        match self {
            Intergrator::DirectIntegratorType(i)=> i.get_spp(),
            Intergrator::AOIntegratorType(i)=>i.get_spp(),
            Intergrator::PathIntegratorType(i)=>i.get_spp(),
        }

    }

    pub fn get_film(&self)->&pararell::Film1{
        match self {
            Intergrator::DirectIntegratorType(i)=> i.get_film(),
            Intergrator::AOIntegratorType(i)=>i.get_film(),
            Intergrator::PathIntegratorType(i)=>i.get_film(),
        } 
    }
    pub fn get_camera(&self)->&PerspectiveCamera{
        match self {
            Intergrator::DirectIntegratorType(i)=> i.get_camera(),
            Intergrator::AOIntegratorType(i)=>i.get_camera(),
            Intergrator::PathIntegratorType(i)=>i.get_camera(),
        } 
    }
    pub fn get_sampler(&self)->&SamplerType{
        match self {
            Intergrator::DirectIntegratorType(i)=> i.get_sampler(),
            Intergrator::AOIntegratorType(i)=>i.get_sampler(),
            Intergrator::PathIntegratorType(i)=>i.get_sampler(),

        } 
    }
    // pub fn integrate(&self, pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, sampler: &mut SamplerType)
    pub fn integrate(& self,pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, sampler: &mut SamplerType)->Srgb {
        match self {
            Intergrator::DirectIntegratorType(i)=> i.integrate(pfilm, lights, scene, depth, sampler),
            Intergrator::AOIntegratorType(i)=>i.integrate(pfilm, lights, scene, depth, sampler),
            Intergrator::PathIntegratorType(i)=>i.integrate(pfilm, lights, scene, depth, sampler),
        } 
    }
    


    // pub fn integrate_serial(&mut self ){

    //     let start = Instant::now();
    //     let scene = self.get_scene();
    //     let res = (scene.width as i64, scene.height as i64);
    //     let spp = self.get_spp();
    //     let film = self.get_film();
    //     let film_arc = Arc::new(film);
    //     let cameraInstance = self.get_camera(); 
    //     let samplertype =  self.get_sampler();
    //     let filtergauss  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0,2.0),3.0));
   
    //    let integrator_impl = &self;
    //    let mut film  = pararell::Film1::init(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))) , filtergauss); 
    //    let mut wholetile = film.init_tile(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))));
    //    let mut sampler = samplertype.seed_instance();
    //    println!("integrate_serial!");
    //     for iy in 0..res.1{
    //         for ix in 0..res.0{
            
    //             sampler.start_pixel(Point2i::new(ix as i64, iy as i64)); 
    //             let samplecamerafilm = sampler.get2d();
            
    //             // let ray = cameraInstance.get_ray(&sampleforcamerauv);
    //             let ray = cameraInstance.get_ray(& samplecamerafilm);
    //             let color =  integrator_impl.integrate( &ray,&scene.lights,scene , 1,&mut sampler);
    //             // if ishit {num_total_of_hits.fetch_add(1, Ordering::SeqCst);} 
    //             wholetile.add_sample(&Point2f::new(samplecamerafilm.0 , samplecamerafilm.1), color);

    //             while sampler.start_next_sample(){
    //                 let samplecamera = sampler.get2d();
    //                 // let sampleforcamerauv = to_map(&samplecamera, res.0 as i64  , res.1 as i64);
    //                 let ray = cameraInstance.get_ray(&samplecamera);
    //                 let color =  integrator_impl.integrate( &ray,&scene.lights,scene , 1,&mut sampler);
                    
    //                 // if ishit {num_total_of_hits.fetch_add(1, Ordering::SeqCst);}
    //                 wholetile.add_sample(&Point2f::new(samplecamera.0 , samplecamera.1), color);
    //             }
            
    //         }
    //     }

    //     film_arc.merge_tile(&wholetile);
    //     film_arc.commit_and_write("filtertest", "miexample_brute_force.png").unwrap();

    //    println!(" time: {} seconds", start.elapsed().as_secs_f64());
    // }
    // pub fn integrate_pararell(&mut self ){
    //     let start = Instant::now();
    //     let scene = self.get_scene();
    //     let res = (scene.width as i64, scene.height as i64);
    //     let spp = self.get_spp();
    //     let film = self.get_film();
    //     let film_arc = Arc::new(film);
    //     let cameraInstance = self.get_camera(); 
    //     let sampler = Arc::new( self.get_sampler());
    //    let integrator_impl = &self;
    //     let num_workers =  12;// num_cpus::get();
    //     crossbeam::scope(|scope| {
    //         let (sendertorender0, receiverfrominitialize0) = crossbeam_channel::bounded(num_workers);
      
    //         let filmref =  film;
    //         let camera = cameraInstance;
    //         let sceneshared = scene;


    //         scope.spawn(move|_|{
    //             let mut area = 0 ;
    //             let sender0clone = sendertorender0.clone();
    //             let vlst =    init_tiles_withworkers(num_workers as i64, filmref.width(), filmref.height());
    //             let mut cnt = 0;
    //             for tilescoords in vlst.into_iter(){
                   
    //                 let tile = filmref.init_tile(BBds2i::new(Point2::new(tilescoords.1.0.0  as  i64 ,tilescoords.1.0.1 as  i64,),Point2::new(tilescoords.1.1.0 as  i64 ,tilescoords.1.1.1 as  i64  )));
                    
    //                 println!("init_tiles_withworkers -> thread id {:?}", std::thread::current().id()) ;
    //                 let packedwork = (tilescoords, tile);
    //                     sender0clone.send(packedwork).unwrap_or_else(|op| panic!("---sender error!!--->{}", op));     
    //             }
    //             // println!("{}", area);
    //             drop(sender0clone)
    //         });

    //         for _ in 0..num_workers {
    //             let mut sampler = sampler.seed_instance();
    //             // let atom_hits_ref =&num_total_of_hits;
    //             // let num_total_of_px_ref = &num_total_of_pixel;
    //             let rxclone = receiverfrominitialize0.clone();
    //              scope.spawn(move|_|{
                    
                 
    //                 for mut packedwork in  rxclone.recv().into_iter() {
                      
    //                     println!("render worker -> thread id {:?}", std::thread::current().id()) ;
    //                     println!("       compute tile {:?}",  packedwork.0) ;
                    
    //                     //unfold the stuff send by thread "parent"
    //                     let mut tile = & mut packedwork.1;
    //                     let tilescoords =   packedwork.0;
    
    //                     let startx = tilescoords.1.0.0;
    //                     let starty = tilescoords.1.0.1;
    //                     let endx = tilescoords.1.1.0 ;
    //                     let endy =tilescoords.1.1.1;
                
    //                    for px in  itertools::iproduct!(startx..endx,starty..endy ){
    //                          let x = px.0  as f64;
    //                          let y = px.1  as f64;
    //                          let r =   (x - startx as f64) as  f32 / (endx - startx) as  f32;
    //                          let g=   (y - starty as f64) as  f32 / (endy - starty)as  f32;
    //                          sampler.start_pixel(Point2i::new(x as i64, y as i64));

    //                          let  samplecamerafilm = sampler.get2d();
                          
    //                          debug_assert_eq!( samplecamerafilm.0.floor()as usize , x as usize);
    //                          debug_assert_eq!( samplecamerafilm.1.floor()as usize , y as usize );
                         
    //                          let ray = cameraInstance.get_ray(& samplecamerafilm);
                            
    //                          let color = integrator_impl.integrate(&ray,&scene.lights,scene , 1,&mut sampler);
    //                  //      let ishit =  get_ray_path_1(&ray,&sceneshared.lights, &sceneshared, 1, &mut sampler );
    //                //           if ishit {atom_hits_ref.fetch_add(1, Ordering::SeqCst);}
    //             //             num_total_of_px_ref.fetch_add(1, Ordering::Acquire);
    //                          tile.add_sample(&Point2f::new(samplecamerafilm.0 , samplecamerafilm.1), color);
    //                          while sampler.start_next_sample(){
    
    //                                let samplecamerafilm = sampler.get2d();
    //                                debug_assert_eq!(samplecamerafilm.0.floor()as usize , x as usize);
    //                                debug_assert_eq!(samplecamerafilm.1.floor()as usize , y as usize );
                        
    //                                let ray = cameraInstance.get_ray(& samplecamerafilm);
    //                                let color =  integrator_impl.integrate(&ray,&scene.lights,scene , 1,&mut sampler); 
    //                                 tile.add_sample(&Point2f::new(samplecamerafilm.0 , samplecamerafilm.1), color);
    //                           }
    //             //                 // println!("num_total_of_px --------------->{:?}", num_total_of_px_ref );
                            
                              
    
    //                     } 
    
    //                     filmref.merge_tile(&tile);
                       
                     
                     
    //                   }
    
                    
    
    //               });
    //         } //  for _ in 0..num_workers {
        
    //     }).unwrap();

    //     film.commit_and_write("filtertest", "new_interface_area_pararellel_pathtracer_integrate_mirror_cylinder.png").unwrap();
      
    //     println!("  integrate_par time: {} seconds", start.elapsed().as_secs_f64());
    // }
}


#[test]
pub fn main_render_release(){
    bdpt2::bdpt2::main_bdpt2();
    // main_render();
}

pub fn main_render(){
    if(true){
        bdpt2::bdpt2::main_bdpt2();
    }else{
        let h  = 512.0;
  let   w  = h * 1.77777777778;
    let res:(usize,usize) = (w as usize, h as usize);
    let spp =4096;
    let config = Config::from_args(spp, w as i64, h as i64, 1, 
        Some("newdir".to_string()), 
        Some("path_tracer_disk.png")
    );
    let scene  = Scene1::make_scene(
            res.0 as usize, 
            res.1 as usize, 
            vec![
                // Light::New_Interface_AreaLightType( New_Interface_AreaLight::new(  Srgb::new(4.0,4.0,4.0),PrimitiveType::CylinderType(   Cylinder::new(Vector3::new( 0.0000, 0.30,2.50000),Matrix3::from_angle_y(Deg( 90.0)),-1.0, 1.0, 0.150,MaterialDescType::NoneType)),4)),
          Light::New_Interface_AreaLightType ( New_Interface_AreaLight::new( Srgb::new(4.0,4.0,4.0),PrimitiveType::SphereType( Sphere::new(Vector3::new(0.0000, 0.0,2.50000),0.35,   MaterialDescType::NoneType)),4)),
           Light::New_Interface_AreaLightType(New_Interface_AreaLight ::new(Srgb::new(4.0,4.0,4.0),PrimitiveType::DiskType(Disk::new(Vector3::new( 0.0000,0.3,2.5000),Vector3::new(0.0, -1.0, 0.0),0.0, 0.5, MaterialDescType::NoneType)),4)),
   
             //  Light::PointLight(PointLight {iu: 4.0,positionws: Point3::new(0.0000,0.3,2.5000),color: Srgb::new(4.190,4.190, 4.190),})
                ], 
            vec![
            //   PrimitiveType::SphereType(Spheref64::new(Vector3f::new(-0.7,-0.75,2.50), 0.250, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,0.0)))),
            // //   PrimitiveType::SphereType(Spheref64::new(Vector3f::new(-0.0,-0.75,3.00), 0.250, MaterialDescType::MetaType(Metal::from_constant_rought(0.01, 0.01)))),
            //   PrimitiveType::SphereType(Spheref64::new(Vector3f::new(-0.0,-0.75,3.00), 0.250, MaterialDescType::MirrorType(Mirror{..Default::default()}))),
            // PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.7,-0.75,2.50), 0.25, MaterialDescType::PlasticType(Plastic::from_albedo(0.0,1.0,1.0)))),
                  PrimitiveType::DiskType( Disk::new(Vector3::new( 0.0000,-1.0,0.0000),Vector3::new(0.0, 1.0, 0.0),0.0, 100.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00))))
                ], 
            spp.clone() as usize,1);

           
    // let rray =      Ray::new(Point3f::new(0.,0.1,0.),Vector3f::new(0.0,-1.0,0.0));
    // let reshit = new_interface_for_intersect_scene(&rray, &scene);
  

    let sampler  = SamplerType::UniformType( SamplerUniform::new(((0, 0), (res.0 as u32, res.1 as u32)),spp.clone() as u32,false,Some(0)) );
    let cameraInstance =  PerspectiveCamera::from_lookat(Point3f::new(0.0,0.000, 0.0), Point3f::new(0.0,0.0,1.00), 1e-2, 100.0, 75.0, (res.0 as u32, res.1 as u32));
    let filtergauss  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0,2.0),3.0));
    let mut film  = pararell::Film1::new(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))) , filtergauss);  
    // let mut integrator= Intergrator::AOIntegratorType(AOIntegrator::new(scene, sampler, film, cameraInstance));
   // let mut integrator = Intergrator::DirectIntegratorType(DirectIntegrator::new(scene, sampler, film, cameraInstance));
    //  let mut integrator = Intergrator::PathIntegratorType(PathIntegrator::new(scene, sampler, film, cameraInstance,3));
    // integrator.preprocess( );
    // integrator.integrate_pararell();
    let mut integrator =  IntergratorBaseType::IntergratorType(Intergrator::PathIntegratorType(PathIntegrator::new(scene, sampler, film, cameraInstance,3, config)));
   integrator.integrate_serial();
   
    }
    
}