use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use cgmath::{Point3, Vector1, Vector3};
use hexf::hexf32;

use palette::Srgb;

use crate::bdpt::{
    bidirectional, cameraTracerStrategy, compute_weights, compute_weights1, convert_static,
    emissionDirectStrategy, init_camera_path, init_light_path, lightTracerStrategy, BdptVtx,
    PathTypes, PathVtx, PathLightVtx,
};
use crate::integrator::isBlack;
use crate::materials::SampleIllumination;
use crate::materials::{MaterialDescType, Pdf, Plastic, RecordSampleIn};
use crate::primitives::prims::HitRecord;
use crate::primitives::prims::Ray;
use crate::primitives::prims::Sphere;
use crate::primitives::{Disk, PrimitiveIntersection};
use crate::raytracerv2::{interset_scene, Scene};
use crate::sampler::{Sampler, SamplerUniform};
use crate::Cameras::PerspectiveCamera;
use crate::Lights::RecordSampleLightEmissionIn;
use crate::Lights::RecordSampleLightEmissionOut;
use crate::Lights::SampleEmission;
use crate::Lights::{Light, LigtImportanceSampling, PointLight};
use crate::{Point3f, Vector3f};

fn PlotChains(vpath: &Vec<PathTypes>) {
    for (id, p) in vpath.iter().enumerate() {
        println!("{}, {:?}", id, p.transport());
        println!("  v.p()       {:?}", p.p());
        println!("  v.n()       {:?}", p.n());
        println!("  v.pdfFwd()  {:?}", p.get_pdfnext());
        println!("  v.pdfRev()  {:?}", p.get_pdfrev());
    }
}

#[test]
pub fn test_bdpt_1() {
    let filmres: (u32, u32) = (512, 512);
    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );

    let sphere2smallptr =
        &(Box::new(sphere) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);
    let primitivesIntersections: Vec<&Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> =
        vec![sphere2smallptr];

    let pointlight = Light::PointLight(PointLight {
        iu: 4.0,
        positionws: Point3::new(0.000, 0.0, 0.000),
        color: Srgb::new(1.0, 1.0, 1.0),
    });
    let scene: &Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![pointlight],
        vec![],
        primitivesIntersections,
        1,
        1,
        None,
    );
    // let mut sampler :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
    // init_light_path(scene, & mut sampler);

    let mut samplercam: Box<dyn Sampler> = Box::new(SamplerUniform::new(
        ((0, 0), (32 as u32, 32 as u32)),
        32,
        false,
        Some(0),
    ));
    let cameraInstance = PerspectiveCamera::new(1e-3, 1000.0, 45.0, filmres);

    //   let vpath = init_camera_path(scene,(256.0,256.0),& mut samplercam, cameraInstance );

    //    for   mut vtx in & mut vpath.clone(). into_iter().peekable() {

    //     // vtx.set_pdfrev(1111.0);
    //     // println!("{:?}",vtx);
    // }
    // println!("{:?}",vpath);0.318309873
    //PlotChains(&vpath);
}

#[test]
pub fn test_bdpt_2() {
    let maxdepth: i32 = 16;
    let filmres: (u32, u32) = (512, 512);
    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );

    let sphere2smallptr =
        &(Box::new(sphere) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);
    let primitivesIntersections: Vec<&Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> =
        vec![sphere2smallptr];

    let pointlight = Light::PointLight(PointLight {
        iu: 4.0,
        positionws: Point3::new(0.000, 0.0, 0.000),
        color: Srgb::new(
            std::f32::consts::PI,
            std::f32::consts::PI,
            std::f32::consts::PI,
        ),
    });
    let scene: &Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![pointlight],
        vec![],
        primitivesIntersections,
        1,
        1,
        None,
    );
    let mut pathlight: Vec<PathTypes> = vec![];
    let mut samplerlight: Box<dyn Sampler> = Box::new(SamplerUniform::new(
        ((0, 0), (32 as u32, 32 as u32)),
        32,
        false,
        Some(0),
    ));
    init_light_path(scene, &mut pathlight, &mut samplerlight, maxdepth as usize);
    //   println!("\n\n\n pathlight ");
    //   PlotChains(&pathlight);
    let mut samplercam: Box<dyn Sampler> = Box::new(SamplerUniform::new(
        ((0, 0), (32 as u32, 32 as u32)),
        32,
        false,
        Some(0),
    ));
    let cameraInstance = PerspectiveCamera::new(1e-3, 1000.0, 45.0, filmres);
    let mut pathcamera: Vec<PathTypes> = vec![];
    init_camera_path(
        scene,
        &(256.0, 256.0),
        &mut pathcamera,
        cameraInstance,
        &mut samplercam,
        maxdepth as usize,
    );
    //     println!("\n\n\n pathcamera");
    //  PlotChains(&pathcamera);
    let mut samplerconnect: Box<dyn Sampler> = Box::new(SamplerUniform::new(
        ((0, 0), (32 as u32, 32 as u32)),
        32,
        false,
        Some(0),
    ));
    let mut Lsum = Srgb::new(0.0, 0.0, 0.0);
    for t in 2..=pathcamera.len() as i32 {
        for s in 0..=pathlight.len() as i32 {
            let depth: i32 = s + t - 2;
            //   println!("s={} t={}", s, t);
            let pfilm = Point3f::new(256.0, 256.0, 0.0);
            if t == 1 && s == 1 {
                continue;
            }
            if depth > maxdepth {
                continue;
            }
            if s == 0 {
                if false {
                    let L = emissionDirectStrategy(
                        t,
                        scene,
                        &mut samplerconnect,
                        &pathlight,
                        &pathcamera,
                    );
                    let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                    let weight =
                        compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                    Lsum = LigtImportanceSampling::addSrgb(
                        LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                        Lsum,
                    );
                }
            } else if t == 1 {
                if false {
                    let L = cameraTracerStrategy(
                        s,
                        scene,
                        &pfilm,
                        &mut samplerconnect,
                        cameraInstance,
                        &pathlight,
                        &pathcamera,
                    );
                    let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                    let weight =
                        compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                    // Lsum = LigtImportanceSampling::addSrgb(
                    //     LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                    //     Lsum,
                    // );
                }
            } else if s == 1 {
                if true {
                    let (L, vtx) =
                        lightTracerStrategy(t, scene, &mut samplerconnect, &pathlight, &pathcamera);

                    let weight = compute_weights1(s,t,scene,vtx.unwrap(),pathlight.as_mut_slice(),pathcamera.as_mut_slice(),);
                    println!("s {} t {} L {:?} weight:{}",s,t,LigtImportanceSampling::mulScalarSrgb(L, weight as f32),weight);
                    Lsum = LigtImportanceSampling::addSrgb(LigtImportanceSampling::mulScalarSrgb(L, weight as f32),Lsum,);
                    // bidirectional(s,t, scene, &Point3f::new(256.0,256.0,0.0),   & pathlight,  & pathcamera );
                    //  let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                    //  compute_weights1(s,t,scene, &sampledvtx,  pathlight.as_mut_slice(),   pathcamera.as_mut_slice());
                }
            } else {
                if false {
                    let Lbidirectional = bidirectional(
                        s,
                        t,
                        scene,
                        &Point3f::new(256.0, 256.0, 0.0),
                        &pathlight,
                        &pathcamera,
                    );
                    let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                    let weight =
                        compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                    println!(
                        "s {} t {} L {:?} weight:{}",
                        s,
                        t,
                        LigtImportanceSampling::mulScalarSrgb(Lbidirectional, weight as f32),
                        weight
                    );
                    Lsum = LigtImportanceSampling::addSrgb(
                        LigtImportanceSampling::mulScalarSrgb(Lbidirectional, weight as f32),
                        Lsum,
                    );
                }
                //  println!("s {} t {} L {:?} weight:{}",s, t, LigtImportanceSampling::mulScalarSrgb( Lbidirectional, weight as f32), weight);
            }
        }
    }
    println!(" L {:?}  ", Lsum);
    // bidirectional(2,2, scene, &Point3f::new(256.0,256.0,0.0),    & pathlight,  & pathcamera);
}

#[test]
pub fn testing_bdpt_tracing() {
    let maxdepth: i32 = 16;
    let filmres: (u32, u32) = (16, 16);
    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 2.000),
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );

    let sphere2smallptr =
        &(Box::new(sphere) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);

    let diskescene = Disk::new(
        Vector3::new(0.0001, -1.0, 0.0001),
        Vector3::new(0.0, 1.0, 0.0),
        0.0,
        1000.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );
    let diskescene =
        &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);

    let primitivesIntersections: Vec<&Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> =
        vec![sphere2smallptr, diskescene];

    let pointlight = Light::PointLight(PointLight {
        iu: 4.0,
        positionws: Point3::new(0.000, 1.0, 0.000),
        color: Srgb::new(
            std::f32::consts::PI,
            std::f32::consts::PI,
            std::f32::consts::PI,
        ),
    });
    let scene: &Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![pointlight],
        vec![],
        primitivesIntersections,
        1,
        1,
        None,
    );

    let mut samplerlight: Box<dyn Sampler> = Box::new(SamplerUniform::new(
        ((0, 0), (32 as u32, 32 as u32)),
        32,
        false,
        Some(0),
    ));
    let cameraInstance = PerspectiveCamera::new(1e-3, 1000.0, 45.0, filmres);

    // init_camera_path(scene,&(256.0, 256.0),&mut pathcamera, cameraInstance, &mut samplercam, maxdepth as usize);

    for yy in 0..filmres.1 {
        for xx in 0..filmres.0 {
            let mut samplerinstance: Box<dyn Sampler> = Box::new(SamplerUniform::new(
                ((0, 0), (32 as u32, 32 as u32)),
                32,
                false,
                Some(0),
            ));
            let p = &(xx as f64, yy as f64);
            //    let p = &(xx as f64, 32.000  as f64);
            let mut pathlight: Vec<PathTypes> = vec![];
            let mut pathcamera: Vec<PathTypes> = vec![];

            println!("\n px : {:?}", p);

            if (true) {
                let ncma = init_camera_path(
                    scene,
                    p,
                    &mut pathcamera,
                    cameraInstance,
                    &mut samplerinstance,
                    maxdepth as usize,
                );
                println!("ncameravertives {:?}", ncma);
                PlotChains(&pathcamera);
            }

            if (false) {
                let nlights =
                    init_light_path(scene, &mut pathlight, &mut samplerlight, maxdepth as usize);
                println!("nvertives {:?}", nlights);
                PlotChains(&pathlight);
            }
        }
    }
}






































#[test]
pub fn testing_bppt_tracing_and_connection() {
   
   
   
    let maxdepth: i32 = 16;
    let filmres: (u32, u32) = (16, 16);



    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 2.000),
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );

    let sphere0smallptr =
        &(Box::new(sphere) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);



        let sphere1 = Sphere::new(
            Vector3::new(0.0, 0.0, -1.5000),
            1.0,
            MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
        );
        let sphere1smallptr =
        &(Box::new(sphere1) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);



    let diskescene = Disk::new(
        Vector3::new(0.0001, -1.0, 0.0001),
        Vector3::new(0.0, 1.0, 0.0),
        0.0,
        1000.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );
    let diskescene =
        &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);

    let primitivesIntersections: Vec<&Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> =
        vec![sphere0smallptr, diskescene, sphere1smallptr];

    let pointlight = Light::PointLight(PointLight {
        iu: 4.0,
        positionws: Point3::new(0.000, 1.0, 0.000),
        color: Srgb::new(
            std::f32::consts::PI,
            std::f32::consts::PI,
            std::f32::consts::PI,
        ),
    });
    let scene: &Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![pointlight],
        vec![],
        primitivesIntersections,
        1,
        1,
        None,
    );

    let mut samplerlight: Box<dyn Sampler> = Box::new(SamplerUniform::new(
        ((0, 0), (32 as u32, 32 as u32)),
        32,
        false,
        Some(0),
    ));
    let cameraInstance = PerspectiveCamera::new(1e-3, 1000.0, 45.0, filmres);

    // init_camera_path(scene,&(256.0, 256.0),&mut pathcamera, cameraInstance, &mut samplercam, maxdepth as usize);
    let mut samplerinstance: Box<dyn Sampler> = Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32,false,Some(0),));
    for yy in 0..filmres.1 {
        for xx in 0..filmres.0 {
            
            let p = &(xx as f64, yy as f64);
            //    let p = &(xx as f64, 32.000  as f64);
            let mut pathlight: Vec<PathTypes> = vec![];
            let mut pathcamera: Vec<PathTypes> = vec![];

             

            

            let ncamera = init_camera_path(
                scene,
                p,
                &mut pathcamera,
                cameraInstance,
                &mut samplerinstance,
                maxdepth as usize,
            );
            let nlights = init_light_path(scene, &mut pathlight, &mut samplerinstance, maxdepth as usize)-1;
            if (nlights>=3 && ncamera >=3){
                //println!("-px {:?}", p);
            }
            if   p.0 != 2.0 || p.1 != 0.0 {

                //  println!("--ncameravertives ");
              //  continue;
                  
              }
            
            //   println!("--ncameravertives {:?}", ncamera);
            //   println!("--nvertives {:?}", nlights);
              if   p.0 == 10.0 && p.1 == 0.0 {

                // PlotChains(&pathcamera);
                // PlotChains(&pathlight);
               //  continue;
                   
               }
         
            if (false) { 
                
              
            }
          
            if (false) {
               
               
             
            }

 
        

// continue;
            let mut Lsum = Srgb::new(0.0, 0.0, 0.0);

            for t in 1..=ncamera as i32 {
                for s in 0..=nlights  as i32 {
                    let depth: i32 = s + t - 2;
                
                    let pfilm = Point3f::new(p.0, p.1,0.0);
                    if t == 1 && s == 1 {
                        continue;
                    }
                    if depth > maxdepth {
                        continue;
                    }
                //    let cameracur =  pathcamera.get(t-1);
           
                    // cameraVertices[t - 1].type == VertexType::Light
                    let vtxcamera = &pathcamera[t as usize -1 ];
                   
                    let islightvtx =   match  vtxcamera {
                         PathTypes::PathLightVtxType(l)=>true,
                       _=>{false}
                   };
                    if t > 1 && s!= 0 && islightvtx{ 
                        continue;
                    }
                    
               
                    if s == 0 {
                        if false{
                            let L = emissionDirectStrategy(
                                t,
                                scene,
                                &mut samplerinstance,
                                &pathlight,
                                &pathcamera,
                            );
                            let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                            let weight =
                                compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                            Lsum = LigtImportanceSampling::addSrgb(
                                LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                                Lsum,
                            );
                        }
                    } else if t == 1 { 
                        if   true {
 
                            let uu = 10;
                            let gg  = uu;
                               
                          
                            if true {
                                let (L, sampledvtx) = cameraTracerStrategy(
                                    s,
                                    scene,
                                    &pfilm,
                                    &mut samplerinstance,
                                    cameraInstance,
                                    &pathlight,
                                    &pathcamera,
                                );
                                
                                if !isBlack(L) {
                                
                                    let weight =        compute_weights1(s, t, scene, sampledvtx.unwrap(), &mut pathlight, &mut pathcamera);
                                    println!("px : {:?}", p);
                                    println!("t=1  s= {} L={:?}, weight  {:?}  Lfinal {:?} ", s, L, weight ,LigtImportanceSampling::mulScalarSrgb(L, weight as f32) );
                                }
                                    
                                //     let weight =
                                //         compute_weights1(s, t, scene, sampledvtx.unwrap(), &mut pathlight, &mut pathcamera);
                                //     Lsum = LigtImportanceSampling::addSrgb(
                                //         LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                                //         Lsum,
                                //     );
                                // }
                          
                            }
                        }
                    } else if s == 1 {
                        if false {
                            let (L, vtx) =
                                lightTracerStrategy(t, scene, &mut samplerinstance, &pathlight, &pathcamera);
                            if !isBlack(L) {
                                // println!(" beta {:?}", vtx.as_ref().unwrap().transport());
                                // println!(" p {:?}", vtx.as_ref().unwrap().p());
                                // println!(" n {:?}", vtx.as_ref().unwrap().n());
                                // println!(" pdfnext {:?}", vtx.as_ref().unwrap().get_pdfnext());
                                // println!(" pdfprev {:?}", vtx.as_ref().unwrap().get_pdfrev());
                                let weight = compute_weights1(
                                    s,
                                    t,
                                    scene,
                                    vtx.unwrap(),
                                    pathlight.as_mut_slice(),
                                    pathcamera.as_mut_slice(),
                                );



                                println!(
                                    "s {} t {} L {:?} weight:{}",
                                    s,
                                    t,
                                    LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                                    weight
                                );
                                Lsum = LigtImportanceSampling::addSrgb(
                                    LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                                    Lsum,
                                );
                            }
                           
                           
                            // bidirectional(s,t, scene, &Point3f::new(256.0,256.0,0.0),   & pathlight,  & pathcamera );
                            //  let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                            //  compute_weights1(s,t,scene, &sampledvtx,  pathlight.as_mut_slice(),   pathcamera.as_mut_slice());
                        }
                    } else {
                         
                        if false{
                            let Lbidirectional = bidirectional(
                                s,
                                t,
                                scene,
                                &Point3f::new(256.0, 256.0, 0.0),
                                &pathlight,
                                &pathcamera,
                            );
                            if !isBlack(Lbidirectional) {
                                let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                                let weight =
                                  compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                               println!(
                                   "s {} t {} L {:?} weight:{}",
                                   s,
                                   t,
                                   LigtImportanceSampling::mulScalarSrgb(Lbidirectional, weight as f32),
                                   weight
                               );
                            }
                          
                            // Lsum = LigtImportanceSampling::addSrgb(
                            //     LigtImportanceSampling::mulScalarSrgb(Lbidirectional, weight as f32),
                            //     Lsum,
                            // );
                        }
                        //  println!("s {} t {} L {:?} weight:{}",s, t, LigtImportanceSampling::mulScalarSrgb( Lbidirectional, weight as f32), weight);
                    }
                }
            }
















           
        }
    }
}
