use std::borrow::Cow;
use std::cell::RefCell;
use std::fs;
use std::ops::Deref;
use std::path::Path;
use std::rc::Rc;
 

use cgmath::{Point3, Vector1, Vector3, InnerSpace};
use hexf::hexf32;

use palette::Srgb;

use crate::bdpt::{
    bidirectional, cameraTracerStrategy,   compute_weights1, convert_static,
    emissionDirectStrategy, init_camera_path, init_light_path, lightTracerStrategy, BdptVtx,
    PathTypes, PathVtx, PathLightVtx, geometricterm, IsLightBdpt,
};
use crate::imagefilm::ImgFilm;
use crate::integrator::isBlack;
use crate::materials::SampleIllumination;
use crate::materials::{MaterialDescType, Pdf, Plastic, RecordSampleIn};
use crate::primitives::prims::{HitRecord, Intersection, IntersectionOclussion};
use crate::primitives::prims::Ray;
use crate::primitives::prims::Sphere;
use crate::primitives::{Disk, PrimitiveIntersection};
use crate::raytracerv2::{interset_scene, Scene};
use crate::sampler::{Sampler, SamplerUniform};
use crate::Cameras::PerspectiveCamera;
use crate::Lights::{RecordSampleLightEmissionIn, AreaLightDisk, PdfEmission, RecordPdfEmissionIn};
use crate::Lights::RecordSampleLightEmissionOut;
use crate::Lights::SampleEmission;
use crate::Lights::{Light, LigtImportanceSampling, PointLight};
use crate::samplerhalton::SamplerHalton;
use crate::{Point3f, Vector3f, Point2i, Point2f};

fn PlotChains(vpath: &Vec<PathTypes>) ->f64{
    let mut  ckcsumpdffwd = 0.0;
    let mut ckcsumpdfpdfRev = 0.0;
    let mut pat :PathTypes ;
    
    for (id, p) in vpath.iter().enumerate() {
        // println!(" vtx{} is light    {:?}",id,   p.is_light() );
      
        ;
        println!("{}, {:?}", id, p.transport());
        println!("  v.p()       {:?}", p.p());
        println!("  v.n()       {:?}", p.n());
        println!("  v.pdfFwd()  {:?}", p.get_pdfnext());
        println!("  v.pdfRev()  {:?}", p.get_pdfrev());
        ckcsumpdffwd+=p.get_pdfnext();
        ckcsumpdfpdfRev +=p.get_pdfrev(); 
    }
    // println!("                  ckcsumpdffwd {:?}", ckcsumpdffwd );
    // println!("                  ckcsumpdfpdfRev {:?}",ckcsumpdfpdfRev);
    ckcsumpdffwd + ckcsumpdfpdfRev
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
    // https://github.com/jan-van-bergen/GPU-Raytracer/tree/master/Src
   //  https://github.com/LuxCoreRender/LuxCore/blob/master/src/luxrays/devices/cudadevice.cpp 
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
                if true {
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
                if false {
                    let (L, vtx) =
                        lightTracerStrategy(t, scene, &mut samplerconnect, &pathlight, &pathcamera);

                    let weight = compute_weights1(s,t,scene,vtx.unwrap(),pathlight.as_mut_slice(),pathcamera.as_mut_slice(),);
                  //   println!("s {} t {} L {:?} weight:{}",s,t,LigtImportanceSampling::mulScalarSrgb(L, weight as f32),weight);
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

































// enum Stategy{
//     EmissionDirect = 0,
//     LightPath = 1,
//     CameraPath = 2,
//     Bidirectional = 3,
// }




#[test]
pub fn testing_bppt_tracing_and_connection() {
   
   
   
    let maxdepth: i32 = 4+2;
    let filmres: (u32, u32) = (32,32);

     
   
  //   let rayTest = Ray::new(Point3f::new(0.01,0.01,0.01), Vector3f::new(0.01,0.01,1.0));

     
    let sphere = Sphere::new(
        Vector3::new(0.0,0.0, 2.000),
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );

    let sphere0smallptr = &(Box::new(sphere) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);
    

     
        let sphereminusx = Sphere::new(
            Vector3::new(-1.5, 0.0, 2.000),
            1.0,
            MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
        );
    
         let  sphereminusx = &(Box::new(sphereminusx) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);
        







        let sphere1 = Sphere::new(
            Vector3::new(0.0, 0.0, -2.0000),
            1.0,
            MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
        );
        let sphere1smallptr =&(Box::new(sphere1) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);



    let diskescene = Disk::new(
        Vector3::new(0.0001, -1.50, 0.0001),
        Vector3::new(0.0, 1.0, 0.0),
        0.0,
        1000.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );
    let diskescene =
        &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);

    let primitivesIntersections: Vec<&Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> =
        vec![
        // sphere0smallptr,
        diskescene,
    //  sphere1smallptr,
    // sphereminusx
          ];

    let pointlight = Light::PointLight(PointLight {
        iu: 4.0,
        positionws: Point3::new(0.000, 1.01, 0.000),
        color: Srgb::new(
            std::f32::consts::PI,
            std::f32::consts::PI,
            std::f32::consts::PI,
        ),
    });



    let arealightdisk = Light::AreaLightDiskType(
        AreaLightDisk::new( 
            Vector3::new(0.000,3.5,2.000 ), 
            Vector3::new(0.0, -1.0, 0.0).normalize(),2.000,
            Srgb::new(  std::f32::consts::PI,
                std::f32::consts::PI,
                std::f32::consts::PI,)));
    
    // for  ( mut a, mut b) in std::iter::zip(0..10, 0..10){
    //      let p = ( a as f64 / 10.0 , b as f64 / 10.0);
    //      let recout = arealightdisk.sample_emission(RecordSampleLightEmissionIn{
    //         psample0: p,
    //         psample1: p.clone()
    //     });
        
    //    let pdfrec =  arealightdisk.pdf_emission( RecordPdfEmissionIn { n:recout.n, ray:recout.ray });
         
    // }
    


    
    


    let scene: &Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![arealightdisk],
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
 
     let cameraInstance = PerspectiveCamera::from_lookat(Point3f::new(0.0,1.00,0.0), Point3f::new(0.0,0.0,100.0),1e-2, 1000.0, 90.0, filmres);
    // let cameraInstance = PerspectiveCamera::new(1e-2, 1000.0, 45.0, filmres);
    // init_camera_path(scene,&(256.0, 256.0),&mut pathcamera, cameraInstance, &mut samplercam, maxdepth as usize);
    let mut samplerinstance: Box<dyn Sampler> = Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32,false,Some(0),));

    let mut L_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_bdir_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_cam_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_lightStra_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut ckcsumpdTOTAL = 0.0;
      let  mut ckcsump = 0.0;
    for yy in 1..filmres.1-1 {
        for xx in 1..filmres.0-1 {
            
            let p = &(xx as f64, yy as f64);
            //    let p = &(xx as f64, 32.000  as f64);
            let mut pathlight: Vec<PathTypes> = vec![];
            let mut pathcamera: Vec<PathTypes> = vec![];

             

            if p.0 == 1.0 &&p.1 ==2.0   {

   //             println!("pixel {:?}", p);
            }
           
             let ncamera = init_camera_path(scene,p,&mut pathcamera,cameraInstance,&mut samplerinstance,maxdepth as usize,);
           let nlights = init_light_path(scene, &mut pathlight, &mut samplerinstance, maxdepth as usize)-1;
           let ncamera = pathcamera.len();
           let nlights  = pathlight.len();
           if p.0 == 1.0 &&p.1 ==2.0   {
                        
          
                //    PlotChains(&pathcamera);
                //    PlotChains(&pathlight);
         
            }
       //     println!("pixel {:?}", p);
        //    PlotChains(&pathcamera);
        //       PlotChains(&pathlight);
        // primero tengo que seguir comparando esta configuracion. No he termnado!
        //     tengo que probar mas cosas ! creo que ya funciona la camara. 
        println!(" {:?}",p);
        // ckcsump = 0.0;
        //    ckcsump +=      PlotChains(&pathcamera);
        //    ckcsump +=      PlotChains(&pathlight);

   println!(" {:?}, {} {}  ",ncamera, nlights, ckcsump);
            
              if  false {
                
                // let  mut ckcsump = 0.0;
                // ckcsump +=      PlotChains(&pathcamera);
                // ckcsump +=      PlotChains(&pathlight);
                // println!(" {:?}, {} ,{}",ncamera, nlights, ckcsump);
                // println!(" {:?}",ckcsumpdTOTAL);
            
                   
               }
  
  
            let mut Lsum = Srgb::new(0.0, 0.0, 0.0);
            let mut  Llightstrategy= Srgb::new(0.0, 0.0, 0.0);
            let mut  Lcamerastrategy= Srgb::new(0.0, 0.0, 0.0);
            let mut  LbirectionalStrategy= Srgb::new(0.0, 0.0, 0.0);
            for t in 1..=ncamera as i32 {
                for s in 0..=nlights as i32 {
                    let depth: i32 = s + t - 2;
                 
                    let pfilm = Point3f::new(p.0, p.1,0.0);
                    if t == 1 && s == 1 {
                        continue;
                    }
                    if depth > maxdepth {
                        continue;
                    }
                    if depth < 0{
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
                    if p.0 == 18.0 &&p.1 ==30.0 && s == 0 && t==3{
                      
                        // let depth2: i32 = s + t - 2;
                        // let  mut ckcsump = 0.0;
                        // ckcsump +=      PlotChains(&pathcamera);
                        // ckcsump +=      PlotChains(&pathlight);
                        println!("");
                    }
                    
                    
                    if s == 0 {
                        if true{
                            let L = emissionDirectStrategy(
                                t,
                                scene,
                                &mut samplerinstance,
                                &pathlight,                       
                                &pathcamera,
                            );
                            
                            if !isBlack(L) {
                                let sampledvtx = PathTypes:: None;
                                let weight =
                                    compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                                Lsum = LigtImportanceSampling::addSrgb(
                                    LigtImportanceSampling::mulScalarSrgb(L, weight as f32),
                                    Lsum,
                                );
                                 //  println!("emission s {}, t {}  final L {:?} ",  s, t,  LigtImportanceSampling::mulScalarSrgb(L, weight as f32).red);
                             }
                             
                            
                        }
                    } else if t == 1 { 
                        if   false {
 
                        
                               
                          
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
                                    let Lcam = LigtImportanceSampling::mulScalarSrgb(L, weight as f32);
                                    L_cam_TOTAL = LigtImportanceSampling::addSrgb(L_cam_TOTAL,Lcam );
                                  //  println!("     Lcam  {:?}", Lcam);
                                    Lcamerastrategy = LigtImportanceSampling::addSrgb(Lcamerastrategy,Lcam);
                                    Lsum = LigtImportanceSampling::addSrgb(Lsum,Lcam);
                                }
                                    
                            
                            }
                        }
                    } else if s == 1 {
                       

                        if false{
                             
                            let (L, vtx) =
                                lightTracerStrategy(t, scene, &mut samplerinstance, &pathlight, &pathcamera);
                            if !isBlack(L) {
                            
                                let weight = compute_weights1(s,t,scene,vtx.unwrap(),pathlight.as_mut_slice(),pathcamera.as_mut_slice(),); 
                                let LPath = LigtImportanceSampling::mulScalarSrgb(L, weight as f32);
                                Llightstrategy = LigtImportanceSampling::addSrgb(Llightstrategy,LPath); 
                                L_lightStra_TOTAL = LigtImportanceSampling::addSrgb(L_lightStra_TOTAL,LPath );
                                Lsum = LigtImportanceSampling::addSrgb(LigtImportanceSampling::mulScalarSrgb(L, weight as f32),Lsum,);
                                println!("       lightTracerStrategy  s {} t {}  Lsum {}", s, t ,LPath.red );
                            }else{
                                 println!("       lightTracerStrategy  s {} t {}  Lsum {}", s, t , 0.0 );
                            }
                        
                            
                        }
                    } else {
                         
                        if false{
                        //     println!("  bidirectional s {} t {}  ", s, t   );
                           if p.0 == 3.0 && p.1 == 1.0 && s == 3 && t==4{
                        //   println!("Deb");
                        //        PlotChains(&pathcamera);
                        //      PlotChains(&pathlight);
                           }
                            let Lbidirectional = bidirectional(s,t,scene,&Point3f::new(p.0, p.1, 0.0),&pathlight,&pathcamera,);
                            if !isBlack(Lbidirectional) {
                              
                                let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                                let weight = compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                                let LPath = LigtImportanceSampling::mulScalarSrgb(Lbidirectional, weight as f32);
                              //   println!(" bidirectional : {:?}", LPath);
                                L_bdir_TOTAL = LigtImportanceSampling::addSrgb(L_bdir_TOTAL,LPath );
                                 
                                LbirectionalStrategy = LigtImportanceSampling::addSrgb(LbirectionalStrategy,LPath); 
                                
                                Lsum = LigtImportanceSampling::addSrgb( LPath,Lsum,);
                              //  println!("       bidirectional   s {} t {} L {}", s, t , LPath.red);
                            }else {
                               // println!("       bidirectional   s {} t {}  Lsum {}", s, t , 0.0 );
                            }
                          
                             
                        }
                        
                    }
                }
            }
         
           
              L_TOTAL =  LigtImportanceSampling::addSrgb(L_TOTAL, Lsum);
              println!("                Px L {:?}", Lsum.red);
           











           
        }
    }
    println!(" {:?} {:?}",ckcsump,L_TOTAL);
    //  println!(" {:?}",L_lightStra_TOTAL);
    //  println!(" {:?}",L_bdir_TOTAL);
    //  println!(" {:?}",L_cam_TOTAL);
    // println!(" {:?}",ckcsumpdTOTAL);
    println!("  ");
    
}


 


















 




































 






pub fn testing_bppt_tracing_and_connection_sampler_with_film(spp:i64, dims: (u32, u32)) {
   
 
 println!("testing_bppt_tracing_and_connection_sampler_with_film!");
   
    let maxdepth: i32 = 4+2;
    let filmres: (u32, u32) = dims;
    let spp = spp;
    let mut  film = RefCell::new( ImgFilm::from_filter_gauss(filmres.0 as usize,filmres.1 as usize,(2.0,2.0),1.0));

    let sphere0 = Sphere::new(
        Vector3::new(0.30, 0.25, 1.300000),
        0.25,
        MaterialDescType::PlasticType(Plastic::from_albedo(0.10, 0.80, 0.10)),
    );

    let sphere0  =&(Box::new(sphere0) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);


    let sphere1 = Sphere::new(
        Vector3::new(-0.30, 0.25, 1.300000),
        0.25,
        MaterialDescType::PlasticType(Plastic::from_albedo(0.1, 0.10, 1.0)),
    );

    let sphere1  =&(Box::new(sphere1) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);


    



        let diskescene = Disk::new(
            Vector3::new(0.0001, -0.0001, 0.0001 ),
            Vector3::new(0.0, 1.0, 0.0),
            0.0,
            1000.0,
            MaterialDescType::PlasticType(Plastic::from_albedo(0.70, 0.10, 0.01)),
        );
    let diskescene = &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);

    let primitivesIntersections: Vec<&Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> =
        vec![
            // sphere0, sphere1, 
         diskescene,
    //    sphere1smallptr
         ];






// // no se que pasa...si funciona o que...
// // pongo point light bien. cambio a area light y tarda mucho, ademas de que no se si produce los paths.
// // reduce el raster screen hasta lo misimo posible



let pw = 4.50;
         let arealightdisk = Light::AreaLightDiskType(
            AreaLightDisk::new( 
                Vector3::new(0.000,1.5,3.000 ), 
                Vector3::new(0.0, -1.0, 0.0).normalize(),
                2.000,
                Srgb::new( pw, pw,pw),));









       
let pw = 15.50;
         let pointlight = Light::PointLight(PointLight {
            iu: 4.0,
            positionws: Point3::new(0.0, 1.200010, 1.20850),
            color: Srgb::new( pw, pw,pw),
        });
    let scene: &Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![arealightdisk],
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
     
   
    let cameraInstance =  PerspectiveCamera::from_lookat(Point3f::new(0.0,-0.500, 0.50), Point3f::new(0.0,0.0,1.00), 1e-2, 100.0, 90.0, filmres);
 
     
    let mut samplerhalton  :  Box<dyn Sampler> = Box::new(SamplerHalton::new(&Point2i::new(0, 0),&Point2i::new(filmres.0 as i64, filmres.1 as i64),spp,false,));
    
 
    let mut L_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_bdir_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_cam_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_lightStra_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut ckcsumpdTOTAL = 0.0;
    for yy in 0..filmres.1 {
        println!("pix {}% ",   (yy as f32  ) / filmres.1  as f32);
      
        for xx in 0..filmres.0  {
            
            let  mut pxsample  =0 ;
            // let p = &(xx as f64, yy as f64);
            samplerhalton.start_pixel(Point2i::new(xx as i64, yy as i64));
           
            let psampleoffset = samplerhalton.get2d();
       
        
               let p = &(  psampleoffset.0,  psampleoffset.1);

 




            let mut pathlight: Vec<PathTypes> = vec![];
            let mut pathcamera: Vec<PathTypes> = vec![];
        let ncamera = init_camera_path(scene,p,&mut pathcamera,cameraInstance,&mut  samplerhalton,maxdepth as usize,);
            let nlights = init_light_path(scene, &mut pathlight, &mut  samplerhalton, maxdepth as usize)-1;
            let ncamera = pathcamera.len();
            let nlights  = pathlight.len();
            
            let mut L_PX_TOTAL = Srgb::new(0.0, 0.0, 0.0);
            while samplerhalton.start_next_sample() {
                
                let psampleoffset = samplerhalton.get2d();
                 
              
                let pnewpointsample = &psampleoffset;
                pxsample+=1;
                let mut pathlight: Vec<PathTypes> = vec![];
                let mut pathcamera: Vec<PathTypes> = vec![];
              let ncamera = init_camera_path(scene,pnewpointsample,&mut pathcamera,cameraInstance,&mut  samplerhalton,maxdepth as usize,);
               let nlights = init_light_path(scene, &mut pathlight, &mut  samplerhalton, maxdepth as usize)-1;
                  let ncamera = pathcamera.len();
                let nlights  = pathlight.len();
             
               


             let mut Lsum = Srgb::new(0.0, 0.0, 0.0);
            let mut  Llightstrategy= Srgb::new(0.0, 0.0, 0.0);
            let mut  Lcamerastrategy= Srgb::new(0.0, 0.0, 0.0);
            let mut  LbirectionalStrategy= Srgb::new(0.0, 0.0, 0.0);
            for t in 1..=ncamera as i32 {
                for s in 0..=nlights as i32 {
                    let depth: i32 = s + t - 2;
                
                    let pfilm = Point3f::new(p.0, p.1,0.0);
                    if t == 1 && s == 1 {
                        continue;
                    }
                    if depth > maxdepth {
                        continue;
                    }
                
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
                                &mut  samplerhalton,
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
                        if   false { 
                                let (L, sampledvtx) = cameraTracerStrategy(
                                    s,
                                    scene,
                                    &Point3f::new(pnewpointsample.0,pnewpointsample.1, 0.0),
                                    &mut  samplerhalton,
                                    cameraInstance,
                                    &pathlight,
                                    &pathcamera,
                                );
                                
                                if !isBlack(L) {
                                
                                    let weight =        compute_weights1(s, t, scene, sampledvtx.unwrap(), &mut pathlight, &mut pathcamera);
                                    let Lcam = LigtImportanceSampling::mulScalarSrgb(L, weight as f32);
                                    L_cam_TOTAL = LigtImportanceSampling::addSrgb(L_cam_TOTAL,Lcam );
                                  //  println!("     Lcam  {:?}", Lcam);
                                    Lcamerastrategy = LigtImportanceSampling::addSrgb(Lcamerastrategy,Lcam);
                                    Lsum = LigtImportanceSampling::addSrgb(Lsum,Lcam);
                                }
                                    
                        }
                    } else if s == 1 {
                       
                    
                        if true{
                            let (L, vtx) =
                                lightTracerStrategy(t, scene, &mut  samplerhalton, &pathlight, &pathcamera);
                            if !isBlack(L) {
                            
                                let weight = compute_weights1(s,t,scene,vtx.unwrap(),pathlight.as_mut_slice(),pathcamera.as_mut_slice(),); 
                                let LPath = LigtImportanceSampling::mulScalarSrgb(L, weight as f32);
                                Llightstrategy = LigtImportanceSampling::addSrgb(Llightstrategy,LPath); 
                                L_lightStra_TOTAL = LigtImportanceSampling::addSrgb(L_lightStra_TOTAL,LPath );
                                Lsum = LigtImportanceSampling::addSrgb(LigtImportanceSampling::mulScalarSrgb(L, weight as f32),Lsum,);
                                //   println!("{:?}", Lsum)
                            }
                            
                        }
                    } else {
                         
                        if false{
                            
                            let Lbidirectional = bidirectional(s,t,scene,&Point3f::new(pnewpointsample.0,pnewpointsample.1, 0.0),&pathlight,&pathcamera,);
                            if !isBlack(Lbidirectional) {
                              
                                let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                                let weight = compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                                let LPath = LigtImportanceSampling::mulScalarSrgb(Lbidirectional, weight as f32);
                              //   println!(" bidirectional : {:?}", LPath);
                                // L_bdir_TOTAL = LigtImportanceSampling::addSrgb(L_bdir_TOTAL,LPath );
                                 
                                // LbirectionalStrategy = LigtImportanceSampling::addSrgb(LbirectionalStrategy,LPath); 
                                
                                Lsum = LigtImportanceSampling::addSrgb( LPath,Lsum,);
                               
                            }
                          
                             
                        }
                        
                    }
                }
            }
 
                
                L_PX_TOTAL =     LigtImportanceSampling::addSrgb(L_PX_TOTAL, Lsum);
                let spp = samplerhalton.get_spp();
                 
                if !isBlack(Lsum){
                    let mut  bind = film.borrow_mut();
                    bind.add_sample(&Point2f::new(pnewpointsample.0,pnewpointsample.1), Lsum );
                 
                }
             
            }
            let spp = samplerhalton.get_spp();
            L_PX_TOTAL =  LigtImportanceSampling::mulScalarSrgb(L_PX_TOTAL, 1.0/(spp as f32));
           // println!(" {:?}", [L_PX_TOTAL.red,   L_PX_TOTAL.green,  L_PX_TOTAL.blue] );
            // if !isBlack(L_PX_TOTAL){
            //     let mut  bind = film.borrow_mut();
            //     bind.add_sample(&Point2f::new(pnewpointsample.0,pnewpointsample.1), L_PX_TOTAL );
             
            // }
        
             continue;  
        }
    }
    
    let dirpath : &Path = Path::new("renderbdpttest");
  if  fs::create_dir_all(dirpath).is_ok() {
    let bind = film.borrow_mut();
    // raytracer_sampler1
    let spath = & format!("bdpt_test_2_{}.png", 1);
    bind.commit_and_write(dirpath .join(spath).as_os_str().to_str().unwrap());
  }
 
  println!("testing_bppt_tracing_and_connection_sampler_with_film! end");
    
} // 



use  std::time::Instant;



#[test]
pub fn bdpt_tracing_DEBUG(){
     testing_bppt_tracing_and_connection_sampler_with_film(2,( 256, 256));
}
pub fn bdpt_tracing_MAIN<T : std::fmt::Display +   std::cmp::PartialEq<str>>(args: Vec<T>){


    if args.len() > 1 {
        
        if &args[1]=="low" {
            
            let start = Instant::now();
            println!("low conf  spp=8 dims = (339, 256)" );
            testing_bppt_tracing_and_connection_sampler_with_film(64,(256/2/2, 256/2/2));
            println!("Frame time: {} ms", start.elapsed().as_millis());
            println!("low conf  spp=8 dims = (339, 256)" );
        }else if &args[1]=="med"  {
            let start = Instant::now();
            println!("med conf  spp=256 dims = (339, 256)" );
            testing_bppt_tracing_and_connection_sampler_with_film(256,(339, 256));
            println!("Frame time: {} ms", start.elapsed().as_millis());
            println!("med conf  spp=256 dims = (339, 256)" );
        }else if &args[1]=="hight"  {
            let start = Instant::now();
            println!("low conf  spp=512 dims = (512, 512)" );
            testing_bppt_tracing_and_connection_sampler_with_film(512,(339, 512));
            println!("Frame time: {} ms", start.elapsed().as_millis());
            println!("low conf  spp=512 dims = (512, 512)" );
        }

      
    }
   
}











