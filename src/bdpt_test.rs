use std::borrow::Cow;
use std::cell::RefCell;
use std::ops::Deref;
use std::rc::Rc;

use cgmath::{Vector3, Point3, Vector1};
use palette::Srgb;

use crate::Cameras::PerspectiveCamera;
use crate::Lights::{Light, PointLight, LigtImportanceSampling};
use crate::bdpt::{init_light_path, init_camera_path, PathTypes, BdptVtx, convert_static, lightTracerStrategy, bidirectional, cameraTracerStrategy,  PathVtx, compute_weights, compute_weights1, emissionDirectStrategy, };
use crate::materials::{RecordSampleIn, Pdf, Plastic, MaterialDescType};
use crate::primitives::{PrimitiveIntersection, Disk};
use crate::primitives::prims::Sphere;
use crate::{Point3f, Vector3f};
use crate::sampler::{Sampler, SamplerUniform};
use crate::primitives::prims::Ray;
use crate::primitives::prims::HitRecord;
use crate::raytracerv2::{Scene, interset_scene};
use crate::materials::SampleIllumination;
use crate::Lights::SampleEmission;
use crate::Lights::RecordSampleLightEmissionIn ;
use crate::Lights::RecordSampleLightEmissionOut ;

fn PlotChains(  vpath : &Vec<PathTypes>){
    
    for (id, p) in vpath.iter().enumerate(){
        println!("{}, {:?}", id,    p.transport());
        // println!("  v.p()       {:?}",     p.p());
        // println!("  v.n()       {:?}",     p.n());
          println!("  v.pdfFwd()  {:?}",     p.get_pdfnext());
          println!("  v.pdfRev()  {:?}",     p.get_pdfrev());
       
    }
}

#[test]
pub fn test_bdpt_1(){

    let filmres: (u32, u32) = (512, 512);
    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0))

    );
  
    let sphere2smallptr = &(
        Box::new(  sphere ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>> );
    let primitivesIntersections  :Vec<&Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>> =vec![ 
        sphere2smallptr,
       
       ];


       let pointlight = Light::PointLight(PointLight { iu: 4.0, positionws: Point3::new(0.000,0.0,0.000),color: Srgb::new(1.0,1.0,1.0 )});
    let scene :&Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![ pointlight],
        vec![],
        primitivesIntersections,
        1,
        1,None
    );
    // let mut sampler :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
    // init_light_path(scene, & mut sampler);

    let mut samplercam :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
    let cameraInstance = PerspectiveCamera::new(1e-3, 1000.0, 45.0,  filmres);
     
 //   let vpath = init_camera_path(scene,(256.0,256.0),& mut samplercam, cameraInstance ); 
   
 
//    for   mut vtx in & mut vpath.clone(). into_iter().peekable() {
  
//     // vtx.set_pdfrev(1111.0);
//     // println!("{:?}",vtx);
// }
// println!("{:?}",vpath);0.318309873
    //PlotChains(&vpath);
}




#[test]
pub fn test_bdpt_2(){




let maxdepth :i32 = 16;
    let filmres: (u32, u32) = (512, 512);
    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0))

    );
    
  
    let sphere2smallptr = &(
        Box::new(  sphere ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>> );
    let primitivesIntersections  :Vec<&Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>> =vec![ 
        sphere2smallptr,
       
       ];
     

       let pointlight = Light::PointLight(PointLight { iu: 4.0, positionws: Point3::new(0.000,0.0,0.000),color: Srgb::new(std::f32::consts::PI, std::f32::consts::PI , std::f32::consts::PI  )});
    let scene :&Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![ pointlight],
        vec![],
        primitivesIntersections,
        1,
        1,None
    );
    let mut pathlight:    Vec<PathTypes> =  vec![    ];
      let mut samplerlight :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
      init_light_path(scene, & mut pathlight, & mut samplerlight, maxdepth as usize);
    //   println!("\n\n\n pathlight ");
    //   PlotChains(&pathlight);
    let mut samplercam :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
      let cameraInstance = PerspectiveCamera::new(1e-3, 1000.0, 45.0,  filmres);
     let mut pathcamera:    Vec<PathTypes> =  vec![    ];
        init_camera_path(scene,&(256.0, 256.0),&mut pathcamera, cameraInstance, &mut samplercam, maxdepth as usize);
    //     println!("\n\n\n pathcamera");
    //  PlotChains(&pathcamera);
     let mut samplerconnect :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
    let mut Lsum = Srgb::new(0.0,0.0,0.0);
    for t   in 2..=pathcamera.len() as i32  {
        for s in 0..=pathlight.len() as i32  {
            let depth : i32 = s + t -2;
         //   println!("s={} t={}", s, t);
            let pfilm = Point3f::new(256.0, 256.0,0.0);
            if t == 1 && s == 1{
                continue;
            }
            if depth > maxdepth {
                continue;
            }
            if s == 0 {
                if false{ 
                    
                    let L =  emissionDirectStrategy(t,scene, & mut samplerconnect, &pathlight, &pathcamera);
                    let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                        let weight =   compute_weights1(s,t,scene, sampledvtx,&mut pathlight,  &mut pathcamera );
                    Lsum =  LigtImportanceSampling::addSrgb(LigtImportanceSampling::mulScalarSrgb( L, weight as f32), Lsum);
                }
            }else if t == 1 {
                if false{
                    let L = cameraTracerStrategy(s, scene, &pfilm,    & mut samplerconnect, cameraInstance, & pathlight,  & pathcamera);
                    let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                    let weight =   compute_weights1(s,t,scene, sampledvtx,&mut pathlight,  &mut pathcamera );
                    Lsum =  LigtImportanceSampling::addSrgb(LigtImportanceSampling::mulScalarSrgb( L, weight as f32), Lsum);
                }
            }else if s==1 {
                if true{
                    let (L, vtx) = lightTracerStrategy(t, scene,   & mut samplerconnect,  &   pathlight,&  pathcamera);
        
                    let weight = compute_weights1(s,t,scene, vtx.unwrap(),  pathlight.as_mut_slice(),   pathcamera.as_mut_slice());
                    println!("s {} t {} L {:?} weight:{}",s, t, LigtImportanceSampling::mulScalarSrgb(L, weight as f32), weight);
                    Lsum =  LigtImportanceSampling::addSrgb(LigtImportanceSampling::mulScalarSrgb( L, weight as f32), Lsum);
                // bidirectional(s,t, scene, &Point3f::new(256.0,256.0,0.0),   & pathlight,  & pathcamera );
                //  let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());  
                //  compute_weights1(s,t,scene, &sampledvtx,  pathlight.as_mut_slice(),   pathcamera.as_mut_slice());
                }
            }else{
                if false{
                    let Lbidirectional =  bidirectional(s,t, scene, &Point3f::new(256.0,256.0,0.0),   & pathlight,  & pathcamera );
                    let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                    let weight =   compute_weights1(s,t,scene, sampledvtx,&mut pathlight,  &mut pathcamera );
                    println!("s {} t {} L {:?} weight:{}",s, t, LigtImportanceSampling::mulScalarSrgb( Lbidirectional, weight as f32), weight);
                    Lsum =  LigtImportanceSampling::addSrgb(LigtImportanceSampling::mulScalarSrgb( Lbidirectional, weight as f32), Lsum);
                }
              //  println!("s {} t {} L {:?} weight:{}",s, t, LigtImportanceSampling::mulScalarSrgb( Lbidirectional, weight as f32), weight);
            } 
        }
    }
    println!(" L {:?}  " , Lsum);
    // bidirectional(2,2, scene, &Point3f::new(256.0,256.0,0.0),    & pathlight,  & pathcamera);

     
}



 

#[test]
pub fn testing_bdpt_tracing(){

    
    let maxdepth :i32 = 16;
    let filmres: (u32, u32) = (16, 16);
    let sphere = Sphere::new(
        Vector3::new(0.0, 0.0, 2.000),
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0))

    );

    let sphere2smallptr = &(
        Box::new(  sphere ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>> );
        
    let diskescene = Disk::new(Vector3::new(0.0001,-1.0001,0.0001),Vector3::new(0.0, 1.0, 0.0),0.0, 1000.001, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,1.0)));
    let diskescene = &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
    
    let primitivesIntersections  :Vec<&Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>> =vec![ 
        sphere2smallptr,
        diskescene
    
    ];


 


    let pointlight = Light::PointLight(PointLight { iu: 4.0, positionws: Point3::new(0.000,1.0,0.000),color: Srgb::new(std::f32::consts::PI, std::f32::consts::PI , std::f32::consts::PI  )});
    let scene :&Scene<f64> = &Scene::make_scene1(
        filmres.0 as usize,
        filmres.1 as usize,
        vec![ pointlight],
        vec![],
        primitivesIntersections,
        1,
        1,None
    );

    let mut samplerlight :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
      let cameraInstance = PerspectiveCamera::new(1e-3, 1000.0, 45.0,  filmres);
      
       // init_camera_path(scene,&(256.0, 256.0),&mut pathcamera, cameraInstance, &mut samplercam, maxdepth as usize);

        for yy in 0..filmres.1{
            for xx in 0..filmres.0{
                let mut samplerinstance :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
                let p = &(xx as f64, yy  as f64);
            //    let p = &(xx as f64, 32.000  as f64);
               let mut pathlight:    Vec<PathTypes> =  vec![    ];
             let mut pathcamera:    Vec<PathTypes> =  vec![    ];
               
               println!("\n px : {:?}", p);
                
          if(false){
            let ncma =  init_camera_path(scene,p ,&mut pathcamera, cameraInstance, &mut samplerinstance, maxdepth as usize);
            println!("ncameravertives {:?}", ncma);
            PlotChains(  &pathcamera );
          }
          
          if(true){
            let nlights =   init_light_path(scene, &mut pathlight, &mut samplerlight,maxdepth as usize);
            println!("nvertives {:?}", nlights);
            PlotChains(  &pathlight );
          }
             
                
            //   let nlights =   init_light_path(scene, &mut pathlight, &mut samplercam,maxdepth as usize);
            //   if nlights>2{
            //     println!("");
            //   }
            //     if true{
            //         PlotChains(  &pathlight);
            //     }
                 
            }
        }
}