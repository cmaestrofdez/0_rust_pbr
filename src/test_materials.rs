
use crate::Lights::{PointLight, BackgroundAreaLight, RecordSampleLightIlumnation, LigtImportanceSampling, generateSamplesWithSamplerGather, Light, AreaLightDisk};
use crate::integrator::{get_ray_direct_light, updateL, estimatelights};
use crate::primitives::{PrimitiveIntersection, Disk};
use crate::sampler::custom_rng::CustomRng;
use crate::sampler::{Sampler, SamplerUniform};
use crate::primitives::prims::HitRecord;
use crate::{primitives, assert_delta, Lights };
use crate::primitives::prims::{self, Intersection};
use crate::primitives::prims::{IntersectionOclussion, Ray};
use crate::raytracerv2::{clamp, lerp, Camera, interset_scene};
use crate::raytracerv2::Scene;
use cgmath::*;
use cgmath::{Deg, Euler, Quaternion};
use num_traits::cast::cast;
use num_traits::Float;
use num_traits::Pow;
use palette::rgb::Srgb;
use rand::Rng;
use std::borrow::Cow;
use std::f32;
use std::f64;
use std::iter::zip;
use crate::materials::{
    Plastic,
    MaterialDescType,
    UniformSampleSemisphere, transform_eta_to_air_to_r0, frSclick, fresnel_diffuse_disney,
     BsdfType, LambertianBsdf, RecordSampleIn, SampleIllumination, Fresnel, beckman_G, microfacet_beckmann, beckman_alpha,
     Eval, FresnelSchlickApproximation, Sin2Phi, SinPhi, CosPhi, FrameShadingUtilites, Cos2Phi, TanTheta, Tan2Theta, 
     MicroFacetReflection, MicroFaceDistribution, BeckmannDistribution, 
     FresnelNop, Mirror, TrowbridgeReitzDistribution,
     Metal, Glass, IsSpecular, Glass2
    };
use crate::materials::UniformSampleCone;
use crate::materials::FrameShading;

use crate::materials;
use crate::texture::{*};






#[test]
fn test_UniformSampleSemisphere() {
    //    let bsdf = Bsdf::new(Vector3::new(0.0, 1.0, 0.0), Vector3::new(0.0, 0.0, 1.0),Vector3::new(1.0, 0.0, 0.0));

    let mut rng = rand::thread_rng();

    for fakerng in zip(1..10, 1..10) {
        let qrotation = Quaternion::from(Euler {
            x: Deg(rng.gen_range(0..90) as f64),
            y: Deg(rng.gen_range(0..90) as f64),
            z: Deg(rng.gen_range(0..90) as f64),
        });
        let rotMat = Matrix4::from(qrotation);
        let n = rotMat.transform_vector(Vector3::new(0.0, 1.0, 0.0));
        let tn = rotMat.transform_vector(Vector3::new(0.0, 0.0, 1.0));
        let bn = rotMat.transform_vector(Vector3::new(1.0, 0.0, 0.0));
        let bsdf = FrameShading::new(n, tn, bn);

        let u = (fakerng.0 as f64 / 10.0, fakerng.1 as f64 / 10.0);
        let (wnextL, pdf) = UniformSampleSemisphere(&u);
        let nextW = bsdf.to_world(&wnextL);
        println!(
            "{:?}, {:?}",
            wnextL.angle(Vector3::new(0.0, 0.0, 1.0)),
            nextW.angle(bsdf.n)
        );
        assert_delta!(
            wnextL.angle(Vector3::new(0.0, 0.0, 1.0)),
            nextW.angle(bsdf.n),
            Rad(0.00001)
        );
    }

    // println!("nextW, {:?}", wnextL);;

    // ;
    //  println!("nextW, {:?}", nextW);;
}









#[test]
fn test_UniformSampleCone() {
    println!("{:?}", UniformSampleCone(&(0.21, 1.0), 0.990).0.z);
    println!("");
    let eps = 0.0001;
    assert_ne!(UniformSampleCone(&(0.1, 1.0), 0.990).0.z, 0.990);
    assert_delta!(UniformSampleCone(&(0.221, 1.0), 0.990).0.z, 0.990, eps);
    assert_delta!(UniformSampleCone(&(0.321, 1.0), 0.990).0.z, 0.990, eps);
    assert_delta!(UniformSampleCone(&(0.421, 1.0), 0.990).0.z, 0.990, eps);
    assert_delta!(UniformSampleCone(&(0.521, 1.0), 0.990).0.z, 0.990, eps);
    assert_delta!(UniformSampleCone(&(0.621, 1.0), 0.990).0.z, 0.990, eps);
    assert_delta!(UniformSampleCone(&(0.721, 1.0), 0.990).0.z, 0.990, eps);
    assert_delta!(UniformSampleCone(&(0.821, 1.0), 0.990).0.z, 0.990, eps);
    assert_delta!(UniformSampleCone(&(1.0, 1.0), 0.990).0.z, 0.990, eps);
}















// algunos valores
#[test]
fn borr(){
 let r0 =  transform_eta_to_air_to_r0(1.00915);
  println!("{}", frSclick(transform_eta_to_air_to_r0(1.00915),0.0));
  println!("{}", frSclick(transform_eta_to_air_to_r0(1.00915),0.1));
  println!("{}", frSclick(transform_eta_to_air_to_r0(1.00915),0.8));
  println!("{}", frSclick(transform_eta_to_air_to_r0(1.00915),1.0));
  println!("{}", frSclick(transform_eta_to_air_to_r0(1.00915),1.1));
  
  println!("{}",fresnel_diffuse_disney(&Vector3::new(0.0,0.0,1.0),&Vector3::new(0.0,0.0,1.0)));
  println!("{}",fresnel_diffuse_disney(&Vector3::new(1.0,0.0,0.0),&Vector3::new(-1.0,0.0,0.0)));
}















#[test]
pub fn TEST_BeckmannDistribution() {






    let vvis =vec![ Vector3::new(0.0, 0.0, 1.0), Vector3::new(1.0, 0.0, 1.0), Vector3::new(1.0, 1.0, 1.0),Vector3::new(1.7, 1.0, 1.0),Vector3::new(2.7, 1.0, 1.0),Vector3::new(3.7, 1.0, 1.0),
    Vector3::new(3.7, 2.0, 1.0),
    Vector3::new(3.7, 3.0, 1.0),
    Vector3::new(3.7, 4.0, 1.0),
    Vector3::new(3.7, 4.0, 2.0),
    Vector3::new(3.7, 4.0, 3.0),
    Vector3::new(3.7, 4.0, 4.0),];
    let vv =vec![
        Vector3::new(0.0, 0.0, 1.0), 
        Vector3::new(1.0, 0.0, 1.0), 
        Vector3::new(1.0, 1.0, 1.0),
    Vector3::new(1.7, 1.0, 1.0),
    Vector3::new(2.7, 1.0, 1.0),
    Vector3::new(3.7, 1.0, 1.0),
    Vector3::new(3.7, 2.0, 1.0),
    Vector3::new(3.7, 3.0, 1.0),
    Vector3::new(3.7, 4.0, 1.0),
    Vector3::new(3.7, 4.0, 2.0),
    Vector3::new(3.7, 4.0, 3.0),
    Vector3::new(3.7, 4.0, 4.0),
    ] ;
    let mut a = 0.4;
    a = beckman_alpha(a);
     
    for mut wi in vvis{
        println!("{:?}",wi);
        for v in &vv{
            let v= v.normalize();
            let R = microfacet_beckmann(v, wi.normalize(), a, a);
             println!("--->{:?} ,{}",v,R);
             // println!("{}",G(&v.normalize(),&Vector3::new(0.0, 0.0, 1.0).normalize(), 0.461760044, 0.461760044));
     
         }
    }
   
    
}











#[test]
pub fn TEST_termgeometry() {
//     std::vector<Vector3f> vs{Vector3f(0, 0, 1.f), Vector3f(1.0, 0, 1.f),
//         Vector3f(1.0, 1.0, 1.f)};
// for(auto i : vs){
// Float g =distrib->G(i, Vector3f(0, 0, 1.f));
// cout << g << endl;
// }
    let vv =vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(1.0, 0.0, 1.0), Vector3::new(1.0, 1.0, 1.0),
    Vector3::new(1.7, 1.0, 1.0),
    Vector3::new(2.7, 1.0, 1.0),
    Vector3::new(3.7, 1.0, 1.0),
    Vector3::new(3.7, 2.0, 1.0),
    Vector3::new(3.7, 3.0, 1.0),
    Vector3::new(3.7, 4.0, 1.0),
    Vector3::new(3.7, 4.0, 2.0),
    Vector3::new(3.7, 4.0, 3.0),
    Vector3::new(3.7, 4.0, 4.0),
    ] ;
    for v in vv{
        println!("{}",beckman_G(&v.normalize(),&Vector3::new(0.0, 0.0, 1.0).normalize(), 0.461760044, 0.461760044));
    }
   
}











#[test]
fn test_FresnelSchlickApproximation() {
    let fresnel = Fresnel::FresnelSchlickApproximation(FresnelSchlickApproximation::new(1.0, 1.5));
    for i in 0..32 {
        let sc = (i as f64).sin_cos();
        let dir = ((i as f64 / 32 as f64) * 2.0 * std::f64::consts::PI).sin_cos();
    }
    println!("{}", fresnel.eval(0.0));
    println!("{}", fresnel.eval(1.0));
    println!("{}", fresnel.eval(1.5));
}












#[test]
fn testing_pt_material_intersection() {
   
   
    let camera = Camera::new(
        Point3::new(0.0, 0.0, 1.5),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        1.0f64 / 1.0f64,
    );
    let sphere = prims::Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::MirrorType(Mirror::default())
    );
     let micro = BsdfType::MicroFacetReflection(MicroFacetReflection{
        frame:Some(FrameShading::default()),
        distri : MicroFaceDistribution::BeckmannDistribution(BeckmannDistribution::new(0.1, 0.1)),
        fresnel:Fresnel::FresnelNop(FresnelNop{})
    });
     
    let primitives = vec![&sphere];
    let scene = &Scene::make_scene(
        128,
        128,
        vec![
            // Light::PointLight(PointLight {
            // iu: 4.0,
            // positionws: Point3::new(1.15,1.15,1.15),
            // color: Srgb::new(2.20,2.20,23.9420),
            // }),
           Lights::Light::PointLight(PointLight {
            iu: 4.0,
            positionws: Point3::new(2.0,2.0,2.0),
            color: Srgb::new(0.20,0.20,0.59420),
            }),
            
          
        ],
        primitives,
        1,
        1,
    );
   
    for iy in 0..32{
    for ix in 0..32{
       
        let x = ix as f64 / 32.0;
        let y = iy as f64 / 32.0;
        
        let x=2.0*x-1.0;
        let y=2.0*y-1.0;
        let orig = Point3::new(0.0,0.0, 2.0);
        let newdir =  Point3::new(x,y, 0.0)-orig;
        let ray = prims::Ray::new(
            orig,
            newdir.normalize()
    
        );
       
        // get_ray_direct_light(&ray,   &scene.lights, scene,0);
        let mut hitop =  interset_scene(&ray, scene);
        if let Some(hit) = hitop {
            let hit = hitop.unwrap();
           
            let rec = hit.material.sample(RecordSampleIn {
                pointW: hit.point,
                prevW: -ray.direction,
                sample:Some((0.0,0.0))
            });
            println!("{},{}, {:?}, next: {:?} , fr :{:?}", ix, iy, hit.point, rec.next,rec.f);
        }
     }
    }   
}


fn estimate_bck_area_lights<'a>( r: &Ray<f64>,
    lights: &Vec<Light>,
    scene: &Scene<'a, f64>,
    depth: i32,
    hit : HitRecord<f64>,
    sampler: &mut Box<dyn Sampler> ){
        let mut rng = CustomRng::new();
        rng.set_sequence(0);
        let nsamples = 32;
                    
        lights.into_iter().for_each(|light| {
            let mut newLd: Vec<f32> = vec![0.0; 3]; 
            let vecsamples =  generateSamplesWithSamplerGather(sampler, nsamples);
            for isample in vecsamples.into_iter(){
                rng.uniform_float();
                 let   p = ( rng.uniform_float(), rng.uniform_float()); 
                 let   usamplebsdf = ( rng.uniform_float(),  rng.uniform_float());
                //  println!("samplelight {:?}  ", p );
                //  println!("samplebsdf  {:?}  ", usamplebsdf);
                //  let   p =  (0.7274880120530725, 0.4938478171825409);
                //  let   usamplebsdf = (0.104709394, 0.329331279);
                
                //  rng.uniform_float();
                //   rng.uniform_float() ;
               
                 let resnewe =  LigtImportanceSampling::samplelight(&r, &scene, &hit, light.clone(), &p );
                 let bsdfsample = LigtImportanceSampling::samplebsdf(r, scene, &hit, light.clone(), &usamplebsdf );
               
                println!("          {:?}", LigtImportanceSampling::addSrgb(resnewe , bsdfsample));
                 updateL(&mut newLd,LigtImportanceSampling::addSrgb(resnewe , bsdfsample));
               
            }        
            newLd[0]=  newLd[0]/(nsamples  as f32);
            newLd[1]=  newLd[1]/(nsamples as f32);
            newLd[2]=  newLd[2]/(nsamples as f32);
            println!("Ldfinal {:?}", newLd )
        });     
        
          
        
}







#[test]
fn testing_path_tracer() {
    use crate::primitives::prims::HitRecord;
    use crate::primitives::prims::Sphere;
    use crate::Lights::Light;
    use crate::Lights::BackgroundAreaLight;
    use crate::Lights::SampleLightIllumination;
    let micro = BsdfType::MicroFacetReflection(MicroFacetReflection{
        frame:Some(FrameShading::default()),
        distri : MicroFaceDistribution::BeckmannDistribution(BeckmannDistribution::new(0.1, 0.1)),
        fresnel:Fresnel::FresnelNop(FresnelNop{})
    });
    let microtrosky = BsdfType::MicroFacetReflection(MicroFacetReflection{
        frame:Some(FrameShading::default()),
        distri : MicroFaceDistribution::TrowbridgeReitzDistribution(TrowbridgeReitzDistribution::new(0.1, 0.1)),
        fresnel:Fresnel::FresnelNop(FresnelNop{})
    });
   
    let camera = Camera::new(
        Point3::new(0.0, 0.0, 1.5),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        1.0f64 / 1.0f64,
    );
    
    let sphere = prims::Sphere::new(
        Vector3::new(0.0, 0.30, 0.0),
        0.3,
        // MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0))
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0))
    );
    
    let sphere2smallptr = &(
        Box::new(  sphere ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );
    let sphere1 = prims::Sphere::new(
        Vector3::new(0.0, -1000.0, 0.0),
        1000.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0))
    );
    let sphereplanesmallptr = &(
        Box::new(  sphere1 ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );









    
    let diskescene = Disk::new(Vector3::new(0.0001,0.0001,0.0001),Vector3::new(0.0, 1.0, 0.0),0.0, 1000.001, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,1.0)));
    let diskescene = &(Box::new(diskescene) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>> >);
    
    let primitivesIntersections  :Vec<&Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>> =vec![ 
        sphere2smallptr,
        diskescene
       //  sphere2smallptr  //, sphere2smallptr,spherebluesmallptr
       ];


       let pointlight = Light::PointLight(PointLight { iu: 4.0, positionws: Point3::new(0.0010,1.0,0.0010),color: Srgb::new(1.0,1.0,1.0 )});
    let infarealight =  Light::BackgroundAreaLightType (BackgroundAreaLight::from_file("arco.png", Srgb::new(1.0,1.0,1.0)));
    let arealightdisk = Light::AreaLightDiskType(AreaLightDisk::new( Vector3::new(0.0010,1.0,0.0010 ), Vector3::new(0.0,-1.0, 0.0).normalize(),10.00,Srgb::new(1.0, 1.0, 1.0)));
    let scene = &Scene::make_scene1(
        128,
        128,
        vec![
            // Light::PointLight(PointLight {
            // iu: 4.0,
            // positionws: Point3::new(1.15,1.15,1.15),
            // color: Srgb::new(2.20,2.20,23.9420),
            // }),
        //    Lights::Light::PointLight(PointLight {
        //     iu: 4.0,
        //     positionws: Point3::new(2.0,2.0,2.0),
        //     color: Srgb::new(0.20,0.20,0.59420),
        //     }),
          arealightdisk
      //  pointlight
        ],
        vec![],
        primitivesIntersections,
        1,
        1,None
    );

 
    
    let w = 32;
    let h = 32;
    let mut sampleruniform :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (w as u32, h as u32)),1, false, Some(0)));
    for iy in 0..h{
    for ix in 0..w{
        let ix = 13;
        let iy = 14;
     
        let x = ix as f64 / w as f64;
        let y = iy as f64 /  h as f64;;
        
        let x=2.0*x-1.0;
        let y=2.0*y-1.0;
        let orig = Point3::new(0.00, 1.310, 2.5);
        let newdir =  Point3::new(x,y, 0.0)-orig;
        let ray = prims::Ray::new(
            orig,
            newdir.normalize()
    
        );
      //   supongo que hay que mirar la imagen! y mirar a ver si puedo copiar el rayo que me hace eso raro
        // get_ray_direct_light(&ray,   &scene.lights, scene,0);
        // println!("{:?} {:?}", ray.origin , ray.direction );
        let mut hitop =  interset_scene(&ray, scene);
       
        if let Some(hit) = hitop {
            let hit = hitop.unwrap();
          //  println!("{},{} {:?}",ix, iy,hit.point);
            
       
       
        
           //  LigtImportanceSampling::estimate_area_light(&ray,&scene.lights[0],scene,0,hit, &mut sampleruniform,32);

            fn trackSampleGenerator(hit:prims::HitRecord<f64>, ray :Ray<f64>, ix:i32, iy:i32, total:i32){
                
                for _x in 0..total{
                    let itest =  (_x / (total / 2) ) as f64/ (total / 2) as f64;
                    let jtest= (_x %  (total / 2)) as f64/ (total / 2)as f64;
                    let rec = hit.material.sample(RecordSampleIn::from_hitrecord(hit, &ray, (itest as f64, jtest as f64)));
                   //  println!("              {},{}, {},{} {:?}, next: {:?} , fr :{:?}, pdf {}", ix, iy, itest, jtest, hit.point, rec.next,rec.f, rec.pdf);
                 }
              
             
            }
         //   trackSampleGenerator(hit, ray, ix, iy, 8);
            fn  walkpath(hitrec:prims::HitRecord<f64>, ray :Ray<f64>, ix:i32, iy:i32, scene:&Scene<f64>,   sampler : &mut  Box<dyn Sampler> ){
             //   print!("{},{} {:?}",ix, iy,hitrec.point);
             let mut L: Vec<f64> = vec![0.0; 3];
               let depth = 5;
               let mut paththrought : Srgb  = Srgb::new(1.0,1.0, 1.0);
              let mut r =  Cow::Borrowed(&ray);
              let mut  hit =  Cow::Borrowed(&hitrec);
            //   println!("{:?}, {:?}", r.origin, r.direction);
         
          //  puedo hacer esto como esta con disco y esfera y con profundidad 
                let mut  longdepth = 0;
               for idepth in 0..depth{ 
                    longdepth = longdepth.max(idepth);
                    
                    let Ld = estimatelights(scene, *hit, &r,  sampler); 
                   // println!("Ld : {:?}", Ld);
                    L = LigtImportanceSampling::addVecBySrgb(L,LigtImportanceSampling::mulSrgb(Ld, paththrought));
                    let rec = hit.material.sample(RecordSampleIn::from_hitrecord(*hit, &r, (0.5,0.5)));
                   // pathThrought*= fr*AbsDot(currentHit.n, wnext) / pdf;
                 
                  let absdot =  hit.normal.dot(rec.next).abs();
                   let pdf  = rec.pdf;
                   let a = absdot / pdf;
                   let fr = rec.f;
                  let current  = LigtImportanceSampling::mulScalarSrgb(fr, a as f32);
                  paththrought= LigtImportanceSampling::mulSrgb(paththrought, current);
                  
                

               
                 r= Cow::Owned(Ray::<f64>::new(hit.point, rec.next));
               


                  if let Some(newhit) =  interset_scene(&r, scene){
                     println!("{},{}  depth {}  current L {:?} Ld {:?}  paththrought{:?}",ix, iy, idepth, L,  Ld, paththrought);
                    hit = Cow::Owned(newhit); 
                   }else{
                    break;
                   }
               }    
               if longdepth >0 {
              //   println!("{}, {} {}",ix, iy, longdepth);
               }
             

           
           }
           //   walkpath(hit, ray, ix, iy, scene, &mut sampleruniform);
            
        }
     }
}   
}









#[test]
fn some_relations_in_tangent_space(){
    fn sintheta(v:Vector3<f64>)->f64{
        ( 1.0 - v.z*v.z).sqrt()
     }
     fn cosphi(v:Vector3<f64>)->f64{
         let sinth = sintheta(v);
         v.x / sinth
     }
     fn sinphi(v:Vector3<f64>)->f64{
         let sinth = sintheta(v);
         v.y / sinth
     }
     
      
     println!("{}",SinPhi(&Vector3::new(1.0,0.0,0.0).normalize()) ) ;
     println!("{}",SinPhi(&Vector3::new(1.0,1.0,0.0).normalize()) );
     println!("{}",SinPhi(&Vector3::new(0.0,1.0,0.0).normalize()) );
     println!("{}",CosPhi(&Vector3::new(1.0,0.0,0.0).normalize()) ) ;
     println!("{}",CosPhi(&Vector3::new(1.0,1.0,0.0).normalize()) );
     println!("{}",CosPhi(&Vector3::new(0.0,1.0,0.0).normalize()) );
     
     println!("");
     println!("{}",sinphi(Vector3::new(1.0,0.0,0.0).normalize()) ) ;
     println!("{}",sinphi(Vector3::new(1.0,1.0,0.0).normalize()) );
     println!("{}",sinphi(Vector3::new(0.0,1.0,0.0).normalize()) );
     println!("{}",cosphi(Vector3::new(1.0,0.0,0.0).normalize()) ) ;
     println!("{}",cosphi(Vector3::new(1.0,1.0,0.0).normalize()) );
     println!("{}",cosphi(Vector3::new(0.0,1.0,0.0).normalize()) );



     
     println!("{}",SinPhi(&Vector3::new(-1.0,0.0,0.0).normalize()) ) ;
     println!("{}",SinPhi(&Vector3::new(-1.0,1.0,0.0).normalize()) );
     println!("{}",SinPhi(&Vector3::new(0.0,1.0,0.0).normalize()) );
     println!("{}",CosPhi(&Vector3::new(-1.0,0.0,0.0).normalize()) ) ;
     println!("{}",CosPhi(&Vector3::new(-1.0,1.0,0.0).normalize()) );
     println!("{}",CosPhi(&Vector3::new(0.0,1.0,0.0).normalize()) );

      
// assert_eq!(Vector3::new(-1.0,0.0,0.0).normalize().sin2phi() == SinPhi(&Vector3::new(-1.0,0.0,0.0).normalize()), true);
 
  assert_eq!( SinPhi(&Vector3::new(-1.0,1.0,0.0).normalize()) , Vector3::new(-1.0,1.0,0.0).normalize().sinphi()  );
assert_eq!(   Sin2Phi(&Vector3::new(0.0,1.0,0.0).normalize()),Vector3::new(0.0,1.0,0.0).normalize().sin2phi());
assert_eq!(   CosPhi(&Vector3::new(-1.0,0.0,0.0).normalize()),Vector3::new(-1.0,0.0,0.0).normalize().cosphi()  );
assert_eq!(   Cos2Phi(&Vector3::new(-1.0,1.0,0.0).normalize()) ,Vector3::new(-1.0,1.0,0.0).normalize().cos2phi() );
assert_eq!(     TanTheta(&Vector3::new(0.0,1.0,0.0).normalize()),Vector3::new(0.0,1.0,0.0).normalize().tantheta() ) ;
assert_eq!(     Tan2Theta(&Vector3::new(0.0,1.0,0.0).normalize()),Vector3::new(0.0,1.0,0.0).normalize().tan2theta() ) ;









}














#[test]
pub fn test_sample_sphere() {
    use prims::Intersection;
    let mut cnt = 0;

    let ray = prims::Ray::new(
        Point3::new(12.0, 0.0, 12.0),
        Vector3::new(0.0, 1.0, 0.0).normalize(),
    );
    let sphere = prims::Sphere::new(
        Vector3::new(12.0, 12.0, 12.0),
        11.0,
        MaterialDescType::PlasticType(Plastic::default())
    );
    let sphere1 = prims::Sphere::new(
        Vector3::new(12.0, -12.0, 12.0),
        11.0,
        MaterialDescType::PlasticType(Plastic::default())
    );
    let scene = Scene::<f64>::make_scene(0, 0, vec![], vec![&sphere, &sphere1], 0, 0);
    // scene.intersect_occlusion(ray, target)
    // if let Some(hit) = scene.intersect(&ray, 0.001, f64::MAX) {
    //     let record = RecordSampleIn {
    //         prevW: Vector3::new(0.0, 0.0, 0.0),
    //         pointW: hit.point,
    //     };
    //     let recout = hit.material.sample(record);
    //     // println!( "cnt {}, hit.t {}, p:{:?} angle n:{:?} {},{}",cnt,hit.t , hit.point, Deg::from(hit.normal.angle(Vector3::new(0.,0.,1.0))) , hit.u,hit.v);
    //     println!("{:?}", recout.newray);
    //     fn inner_itc(r: &Ray<f64>, scene: &Scene<f64>, depth: i32, cnt: i32) -> i32 {
    //         if depth == 0 {
    //             return cnt;
    //         }
    //         if let Some(hit) = scene.intersect(&r, 0.001, f64::MAX) {
    //             let record = RecordSampleIn {
    //                 prevW: Vector3::new(0.0, 0.0, 0.0),
    //                 pointW: hit.point,
    //             };
    //             let recout = hit.material.sample(record);
    //             println!("{:?}", recout.newray);
    //             return inner_itc(&recout.newray.unwrap(), &scene, depth - 1, cnt + 1);
    //         }
    //         return cnt;
    //     }
    //     let depth = 10;
    //     let result_cnt = inner_itc(&ray, &scene, depth, 0);
    //     println!("{}", result_cnt);
    //     assert_eq!(depth, result_cnt);
    // }
}


#[test]
fn test_bsdf_world_local() {
    for i in 0..100 {
        let bsdf = FrameShading::new(
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(1.0, 0.0, 0.0),
        );
        //    let vnew = Vector3::new(1.0,-10.0,1.0).normalize();

        let vnew = FrameShading::random(-1.0f64, 1.0f64).normalize();
        let vlocal = bsdf.to_local(&vnew).normalize();

        let vworld = bsdf.to_world(&vlocal).normalize();

        assert_delta!(vnew.x, vworld.x, 0.0000000000001);
        assert_delta!(vnew.y, vworld.y, 0.0000000000001);
        assert_delta!(vnew.z, vworld.z, 0.0000000000001);
    }
}







#[test]
pub fn testing_glass_material() {

    use crate::primitives::prims::HitRecord;
    use crate::primitives::prims::Sphere;
    use crate::Lights::Light;
    use crate::Lights::BackgroundAreaLight;
    use crate::Lights::SampleLightIllumination;
     
    let sphere = prims::Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        // MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0))
        MaterialDescType::Glass2Type(Glass2::new())
    );
  
    
    let sphere2smallptr = &(
        Box::new(  sphere ) as Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
    );
    








    
    
    let primitivesIntersections  :Vec<&Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>> =vec![ 
        sphere2smallptr,
       
       ];


       let pointlight = Light::PointLight(PointLight { iu: 4.0, positionws: Point3::new(0.0010,1.0,0.0010),color: Srgb::new(1.0,1.0,1.0 )});
    let scene :&Scene<f64> = &Scene::make_scene1(
        128,
        128,
        vec![ pointlight],
        vec![],
        primitivesIntersections,
        1,
        1,None
    );
    let mut sampleruniform :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),32, false, Some(0)));
    // Ray rv2(Point3f(0.00f, .0f, -2.f), Vector3f(0,0,1));
    for iy in 0..32 {
        let yf = (iy as f64 / 32.0) * 2.0 -1.0;
        let ray = Ray::new(Point3::new(0.0,yf,-2.0), Vector3::new(0.0,0.0, 1.0));
        let mut hitop =  interset_scene(&ray, scene);
        

        if let Some(hit) = hitop {
            let hit = hitop.unwrap();
           if hit.material.is_specular(){
            assert!(true);
           }
           let psample = sampleruniform.get2d();
            let recordout  = hit.material.sample(RecordSampleIn::from_hitrecord(hit, &ray,  psample));
          
             println!("1er pt :{} {:?} psample {:?}",iy, hit.point, psample);
              println!("          next {:?}", recordout.next);
              println!("          res {:?}", recordout.f);
            // println!("          pdf {:?}",  recordout. pdf);

            let ray2 = Ray::new(hit.point , recordout.next);
             let mut hitop =  interset_scene(&ray2, scene);
            if let Some(hit) = hitop {
                let psample = sampleruniform.get2d();
                    println!("         2er pt :{} {:?}  psample {:?}",iy, hit.point, psample);
                    let recordout  = hit.material.sample(RecordSampleIn::from_hitrecord(hit, &ray2,  psample)); 
                    println!("            next {:?}", recordout.next);
                    println!("            res {:?}", recordout.f);
                   //  println!("          pdf {:?}",  recordout. pdf);
                    let ray2 = Ray::new(hit.point , recordout.next);
                    let mut hitop =  interset_scene(&ray2, scene);
                    if let Some(hit) = hitop {
                    println!("                          3er pt :{} {:?}",iy, hit.point);
                    }
             }
      


       // println!("{} , {:?}  next {:?}",yf, hit.point, recordout.next);
        // println!("          res{:?}", recordout.f);
        // println!("          pdf{:?}",  recordout. pdf);


        }
      
    }
     
      

    
}