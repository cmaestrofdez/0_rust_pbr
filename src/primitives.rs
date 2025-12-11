// extern crate nalgebra as na;
// extern crate nalgebra_glm as glm;

// use glm::Mat4;
// use glm::Vec4;
// use na::Orthographic3;
// use na::Perspective3;
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unused_parens)]

extern crate cgmath;
use std::cmp::min;
use std::iter::zip;
use crate::raytracerv2::lerp;
use crate::scene::Scene1;
use crate::threapoolexamples1::{new_interface_for_intersect_occlusion, new_interface_for_intersect_scene};
use crate::volumentricpathtracer::volumetric::MediumOutsideInside;
use crate::{Lights, Vector3f, Point3f, Spheref64};
use crate::Lights::{SampleLightIllumination, New_Interface_AreaLight, RecordSampleLightEmissionIn, RecordSampleLightIlumnation};
use crate::assert_delta;
use crate::materials::{self, SampleIllumination, Fr, MaterialDescType, Plastic, FrameShading, MaterialDescriptor, BsdfType, ConcentricSampleDisk};
use crate::primitives::prims::Intersection;
use crate::texture::{*};
use crate::texture::{MapConstant,Texture2D,Texture2DSRgbMapConstant};
use palette::Srgb;
use cgmath::*;
use num_traits::Float;
use std::f64::consts;
use std::time::Instant;

use self::prims::{IntersectionOclussion, Ray, HitRecord, Sphere};

pub mod prims {
    use cgmath::BaseFloat;
    use cgmath::Matrix4;
    use cgmath::Point3;
    use cgmath::Vector3;

    use cgmath::*;
    use num_traits::Float;
    use num_traits::ToPrimitive;
    use palette::Srgb;

    use crate::Lights;
    use crate::Lights::Light;
    use crate::Lights::SampleLightIllumination;
    use crate::Lights::generic_shape_pdfillumination;
    use crate::Point3f;
    use crate::Vector3f;
    use crate::materials;
    use crate::materials::CosSampleH2;
    use crate::materials::FrameShading;
    use crate::materials::BsdfType;

    use crate::materials::LambertianBsdf;
    use crate::materials::MaterialDescType;
    use crate::materials::MaterialDescriptor;
    use crate::materials::Plastic;
    use crate::materials::UniformSampleConePdf;
    use crate::scene::Scene1;
    use crate::texture::MapConstant;
    use crate::texture::Texture2D;
    
    use crate::raytracerv2::clamp;
    use crate::volumentricpathtracer;
    use crate::volumentricpathtracer::volumetric::MediumOutsideInside;
    use crate::volumentricpathtracer::volumetric::MediumType;
    #[derive(Debug, Clone )]
    pub struct Sphere< Scalar> {
        pub world: Matrix4<Scalar>,
        pub local: Matrix4<Scalar>,
        pub radius: Scalar,

        pub material: Option<materials::MaterialDescType >,
        pub medium : Option<MediumOutsideInside>
    }
    impl<Scalar: BaseFloat> Sphere<Scalar> {
        pub fn new(
            translation: Vector3<Scalar>,
            scale: Scalar,
            material: materials::MaterialDescType ,
        ) -> Sphere<Scalar> {
            let t = Matrix4::from_translation(translation);
            Sphere {
                world: t,
                local: t.inverse_transform().unwrap(),
                radius: scale,
                material: Some(material),
                medium:None,
            }
        }
        pub fn new_with_medium(
            translation: Vector3<Scalar>,
            scale: Scalar,
            material: materials::MaterialDescType ,
            outside_inside_medium : MediumOutsideInside,
        ) -> Sphere<Scalar> {
            let t = Matrix4::from_translation(translation);
            match material {
                MaterialDescType::NoneType=>{
                    Sphere {world: t,local: t.inverse_transform().unwrap(),radius: scale,material:None,medium:Some(outside_inside_medium)}
                },
                _=>{
                    Sphere {world: t,local: t.inverse_transform().unwrap(),radius: scale,material:Some(material),medium:Some(outside_inside_medium)}
                }
            }
            
        }
        

        pub fn with_box(
            translation: Vector3<Scalar>,
            scale: Scalar,
            material: materials::MaterialDescType ,
        ) -> Box<Sphere<Scalar>> {
            let t = Matrix4::from_translation(translation);
            Box::new(
                Sphere {
                    world: t,
                    local: t.inverse_transform().unwrap(),
                    radius: scale,
                    material: Some(material),
                    medium:None,
                }
            )
           
        }
        pub fn trafo_to_local(&self, pworld: Point3<Scalar>) -> Point3<Scalar> {
            self.local.transform_point(pworld)
        }
        pub fn trafo_to_world(&self, plocal: Point3<Scalar>) -> Point3<Scalar> {
            self.world.transform_point(plocal)
        }
        pub fn trafo_vec_to_local(&self, pworld: Vector3<Scalar>) -> Vector3<Scalar> {
            self.local.transform_vector(pworld)
        }
        pub fn trafo_vec_to_world(&self, plocal: Vector3<Scalar>) -> Vector3<Scalar> {
            self.world.transform_vector(plocal)
        }
        pub fn area(&self)->f64{
            let r =  num_traits:: cast::<Scalar, f64 >(self.radius).unwrap();
             (r    *2.0*std::f64::consts::PI  )
         }
         pub fn sample_ref(&self, phit:Point3f, psample: (f64, f64)) ->(Point3<f64>, Vector3<f64>, f64) {
            let zero = num_traits::cast::cast::<f64, Scalar>(0.0).unwrap();
            let pcenterinworld =  self.trafo_to_world(Point3::new( zero, zero, zero));
           let p =  cast_point_to_point_to_vf64(pcenterinworld);
           let res =  sample_sphere_extern(num_traits::cast::cast::< Scalar , f64>(self.radius).unwrap(),p, phit,psample) ;
          
           res
            
         }
         pub fn sample(&self,  psample: (f64, f64))->(Point3<f64>, Vector3<f64>, f64){
            let zero = num_traits::cast::cast::<f64, Scalar>(0.0).unwrap();
           let pcenterinworld =  self.trafo_to_world(Point3::new( zero, zero, zero));
          let p =  cast_point_to_point_to_vf64(pcenterinworld);
           let res =  sample_sphere_extern(num_traits::cast::cast::< Scalar , f64>(self.radius).unwrap(),p, Point3::new(0.0,0.0,0.0),psample) ;
          
           res
         }

         pub fn pdfIllumination1( &self, hit: &HitRecord<f64>, scene: &Scene1, next: Vector3<f64>)->f64{
            let zero = num_traits::cast::cast::<f64, Scalar>(0.0).unwrap();
            let pcenterinworld =  self.trafo_to_world(Point3::new( zero, zero, zero));
         
           if pcenterinworld.distance2(   cast_to_pScalar(hit.point))<=self.radius*self.radius{
                todo!("shpere::pdfindide");
               let pdf =  generic_shape_pdfillumination(hit, scene, next, self.area());
               return  pdf;
           }
           pdf_illumination_extern(num_traits::cast::cast::< Scalar , f64>(self.radius).unwrap(),  cast_point_to_point_to_vf64(pcenterinworld), hit.point)
           
            
         }
      
         
    }
    fn cast_to_vf64<Scalar:BaseFloat>(v:Vector3<Scalar>)->Vector3<f64>{
        Vector3::new(
                num_traits:: cast::<Scalar, f64>(v.x).unwrap(), 
                    num_traits:: cast::<Scalar, f64>(v.y).unwrap(), 
                num_traits:: cast::<Scalar, f64>(v.z) .unwrap())
        
       }
       fn cast_point_to_point_to_vf64<Scalar:BaseFloat>(v:Point3<Scalar>)->Point3<f64>{
        Point3::new(
                num_traits:: cast::<Scalar, f64>(v.x).unwrap(), 
                    num_traits:: cast::<Scalar, f64>(v.y).unwrap(), 
                num_traits:: cast::<Scalar, f64>(v.z) .unwrap())
        
       }
       fn cast_to_vScalar<Scalar:BaseFloat>(v:Vector3<f64>)->Vector3<Scalar>{
        Vector3::new(
                num_traits:: cast::<f64,Scalar >(v.x).unwrap(), 
                    num_traits:: cast::< f64,Scalar >(v.y).unwrap(), 
                num_traits:: cast:: <f64,Scalar>(v.z) .unwrap())
        
       }
       fn cast_to_pScalar<Scalar:BaseFloat>(v:Point3<f64>)->Point3<Scalar>{
        Point3::new(
                num_traits:: cast::<f64,Scalar >(v.x).unwrap(), 
                    num_traits:: cast::<f64,Scalar >(v.y).unwrap(), 
                num_traits:: cast:: <f64,Scalar >(v.z) .unwrap())
        
       }
    // hay una forma de hacer esto sin sqrt.TODO:buscar
    pub fn coord_system( v :  Vector3<f64> )->(Vector3<f64>,Vector3<f64>){
        // *v2 = Vector3<T>(-v1.z, 0, v1.x) / std::sqrt(v1.x * v1.x + v1.z * v1.z);
        let  mut a ;
        if v.x.abs()>v.y.abs(){
            let s =( v.x *v.x + v.z*v.z).sqrt();
             a = Vector3::new(-v.z / s, 0.0, v.x / s);
        }else{
            let s =( v.y *v.y + v.z*v.z).sqrt();
             a = Vector3::new(0.0, v.z / s, -v.y / s);
        }
        (a, v.cross(a))
    }

    pub fn pdf_illumination_extern( radius : f64 ,  center_sphere : Point3<f64>, pref : Point3<f64>)->f64{

        let sintmax2 =( radius *  radius) / center_sphere.distance2(pref);
       let costmax2 = ( (1.0 - sintmax2).max(0.0)).sqrt();
       return  UniformSampleConePdf(costmax2);
    }
    
    pub fn sample_sphere_extern(r : f64 ,pcenter_sphere : Point3<f64>, pref : Point3<f64> ,psample : (f64,f64))->(Point3<f64>, Vector3<f64>, f64){
       let vc =  pcenter_sphere  - pref;
       let d = vc.magnitude();
       let invd = 1.0/vc.magnitude();
       let vc =vc.normalize();




       let (vx, vy)=coord_system(vc);
      let sintheta =   r   *invd ;
       let sintheta2 = sintheta*sintheta;
       let invsintheta = 1.0 / sintheta;
       let costheta2 =(1.0-sintheta2).max(0.0).sqrt();
       
       let costhetasample = (1.0-psample.0)+psample.0*costheta2;
       let costhetasample = (costheta2-1.0)*psample.0+1.0;
       let sinthetasample = (1.0-costhetasample*costhetasample) .max(0.0) ;
       let phi = psample.1*2.0*std::f64::consts::PI;
    
    //   let ds = d*costhetasample - (r*r -d *d *sinthetasample*sinthetasample).max(0.0).sqrt()  ;
    //   let cosalpha = ( d * d + r*r - ds*ds)/(2.0 * d * r);
    //   let sinalpha = (1.0-cosalpha*cosalpha).max(0.0).sqrt();


     let cosalpha  =  sinthetasample *invsintheta + costhetasample * ((1.0 - sinthetasample * invsintheta*invsintheta).max(0.0).sqrt());
     let sinalpha = (1.0 - cosalpha*cosalpha).max(0.0).sqrt();
   // spherical direction
     let ja =  Vector3::new(- sinalpha* phi.cos() * vx.x,- sinalpha* phi.cos()* vx.y ,-sinalpha* phi.cos() * vx.z);
     let jb =  Vector3::new(- sinalpha* phi.sin() * vy.x,- sinalpha* phi.sin()* vy.y ,-sinalpha* phi.sin() * vy.z);
     let jc  =  Vector3::new(- cosalpha * vc.x,-cosalpha* vc.y ,-cosalpha  * vc.z);
     let n  =  ja + jb  + jc;
     // translate point to sphere center and
      let pSpheresampleInWorlds = Point3::new( pcenter_sphere.x + n.x*r, pcenter_sphere.y + n.y*r, pcenter_sphere.z + n.z*r);
      

      (pSpheresampleInWorlds, n.normalize(),  1.0 /(2.0 * std::f64::consts::PI * (1.0 -costheta2 )))
   
    
       
     }
    
 
    
    #[derive(
        Debug,
        Clone,
         Copy
        // Deserialize, Serialize
    )]
    pub struct Ray<Scalar> {
        pub origin: Point3<Scalar>,
        pub direction: Vector3<Scalar>,
        pub is_in_medium : bool,
    
        pub medium:Option<MediumType>,
        // pub tmax: f64,
    }

    impl<Scalar: BaseFloat> Ray<Scalar> {
        pub fn new(origin: Point3<Scalar>, direction: Vector3<Scalar>) -> Ray<Scalar> {
            Ray { origin, direction ,is_in_medium: false ,  medium:None}
        }
        pub fn new_with_medium_active(origin: Point3<Scalar>, direction: Vector3<Scalar>) -> Ray<Scalar> {
            Ray { origin, direction ,is_in_medium: true  ,medium:None}
        }
        pub fn new_with_state(origin: Point3<Scalar>, direction: Vector3<Scalar> ) -> Ray<Scalar> {
            Ray { origin, direction ,is_in_medium: true  ,medium:None}
        }
        pub fn new_with_state_medium(origin: Point3<Scalar>, direction: Vector3<Scalar> , medium:MediumType) -> Ray<Scalar> {
            Ray { origin, direction ,is_in_medium: true  , medium:Some(medium)}
        }
        pub fn at(&self, p: Scalar) -> Point3<Scalar> {
            self.origin + self.direction * p
        }
        pub fn from_origin_to_target(
            origin: Point3<Scalar>,
            target: Point3<Scalar>,
        ) -> Ray<Scalar> {
            let direction = (target - origin) ;
            Ray { origin, direction, is_in_medium: false ,medium:None }
        }
    }


    
    #[derive(
        Debug,
        Clone,
      Copy,
        // Deserialize, Serialize
    )]
    pub struct HitRecord<Scalar> {
        pub t: Scalar,
        pub point: Point3<Scalar>,
        pub normal: Vector3<Scalar>,
        pub tn: Vector3<Scalar>,
        // pub front_face: bool,
        pub material: materials::BsdfType ,
        pub bn: Vector3<Scalar>,
        pub u: Scalar,
        pub v: Scalar,
        pub prev: Option<Vector3<Scalar>>,
        pub is_hit_inside: Option<bool>,
        pub is_emision : Option<bool>,
        pub emission : Option<Srgb>,
        pub medium : Option< MediumOutsideInside >,
         
       
    }
    impl HitRecord<f64> {
      
        pub fn init_empty()->Self{
          
            HitRecord{
                t:0.0, 
                point:Point3::new(0.0,0.0,0.0),
                normal:Vector3::new(0.0,0.0,0.0),
                tn:Vector3::new(0.0,0.0,0.0),
                material: BsdfType::None,
                bn:Vector3::new(0.0,0.0,0.0),
                u:0.0,
                v:0.0,
                prev: Some(Vector3::new(0.0,0.0,0.0)),
                is_hit_inside:Some(false),
                is_emision :Some(false),
                emission : None,
                medium:None
              
                
            }
        }
        // auxiliary method for bidrectional path tracing.
        // this method is used when it`s create  ligt path vertex
        pub fn from_point(p : Point3<f64>, n : Vector3<f64>)->Self{

            HitRecord{
                t:0.0, 
                point:p,
                normal:n,
                tn:Vector3::new(0.0,0.0,0.0),
                material: BsdfType::None,
                bn:Vector3::new(0.0,0.0,0.0),
                u:0.0,
                v:0.0,
                prev: Some(Vector3::new(0.0,0.0,0.0)),
                is_hit_inside:Some(false),
                is_emision :Some(false),
                emission : None,
                // light:None,
                medium:None
              

            }
        }
     
        pub fn has_medium(&self)->Option<(MediumType,MediumType)>{
            self.medium
        }
        // pub fn is_in_medium_interface(&self, ray:&Ray<f64>)->bool{
        //      match self.material {
        //         BsdfType::None=>{
        //             match ray.state {
        //                 RayState::OutsideEntering =>{  true},
        //                 _=>false
        //             }
        //         },
        //         _=>false
        //      } 
           
        // }
        pub fn has_material(&self)->bool{
            match self.material {
               BsdfType::None=>{
                  false
               },
               _=>true
            } 
          
       }
        pub fn spawn_ray(&self,v:Vector3f )->Ray<f64>{
            let medium = self.get_medium_1(v);
            Ray ::new_with_state_medium(self.point, v,self.get_medium_1(v)) 
        }
        pub fn get_medium_1(&self, v:Vector3f)-> MediumType{
          
            if v.dot(self.normal) <0.0{
               return   self.medium.unwrap().1;
            }
            self.medium.unwrap().0 
        }
        pub fn get_medium(&self)-> MediumType{
            if self.prev.unwrap().dot(self.normal) <0.0{
               return   self.medium.unwrap().1;
            }
            self.medium.unwrap().0 
        }
        //true is surface entering, 
        // false : is surface out
        
        pub fn is_surface_interaction(&self)->bool{
            self.prev.unwrap().dot(self.normal) > 0.0
        }
    }


     

    #[test]
    pub fn test_sphere_transformations() {
        let sphere = Sphere::new(
            Vector3::new(1.0, 1.0, 1.0),
            1.0,
            MaterialDescType::PlasticType(Plastic ::default())
        );
        
        assert_eq!(
            sphere.trafo_to_world(sphere.trafo_to_local(Point3::new(1.0, 1.0, 1.0))),
            Point3::new(1.0, 1.0, 1.0)
        );
    }

   
    pub trait Intersection<Scalar> {
        type Output;
        fn intersect(&self, ray: &Ray<Scalar>, tmin: Scalar, tmax: Scalar) -> Option<Self::Output>;
    }
   //     |y   /z
    //     |  /
    //     |/______x
    //    
    //+ 

    impl<Scalar: BaseFloat + rand::distributions::uniform::SampleUniform + PartialOrd>
        Intersection<Scalar> for Sphere<Scalar>
    {
        type Output = HitRecord<Scalar>;
        fn intersect(&self, ray: &Ray<Scalar>, tmin: Scalar, tmax: Scalar) -> Option<Self::Output> {
            let origin = self.trafo_to_local(ray.origin);
            let direction = self.trafo_vec_to_local(ray.direction);
            let oc = origin;

            //  println!("{:?}", oc);
            let A = direction.dot(direction);
            let B = oc.dot(direction);
            let C = oc.dot(oc.to_vec()) - self.radius * self.radius; // -1 - self.radius * self.radius;
            let disc = B * B - A * C;

            if disc.to_f64().unwrap() > 0.0 {
                let sqrtd = disc.sqrt();
                let root_a = ((-B) - sqrtd) / A;
                let root_b = ((-B) + sqrtd) / A;
                for root in [root_a, root_b].iter() {
                    if *root > tmin && *root < tmax {
                        // trafo ray in unit space
                        let mut p = origin + direction * (*root);

                        if p.x.to_f64().unwrap() == 0.0 && p.y.to_f64().unwrap() == 0.0 {
                            p.x =
                                num_traits::cast::cast::<f64, Scalar>(1e-5).unwrap() * self.radius;
                           // println!("ups");
                        }
                        let mut phi = p.y.to_f64().unwrap().atan2(p.x.to_f64().unwrap());

                        if phi < 0.0 {
                            phi += 2.0f64 * std::f64::consts::PI;
                        };


                       let uu =  num_traits::cast::cast::<f64, Scalar>(phi / (2.0f64 * std::f64::consts::PI));
                       let u =  num_traits::cast::cast::<f64, Scalar>(phi / (2.0f64 * std::f64::consts::PI)).unwrap();

                        let theta =  clamp(p.z.to_f64().unwrap() / self.radius.to_f64().unwrap(),-1.0,1.0).acos()  ;
                       
                       let v = num_traits::cast::cast::<f64, Scalar>(( (theta - std::f64::consts::PI) / -std::f64::consts::PI)).unwrap();
                        let twopi =
                            num_traits::cast::cast::<f64, Scalar>(2.0f64 * std::f64::consts::PI)
                                .unwrap();
                        let pi =
                            num_traits::cast::cast::<f64, Scalar>(std::f64::consts::PI).unwrap();
                        let zero = num_traits::cast::cast::<f64, Scalar>(0.0).unwrap();
                        let invradius = num_traits::cast::cast::<f64, Scalar>(1.0).unwrap()
                            / (p.y * p.y + p.x * p.x).sqrt();
                        let mut du = Vector3::new(-twopi * p.y,  twopi * p.x,zero);
                        //  println!("{:?}", du);
                        let cosphi = p.x * invradius;
                        let sinphi = p.y * invradius;
                        
                        // let theta = num_traits::cast::cast::<f64, Scalar>(
                        //     ((p.z / self.radius).to_f64().unwrap()).acos(),
                        // )
                        // .unwrap();
                        // Vector3f dpdv =
                        // (thetaMax - thetaMin) *
                        // Vector3f(pHit.z * cosPhi, pHit.z * sinPhi, -radius * std::sin(theta));
                        let mut dv =
                            Vector3::new(p.z * cosphi*(-pi),  p.z * sinphi*(-pi),-self.radius * num_traits::cast::cast::<f64, Scalar>(theta.sin()).unwrap()*(-pi));
                      
                        let mut dn = du.cross(dv).normalize();
                        let is_hit_inside = ray.direction.dot(dn)
                            > num_traits::cast::cast::<f64, Scalar>(0.0).unwrap();
                 //       println!("{:?}, {:?}, {:?}",dn, dv,du);
                        if (is_hit_inside && false) {
                            dn = -dn;
                            du = -du;
                            dv = -dv;
                        }
                       
                      
                    //    fn cast_to_vf64<Scalar:BaseFloat>(v:Vector3<Scalar>)->Vector3<f64>{
                    //     Vector3::new(
                    //             num_traits:: cast::<Scalar, f64>(v.x).unwrap(), 
                    //                 num_traits:: cast::<Scalar, f64>(v.y).unwrap(), 
                    //             num_traits:: cast::<Scalar, f64>(v.z) .unwrap())
                        
                    //    }
                    //    fn cast_point_to_point_to_vf64<Scalar:BaseFloat>(v:Point3<Scalar>)->Point3<f64>{
                    //     Point3::new(
                    //             num_traits:: cast::<Scalar, f64>(v.x).unwrap(), 
                    //                 num_traits:: cast::<Scalar, f64>(v.y).unwrap(), 
                    //             num_traits:: cast::<Scalar, f64>(v.z) .unwrap())
                        
                    //    }
                    //    fn cast_to_vScalar<Scalar:BaseFloat>(v:Vector3<f64>)->Vector3<Scalar>{
                    //     Vector3::new(
                    //             num_traits:: cast::<f64,Scalar >(v.x).unwrap(), 
                    //                 num_traits:: cast::< f64,Scalar >(v.y).unwrap(), 
                    //             num_traits:: cast:: <f64,Scalar>(v.z) .unwrap())
                        
                    //    }
                     let invertnormal = false;
                         if invertnormal {
                            dn = -dn;
                         }
                        let frame = FrameShading ::new(
                            cast_to_vf64(self.trafo_vec_to_world(dn)),
                            cast_to_vf64(self.trafo_vec_to_world(dn.cross(du.normalize())).normalize()),
                            cast_to_vf64(self.trafo_vec_to_world(du.normalize() ).normalize()),
                        );

                      
                        let mut bsdftype :BsdfType =  BsdfType::None;
                        if  self.material.is_some() {
                            bsdftype =   self.material.as_ref().unwrap().instantiate( &cast_point_to_point_to_vf64(self.trafo_to_world(p )), &( num_traits:: cast:: <Scalar,f64>(u).unwrap(),num_traits:: cast:: <Scalar,f64>(v).unwrap()),&frame);
                        }
                      
                    
                        return Some(HitRecord {
                            t: *root,
                            point: self.trafo_to_world(p),
                            normal: cast_to_vScalar(frame.n),
                            tn: cast_to_vScalar(frame.tn),
                            bn: cast_to_vScalar( frame.bn),
 
                            u,
                             
                            v,
                            material:bsdftype,
                            prev: Some(-ray.direction),
                            is_hit_inside: Some(is_hit_inside),
                            is_emision:Some(false),
                            emission :None,
                            medium:self.medium
              
                        });
                    }
                }
            }

            None
        }

       
    }

    pub trait IntersectionOclussion<Scalar> {
       
        fn intersect_occlusion(
            &self,
            ray: &Ray<Scalar>,
            target: &Point3<Scalar>,
        ) ->  Option<(bool,Scalar, Point3<Scalar>,  Option<MediumOutsideInside>)>;
    }
    //     |y   /z
    //     |  /
    //     |/______x
    //    
    //+ 


    pub trait IsEmptySpace {
       
       // return true si entre p0  y p1 solo existe el vacio
        fn is_empty_space(
            &self,
            p0:  Point3<f64>,
            p1 : Point3<f64>,
        ) ->  bool;
    }
 
    impl<Scalar: BaseFloat + rand::distributions::uniform::SampleUniform + PartialOrd>
        IntersectionOclussion<Scalar> for Sphere<Scalar>
    {
       
        fn intersect_occlusion(
            &self,
            ray: &Ray<Scalar>,
            target: &Point3<Scalar>,
        ) ->  Option< (bool, Scalar, Point3<Scalar>,  Option<MediumOutsideInside>)> {
            let origin = self.trafo_to_local(ray.origin);
            let direction = self.trafo_vec_to_local(ray.direction);
            let oc = origin;
            let tmin = num_traits::cast::cast::<f64, Scalar>(0.000000001).unwrap();
            let tmax = num_traits::cast::cast::<f64, Scalar>(f64::MAX).unwrap();
            //  println!("{:?}", oc);
            let A = direction.dot(direction);
            let B = oc.dot(direction);
            let C = oc.dot(oc.to_vec()) - self.radius * self.radius; // -1 - self.radius * self.radius;
            let disc = B * B - A * C;
            if disc.to_f64().unwrap() > 0.0 {
                let sqrtd = disc.sqrt();
                let root_a = ((-B) - sqrtd) / A;
                let root_b = ((-B) + sqrtd) / A;
                for root in [root_a, root_b].iter() {
                    if (*root > tmin) && (*root < tmax) {
                        let mut p = origin + direction * (*root);

                        return Some((true, *root, self.trafo_to_world(p), self.medium));
                    }
                }
            }
            None
        }
    } // end interface
} // end prims
 
#[derive(Debug, Clone )]
pub struct Plane  {
    
    pub world: Decomposed<Vector3<f64>,Quaternion<f64>>,
    pub local: Decomposed<Vector3<f64>,Quaternion<f64>>,
    
    
    pub halfheight: f64 ,
    pub halfwidth:f64,
    pub material: Option<materials::MaterialDescType >,
}
impl Plane {
    pub fn new(
        translation: Vector3<f64>,
        direction   :Vector3<f64>,
        halfwidth:f64,
        halfheight: f64,
        
        material: materials::MaterialDescType ,
    ) ->Plane  {
        
        
        let t1 =  Decomposed{
            disp:translation,
            rot :Quaternion::from_arc(Vector3::new(0.0,0.0,1.0), direction, None),
            scale:1.0
        };
         
        Plane  {
            local:t1.inverse_transform().unwrap(),
            world:t1,
            halfheight, // half height
            halfwidth, // half with
            material: Some(material)
        }
    }
    pub fn trafo_to_local(&self, pworld: Point3<f64>) -> Point3<f64> {
        


        self.local.transform_point(pworld)
    }
    pub fn trafo_to_world(&self, plocal: Point3<f64>) -> Point3<f64> {
        self.world.transform_point(plocal)
    }
    pub fn trafo_vec_to_local(&self, pworld: Vector3<f64>) -> Vector3<f64> {
        self.local.transform_vector(pworld)
    }
    pub fn trafo_vec_to_world(&self, plocal: Vector3<f64>) -> Vector3<f64> {
        self.world.transform_vector(plocal)
    }
    pub fn intersect_P(&self, ray :&Ray<f64>) -> bool {
        let origin = self.trafo_to_local(ray.origin);
            let direction = self.trafo_vec_to_local(ray.direction);
        println!("ray : {:?}, {:?}", origin, direction );
        let d = Vector3::new(0.0,0.0,-1.0).dot(direction);
        if d < 0.0{
            println!("no itc ");
            return false
        }
        println!("{:?}",  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) );
        let t =  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) /  d;
        if t < 0.0 {return false}
       let pr =   origin + (t * direction);
      
       if   pr.x.abs() > self.halfwidth || pr.y.abs() > self.halfheight {
        println!("no itc ");
        return false
       }

       let w = self.halfwidth * 2.0;
       let h = self.halfheight * 2.0;
       let u = (pr.x + self.halfwidth) / w;
       let v = (pr.y + self.halfheight) / h;
 
       println!("itc {:?} , u,v {:?}", self.trafo_to_world(pr), (u, v));
       self.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0));
       self.trafo_vec_to_world(Vector3::new(0.0,1.0,0.0));
       self.trafo_vec_to_world(Vector3::new(1.0,0.0,0.0));
       FrameShading::new(
        Vector3::new(0.0,0.0,1.0),
        Vector3::new(1.0,0.0,0.0),
        Vector3::new(0.0,1.0,0.0),
       );
       panic!("-----------------------cambia el frame-----------------------!");

       return true
    }
    pub fn intersect(&self, ray: &Ray< f64>, tmin: f64, tmax:  f64)->Option<HitRecord<f64>>{

        let origin = self.trafo_to_local(ray.origin);
        let direction = self.trafo_vec_to_local(ray.direction);
      //   println!("ray : {:?}, {:?}", origin, direction );
        let d = Vector3::new(0.0,0.0,-1.0).dot(direction);
        if d < 0.0{
          //  println!("no itc ");
            return None
        }
       
        let t =  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) /  d;
        if t < 0.0 { return None }
        if (t  < tmin) ||  (t > tmax) { return None }
        let pr =   origin + (t * direction); 
        if   pr.x.abs() > self.halfwidth || pr.y.abs() > self.halfheight { 
            return None
        }

        let w = self.halfwidth * 2.0;
        let h = self.halfheight * 2.0;
        let u = (pr.x + self.halfwidth) / w;
        let v = (pr.y + self.halfheight) / h;

      //   println!("itc {:?} , u,v {:?}", self.trafo_to_world(pr), (u, v));
      /*
         let frame = FrameShading ::new(
                            cast_to_vf64(self.trafo_vec_to_world(dn)),
                            cast_to_vf64(self.trafo_vec_to_world(dn.cross(du.normalize())).normalize()),
                            cast_to_vf64(self.trafo_vec_to_world(du.normalize() ).normalize()),
                        );
       */
        let n = self.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0));
        let bn = self.trafo_vec_to_world(Vector3::new(-1.0,0.0,0.0));
        let tn =  self.trafo_vec_to_world( Vector3::new(0.0,-1.0,0.0));
        
       
        let frame =  FrameShading::new(n,tn,bn);

         
        let pworld = self.trafo_to_world(pr);
       
        let  mut bsdftype :BsdfType =  match self.material.as_ref(){
           Some(MaterialDescType::NoneType)=>{ materials::BsdfType::None }
            Some(materialDescType)=>{ materialDescType.instantiate(&pworld, &(u,v), &frame)},
            None=> materials::BsdfType::None
            
        };
        
      
       

        return Some(HitRecord {
            t,
            point: pworld,
            normal: frame.n,
            tn: frame.tn,
            bn:  frame.bn,

            u,
             
            v,
            material:bsdftype,
            prev: Some(-ray.direction),
            is_hit_inside: Some(true),
            is_emision:Some(false),
            emission:None,
            medium:None
              
        })
    }


    /**
     * 
     * impl<Scalar: BaseFloat + rand::distributions::uniform::SampleUniform + PartialOrd>
        IntersectionOclussion<Scalar> for Sphere<Scalar>
    {
        type Output = (Scalar, Point3<Scalar>);
        fn intersect_occlusion(
            &self,
            ray: &Ray<Scalar>,
            target: &Point3<Scalar>,
        ) ->  Option<Self::Output> {
            let origin = self.trafo_to_local(ray.origin);
            let direction = self.trafo_vec_to_local(ray.direction);
            let oc = origin;
            let tmin = num_traits::cast::cast::<f64, Scalar>(0.000000001).unwrap();
            let tmax = num_traits::cast::cast::<f64, Scalar>(f64::MAX).unwrap();
            //  println!("{:?}", oc);
            let A = direction.dot(direction);
            let B = oc.dot(direction);
            let C = oc.dot(oc.to_vec()) - self.radius * self.radius; // -1 - self.radius * self.radius;
            let disc = B * B - A * C;
            if disc.to_f64().unwrap() > 0.0 {
                let sqrtd = disc.sqrt();
                let root_a = ((-B) - sqrtd) / A;
                let root_b = ((-B) + sqrtd) / A;
                for root in [root_a, root_b].iter() {
                    if (*root > tmin) && (*root < tmax) {
                        let mut p = origin + direction * (*root);

                        return Some((*root, self.trafo_to_world(p)));
                    }
                }
            }
            None
        }
    } // end interface
     */
    pub fn internal_intersect_occlusion(  &self,ray: &Ray<f64>,target: &Point3<f64>)->Option< (bool, f64, Point3<f64>, Option<MediumOutsideInside>)>{

        let origin = self.trafo_to_local(ray.origin);
        let direction = self.trafo_vec_to_local(ray.direction);
      //   println!("ray : {:?}, {:?}", origin, direction );
        let d = Vector3::new(0.0,0.0,-1.0).dot(direction);
        if d < 0.0{
          //  println!("no itc ");
            return None
        }
        let tmin = 0.00001;
        let tmax = std::f64::MAX;
        let t =  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) /  d;
        if t < 0.0 { return None }
        if (t  < tmin) ||  (t > tmax) { return None }
        let pr =   origin + (t * direction); 
        if   pr.x.abs() > self.halfwidth || pr.y.abs() > self.halfheight { 
            return None
        }
        return Some((true,t, self.trafo_to_world(pr),None));
    }
    pub fn area(&self)->f64{
        (self.halfheight * 2.0) * (self.halfwidth * 2.0)
    }
    pub fn sample(&self, psample : (f64,f64))->(Point3<f64>, Vector3<f64>, f64){
       
        let w = self.halfwidth*2.0;
        let h = self.halfheight*2.0;
        
       let x  =- self.halfwidth  + w * psample.0;
       let y  = -self.halfheight  + h * psample.1;
       
       let v = self.trafo_vec_to_world( Point3::new(x, y, 0.0).to_vec());
       // hacemos la translacion  a mano? esto es absurdo
       let p =  Point3::new(self.world.disp.x + v.x, self.world.disp.y + v.y, self.world.disp.z + v.z);
        
 

       // for testing
       if false {
        fn uniformTriangle(u:&(f64, f64))->(f64, f64){
            let s = u.0.sqrt();
           ( 1.0-s, u.1*s)
        } 
       fn  interpolateTriangle(a:Vector3<f64>, b:Vector3<f64>,c:Vector3<f64>, sample : &(f64, f64))->Point3<f64>{

        
        let r = a * sample.0 +        b * sample.1 +  c * (1.0-sample.0-sample.1);
        Point3::new(r.x, r.y, r.z)
       
       }
       let trisample  = uniformTriangle(&psample);
       let p0 = Vector3::new(-5.0, 1.0, -5.0);
       let p1 = Vector3::new(5.0, 1.0, -5.0 );
       let p2 = Vector3::new(-5.0, 1.0, 5.0 );
       let p  =  interpolateTriangle(p0, p1,p2,&trisample);
       println!("p--->{:?}", p);
       }
    
       (p, self.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0)), 1.0 / 50.0)
      
      
    }

    // pdf 
    pub fn internal_pdf(&self, p : Point3<f64>, phit: &Point3<f64>, wi : &Vector3<f64>)->f64{
        fn AbsDot(v0:Vector3<f64>, v1 : Vector3<f64>){
            v0.dot(v1).abs();

        }
        let normal = self.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0));
      //  let absd =  normal.dot(-wi).abs();
        let absd =  normal.dot( *wi).abs();
        let pdf = (p-phit).magnitude2() /(  absd* self.area());
        pdf

    }

}
// Intersection<Scalar> for Sphere<Scalar>
impl Intersection<f64> for Plane{
    type Output = HitRecord<f64>;
    fn intersect(&self, ray: &Ray<f64>, tmin: f64, tmax: f64) -> Option<Self::Output> {
        self.intersect(ray, tmin, tmax)
        
    }
}
 
impl IntersectionOclussion<f64> for Plane 
{
     
    fn intersect_occlusion(&self, ray: &Ray<f64>,target: &Point3<f64>)->Option<(bool, f64, Point3<f64>, Option<MediumOutsideInside>)>{
        self.internal_intersect_occlusion(ray, target)
    }
}
  
pub trait  PrimitiveIntersection : Intersection<f64> + IntersectionOclussion<f64> {}
impl PrimitiveIntersection for Plane {}
impl PrimitiveIntersection for Sphere<f64> {} 
impl PrimitiveIntersection for Disk {}








#[derive(Debug, Clone )]
pub struct Disk  {
    
    pub world: Decomposed<Vector3<f64>,Quaternion<f64>>,
    pub local: Decomposed<Vector3<f64>,Quaternion<f64>>,
    
    
    pub height: f64 ,
    
    pub radius :f64,
    pub material: Option<materials::MaterialDescType >,
}
impl Disk {
    pub fn new(
        translation: Vector3<f64>,
        direction   :Vector3<f64>,
      height : f64,
        radius :f64, 
        material: materials::MaterialDescType ,
    ) ->Disk {
        
        
        let t1 =  Decomposed{
            disp:translation,
            rot :Quaternion::from_arc(Vector3::new(0.0,0.0,1.0), direction, None),
            scale:1.0
        };
         
        Disk {
            local:t1.inverse_transform().unwrap(),
            world:t1,
             height,
            material: Some(material),
            radius:radius
        }
    }
    pub fn trafo_to_local(&self, pworld: Point3<f64>) -> Point3<f64> {
        


        self.local.transform_point(pworld)
    }
    pub fn trafo_to_world(&self, plocal: Point3<f64>) -> Point3<f64> {
        self.world.transform_point(plocal)
    }
    pub fn trafo_vec_to_local(&self, pworld: Vector3<f64>) -> Vector3<f64> {
        self.local.transform_vector(pworld)
    }
    pub fn trafo_vec_to_world(&self, plocal: Vector3<f64>) -> Vector3<f64> {
        self.world.transform_vector(plocal)
    }
    pub fn intersect_P(&self, ray :&Ray<f64>) -> bool {
    //     let origin = self.trafo_to_local(ray.origin);
    //         let direction = self.trafo_vec_to_local(ray.direction);
    //     println!("ray : {:?}, {:?}", origin, direction );
    //     let d = Vector3::new(0.0,0.0,-1.0).dot(direction);
    //     if d < 0.0{
    //         println!("no itc ");
    //         return false
    //     }
    //     println!("{:?}",  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) );
    //     let t =  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) /  d;
    //     if t < 0.0 {return false}
    //    let pr =   origin + (t * direction);
      
    //    if   pr.x.abs() > self.halfwidth || pr.y.abs() > self.halfheight {
    //     println!("no itc ");
    //     return false
    //    }

    //    let w = self.halfwidth * 2.0;
    //    let h = self.halfheight * 2.0;
    //    let u = (pr.x + self.halfwidth) / w;
    //    let v = (pr.y + self.halfheight) / h;
 
    //    println!("itc {:?} , u,v {:?}", self.trafo_to_world(pr), (u, v));
    //    self.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0));
    //    self.trafo_vec_to_world(Vector3::new(0.0,1.0,0.0));
    //    self.trafo_vec_to_world(Vector3::new(1.0,0.0,0.0));
    //    FrameShading::new(
    //     Vector3::new(0.0,0.0,1.0),
    //     Vector3::new(1.0,0.0,0.0),
    //     Vector3::new(0.0,1.0,0.0),
    //    );
    //    panic!("-----------------------cambia el frame-----------------------!");

       return true
    }
    pub fn intersect(&self, ray: &Ray< f64>, tmin: f64, tmax:  f64)->Option<HitRecord<f64>>{
        // let start = Instant::now();
        let origin = self.trafo_to_local(ray.origin);
        let direction = self.trafo_vec_to_local(ray.direction) ;
      //   println!("ray : {:?}, {:?}", origin, direction );
        let d = Vector3::new(0.0,0.0,-1.0).dot(direction);
        if d < 0.0 {
          //  println!("no itc ");
            return None
        }
       
        let t =  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) /  d;
        if t < 0.0    { return None }
        if (t  < tmin) ||  (t > tmax) { return None }
        let phit =   origin + (t * direction); 
        let l = phit.x * phit.x + phit.y*phit.y;
        if l >self.radius*self.radius{
            return None
        }
        let mut phi = phit.y .atan2(phit.x);
        if phi < 0.0  {  phi +=2.0*std::f64::consts::PI};
       let u  =   phi / std::f64::consts::TAU;
        let dist =  l.sqrt();
        let mut  v =  dist / self.radius;
            v = 1.0 - v;


          
                      
                    
       let du =  Vector3::new(-2.0 * std::f64::consts::PI * phit.y , 2.0 * std::f64::consts::PI * phit.x, 0.0);
        let b =   - self.radius / dist;
        let dv =  Vector3::new(   phit.x *b   ,     phit.y*b, 0.0);
        let mut dn = du.cross(dv).normalize();
        
        let frame = FrameShading ::new(
            self.trafo_vec_to_world(dn),
             self.trafo_vec_to_world(dn.cross(du.normalize())).normalize(),
            self.trafo_vec_to_world(du.normalize() ).normalize(),
        );
     
       

         
        let pworld = self.trafo_to_world( phit);
       
        let  mut bsdftype :BsdfType =  match self.material.as_ref(){
           Some(MaterialDescType::NoneType)=>{ materials::BsdfType::None }
            Some(materialDescType)=>{ materialDescType.instantiate(&pworld, &(u,v), &frame)},
            None=> materials::BsdfType::None
            
        };
        
        // println!(" time: {} nanos", start.elapsed().as_nanos());
        return Some(HitRecord {
            t,
            point: pworld,
            normal: frame.n,
            tn: frame.tn,
            bn:  frame.bn,

            u,
             
            v,
            material:bsdftype,
            prev: Some(-ray.direction),
            is_hit_inside: Some(true),
            is_emision:Some(false),
            emission:None,
            medium:None
              
        })
    }
 
    pub fn internal_intersect_occlusion(  &self,ray: &Ray<f64>,target: &Point3<f64>)->Option< (bool, f64, Point3<f64>, Option<MediumOutsideInside>)>{

        let origin = self.trafo_to_local(ray.origin);
        let direction = self.trafo_vec_to_local(ray.direction);
      
      
         let mut d = Vector3::new(0.0,0.0,-1.0).dot(direction) ;
        
         if d < 0.0{ 
            return None;
          }
          let tmin = 0.00001;
        let tmax = std::f64::MAX;
         let t =  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) /  d;


     
        if t < 0.0 { return None }
        if (t  < tmin) ||  (t > tmax) { return None }
          let phit =   origin + (t * direction); 
          let l = phit.x * phit.x + phit.y*phit.y;
          if l >self.radius{
              return None
          }
         return Some((true,t, self.trafo_to_world(phit),None));
    }
    pub fn area(&self)->f64{
        (self.radius * self.radius  *std::f64::consts::PI  )
    }
    pub fn sample(&self, psample : (f64,f64))->(Point3<f64>, Vector3<f64>, f64){
        let csample = ConcentricSampleDisk(&psample); 
       let n = self.trafo_vec_to_world( Point3::new(0.0,0.0, 1.0).to_vec());
       
       let p =  Point3::new(self.radius * csample.0 , self.radius * csample.1, 0.0);  
        

       (self.trafo_to_world(p), self.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0)), 1.0/self.area())
      
      
    }

    // pdf 
    pub fn internal_pdf(&self, p : Point3<f64>, phit: &Point3<f64>, wi : &Vector3<f64>)->f64{
        fn AbsDot(v0:Vector3<f64>, v1 : Vector3<f64>){
            v0.dot(v1).abs();

        }
        let normal = self.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0));
      //  let absd =  normal.dot(-wi).abs();
        let absd =  normal.dot( *wi).abs();
        let pdf = (p-phit).magnitude2() /(  absd* self.area());
        pdf

    }

}





// Intersection<Scalar> for Sphere<Scalar>
impl Intersection<f64> for Disk{
    type Output = HitRecord<f64>;
    fn intersect(&self, ray: &Ray<f64>, tmin: f64, tmax: f64) -> Option<Self::Output> {
        self.intersect(ray, tmin, tmax)
        
    }
}
 
impl IntersectionOclussion<f64> for Disk
{
     
    fn intersect_occlusion(&self, ray: &Ray<f64>,target: &Point3<f64>)->Option<(bool, f64, Point3<f64>, Option<MediumOutsideInside>)>{
        self.internal_intersect_occlusion(ray, target)
    }
}
  












 
#[derive(Debug, Clone )]
 pub struct Bounds3f{
    pub pmin:Point3f,
   pub pmax:Point3f
 }
impl Bounds3f {
    pub fn new(pmin:Point3f,pmax:Point3f)->Self{
        Bounds3f{pmin, pmax}
    }
    pub fn  intersect(&self, ray: &Ray< f64>, tmin: f64, tmax:  f64)->(bool,f64, f64) {
        let mut t0= tmin;
        let mut t1=tmax;

        for component in 0..3 {
            let inv = 1.0/ ray.direction[component];
            let mut  tn = (self.pmin[component]-ray.origin[component]) * inv;
            let mut  tf = (self.pmax[component]-ray.origin[component]) * inv;
            if tn>tf {std::mem::swap(&mut tn, &mut  tf)}
     
            if tn > t0 {
               t0 = tn; 
            } 
            if tf < t1 {
                t1 = tf;
            }
             if t0>t1{return  (false, std::f64::INFINITY, -std::f64::INFINITY);}
        }
        (true, t0,t1)
    }
}
#[derive(Debug, Clone )]
pub struct Cylinder  {
    
    pub world: Decomposed<Vector3<f64>,Quaternion<f64>>,
    pub local: Decomposed<Vector3<f64>,Quaternion<f64>>,
    
    
    pub minh: f64 ,
    pub maxh: f64 ,
    pub radius :f64,
    pub material: Option<materials::MaterialDescType >,
    pub worldbound3f : Option<Bounds3f>,
    pub objectbound3f : Bounds3f,
    
}
impl Cylinder {
    pub fn new(
        translation: Vector3<f64>,
        rot   :Matrix3<f64>,
        minh: f64 ,
   maxh: f64 ,
        radius :f64, 
        material: materials::MaterialDescType ,
    ) ->Cylinder {
        
        
        let t1 =  Decomposed{
            disp:translation,
            rot :Quaternion::from(rot),
            scale:1.0
        };
       let objecspacetbound =  Bounds3f{
            pmin: Point3 { x : -radius, y:-radius,z: minh } ,
            pmax: Point3 { x : radius, y:radius,z: maxh } ,
        };
       let worldbound =  Self::compute_world_bounds(t1,   objecspacetbound.clone());
      
        
         
        Cylinder{
            local:t1.inverse_transform().unwrap(),
            world:t1,
             minh,
             maxh,
            material: Some(material),
            radius:radius,
            objectbound3f:  objecspacetbound,
            worldbound3f:Some(worldbound)
        }
    }
    pub fn world_bound(&self)->Bounds3f{
        if self.worldbound3f.is_some() {return  self.worldbound3f.clone(). unwrap()}

        let p0 = self.trafo_to_world(Point3f::new(self.objectbound3f.pmin.x, self.objectbound3f.pmin.y, self.objectbound3f.pmin.z));
        let p1 = self.trafo_to_world(Point3f::new(self.objectbound3f.pmax.x, self.objectbound3f.pmin.y, self.objectbound3f.pmax.z));
        let p2 =self.trafo_to_world( Point3f::new(self.objectbound3f.pmin.x, self.objectbound3f.pmin.y, self.objectbound3f.pmin.z));
        let p3 = self.trafo_to_world(Point3f::new(self.objectbound3f.pmax.x, self.objectbound3f.pmin.y, self.objectbound3f.pmax.z));
        let p4 = self.trafo_to_world(Point3f::new(self.objectbound3f.pmin.x, self.objectbound3f.pmax.y, self.objectbound3f.pmin.z));
        let p5 = self.trafo_to_world(Point3f::new(self.objectbound3f.pmax.x, self.objectbound3f.pmax.y, self.objectbound3f.pmax.z));
        let p6 = self.trafo_to_world(Point3f::new(self.objectbound3f.pmin.x, self.objectbound3f.pmax.y, self.objectbound3f.pmin.z));
        let p7 =self.trafo_to_world( Point3f::new(self.objectbound3f.pmax.x, self.objectbound3f.pmax.y, self.objectbound3f.pmax.z));
      let r = vec![p0,p1, p2, p3, p4, p5, p6, p7];
      let pminworld =  r.clone().into_iter().reduce(|a, b| Point3f::new(a.x.min(b.x), a.y.min(b.y),a.z.min(b.z) ));
      let pmaxworld =  r.into_iter().reduce(|a, b| Point3f::new(a.x.max(b.x), a.y.max(b.y),a.z.max(b.z) ));
        Bounds3f{
            pmin:  pminworld.unwrap(),
            pmax:  pmaxworld.unwrap(),
        }
      
    }
    pub fn compute_world_bounds( trafo : Decomposed<Vector3<f64>, Quaternion<f64>>, objectbound3f : Bounds3f)->Bounds3f{

        let p0 = trafo.transform_point(Point3f::new(objectbound3f.pmin.x, objectbound3f.pmin.y, objectbound3f.pmin.z));
        let p1 = trafo.transform_point(Point3f::new(objectbound3f.pmax.x, objectbound3f.pmin.y, objectbound3f.pmax.z));
        let p2 =trafo.transform_point( Point3f::new(objectbound3f.pmin.x, objectbound3f.pmin.y, objectbound3f.pmin.z));
        let p3 = trafo.transform_point(Point3f::new(objectbound3f.pmax.x, objectbound3f.pmin.y, objectbound3f.pmax.z));
        let p4 = trafo.transform_point(Point3f::new(objectbound3f.pmin.x, objectbound3f.pmax.y, objectbound3f.pmin.z));
        let p5 = trafo.transform_point(Point3f::new(objectbound3f.pmax.x, objectbound3f.pmax.y, objectbound3f.pmax.z));
        let p6 = trafo.transform_point(Point3f::new(objectbound3f.pmin.x, objectbound3f.pmax.y, objectbound3f.pmin.z));
        let p7 =trafo.transform_point( Point3f::new(objectbound3f.pmax.x, objectbound3f.pmax.y, objectbound3f.pmax.z));
     let r = vec![p0,p1, p2, p3, p4, p5, p6, p7];
     let pminworld =  r.clone().into_iter().reduce(|a, b| Point3f::new(a.x.min(b.x), a.y.min(b.y),a.z.min(b.z) ));
     let pmaxworld =  r.into_iter().reduce(|a, b| Point3f::new(a.x.max(b.x), a.y.max(b.y),a.z.max(b.z) ));
        Bounds3f{
            pmin:  pminworld.unwrap(),
            pmax:  pmaxworld.unwrap(),
        }
      
    }
    pub fn area(&self)->f64{
       ( self.maxh - self.minh) *(self.radius  *2.0*std::f64::consts::PI  )
    }
    pub fn sample(&self, psample : (f64,f64))->(Point3<f64>, Vector3<f64>, f64){
         let h = lerp(psample.0, self. minh, self.maxh);
         let phi = 2.0f64*std::f64::consts::PI * psample.1;
       let sc =  phi.sin_cos();
      let p = Point3f::new( sc.1*self.radius ,sc.0*self.radius,h);
      
        let r = (p.x * p.x + p.y *p.y ).sqrt();
        let rt = self.radius / r;
        (
            self.trafo_to_world(Point3f::new( p.x*rt,p.y*rt,p.z)),
            self.trafo_vec_to_world(Vector3f::new(sc.1*self.radius ,sc.0*self.radius,0.0)),
            1.0/self.area()

        )
       
      
    }
    pub fn trafo_to_local(&self, pworld: Point3<f64>) -> Point3<f64> {
        


        self.local.transform_point(pworld)
    }
    pub fn trafo_to_world(&self, plocal: Point3<f64>) -> Point3<f64> {
        self.world.transform_point(plocal)
    }
    pub fn trafo_vec_to_local(&self, pworld: Vector3<f64>) -> Vector3<f64> {
        self.local.transform_vector(pworld)
    }
    pub fn trafo_vec_to_world(&self, plocal: Vector3<f64>) -> Vector3<f64> {
        self.world.transform_vector(plocal)
    }
    /**
     * 
     * NOTA NOTA:
     *  el cilindro esta puesto sobre z es decir z atraviesa las dos tapas
     */
    pub fn intersect(&self, ray: &Ray< f64>, tmin: f64, tmax:  f64)->Option<HitRecord<f64>>{

        let origin = self.trafo_to_local(ray.origin);
        let direction = self.trafo_vec_to_local(ray.direction) ;

       
  
        // d.x^2 + d.y^2
       let A =  direction.x*direction.x + direction.y*direction.y;
        // 2*dot(d,o) 
        let B = 2.0* (direction.x * origin.x + direction.y*origin.y);
        // dot(o,o) - r^2 
        let C =  origin.x * origin.x + origin.y*origin.y - self.radius*self.radius;


// resolve quadratic. diferent method from above
        let disc = B * B -  4.0 * A * C;
        if disc<0.0 {return None};
        let sqrtd = disc.sqrt();
        let q : f64;
        if B < 0.0 {
            q = -0.5 * (B-sqrtd);
        }else{
             q =  -0.5 * (B+sqrtd);
        }
        let mut root_a = q / A;
        let mut  root_b = C / q;
        if root_a>root_b{ std::mem::swap(&mut root_a, &mut root_b);}

        


        for mut root in [root_a, root_b].iter() {

            if *root >= tmin && *root < tmax {
               let mut p = origin + direction * (*root);
               let hit = (p.x * p.x + p.y *p.y).sqrt();
               p.x*= self.radius / hit ;
               p.y*=  self.radius / hit ;
               let mut  phi = (p.y / p.x ).atan( ); 
               if phi < 0.0 {    phi += (2.0f64 * std::f64::consts::PI); }


                //taps
                
                if  p.z < self.minh || p.z > self.maxh {
                    if *root == root_b {return  None;}
                     root = &root_b;
                     if root_b > tmax {return  None;}
                    
                    p = origin + direction * (*root);
                    let hit = (p.x * p.x + p.y *p.y).sqrt();
                    p.x*= self.radius / hit ;
                    p.y*=  self.radius / hit ;
                    phi = (p.y / p.x ).atan( ); 
                    if phi < 0.0 {    phi += (2.0f64 * std::f64::consts::PI); }
                    if  p.z < self.minh || p.z > self.maxh {return  None;}
                }


                let u = phi / (2.0f64 * std::f64::consts::PI);
                let v = (p.z - self.minh)/(self.maxh-self.minh);
                let du = Vector3f::new(- std::f64::consts::PI * 2.0  * p.y , std::f64::consts::PI * 2.0 *p.x,0.0 );
                let dv = Vector3f::new( 0.0 , 0.0 , (self.maxh-self.minh) );

              
                let mut dn = du.cross(dv).normalize();
                let frame = FrameShading ::new(
                    self.trafo_vec_to_world(dn),
                     self.trafo_vec_to_world(dn.cross(du.normalize())).normalize(),
                    self.trafo_vec_to_world(du.normalize() ).normalize(),
                );
                let pworld = self.trafo_to_world( p);


                let  mut bsdftype :BsdfType =  match self.material.as_ref(){
                    Some(MaterialDescType::NoneType)=>{ materials::BsdfType::None }
                     Some(materialDescType)=>{ materialDescType.instantiate(&pworld, &(u,v), &frame)},
                     None=> materials::BsdfType::None
                     
                 };
                
                return Some(HitRecord {
                    t:*root,
                    point: pworld,
                    normal: frame.n,
                    tn: frame.tn,
                    bn:  frame.bn,
        
                    u,
                     
                    v,
                    material:bsdftype,
                    prev: Some(-ray.direction),
                    is_hit_inside: Some(true),
                    is_emision:Some(false),
                    emission:None,
                    medium:None
              
                })
            }
          

        }
        None
    }

    pub fn internal_intersect_occlusion(  &self,ray: &Ray<f64>,target: &Point3<f64>)->Option< (bool, f64, Point3<f64>, Option<MediumOutsideInside>)>{
        let tmin = 0.00001;
        let tmax = std::f64::MAX;
           let origin = self.trafo_to_local(ray.origin);
        let direction = self.trafo_vec_to_local(ray.direction) ;

       
  
        // d.x^2 + d.y^2
       let A =  direction.x*direction.x + direction.y*direction.y;
        // 2*dot(d,o) 
        let B = 2.0* (direction.x * origin.x + direction.y*origin.y);
        // dot(o,o) - r^2 
        let C =  origin.x * origin.x + origin.y*origin.y - self.radius*self.radius;


        let disc = B * B -  4.0 * A * C;
        if disc<0.0 {return None};
        let sqrtd = disc.sqrt();
        let q : f64;
        if B < 0.0 {
            q = -0.5 * (B-sqrtd);
        }else{
             q =  -0.5 * (B+sqrtd);
        }
        let mut root_a = q / A;
        let mut  root_b = C / q;
        if root_a>root_b{ std::mem::swap(&mut root_a, &mut root_b);}



        
        for mut root in [root_a, root_b].iter() {

            if *root >= tmin && *root < tmax {
               let mut p = origin + direction * (*root);
               let hit = (p.x * p.x + p.y *p.y).sqrt();
               p.x*= self.radius / hit ;
               p.y*=  self.radius / hit ;
               let mut  phi = (p.y / p.x ).atan( ); 
               if phi < 0.0 {    phi += (2.0f64 * std::f64::consts::PI); }


                //taps
                
                if  p.z < self.minh || p.z > self.maxh {
                    if *root == root_b {return  None;}
                     root = &root_b;
                     if root_b > tmax {return  None;}
                    
                    p = origin + direction * (*root);
                    let hit = (p.x * p.x + p.y *p.y).sqrt();
                    p.x*= self.radius / hit ;
                    p.y*=  self.radius / hit ;
                    phi = (p.y / p.x ).atan( ); 
                    if phi < 0.0 {    phi += (2.0f64 * std::f64::consts::PI); }
                    if  p.z < self.minh || p.z > self.maxh {return  None;}
                }
                let pworld = self.trafo_to_world( p);
                return Some((true,*root,pworld, None));
            }
        }
        None
    }
   
}




impl IntersectionOclussion<f64> for Cylinder
{
     
    fn intersect_occlusion(&self, ray: &Ray<f64>,target: &Point3<f64>)->Option<(bool, f64, Point3<f64>, Option<MediumOutsideInside>)>{
        self.internal_intersect_occlusion(ray, target)
    }
}
  







#[test]
fn test_cylinder_and_box(){
 
   
    let numitec=100;
    let mat3rot =  Matrix3::from_angle_x(Deg(0.0));
    let  c = Cylinder::new(Vector3f::new(0.0, 4.0,  0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType);
 
    let mut ps : Point3f;
    let bf3 = c.world_bound();

    if(false){
        
         let mut vdot = 0.0;
        let mut cnt = 0;
        let mut sumuv = 0.0;
        println!("{:?}", bf3);

            let start = Instant::now();
            for i in 0..numitec{
                let fi =  2.0*(i as f64 / (numitec ) as f64) -1.0;
            let sc = (fi * 2.0 * std::f64::consts::PI).sin_cos();
                let r = Ray::new(Point3f::new(0.0,0.0,0.0), Vector3f::new(0.0,1.0, 0.0).normalize());
                let hitbox = c.world_bound().intersect(&r.clone(), 0.00001, std::f64::MAX);
                let hits =  c.intersect(&r, 0.00001, std::f64::MAX);
        
                if let Some(h) = hits{
                    println!("hit  {} p {:?} uv {},{} hitbox {}",i, h.point, h.u,h.v,hitbox.0);
                vdot+= h.normal.dot(Vector3f::new(1.0,1.0,1.0)) + h.bn.dot(Vector3f::new(1.0,1.0,1.0)) +  h.tn.dot(Vector3f::new(1.0,1.0,1.0));
                    sumuv +=  h.u+h.v;
                    cnt+=1;
                }else{
                    println!("no hit {} hit box {}",i, hitbox.0);
                }
            
            
            
            }
        println!("{} {} dots: {}", cnt, sumuv, vdot);
        println!(" time: {} seconds", start.elapsed().as_secs_f64());
    }

    if false{
        
        // oclusion
        let scene  = Scene1::make_scene(
            8,8 ,
            vec![], 
            vec![PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 4.0,  0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType))], 
            1,1);
        let r = Ray::new(Point3f::new(0.0,0.0,0.0), Vector3f::new(0.0,1.0, 0.0).normalize());
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,20.0,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwrap must be true");
        }
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,1.0,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwrap must be false");
        }
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,3.0-0.0001,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwrap must be false");
        }
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,3.0+0.0001,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwrap must be true");
        }

    }
    
    use crate::Lights::Light;
    use crate::Lights::PdfIllumination1;
   if false {
    let scene  = Scene1::make_scene(
        8,8 ,
        vec![
            Light::New_Interface_AreaLightType(
                New_Interface_AreaLight ::new(  
                    Srgb::new(1.0,1.0,1.0),
                    PrimitiveType::CylinderType(  
                        Cylinder::new(Vector3::new( 0.0000,4.0,0.0000),Matrix3::from_angle_x(Deg( 0.0)),-1.0, 1.0, 1.0,MaterialDescType::NoneType)),1))

        ], 
        vec![
            PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.0,10.0,0.0), 1.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,0.0)))),
            PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 4.0,  0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType))], 
        1,1);
        let r = Ray::new(Point3f::new(0.0,0.0,0.0), Vector3f::new(0.0,1.0, 0.0).normalize());
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,20.0,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwraphay occlusion");
        }
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,0.30,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwrap  hay occlusion");
        }

        let r = Ray::new(Point3f::new(0.0,8.0,0.0), Vector3f::new(0.0,1.0, 0.0).normalize());
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,20.0,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwrap   hay occlusion");
        }

        let r = Ray::new(Point3f::new(0.0,18.0,0.0), Vector3f::new(0.0,1.0, 0.0).normalize());
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,20.0,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwrap NO hay occlusion");
        }

        let r = Ray::new(Point3f::new(0.0,0.0,0.0), Vector3f::new(0.0,0.0, 1.0).normalize());
        let isocclusion =    new_interface_for_intersect_occlusion(&r, &scene, &Point3f::new(0.0,20.0,0.0));
        if isocclusion.unwrap().0{
            println!("isocclusion.unwrap NO hay occlusion");
        }
 
   }

    if false{
        // sample interface test
    let mat3rot =  Matrix3::from_angle_x(Deg(-90.0));
    let c= PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 4.0,  0.0), mat3rot,-1.0, 1.0,2.0, materials::MaterialDescType::NoneType));

    for i in 0..32{
            let  xs = i as f64/ 32.0;
            let psamplecl =  c.sample((xs, xs));
            println!("{} {:?} {:?} {:?}", i, psamplecl.0, psamplecl.1, psamplecl.2);
    
    }
    println!(" ");
    }
  

//    let scene  = Scene1::make_scene(
//     8,8 ,
//     vec![], 
//     vec![
//         // PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.0,10.0,0.0), 1.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,0.0)))),
//         PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 4.0,  0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType)),
//         PrimitiveType::DiskType( Disk::new(Vector3::new( 0.00001, 0.0000,0.00001),Vector3::new(0.0, 1.0, 0.0),0.0, 1.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00)))) 
//         ]
    
        
//         , 
//     1,1);
//      let r = Ray::new(Point3f::new(0.0,0.1,0.0), Vector3f::new(0.0,-1.0, 0.0).normalize());
//      let hit=new_interface_for_intersect_scene(&r, &scene);
//      if let Some(h)=hit {
//         let recout =  scene.lights[0].sampleIllumination(&RecordSampleLightIlumnation::new((0.5, 0.5), h));
         
//      }

if true {
        let mat3rot =  Matrix3::from_angle_x(Deg(0.0));
        let scene  = Scene1::make_scene(
        8,8 ,
        vec![ 

        Light::New_Interface_AreaLightType(
            New_Interface_AreaLight ::new(  
                Srgb::new(1.0,1.0,1.0),
                PrimitiveType::CylinderType(  
                    Cylinder::new(Vector3::new( 0.0000,4.0,0.0000),Matrix3::from_angle_x(Deg( 0.0)),-1.0, 1.0, 1.0,MaterialDescType::NoneType)),1))
        ], 
        vec![
            PrimitiveType::DiskType( Disk::new(Vector3::new( 0.00001, 0.0000,0.00001),Vector3::new(0.0, 1.0, 0.0),0.0, 1.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00)))) ,
            // PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 0.0,  0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType))
            ], 
        1,1);
        
        let r = Ray::new(Point3f::new(0.0,0.0,0.0), Vector3f::new(0.0,0.0, -1.0).normalize());
        if let Some(h) = new_interface_for_intersect_scene(&r, &scene){
            println!("no hay intersection hit  {:?}   uv {},{}  ",h.point, h.u,h.v);
        }





        // giro 90 grados sobre x

        let mat3rot =  Matrix3::from_angle_x(Deg(-90.0));
        let scene  = Scene1::make_scene(
        8,8 ,
        vec![], 
        vec![
            PrimitiveType::CylinderType(Cylinder::new(Vector3f::new(0.0, 0.0,  0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType))], 
        1,1);
       
        let r = Ray::new(Point3f::new(0.0,0.0,0.0), Vector3f::new(0.0,0.0, 1.0).normalize());
        if let Some(h) = new_interface_for_intersect_scene(&r, &scene){
            println!("sihay intersection hit  {:?}   uv {},{}  ",h.point, h.u,h.v);
        }

    }






}








#[test]
fn test_cylinder_and_samples(){
    let numitec=32;
for i in 0..numitec{
    let fi =  (i as f64 / (numitec ) as f64) ;
    let mat3rot =  Matrix3::from_angle_z(Deg(40.0));
    
    let  c = Cylinder::new(Vector3f::new(0.0, 0.0, 0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType);
    
    
    let interaction =  c.sample((fi,fi));

    println!("i {:?}", i);
    println!("  p {:?}", interaction.0);
    println!("  n {:?}", interaction.1);
    println!("  pdf {:?}", interaction.2);

}

}


// #[test]
// fn test_cylinder_and_rotations(){
//     let mut cnt = 0;
// //rot cylinder
// let numitec=32;
// for i in 0..numitec{
//     let fi = 2.0 *(i as f64 / (numitec ) as f64)-1.0;
//     let mat3rot =  Matrix3::from_angle_x(Deg(-90.0));
//    let v =  mat3rot * Vector3f::new(0.0,0.0,1.0);
//     let  c = Cylinder::new(Vector3f::new(0.0, 0.0, 0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType);

//     // println!("{:?}", i);
//     if let Some(hit) = c.intersect(&Ray::<f64>{is_in_medium: false,origin:Point3::new(3.5, 0.99000,0.0), direction:Vector3::new(-1.0,0.0, 0.0).normalize()},0.00001, f64::MAX){
//           println!("{:?}", hit.point);
//         println!("{:?}", hit.normal);
//         println!("{:?}", hit.bn);
//         println!("{:?}", hit.tn);
        
//         println!("{:?} {:?} ", hit.u, hit.v);
//         cnt+=1;
//     }else{
//         println!("no intersection");
//     }
// }

// }
 






// #[test]
// fn test_cylinder(){
//     let mat3rot =  Matrix3::from_angle_z(Deg(90.0));
//     let  c = Cylinder::new(Vector3f::new(0.0, 0.0, 0.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType);
//     let mut cnt = 0;
//     let b = Vector3f::new(1.0,1.0,1.0);
     
//     // for i in 0..32{
//     //     let fi = 2.0 *(i as f64 / 32.0)-1.0;
//     //     if let Some(hit) = c.intersect(&Ray::<f64>{origin:Point3::new(0.0, 2.0,fi), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){
//     //         println!("{:?}", hit.point);
//     //         println!("{:?}", hit.normal);
//     //         println!("{:?}", hit.tn);
//     //         println!("{:?}", hit.bn);
//     //         println!("{:?} {:?} ", hit.u, hit.v);
//     //         cnt+=1;
//     //     }else{
//     //         println!("no intersection");
//     //     }
//     // }
//     let numitec=100;
//     for i in 0..numitec{
//         let fi = 2.0 *(i as f64 / numitec as f64)-1.0;
//         // println!("{:?}", i);
//         if let Some(hit) = c.intersect(&Ray::<f64>{ is_in_medium: false,
//             origin:Point3::new(-2.0, 0.1,fi), direction:Vector3::new(1.0,0.0, 0.0).normalize()},0.00001, f64::MAX){
//             // println!("{:?}", hit.point);
//             // println!("{:?}", hit.normal);
//             // println!("{:?}", hit.bn);
//             // println!("{:?}", hit.tn);
            
//             // println!("{:?} {:?} ", hit.u, hit.v);
//             cnt+=1;
//         }else{
//             println!("no intersection");
//         }
//     }
//     // mov ray
//     let numitec=32;
//     for i in 0..numitec{
//         let fi = 2.0 *(i as f64 / numitec as f64)-1.0;
//         // println!("{:?}", i);
//         if let Some(hit) = c.intersect(&Ray::<f64>{
//             is_in_medium: false,
//             origin:Point3::new(0.0, 0.0,fi), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){
//             //   println!("{:?}", hit.point);
//             // println!("{:?}", hit.normal);
//             // println!("{:?}", hit.bn);
//             // println!("{:?}", hit.tn);
            
//             // println!("{:?} {:?} ", hit.u, hit.v);
//             cnt+=1;
//         }else{
//             // println!("no intersection");
//         }
//     }
    
//     println!("{}", cnt);
   




// }









// #[test]
// fn test_Disk(){
     
//     let p = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0, 1.0),0.0, 1.0, MaterialDescType::NoneType);
//     if let Some(t) = p.intersect(& Ray::<f64>{
//         is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(9.0,9.0, -1.0).normalize()},0.00001, f64::MAX){
//         println!("{:?}",t);
//         assert!(false)
//     }
//     if let Some(t) = p.intersect(& Ray::<f64>{
//         is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(9.0,9.0, -1.0).normalize()},0.00001, f64::MAX){
//         println!("{:?}",t);
//         assert!(false)
//     }
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(-19.0,-19.0, -1.0).normalize()},0.00001, f64::MAX){
//         println!("{:?}",t);
//         assert!(false)
//     }
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,-1.0), direction:Vector3::new(-19.0,-19.0, -1.0).normalize()},0.00001, f64::MAX){
//         println!("{:?}",t);
//         assert!(false)}
    
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,-1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,0.0), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     let p = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0, 1.0),0.0, 1.0, MaterialDescType::NoneType);
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(1.0,1.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,0.50), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     // traslation
//     let p = Disk::new(Vector3::new(0.0,0.0,10.0),Vector3::new(0.0,0.0, 1.0),0.0, 1.0, MaterialDescType::NoneType);
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,10.50), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     let p = Disk::new(Vector3::new(0.0,0.0,10.0),Vector3::new(0.0,0.0, -1.0),0.0, 1.0, MaterialDescType::NoneType);
//     if let None= p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let None = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,10.50), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,0.0,10.50), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}

//     let p = Disk::new(Vector3::new(0.0,0.0,10.0),Vector3::new(0.0,0.0, -1.0),0.0, 1.0, MaterialDescType::NoneType);
//     if let None = p.intersect(& Ray::<f64> {is_in_medium: false,origin:Point3::new(0.0,1.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let None = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,2.10, 1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let None = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,1.0,10.50), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let Some(t) = p.intersect(& Ray::<f64>{ is_in_medium: false,origin:Point3::new(0.0,1.0,10.50), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}



//      // rotation
//      let p = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,1.0, 0.0),0.0, 1.0, MaterialDescType::NoneType);
//      if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,1.0,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}
//      if let None = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,1.0,0.0), direction:Vector3::new(0.0,1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}

//      // rotation + translation
//      let p = Disk::new(Vector3::new(0.0,1.0,0.0),Vector3::new(0.0,1.0, 0.0),0.0, 1.0, MaterialDescType::NoneType);
//      if let None = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.90,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}
//      if let None = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.990,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}
//      if let None = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.99990,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}
//      if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,1.01,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){
//         println!("true : {:?}",t.point);
//         assert!(true)}
//      if let  Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,1.001,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){
//         println!("true : {:?}",t.point);
//         assert!(true)}
//      if let  Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,1.0001,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){
//         println!("true : {:?}",t.point);
//         assert!(true)}

//         let p = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0, 1.0),0.0, 1.0, MaterialDescType::NoneType);
//         for i in 0..32{
//             let ps =  (i as f64 / 32.0);
//             println!(" i -------{:?}",i );
//             if let  Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(ps +0.0010   , ps+0.0010 ,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){
//                 println!("p {:?}", t.point);
//                 println!("uv {:?}", (t.u, t.v));
//                 println!("dpdu {:?}", t.bn );
//                 println!("dpdv {:?}", t.tn);


//             }
//             let rsample =  p.sample( (ps, ps));
//             println!("  p sample{:?}", rsample.0);
//             // println!("p normal {:?}",rsample.1 );
//             println!(   "pdf {:?}", rsample.2 );
//             println!("  " );

//         } // end for
    
// }





















// #[test]
// fn test_plane(){
    
 
//     let p =  Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0,1.0),10.0, 10.0,MaterialDescType::PlasticType(Plastic::default()));
//     if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(9.0,9.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(9.0,9.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//     if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(-19.0,-19.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,-1.0), direction:Vector3::new(-19.0,-19.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    
//     if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,-1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//     if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,0.0), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}

    
    

   
//   //   translaction 
//         let p =  Plane::new(Vector3::new(0.0,0.0,11.0),Vector3::new(0.0,0.0,1.0),10.0, 10.0,MaterialDescType::PlasticType(Plastic::default()));
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,11.1000), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)};
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,11.01000), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,11.001000), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)};
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,11.0001000), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)};

//         let p =  Plane::new(Vector3::new(0.0,0.0,11.0),Vector3::new(0.0,0.0,-1.0),10.0, 10.0,MaterialDescType::PlasticType(Plastic::default()));
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,10.1000), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)};
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,10.01000), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,10.001000), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)};
//         if let Some(t) = p.intersect(& Ray::<f64>{is_in_medium: false,origin:Point3::new(0.0,0.0,10.0001000), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)};

// //    //  rotatation
//     let p =  Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,1.0,1.0).normalize(),10.0, 10.0,MaterialDescType::PlasticType(Plastic::default()));
//     for i in 0..32{
//         let dir =  (i as f64 / 32.0) * std::f64::consts::PI;
//         println!("{:?}", dir.sin_cos().0);
//         println!("{:?}",  Vector3::new( dir.sin_cos().0 , 0.0, 1.0).normalize());
//         println!("{:?}",  Point3::new( dir.sin_cos().0*10.0 , 0.0, 1.0*10.0) );
//         let ptray = Point3::new( dir.sin_cos().0*10.0 , 0.0, 1.0*10.0);
//         let ray = Ray::<f64>::new(ptray ,  -Vector3::new( dir.sin_cos().0 , 0.0, 1.0).normalize());
//         let p =  Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new( dir.sin_cos().0 , 0.0, 1.0).normalize(),10.0, 10.0,MaterialDescType::NoneType);
//         if let Some(t) =   p.intersect(&ray, 0.0001,std::f64::MAX){assert!(true)};
      
//         let pstatic =  Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new( 0.0,0.0, 1.0).normalize(),10.0, 10.0,MaterialDescType::NoneType);
//        match  pstatic.intersect(&ray, 0.0001,std::f64::MAX){
//          Some(h)=>assert!(true), 
//          None =>assert!(false)
//        };
        
        
//     }
     
//     //sample method

//     let h = 10.0;
//     let w = 10.0;
//     // let p = Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0,1.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
//     // println!("{:?}", p.sample((0.0,0.0)));
//     // println!("{:?}", p.sample((0.5,0.5)));
//     // println!("{:?}", p.sample((1.0,1.0)));
//     // let p = Plane::new(Vector3::new(0.0,0.0,10.0),Vector3::new(0.0,0.0,1.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
//     // println!("{:?}", p.sample((0.0,0.0)));
//     // println!("{:?}", p.sample((0.5,0.5)));
//     // println!("{:?}", p.sample((1.0,1.0)));


//     let p = Plane::new(Vector3::new(110.0,111.0,110.0),Vector3::new(0.0,1.0,0.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
 
//     assert!( p.sample((0.0,0.0)).0.x == 110.0 - w/2.0);
//     assert!( p.sample((0.5,0.5)).0.x == 110.0  );
//     assert!(p.sample((1.0,1.0)).0.x == 110.0 + w/2.0);
//     assert!( p.sample((0.0,0.0)).0.z == 110.0 + h/2.0);
//     assert!(p.sample((0.5,0.5)).0.z == 110.0  );
//     assert!( p.sample((1.0,1.0)).0.z == 110.0 - h/2.0);

//     assert!( p.sample((0.0,0.0)).0.y == 111.0  );
//     assert!(p.sample((0.5,0.5)).0.y == 111.0  );
//     assert!( p.sample((1.0,1.0)).0.y == 111.0  );

//     assert!( p.sample((1.0,1.0)).1.z ==  1.0  ); 
//     println!("{:?}", p.sample((0.0,0.0)).0.x == 110.0 - w/2.0);
//     println!("{:?}", p.sample((0.5,0.5)).0.x == 110.0  );
//     println!("{:?}", p.sample((1.0,1.0)).0.x == 110.0 + w/2.0);
//     println!("{:?}", p.sample((0.0,0.0)).0.z == 110.0 + h/2.0);
//     println!("{:?}", p.sample((0.5,0.5)).0.z == 110.0  );
//     println!("{:?}", p.sample((1.0,1.0)).0.z == 110.0 - h/2.0);

//     println!("{:?}", p.sample((0.0,0.0)).0.y == 111.0  );
//     println!("{:?}", p.sample((0.5,0.5)).0.y == 111.0  );
//     println!("{:?}", p.sample((1.0,1.0)).0.y == 111.0  );


    
//     let p = Plane::new(Vector3::new(110.0,110.0,110.0),Vector3::new(0.0,0.0,1.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
//     assert_abs_diff_eq!(p.sample((0.0,0.0)).1 , Vector3::new(0.0,0.0,1.0));
 
//     assert!( p.sample((0.0,0.0)).0.x == 110.0 - w/2.0);
//     assert!( p.sample((0.5,0.5)).0.x == 110.0  );
//     assert!(p.sample((1.0,1.0)).0.x == 110.0 + w/2.0);
//     assert!( p.sample((0.0,0.0)).0.y == 110.0  - h/2.0);
//     assert!(p.sample((0.5,0.5)).0.y == 110.0  );
//     assert!( p.sample((1.0,1.0)).0.y == 110.0 + h/2.0);

//     assert!( p.sample((0.0,0.0)).0.z == 110.0  );
//     assert!(p.sample((0.5,0.5)).0.z == 110.0  );
//     assert!( p.sample((1.0,1.0)).0.z == 110.0  );

//     let negp = Plane::new(Vector3::new(110.0,110.0,110.0),Vector3::new(0.0,0.0,-1.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
//     assert_abs_diff_eq!(negp.sample((0.0,0.0)).1 , Vector3::new(0.0,0.0,-1.0));
//    //  assert!(  negp.sample((0.0,0.0)).1 ==  );
//     assert!( negp.sample((0.0,0.0)).0.x == 110.0 + w/2.0);
    
//     assert!( negp.sample((0.5,0.5)).0.x == 110.0  );
//     assert!(negp.sample((1.0,1.0)).0.x == 110.0 - w/2.0);
//     assert!(negp.sample((0.0,0.0)).0.y == 110.0  - h/2.0);
//     assert!(negp.sample((0.5,0.5)).0.y == 110.0  );
//     assert!( negp.sample((1.0,1.0)).0.y == 110.0 + h/2.0);

//     assert!( negp.sample((0.0,0.0)).0.z == 110.0  );
//     assert!(negp.sample((0.5,0.5)).0.z == 110.0  );
//     assert!( negp.sample((1.0,1.0)).0.z == 110.0  );
    
//     assert!( negp.sample((1.0,1.0)).1.z == -1.0  );



// }



#[test]
fn test_is_hit_inside() {
    use crate::iter::zip;
    use materials::BsdfType;
    use std::f64::consts;
    
    let sphere = prims::Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default())
    );

    let ray = prims::Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(0.0, 0.0, 1.0));
    if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
       let recout =  hit.material.sample(materials::RecordSampleIn { prevW: hit.prev.unwrap(), pointW: hit.point,sample:Some((0.0,0.0)) });
    }
    let ndivs = 120;
    let fndivs = ndivs as f64;
    for (xx, zz) in std::iter::zip(0..ndivs, 0..ndivs) {
      
        // let dir = ((xx as f64 / fndivs)*2.0*std::f64::consts::PI).sin_cos();
        // let ray = prims::Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(dir.0, 0.0, dir.1));
        // if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
        //     // match hit.material {
        //     //     BsdfType::Lambertian(l) => {
        //     //         let bsdf = l.bsdf.unwrap();
        //     //         // println!("{:?}, {:?}  {:?}", bsdf.n, bsdf.tn, bsdf.bn);
        //     //         // tiene que aumentar el angulo hasta vuelve a bajar
        //     //         // the angle amount goes up and then goes down again
        //     //         println!("{:?}",Deg::from(bsdf.n.angle(Vector3::new(0.0,0.0,-1.0))));
        //     //         // el angulo has to be 90
        //     //         assert_delta!(Deg::from(bsdf.n.angle(bsdf.tn) ).0,90.0, 0.00001);
                    
        //     //     }
        //     // }
        // }
    }
    let ray = prims::Ray::new(Point3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0));

    if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
        // match hit.material {
        //     BsdfType::Lambertian(l) => {
        //         let bsdf = l.bsdf.unwrap();
        //         println!("{:?}, {:?}  {:?}", bsdf.n, bsdf.tn, bsdf.bn);
        //     }
        // }
        //  println!("{:?}, {:?}", ray, hit.is_hit_inside.unwrap());
    }

    use rand::Rng;

    let mut rng = rand::thread_rng();
    let min = -1.0;
    let max = 1.0;

    // every ray has a hit inside the sphere.
    for i_ in 1..100 {
        let rngdir = Vector3::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        );
        let ray = prims::Ray::new(Point3::new(0.0, 0.0, 0.0), rngdir.normalize());

        let sphere = prims::Sphere::new(
            Vector3::new(0.0, 0.0, 0.0),
            1.0,
            MaterialDescType::PlasticType(Plastic::default())
                
        );

        if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
            //  println!("{:?}, {:?}", ray, hit.is_hit_inside.unwrap());
            assert_eq!(hit.is_hit_inside.unwrap(), true);
        }
    }

    // every ray has a hit outside the sphere.
    for i_ in 1..100 {
        let rngdir = Vector3::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        );
        for centerray in vec![
            Point3::new(1.3, 1.2, 1.10),
            Point3::new(0.0, 1.2, 1.10),
            Point3::new(0.0, 0.0, 1.10),
        ] {
            let ray = prims::Ray::new(centerray, rngdir.normalize());

            let sphere = prims::Sphere::new(
                Vector3::new(0.0, 0.0, 0.0),
                1.0,
                MaterialDescType::PlasticType(Plastic::default())
                    
            );

            if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
                println!("{:?}, {:?}", ray, hit.is_hit_inside.unwrap());
                assert_eq!(hit.is_hit_inside.unwrap(), false)
            }
        }
    }
    println!("end of test is hit inside");
}

#[test]
pub fn test_intersection() {
    use rand;
    use rand::Rng;
    let sphere = prims::Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default())
            
    );
    let mut cnt = 0;
    for i in 0..100 {
        let pos = (i as f64 / 100.0) * 2.0 - 1.0;
        let ray = prims::Ray::new(
            Point3::new(0.0, pos, 2.0),
            Vector3::new(0.0, 0.0, -1.0).normalize(),
        );
        if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
            cnt = cnt + 1;
            // println!( "cnt {}, hit.t {}, p:{:?} angle n:{:?} {},{}",cnt, hit.t , hit.point, Deg::from(hit.normal.angle(Vector3::new(0.,0.,1.0))) , hit.u,hit.v);
        };
    }
    assert_eq!(cnt, 99);
    cnt = 0;
    for i in 0..100 {
        let pos = (i as f64 / 100.0) * 2.0 - 1.0;
        let ray = prims::Ray::new(
            Point3::new(pos, 0.0, 2.0),
            Vector3::new(0.0, 0.0, -1.0).normalize(),
        );
        if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
            cnt = cnt + 1;
            // println!( "cnt {}, hit.t {}, p:{:?} angle n:{:?} {},{}",cnt,hit.t , hit.point, Deg::from(hit.normal.angle(Vector3::new(0.,0.,1.0))) , hit.u,hit.v);
        }
    }
    assert_eq!(cnt, 99);

    cnt = 0;
    let ray = prims::Ray::new(
        Point3::new(0.0, -2.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0).normalize(),
    );
    if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
        cnt = cnt + 1;
        println!(
            "cnt {}, hit.t {}, p:{:?} angle n:{:?} {},{}",
            cnt,
            hit.t,
            hit.point,
            Deg::from(hit.normal.angle(Vector3::new(0., 0., 1.0))),
            hit.u,
            hit.v
        );
    }

    cnt = 0;
    let ray = prims::Ray::new(
        Point3::new(0.0, 2.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0).normalize(),
    );
    if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
        cnt = cnt + 1;
        println!(
            "cnt {}, hit.t {}, p:{:?} angle n:{:?} {},{}",
            cnt,
            hit.t,
            hit.point,
            Deg::from(hit.normal.angle(Vector3::new(0., 0., 1.0))),
            hit.u,
            hit.v
        );
    }

    println!("");
}

#[test]
pub fn test_interect_sphere() {
    let mut cnt = 0;
    use materials::SampleIllumination;
    let ray = prims::Ray::new(
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, 1.0).normalize(),
    );
    let sphere = prims::Sphere::new(
        Vector3::new(0.0, 0.0, 12.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default())
            
    );
    if let Some(hit) = sphere.intersect(&ray, 0.001, f64::MAX) {
        println!(
            "cnt {}, hit.t {}, p:{:?} angle n:{:?} {},{}",
            cnt,
            hit.t,
            hit.point,
            Deg::from(hit.normal.angle(Vector3::new(0., 0., 1.0))),
            hit.u,
            hit.v
        );
    }
}
#[test]
fn test_interset_with_Scene() {
    use crate::materials::RecordSampleIn;
    use crate::materials::RecordSampleOut;
    use crate::raytracerv2::Scene;
    let ray = prims::Ray::new(
        Point3::new(0.0, 0.0, 5.0),
        Vector3::new(0.0, 0.0, -1.0).normalize(),
    );
    let sphere = prims::Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default())
            
    );

    let sphere1 = prims::Sphere::new(
        Vector3::new(0.0, 0.0, 10.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default())
            
    );

    let scene = Scene::<f64>::make_scene(0, 0, vec![], vec![&sphere, &sphere1], 0, 0);

    fn interset_scene(r: &Ray<f64>, scene: &Scene<f64>) -> Option<Ray<f64>> {
        for sphere_i in &scene.primitives2{
            if let Some(hit) = sphere_i.intersect(r, 0.0001, f64::MAX) {
                //  println!("t:{}, p:{:?} , n :{:?}", hit.t, hit.point, hit.normal);

                let newrec = hit.material.sample(RecordSampleIn {
                    prevW: hit.normal,
                    pointW: hit.point,
                    sample:Some((0.0,0.0))
                });
                return Some(Ray::new(hit.point, newrec.next));
            }
        }
        None
    }
    let mut newra = interset_scene(&ray, &scene).unwrap();
    for i in 0..10 {
        println!("{:?}", newra);
        let newrb = interset_scene(&newra, &scene).unwrap();
        println!("{:?}", newrb);
        newra = newrb;
    }
    println!("ping pong");
}

#[test]
pub fn test_insersect_oclussion() {
    use std::iter::zip;
    let ray = prims::Ray::new(
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 0.0, -1.0).normalize(),
    );
    let sphere = prims::Sphere::new(
        Vector3::new(0.0, 0.0, 0.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default())
    );
    let target = Point3::<f64>::new(0.0, 0.0, 1.0);
    let points = vec![
        Point3::<f64>::new(0.0, 0.0, 1.0),
        Point3::<f64>::new(0.0, 1.0, 1.0),
        Point3::<f64>::new(1.0, 0.0, 1.0),
    ];
    let results = vec![true, false, false];

    for (target, res) in zip(points, results) {
        if let Some(hit) = sphere.intersect_occlusion(&ray, &target) {
            println!("{:?}", hit.0);
            assert_eq!(hit.2 == target, res);
        }
    }

    println!("end oclussion");
}
pub fn main_prim() {
    let sp = prims::Sphere::new(
        Vector3::new(1.0, 1.0, 1.0),
        1.0,
        MaterialDescType::PlasticType(Plastic::default())
    );
    Point3::new(1.0, 1.0, 1.0);

    // let m =  Mat4::new_translation()

    let m4x = Matrix4::<f32>::identity();
    let invmaT = m4x.inverse_transform().unwrap();
    let mut m4a = Matrix4::<f32>::identity();
    let mat_e = Matrix4::new(
        0.409936f64,
        0.683812f64,
        -0.603617f64,
        0.0f64,
        0.,
        0.661778f64,
        0.7497f64,
        0.0f64,
        0.912114f64,
        -0.307329f64,
        0.271286f64,
        0.0f64,
        -0.,
        -1.323555f64,
        -1.499401f64,
        1.0f64,
    );

    let mtt = Matrix4::from_translation(Vector3::new(1.0, 1., 1.));
    let Scale = Matrix4::from_scale(2.0);

    let R = Matrix4::from_angle_x(Rad(90.0));

    let TSR = mtt * Scale * R;
    let inv = TSR.inverse_transform().unwrap();
    inv.transform_vector(Vector3::new(0.0, 1.0, 0.));
    //    println!("{:?}",  inv.transform_point(Point3::new(1.0,1.0,1.0)));
    println!(
        "{:?}",
        TSR.transform_vector(Vector3::new(0.0, 1.0, 0.)).normalize()
    );
    println!(
        "{:?}",
        inv.transform_vector(TSR.transform_vector(Vector3::new(0.0, 1.0, 0.)))
    );
}























 

 
#[derive(Debug, Clone   )]
pub enum PrimitiveType {
    CylinderType(Cylinder),
    DiskType(Disk),
    SphereType(Spheref64),
} 
impl PrimitiveType {
    pub fn intersect(&self,ray: &Ray< f64>, tmin: f64, tmax:  f64)->Option<HitRecord<f64>>{
        match self { 
            PrimitiveType::CylinderType(p)=>p.intersect(ray, tmin, tmax),
            PrimitiveType::DiskType(p)=>p.intersect(ray, tmin, tmax),
            PrimitiveType::SphereType(p)=>p.intersect(ray, tmin, tmax),
            _=>None
        }
    }
    pub fn area(&self )->f64{
        match self { 
            PrimitiveType::CylinderType(p)=>p.area(),
            PrimitiveType::DiskType(p)=>p.area(),
            PrimitiveType::SphereType(p)=>p.area(),
            _=>panic!("")
        }
    }
    pub  fn intersect_occlusion(
        &self,
        ray: &Ray<f64>,
        target: &Point3<f64>,
    ) ->  Option<(bool,f64, Point3<f64>,  Option<MediumOutsideInside>)>{
        match self { 
       
       
           PrimitiveType::SphereType(p)=>p.intersect_occlusion(ray, target),
           PrimitiveType::DiskType(p)=>p.intersect_occlusion(ray, target),
            PrimitiveType::CylinderType(p)=>p.intersect_occlusion(ray, target),
       
            _=>panic!("")

        }
    }
        

    // pub fn world_bound(&self)->Bounds3f{
    //     match self { 
    //           PrimitiveType::CylinderType(p)=>p.world_bound(),
    //           PrimitiveType::DiskType(p)=>p.world_bound(),
    //           PrimitiveType::SphereType(p)=>p.world_bound(),
    //         _=>panic!("")
    //     }
    // } 
    // pub fn compute_world_bounds( trafo : Decomposed<Vector3<f64>, Quaternion<f64>>, objectbound3f : Bounds3f)->Bounds3f {
    //     // match self { 
    //     //     // PrimitiveType::CylinderType(p)=>p.area(),
    //     //     _=>panic!("")
    //     // }
    // }
    pub fn sample_ref(&self, phit:Point3f , psample : (f64,f64))->(Point3<f64>, Vector3<f64>, f64) {
        match self { 
            PrimitiveType::CylinderType(p)=>p.sample(psample),
            PrimitiveType::DiskType(p)=>p.sample(psample),
            PrimitiveType::SphereType(p)=>p.sample_ref(phit, psample),
              _=>panic!("")
          }

    }
    pub fn sample(&self, psample : (f64,f64))->(Point3<f64>, Vector3<f64>, f64) {
        match self { 
          PrimitiveType::CylinderType(p)=>p.sample(psample),
          PrimitiveType::DiskType(p)=>p.sample(psample),
          PrimitiveType::SphereType(p)=>p.sample(psample),
            _=>panic!("")
        }
    }
    pub fn trafo_to_local(&self, pworld: Point3<f64>) -> Point3<f64>  {
        match self { 
           PrimitiveType::CylinderType(p)=>p.trafo_to_local(pworld),
           PrimitiveType::DiskType(p)=>p.trafo_to_local(pworld),
           PrimitiveType::SphereType(p)=>p.trafo_to_local(pworld),
            _=>panic!("")
        }
    }
    pub fn trafo_to_world(&self, plocal: Point3<f64>) -> Point3<f64>  {
        match self { 
             PrimitiveType::CylinderType(p)=>p.trafo_to_world(plocal),
             PrimitiveType::DiskType(p)=>p.trafo_to_world(plocal),
             PrimitiveType::SphereType(p)=>p.trafo_to_world(plocal),
            _=>panic!("")
        }
    }
    pub fn trafo_vec_to_local(&self, pworld: Vector3<f64>) -> Vector3<f64>   {
        match self { 
           PrimitiveType::CylinderType(p)=>p.trafo_vec_to_local(pworld),
           PrimitiveType::DiskType(p)=>p.trafo_vec_to_local(pworld),
           PrimitiveType::SphereType(p)=>p.trafo_vec_to_local(pworld),
            _=>panic!("")
        }
    }
    pub fn trafo_vec_to_world(&self, plocal: Vector3<f64>) -> Vector3<f64>  {
        match self { 
             PrimitiveType::CylinderType(p)=>p.trafo_vec_to_world(plocal),
             PrimitiveType::DiskType(p)=>p.trafo_vec_to_world(plocal),
             PrimitiveType::SphereType(p)=>p.trafo_vec_to_world(plocal),
            _=>panic!("")
        }
    }
    pub fn get_medium(&self ) -> Option<MediumOutsideInside>{
        match self { 
            // PrimitiveType::CylinderType(p)=>p.trafo_vec_to_world(plocal),
          //   PrimitiveType::DiskType(p)=>p.trafo_vec_to_world(plocal),
             PrimitiveType::SphereType(p)=>p.medium,
            _=>panic!("")
        }
    }
 
}