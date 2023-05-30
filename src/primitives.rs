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
use std::iter::zip;

use crate::Lights::SampleLightIllumination;
use crate::assert_delta;
use crate::materials::{self, SampleIllumination, Fr, MaterialDescType, Plastic, FrameShading, MaterialDescriptor, BsdfType, ConcentricSampleDisk};
use crate::primitives::prims::Intersection;
use crate::texture::{*};
use crate::texture::{MapConstant,Texture2D,Texture2DSRgbMapConstant};
use palette::Srgb;
use cgmath::*;
use num_traits::Float;
use std::f64::consts;

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

    use crate::Lights::SampleLightIllumination;
    use crate::materials;
    use crate::materials::FrameShading;
    use crate::materials::BsdfType;

    use crate::materials::LambertianBsdf;
    use crate::materials::MaterialDescType;
    use crate::materials::MaterialDescriptor;
    use crate::materials::Plastic;
    use crate::texture::MapConstant;
    use crate::texture::Texture2D;
    
    use crate::raytracerv2::clamp;
    #[derive(Debug, Clone )]
    pub struct Sphere< Scalar> {
        pub world: Matrix4<Scalar>,
        pub local: Matrix4<Scalar>,
        pub radius: Scalar,

        pub material: Option<materials::MaterialDescType >,
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
                material: Some(material)
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
                    material: Some(material)
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
         pub fn sample(&self,  pref : Point3<f64> ,psample: (f64, f64))->(Point3<f64>, Vector3<f64>, f64){
            let zero = num_traits::cast::cast::<f64, Scalar>(0.0).unwrap();
           let pcenterinworld =  self.trafo_to_world(Point3::new( zero, zero, zero));
          let p =  cast_point_to_point_to_vf64(pcenterinworld);
           let res =  sample_sphere_extern(num_traits::cast::cast::< Scalar , f64>(self.radius).unwrap(),p, Point3::new(0.0,0.0,0.0),psample) ;
          
           res
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
    pub fn sample_sphere_extern(r : f64 ,pcenter_sphere : Point3<f64>, pref : Point3<f64> ,psample : (f64,f64))->(Point3<f64>, Vector3<f64>, f64){
       let vc =  pcenter_sphere  - pref;
       let d = vc.magnitude();
       let vc =vc.normalize();





       let (vx, vy)=coord_system(vc);
       let sintheta2 =  (r * r) /(pref.distance2( pcenter_sphere));
       let costheta2 =(1.0-sintheta2).max(0.0).sqrt();
       let costhetasample = (1.0-psample.0)+psample.0*costheta2;
       let sinthetasample = (1.0-costhetasample*costhetasample) .max(0.0).sqrt();
       let phi = psample.1*2.0*std::f64::consts::PI;
    
      let ds = d*costhetasample - (r*r -d *d *sinthetasample*sinthetasample).max(0.0).sqrt()  ;
      let cosalpha = ( d * d + r*r - ds*ds)/(2.0 * d * r);
      let sinalpha = (1.0-cosalpha*cosalpha).max(0.0).sqrt();
   // spherical direction
     let ja =  Vector3::new(- sinalpha* phi.cos() * vx.x,- sinalpha* phi.cos()* vx.y ,-sinalpha* phi.cos() * vx.z);
     let jb =  Vector3::new(- sinalpha* phi.sin() * vy.x,- sinalpha* phi.sin()* vy.y ,-sinalpha* phi.sin() * vy.z);
     let jc  =  Vector3::new(- cosalpha * vc.x,-cosalpha* vc.y ,-cosalpha  * vc.z);
     let n  =  ja + jb  + jc;
     // translate point to sphere center and
      let pSpheresampleInWorlds = Point3::new( pcenter_sphere.x + n.x*r, pcenter_sphere.y + n.y*r, pcenter_sphere.z + n.z*r);
      

      (pSpheresampleInWorlds, n,  1.0 /(2.0 * std::f64::consts::PI * (1.0 -costheta2 )))
   
    
       
     }
    #[derive(
        Debug,
        Clone,
        Copy,
        // Deserialize, Serialize
    )]
    
    pub struct Ray<Scalar> {
        pub origin: Point3<Scalar>,
        pub direction: Vector3<Scalar>,
    }

    impl<Scalar: BaseFloat> Ray<Scalar> {
        pub fn new(origin: Point3<Scalar>, direction: Vector3<Scalar>) -> Ray<Scalar> {
            Ray { origin, direction }
        }
        pub fn at(&self, p: Scalar) -> Point3<Scalar> {
            self.origin + self.direction * p
        }
        pub fn from_origin_to_target(
            origin: Point3<Scalar>,
            target: Point3<Scalar>,
        ) -> Ray<Scalar> {
            let direction = (target - origin).normalize();
            Ray { origin, direction }
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
        pub emission : Option<Srgb>
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
                emission : None
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
                emission : None
            }
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

                       // println!("{:?}", bsdf);
                        // let material = match self.material.unwrap() {
                        //     BsdfType::Lambertian(l) => {
                        //         l.albedo. eval();
                        //         BsdfType::Lambertian(LambertianBsdf ::new(bsdf, Some(l.albedo)))
                        //     },
                        //     BsdfType::MicroFacetReflection(mut micro)=>{
                        //         micro.frame =Some( bsdf );
                        //         BsdfType::MicroFacetReflection(micro)
                        //     },
                        //     BsdfType::LambertianDisneyBsdf(mut l)=>{
                        //         l.frame =Some( bsdf );
                        //         BsdfType::LambertianDisneyBsdf(l)
                        //     }
                        //     _=> BsdfType::None
                        // };
                      
                      let bsdftype =   self.material.as_ref().unwrap().instantiate( 
                        &cast_point_to_point_to_vf64(self.trafo_to_world(p )), 
                      &( num_traits:: cast:: <Scalar,f64>(u).unwrap(),num_traits:: cast:: <Scalar,f64>(v).unwrap()), 
                        &frame);
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
                            emission :None
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
        ) ->  Option<(bool,Scalar, Point3<Scalar>)>;
    }
    //     |y   /z
    //     |  /
    //     |/______x
    //    
    //+ 

 
    impl<Scalar: BaseFloat + rand::distributions::uniform::SampleUniform + PartialOrd>
        IntersectionOclussion<Scalar> for Sphere<Scalar>
    {
       
        fn intersect_occlusion(
            &self,
            ray: &Ray<Scalar>,
            target: &Point3<Scalar>,
        ) ->  Option< (bool, Scalar, Point3<Scalar>)> {
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

                        return Some((true, *root, self.trafo_to_world(p)));
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
            emission:None
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
    pub fn internal_intersect_occlusion(  &self,ray: &Ray<f64>,target: &Point3<f64>)->Option< (bool, f64, Point3<f64>)>{

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
        return Some((true,t, self.trafo_to_world(pr)));
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
     
    fn intersect_occlusion(&self, ray: &Ray<f64>,target: &Point3<f64>)->Option<(bool, f64, Point3<f64>)>{
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

        let origin = self.trafo_to_local(ray.origin);
        let direction = self.trafo_vec_to_local(ray.direction).normalize();
      //   println!("ray : {:?}, {:?}", origin, direction );
        let d = Vector3::new(0.0,0.0,-1.0).dot(direction);
        if d < 0.0{
          //  println!("no itc ");
            return None
        }
       
        let t =  (Point3::new(0.0,0.0,0.0) - origin).dot( Vector3::new(0.0,0.0,-1.0)) /  d;
        if t < 0.0 { return None }
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
            emission:None
        })
    }
 
    pub fn internal_intersect_occlusion(  &self,ray: &Ray<f64>,target: &Point3<f64>)->Option< (bool, f64, Point3<f64>)>{

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
         return Some((true,t, self.trafo_to_world(phit)));
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
     
    fn intersect_occlusion(&self, ray: &Ray<f64>,target: &Point3<f64>)->Option<(bool, f64, Point3<f64>)>{
        self.internal_intersect_occlusion(ray, target)
    }
}
  








#[test]
fn test_Disk(){
     
    let p = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0, 1.0),0.0, 1.0, MaterialDescType::NoneType);
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(9.0,9.0, -1.0).normalize()},0.00001, f64::MAX){
        println!("{:?}",t);
        assert!(false)
    }
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(9.0,9.0, -1.0).normalize()},0.00001, f64::MAX){
        println!("{:?}",t);
        assert!(false)
    }
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(-19.0,-19.0, -1.0).normalize()},0.00001, f64::MAX){
        println!("{:?}",t);
        assert!(false)
    }
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,-1.0), direction:Vector3::new(-19.0,-19.0, -1.0).normalize()},0.00001, f64::MAX){
        println!("{:?}",t);
        assert!(false)}
    
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,-1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,0.0), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    let p = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0, 1.0),0.0, 1.0, MaterialDescType::NoneType);
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(1.0,1.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,0.50), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    // traslation
    let p = Disk::new(Vector3::new(0.0,0.0,10.0),Vector3::new(0.0,0.0, 1.0),0.0, 1.0, MaterialDescType::NoneType);
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,10.50), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    let p = Disk::new(Vector3::new(0.0,0.0,10.0),Vector3::new(0.0,0.0, -1.0),0.0, 1.0, MaterialDescType::NoneType);
    if let None= p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let None = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,10.50), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,10.50), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}

    let p = Disk::new(Vector3::new(0.0,0.0,10.0),Vector3::new(0.0,0.0, -1.0),0.0, 1.0, MaterialDescType::NoneType);
    if let None = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,1.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let None = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,2.10, 1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let None = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,1.0,10.50), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,1.0,10.50), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}



     // rotation
     let p = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,1.0, 0.0),0.0, 1.0, MaterialDescType::NoneType);
     if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,1.0,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}
     if let None = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,1.0,0.0), direction:Vector3::new(0.0,1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}

     // rotation + translation
     let p = Disk::new(Vector3::new(0.0,1.0,0.0),Vector3::new(0.0,1.0, 0.0),0.0, 1.0, MaterialDescType::NoneType);
     if let None = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.90,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}
     if let None = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.990,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}
     if let None = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.99990,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){assert!(true)}
     if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,1.01,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){
        println!("true : {:?}",t.point);
        assert!(true)}
     if let  Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,1.001,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){
        println!("true : {:?}",t.point);
        assert!(true)}
     if let  Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,1.0001,0.0), direction:Vector3::new(0.0,-1.0, 0.0).normalize()},0.00001, f64::MAX){
        println!("true : {:?}",t.point);
        assert!(true)}

        let p = Disk::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0, 1.0),0.0, 1.0, MaterialDescType::NoneType);
        for i in 0..32{
            let ps =  (i as f64 / 32.0);
            println!(" i -------{:?}",i );
            if let  Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(ps +0.0010   , ps+0.0010 ,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){
                println!("p {:?}", t.point);
                println!("uv {:?}", (t.u, t.v));
                println!("dpdu {:?}", t.bn );
                println!("dpdv {:?}", t.tn);


            }
            let rsample =  p.sample( (ps, ps));
            println!("  p sample{:?}", rsample.0);
            // println!("p normal {:?}",rsample.1 );
            println!(   "pdf {:?}", rsample.2 );
            println!("  " );

        } // end for
    
}





















#[test]
fn test_plane(){
    
 
    let p =  Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0,1.0),10.0, 10.0,MaterialDescType::PlasticType(Plastic::default()));
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(9.0,9.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(9.0,9.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(-19.0,-19.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,-1.0), direction:Vector3::new(-19.0,-19.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,-1.0), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0, 1.0), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}
    if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,0.0), direction:Vector3::new(1.0,1.0, 1.0).normalize()},0.00001, f64::MAX){assert!(false)}

    
    

   
  //   translaction 
        let p =  Plane::new(Vector3::new(0.0,0.0,11.0),Vector3::new(0.0,0.0,1.0),10.0, 10.0,MaterialDescType::PlasticType(Plastic::default()));
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,1.0), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(false)}
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,11.1000), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)};
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,11.01000), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)}
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,11.001000), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)};
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,11.0001000), direction:Vector3::new(0.0,0.0, -1.0).normalize()},0.00001, f64::MAX){assert!(true)};

        let p =  Plane::new(Vector3::new(0.0,0.0,11.0),Vector3::new(0.0,0.0,-1.0),10.0, 10.0,MaterialDescType::PlasticType(Plastic::default()));
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,10.1000), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)};
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,10.01000), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)}
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,10.001000), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)};
        if let Some(t) = p.intersect(& Ray::<f64>{origin:Point3::new(0.0,0.0,10.0001000), direction:Vector3::new(0.0,0.0, 1.0).normalize()},0.00001, f64::MAX){assert!(true)};

//    //  rotatation
    let p =  Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,1.0,1.0).normalize(),10.0, 10.0,MaterialDescType::PlasticType(Plastic::default()));
    for i in 0..32{
        let dir =  (i as f64 / 32.0) * std::f64::consts::PI;
        println!("{:?}", dir.sin_cos().0);
        println!("{:?}",  Vector3::new( dir.sin_cos().0 , 0.0, 1.0).normalize());
        println!("{:?}",  Point3::new( dir.sin_cos().0*10.0 , 0.0, 1.0*10.0) );
        let ptray = Point3::new( dir.sin_cos().0*10.0 , 0.0, 1.0*10.0);
        let ray = Ray::<f64>::new(ptray ,  -Vector3::new( dir.sin_cos().0 , 0.0, 1.0).normalize());
        let p =  Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new( dir.sin_cos().0 , 0.0, 1.0).normalize(),10.0, 10.0,MaterialDescType::NoneType);
        if let Some(t) =   p.intersect(&ray, 0.0001,std::f64::MAX){assert!(true)};
      
        let pstatic =  Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new( 0.0,0.0, 1.0).normalize(),10.0, 10.0,MaterialDescType::NoneType);
       match  pstatic.intersect(&ray, 0.0001,std::f64::MAX){
         Some(h)=>assert!(true), 
         None =>assert!(false)
       };
        
        
    }
     
    //sample method

    let h = 10.0;
    let w = 10.0;
    // let p = Plane::new(Vector3::new(0.0,0.0,0.0),Vector3::new(0.0,0.0,1.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
    // println!("{:?}", p.sample((0.0,0.0)));
    // println!("{:?}", p.sample((0.5,0.5)));
    // println!("{:?}", p.sample((1.0,1.0)));
    // let p = Plane::new(Vector3::new(0.0,0.0,10.0),Vector3::new(0.0,0.0,1.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
    // println!("{:?}", p.sample((0.0,0.0)));
    // println!("{:?}", p.sample((0.5,0.5)));
    // println!("{:?}", p.sample((1.0,1.0)));


    let p = Plane::new(Vector3::new(110.0,111.0,110.0),Vector3::new(0.0,1.0,0.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
 
    assert!( p.sample((0.0,0.0)).0.x == 110.0 - w/2.0);
    assert!( p.sample((0.5,0.5)).0.x == 110.0  );
    assert!(p.sample((1.0,1.0)).0.x == 110.0 + w/2.0);
    assert!( p.sample((0.0,0.0)).0.z == 110.0 + h/2.0);
    assert!(p.sample((0.5,0.5)).0.z == 110.0  );
    assert!( p.sample((1.0,1.0)).0.z == 110.0 - h/2.0);

    assert!( p.sample((0.0,0.0)).0.y == 111.0  );
    assert!(p.sample((0.5,0.5)).0.y == 111.0  );
    assert!( p.sample((1.0,1.0)).0.y == 111.0  );

    assert!( p.sample((1.0,1.0)).1.z ==  1.0  ); 
    println!("{:?}", p.sample((0.0,0.0)).0.x == 110.0 - w/2.0);
    println!("{:?}", p.sample((0.5,0.5)).0.x == 110.0  );
    println!("{:?}", p.sample((1.0,1.0)).0.x == 110.0 + w/2.0);
    println!("{:?}", p.sample((0.0,0.0)).0.z == 110.0 + h/2.0);
    println!("{:?}", p.sample((0.5,0.5)).0.z == 110.0  );
    println!("{:?}", p.sample((1.0,1.0)).0.z == 110.0 - h/2.0);

    println!("{:?}", p.sample((0.0,0.0)).0.y == 111.0  );
    println!("{:?}", p.sample((0.5,0.5)).0.y == 111.0  );
    println!("{:?}", p.sample((1.0,1.0)).0.y == 111.0  );


    
    let p = Plane::new(Vector3::new(110.0,110.0,110.0),Vector3::new(0.0,0.0,1.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
    assert_abs_diff_eq!(p.sample((0.0,0.0)).1 , Vector3::new(0.0,0.0,1.0));
 
    assert!( p.sample((0.0,0.0)).0.x == 110.0 - w/2.0);
    assert!( p.sample((0.5,0.5)).0.x == 110.0  );
    assert!(p.sample((1.0,1.0)).0.x == 110.0 + w/2.0);
    assert!( p.sample((0.0,0.0)).0.y == 110.0  - h/2.0);
    assert!(p.sample((0.5,0.5)).0.y == 110.0  );
    assert!( p.sample((1.0,1.0)).0.y == 110.0 + h/2.0);

    assert!( p.sample((0.0,0.0)).0.z == 110.0  );
    assert!(p.sample((0.5,0.5)).0.z == 110.0  );
    assert!( p.sample((1.0,1.0)).0.z == 110.0  );

    let negp = Plane::new(Vector3::new(110.0,110.0,110.0),Vector3::new(0.0,0.0,-1.0).normalize(),w/2.0, h/2.0,MaterialDescType::NoneType);
    assert_abs_diff_eq!(negp.sample((0.0,0.0)).1 , Vector3::new(0.0,0.0,-1.0));
   //  assert!(  negp.sample((0.0,0.0)).1 ==  );
    assert!( negp.sample((0.0,0.0)).0.x == 110.0 + w/2.0);
    
    assert!( negp.sample((0.5,0.5)).0.x == 110.0  );
    assert!(negp.sample((1.0,1.0)).0.x == 110.0 - w/2.0);
    assert!(negp.sample((0.0,0.0)).0.y == 110.0  - h/2.0);
    assert!(negp.sample((0.5,0.5)).0.y == 110.0  );
    assert!( negp.sample((1.0,1.0)).0.y == 110.0 + h/2.0);

    assert!( negp.sample((0.0,0.0)).0.z == 110.0  );
    assert!(negp.sample((0.5,0.5)).0.z == 110.0  );
    assert!( negp.sample((1.0,1.0)).0.z == 110.0  );
    
    assert!( negp.sample((1.0,1.0)).1.z == -1.0  );



}



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
