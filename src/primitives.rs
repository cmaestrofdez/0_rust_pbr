// extern crate nalgebra as na;
// extern crate nalgebra_glm as glm;

// use glm::Mat4;
// use glm::Vec4;
// use na::Orthographic3;
// use na::Perspective3;

extern crate cgmath;

use cgmath::*;

use crate::primitives::prims::Intersection;

use self::prims::Ray;
pub
mod prims {
    use cgmath::BaseFloat;
    use cgmath::Matrix4;
    use cgmath::Point3;
    use cgmath::Vector3;

    use cgmath::*;
    use num_traits::ToPrimitive;

    
   pub struct Sphere<Scalar> {
        world: Matrix4<Scalar>,
        local : Matrix4<Scalar>
    }
    impl<Scalar: BaseFloat> Sphere<Scalar> {
        pub fn new(translation : Vector3<Scalar> , scale : Scalar)->Sphere<Scalar> {
            let t = Matrix4::from_translation(translation)*Matrix4::from_scale(scale);
            Sphere { world :  t, local: t.inverse_transform().unwrap() }
        }
        pub fn trafo_to_local(&self, pworld: Point3<Scalar>)->Point3<Scalar>{
            self.local.transform_point(pworld)
        }
        pub fn trafo_to_world(&self, plocal: Point3<Scalar>)->Point3<Scalar>{
            self.world.transform_point(plocal)
        }
        pub fn trafo_vec_to_local(&self, pworld: Vector3<Scalar>)->Vector3<Scalar>{
            self.local.transform_vector(pworld)
        }
        pub fn trafo_vec_to_world(&self, plocal: Vector3<Scalar>)->Vector3<Scalar>{
            self.world.transform_vector(plocal)
        }
    }
    pub struct Ray<Scalar> {
        origin  : Point3<Scalar>,
        direction : Vector3<Scalar>
    }

    impl<Scalar:BaseFloat > Ray <Scalar>{
        pub fn new(origin : Point3<Scalar>, direction : Vector3<Scalar>)->Ray<Scalar>{
            Ray{origin, direction}
        }
        pub fn at( &self,p : Scalar)->Point3<Scalar>{
            self.origin + self.direction * p
            
        }
    }
    pub struct HitRecord<Scalar> {
        pub t: Scalar,
        pub point: Point3<Scalar>,
         pub normal: Vector3<Scalar>,
        // pub front_face: bool,
        // // pub material: Material,
    
        pub u: Scalar,
         pub v: Scalar,
    }


    #[test]
    pub fn test_sphere_transformations(){
       let sphere  =  Sphere::new(Vector3::new(1.0,1.0,1.0), 1.0);
       assert_eq!( sphere.trafo_to_world(sphere.trafo_to_local(Point3::new(1.0,1.0, 1.0))),Point3::new(1.0,1.0, 1.0));
    }



   pub trait   Intersection<Scalar>{
        type Output;
        fn intersect(self, ray :&Ray<Scalar>, tmin: Scalar, tmax: Scalar)->Option<Self::Output>;
    }

    impl <Scalar:BaseFloat> Intersection<Scalar> for Sphere<Scalar> {
        type Output = HitRecord<Scalar>;
        fn intersect(self, ray :&Ray<Scalar>, tmin: Scalar, tmax: Scalar) ->Option<Self::Output> {
           let origin =  self.trafo_to_local(ray.origin);
           let direction =   self.trafo_vec_to_local(ray.direction);
           let oc = origin ;

           //  println!("{:?}", oc);
           let A = direction.dot(direction);
           let B = oc.dot(direction);
           let C = oc.dot(oc.to_vec())- num_traits::cast::cast::<f64, Scalar>(1.0).unwrap();// -1 - self.radius * self.radius;
           let disc = B * B - A * C;
           
           if disc.to_f64().unwrap() >= 0.0 {
            let sqrtd = disc.sqrt();
            let root_a = ((-B) - sqrtd) / A;
            let root_b = ((-B) + sqrtd) / A;
            for root in [root_a, root_b].iter() {
                if *root > tmin && *root < tmax {
                    let p = ray.at(*root);  
                    num_traits::cast::cast::<f64, Scalar>(0.0);
                    
                    let n = Vector3::new(
                        num_traits::cast::cast::<f64, Scalar>(0.0).unwrap(),
                        num_traits::cast::cast::<f64, Scalar>(0.0).unwrap(),
                        num_traits::cast::cast::<f64, Scalar>(0.0).unwrap());
                  let is_face = ray.direction.dot(n) ;
                 
                     return Some(
                        HitRecord{
                            t : *root,
                            point : p,
                            normal : n,
                            u:num_traits::cast::cast::<f64, Scalar>(0.0).unwrap(),
                           v : num_traits::cast::cast::<f64, Scalar>(0.0).unwrap()
                          }
                     );
               }
             }
            }
            

            None
        }
    }
   
    
}

#[test]
pub fn test_intersection(){
    use crate::primitives::*;
   let sphere  = prims::Sphere::new(Vector3::new(0.0,0.0,0.0), 1.0);
   let ray  = prims::Ray::new(Point3::new(0.0,0.0,0.0), Vector3::new(0.0, 0.0, 1.0));

    let hit  = sphere.intersect(&ray, 0.001, f64::MAX).unwrap();
}
pub fn main_prim() {
    let sp  = prims::Sphere::new(Vector3::new(1.0, 1.0, 1.0 ), 1.0);
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
    // let matmie = Matrix4::new(
    //     1.0, 0.0, 0.0, 0.0, 0., 1.0, 0.0, 0., 0., 0., 1., 0., 1., 1., 1., 0.0,
    // );
    // let p = Point3::new(1., 1., 1.);
    // let muuu = Matrix4::from_translation(Vector3::new(1.0, 1., 1.));
    // let tr = Matrix4::from_scale(2.0) * Matrix4::from_translation(Vector3::new(1.0, 1., 1.));
    // let invtr = tr.inverse_transform().unwrap();

    // println!("{:?}", tr.transform_point(p));
    // tr.transform_vector(Vector3::new(1., 1., 1.));
    // // mat_mi.transform_point(point)
    // m4a.transpose_self();
    let mtt = Matrix4::from_translation(Vector3::new(1.0, 1., 1.));
    let Scale = Matrix4::from_scale(2.0);
    
    let R = Matrix4::from_angle_x(Rad(90.0));

    let TSR = mtt * Scale *R;
    let inv = TSR.inverse_transform().unwrap();
    inv.transform_vector(Vector3::new(0.0,1.0, 0.));
//    println!("{:?}",  inv.transform_point(Point3::new(1.0,1.0,1.0)));
   println!("{:?}",   TSR.transform_vector(Vector3::new(0.0,1.0, 0.)).normalize());
   println!("{:?}",   inv.transform_vector(TSR.transform_vector(Vector3::new(0.0,1.0, 0.))));


}
