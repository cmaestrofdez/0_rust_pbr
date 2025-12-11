use cgmath::{Point3, InnerSpace};

use crate::{Lights::{Light, IsAreaLight, IsBackgroundAreaLight, GetShape1}, primitives::PrimitiveType, Point3f};
use crate::primitives::prims::Ray;
#[derive( Clone )]
pub struct Scene1{
    pub num_samples: usize,
//     pub primitives: Vec<&'a Sphere<Scalar>>,
//    pub primitives1: Vec<&'a  Box<dyn Intersection<Scalar, Output = HitRecord<Scalar>>>>,
//    pub primitives2:  Vec<& 'a Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>>,
    pub max_depth: usize,
    pub width: usize,
    pub height: usize,
    pub lights: Vec<Light>,
    pub  prims : Vec<PrimitiveType>,
    // pub sky: Option<Sphere<Scalar>>,
    // pub background: Option<TexType>,
    
}
impl Scene1 {
    pub fn make_scene(
        width: usize,
        height: usize,
        lights: Vec<Light>,
        prims : Vec<PrimitiveType>,
        // primitives: Vec<&'a Sphere<Scalar>>,
        // primitives1: Vec<&'a  Box<dyn Intersection<Scalar, Output = HitRecord<Scalar>>>>,
        num_samples: usize,
        max_depth: usize,
    ) -> Scene1{

        Scene1 {
            num_samples,
            max_depth,
            width,
            height,
            lights,
            prims
            // primitives,
            // sky: None,
            // background: None,
            // // primitives1:vec![],
            // primitives2:vec![],
        }
    }
    

    // pub fn make_scene_with_film(
    //     width: usize,
    //     height: usize,
    //     lights: Vec<Light>,
    //     primitives: Vec<&'a Sphere<Scalar>>,
       
    //     prims2 :  Vec<& 'a Box<dyn PrimitiveIntersection<Output =  HitRecord<f64> >>>,
    //     num_samples: usize,
    //     max_depth: usize,
    //     bck: Option<TexType>,
    //     film:Option<RefCell<ImgFilm>>
    // ) -> Scene< 'a ,Scalar> {
    //     Scene {
    //         num_samples,
    //         max_depth,
    //         width,
    //         height,
    //         lights,
    //         // primitives,
    //         sky: None,
    //         background: bck,
    //     //    primitives1:vec![],
    //        primitives2:prims2,
    //     }
    // }
    // pub fn get_height(&self)->usize{
    //     self.height
    // }

pub
    fn intersect_occlusion(
        &self,
        ray: &Ray<f64>,
        target: &Point3f,
    ) -> Option<(bool, f64, Point3<f64>)> {
        //panic!("porque ?????????? tiene en tmax=0.99999");
        let mut _tcurrent = f64::MAX;
        let mut _currentpoint: Option<Point3<f64>> = None;
        for iprim in &self.prims {
            if let Some((ishit, _thit, phit, _)) = iprim.intersect_occlusion(ray, target) {
                if (phit - ray.origin).magnitude2() < (*target - ray.origin).magnitude2() {
                    return Some((true, _thit, phit));
                }
            }
        }
        Some((false, f64::MAX, Point3::new(f64::MAX,f64::MAX, f64::MAX)))
    }

    pub
    fn is_empty_space(&self,p0: Point3f,p1 : Point3f) ->  bool {
         //     panic!("porque ?????????? tiene en tmax=0.99999");
        let ray =   Ray::from_origin_to_target(p0, p1);
        for iprim in &self.prims {
            let hit =   iprim.intersect(&ray, 0.00001,0.999999);  
            if let Some(hit) = hit  {
             
               return  false;
            }
        }
 
        for light in &self.lights{
            if  light.is_arealight() && !light.is_background_area_light() {
               let prim =   light .get_shape1() ;
             
               if let Some(mut hit) = prim.intersect(&ray, 0.0001, 0.999){ 
              
                return  false;
               }
            }
        }
    
        true 

}




    
}




 