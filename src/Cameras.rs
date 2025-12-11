
use core::panic;

use crate::Vector4f;
use crate::primitives::prims::Ray;
use crate::primitives::prims::HitRecord;
use crate::sampler::Sampler;
use crate::{Matrix4f, Point3f, Vector3f};
use cgmath::MetricSpace;
use cgmath::Vector4;
use cgmath::{InnerSpace, Matrix, Matrix4, SquareMatrix, Transform};
use num_traits::float::FloatCore;
use palette::Srgb;








#[derive(Debug, Clone, Copy)]
pub struct RecordSampleImportanceIn  {
    pub praster:Point3f,
    pub psample: (f64, f64),
    pub hit: HitRecord<f64>,
}
impl  RecordSampleImportanceIn{
    pub fn from_hit(  h: HitRecord<f64>,praster:Point3f,  sampler: &mut Box<dyn Sampler>)->Self {
        RecordSampleImportanceIn{
            hit:h,
            psample : sampler.get2d(),
            praster 

        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RecordSampleImportanceOut  {
    pub prasterinworld: Point3f,
    pub weight : Srgb,
    pub  pdf : f64, 
    pub  n : Vector3f,
    pub  ray : Option<Ray<f64>>
    
}
 

impl  RecordSampleImportanceOut {
    pub fn from_hitrecord(  praster: Point3f  ,weight : Srgb,pdfpos : f64, pdf : f64,n : Vector3f,ray : Option<Ray<f64>>) ->Self{
        RecordSampleImportanceOut{prasterinworld:praster, weight, pdf,  n, ray}
    }
    pub fn checkIfZero(&self)->bool{
      
       ( self.weight.red == 0.0 && self.weight.green == 0.0 && self.weight.blue == 0.0) || self.pdf == 0.0
    }
}









#[derive(Debug, Clone, Copy)]
pub struct PerspectiveCamera {
    fovy: f64,
    focal_length: f64, // 1e6
    near: f64,         //1e-3
    far: f64,
    filmres: (u32, u32),            //1e3
    screenToRasterMatrix: Matrix4f, // ndc -> resraster
    rasterToScreenMatrix: Matrix4f, // resraster-> ndc
    perspectiveMatrix: Matrix4f,
    rasterToCameraMatrix: Matrix4f,
    cameraToWorldMatrix: Matrix4f,
    areaVS: f64,
}
impl PerspectiveCamera {
    pub fn lookAt( from:Point3f, to:Point3f)->Matrix4f{
      let dir =  ( to - from ).normalize();
      if  Vector3f::new(0.0,1.0,0.0).cross(dir).magnitude() ==0.0 {
        panic!("vector look at points in the same direction");
      }
      
      let r =         Vector3f::new(0.0,1.0,0.0).cross(dir) .normalize();
      let up =dir.cross(r).normalize();
      let c0 = Vector4f::new(r.x   ,r.y  , r.z,0.0);
      let c1 = Vector4f::new(up.x  ,up.y , up.z,0.0);
      let c2 =  Vector4f::new(dir.x,dir.y, dir.z,0.0);
      let c3 = Vector4f::new(from.x,from.y, from.z,1.0);
      let lookmat = Matrix4f::from_cols(c0, c1, c2, c3);
    
//    let lookmat = Matrix4f::from_cols(Vector4f::new(r,0.0), Vector4f::new(up,0.0), Vector4f::new(dir,0.0), Vector4f::new(translation,1.0) );
    //  println!("{:?}", lookmat);
    lookmat.invert().unwrap()
      
    }
    pub fn perspective(n: f64, f: f64, fov: f64) -> Matrix4f {
        let inv = 1.0 / (f - n);
        let a02 = f * inv;
        let a03 = -f * n * inv;
        let zmatrix = Matrix4f::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, a02, 1.0, 0.0, 0.0, a03, 0.0,
        );
        // Matrix4f::from_nonuniform_scale(x, y, z)
        let invtan = 1.0 / ((fov.to_radians() / 2.0).tan());
        Matrix4f::from_nonuniform_scale(invtan, invtan, 1.0) * zmatrix
    } //min.xy       //max.xy
    pub fn fromperspectiveSpacetoRaster(
        filmres: &(u32, u32),
        window: &((f64, f64), (f64, f64)),
    ) -> Matrix4f {
        let ffilmres = (filmres.0 as f64, filmres.1 as f64);
        Matrix4f::from_nonuniform_scale(ffilmres.0, ffilmres.1, 1.0)
            * Matrix4f::from_nonuniform_scale(
                1.0 / (window.1 .0 - window.0 .0),
                1.0 / (window.0 .1 - window.1 .1),
                1.0,
            )
            * Matrix4f::from_translation(Vector3f::new(-window.0 .0, -window.1 .1, 0.0))
    }
    pub fn new(n: f64, f: f64, fovy: f64, filmres:  (u32, u32)) -> Self {
        let perspectiveMatrix = PerspectiveCamera::perspective(n, f, fovy);
        let ndcTofilm =
            PerspectiveCamera::fromperspectiveSpacetoRaster(&filmres, &((-1.0, -1.0), (1.0, 1.0))); //
        let filmtondc = ndcTofilm.inverse_transform().unwrap();
        let filmtoviewMatrix = perspectiveMatrix.inverse_transform().unwrap() * filmtondc;

        let rasterPmin = filmtoviewMatrix.transform_point(Point3f::new(0.0, 0.0, 0.0));
        let rasterPmax =
            filmtoviewMatrix.transform_point(Point3f::new(filmres.0 as f64, filmres.1 as f64, 0.0));
        let pfilmVSmin = rasterPmin / rasterPmin.z;
        let pfilmVSmax = rasterPmax / rasterPmin.z;
        let areaVS = ((pfilmVSmax.x - pfilmVSmin.x) * (pfilmVSmax.y - pfilmVSmin.y)).abs();
        PerspectiveCamera {
            focal_length: 1e6,
            far: f,
            near: n,
            fovy,
            filmres,
            screenToRasterMatrix: ndcTofilm,
            rasterToCameraMatrix: filmtoviewMatrix,
            perspectiveMatrix,
            rasterToScreenMatrix: filmtondc,
            cameraToWorldMatrix: Matrix4::identity(),
            areaVS: areaVS,
        }
    }
    pub fn from_lookat(from:Point3f, to:Point3f , n: f64, f: f64, fovy: f64, filmres:  (u32, u32)) -> Self {
        let aspect = filmres.0 as f64 / filmres.1 as f64;
        let mut window ;
        if aspect>1.0{
            window = ((-aspect, -1.0), (aspect, 1.0));
        }else{
            window = ((-1.0, -1.0 / aspect), (1.0, 1.0/aspect));
        }
       
        let  cameraToWorldMatrix = Self::lookAt(from, to);
        let perspectiveMatrix = PerspectiveCamera::perspective(n, f, fovy);
        let ndcTofilm =
            PerspectiveCamera::fromperspectiveSpacetoRaster(&filmres, &window); //
        let filmtondc = ndcTofilm.inverse_transform().unwrap();

        let filmtoviewMatrix = perspectiveMatrix.inverse_transform().unwrap() * filmtondc;
       
        let rasterPmin = filmtoviewMatrix.transform_point(Point3f::new(0.0, 0.0, 0.0));
        let rasterPmax =
            filmtoviewMatrix.transform_point(Point3f::new(filmres.0 as f64, filmres.1 as f64, 0.0));
        let pfilmVSmin = rasterPmin / rasterPmin.z;
        let pfilmVSmax = rasterPmax / rasterPmin.z;
        let areaVS = ((pfilmVSmax.x - pfilmVSmin.x) * (pfilmVSmax.y - pfilmVSmin.y)).abs();
        PerspectiveCamera {
            focal_length: 1e6,
            far: f,
            near: n,
            fovy,
            filmres,
            screenToRasterMatrix: ndcTofilm,
            rasterToCameraMatrix: filmtoviewMatrix,
            perspectiveMatrix,
            rasterToScreenMatrix: filmtondc,
            cameraToWorldMatrix: cameraToWorldMatrix,
            areaVS: areaVS,
        }
    }
    pub fn area(&self) -> f64 {
        self.areaVS
    }

    // prastersampler =(filres.x+sampler, filmres.y +sampler)
    pub fn get_ray(&self, prastersampler: &(f64, f64)) -> Ray<f64> {
     
        let pcamera = self.rasterToCameraMatrix.transform_point(Point3f::new(
            prastersampler.0,
            prastersampler.1,
            0.0,
        ));
     
        Ray::<f64>::new(
          self.cameraToWorldMatrix.transform_point(Point3f::new(0.0, 0.0, 0.0)),
          self.cameraToWorldMatrix.transform_vector( Vector3f::new(pcamera.x, pcamera.y, pcamera.z).normalize())
        )
        // Ray::<f64>::new(
        //      Point3f::new(0.0, 0.0, 0.0),
        //     Vector3f::new(0.0, 0.0, 0.0).normalize()

        //   )
    }
    fn is_point_inside_film(&self, pWS: Point3f) -> bool {
        let pfocusVS = self
            .cameraToWorldMatrix
            .inverse_transform()
            .unwrap()
            .transform_point(pWS);
        let to_raster = self.rasterToCameraMatrix.inverse_transform().unwrap();
        // let w2c = self.cameraToWorldMatrix.inverse_transform().unwrap();
      
      
       let praster = to_raster.transform_point(pfocusVS);


        // let cameratoraster = to_raster * self.cameraToWorldMatrix.inverse_transform().unwrap();
        // let praster = cameratoraster.transform_point(pfocusVS);
        let siestrue =  praster.y < 0.0 ;
        let is_praster_y_lesszero  = is_zero_fixed(praster.y) < 0.0 ;
        let is_praster_x_lesszero  = is_zero_fixed(praster.x) < 0.0 ;
       let is_praster_x = praster.x<0.0;
       let is_praster_y =  praster.y<0.0;
        let filmmaxx = self.filmres.0 as f64;
        let filmmaxy = self.filmres.1 as f64;
     
        if  is_praster_x || praster.x >= filmmaxx ||  is_praster_y || praster.y >= filmmaxy {
            return false;
        }
        return true;
    }

    /**
     * sample importance of incident light
     * 
     */
    pub fn sample_importance(&self, recin:RecordSampleImportanceIn)->RecordSampleImportanceOut{

        let pcamera = self.cameraToWorldMatrix .transform_point( Point3f::new(0.0,0.0,0.0) );
         
        let normaldircamera = self
            .cameraToWorldMatrix
            .transform_vector(Vector3f::new(0.0, 0.0, 1.0)).normalize();
        let next  = ( pcamera-recin.hit.point).normalize();
     //   println!("{:?}", "hay que hacer una interaction de visibiiilida para hacer el sample_importance");
        

        let dist = recin.hit.point.distance( pcamera );
        let d2 = dist*dist;
        let absdot = normaldircamera.dot(next).abs();
        let pdf =  d2 / absdot;
        let negnext = -next;
        
        let (weight, pworldraster )  =self.weight(Ray::<f64>::new(pcamera,  negnext ));
        RecordSampleImportanceOut{
            n:next,
            pdf,
            prasterinworld: pworldraster,
            ray:Some(Ray::<f64>::new(pcamera,  negnext)),
            weight
        }
        
    }
    pub fn weight(&self, r : Ray<f64> )->(Srgb, Point3f){
        let dircamera = self.cameraToWorldMatrix.transform_vector(Vector3f::new(0.0, 0.0, 1.0));
         let costheta = dircamera.dot( r.direction );
         if costheta <= 0.0 {
          return   (Srgb::new(0.0,0.0,0.0), Point3f::new(0.0,0.0,0.0));
            
        }
        let focusworld = r.at(1.0 / costheta);
        if !self.is_point_inside_film(focusworld){
            return  (Srgb::new(0.0,0.0,0.0),   Point3f::new(0.0,0.0,0.0));
        }
        let costheta2 = costheta*costheta;
        let a = self.area();
       let wgh =  1.0 /(a * costheta2*costheta2);
     (  Srgb::new(wgh as f32, wgh as f32, wgh as f32), focusworld)

    }
    /**
     * 
     * return  (pdfpos, pdfdir)
     */
    pub fn pdfWemission(&self, r: &Ray<f64>) -> (f64, f64) {
        let dircamera = self
            .cameraToWorldMatrix
            .transform_vector(Vector3f::new(0.0, 0.0, 1.0));
        let costheta = dircamera.dot(r.direction);
        if costheta <= 0.0 {
            let pdfpos = 0.0;
            let pdfdir = 0.0;
            return (pdfpos, pdfdir);
        }
        let focusworld = r.at(1.0 / costheta);
        if !self.is_point_inside_film(focusworld){
          let pdfpos = 0.0;
          let pdfdir = 0.0;
          return (pdfpos, pdfdir);

        }
        let pdfpos = 1.0;
    
        let pdfdir = 1.0 / (self.area() * costheta * costheta * costheta);
        (pdfpos, pdfdir)
    }
}

#[test]

pub fn debug_camera() {
    
    let filmres: (u32, u32) = (512, 512);
    let perspectiveMatrix = PerspectiveCamera::perspective(1e-3, 1000.0, 45.0);
    // println!("{:?}", permat)
    let screenToRaster =
        PerspectiveCamera::fromperspectiveSpacetoRaster(&filmres, &((-1.0, -1.0), (1.0, 1.0))); //
    let rasterToScreenMatrix = screenToRaster.inverse_transform().unwrap();
    let rasterToCameraMatrix =
        perspectiveMatrix.inverse_transform().unwrap() * rasterToScreenMatrix;
    println!("perspectiveMatrix {:?}", perspectiveMatrix);
    println!("screenToRaster {:?}", screenToRaster);
    println!("rasterToScreenMatrix {:?}", rasterToScreenMatrix);
    println!("screenToRaster {:?}", screenToRaster);
    println!("rasterToCameraMatrix{:?}", rasterToCameraMatrix);

    let rasterPmin = rasterToCameraMatrix.transform_point(Point3f::new(0.0, 0.0, 0.0));
    let rasterPmax =
        rasterToCameraMatrix.transform_point(Point3f::new(filmres.0 as f64, filmres.1 as f64, 0.0));

    println!(" rasterPmin {:?}", rasterPmin / rasterPmin.z);
    println!(" rasterPmax{:?}", rasterPmax / rasterPmin.z);
    let pfilmVSmin = rasterPmin / rasterPmin.z;
    let pfilmVSmax = rasterPmax / rasterPmin.z;
    let areaVS = ((pfilmVSmax.x - pfilmVSmin.x) * (pfilmVSmax.y - pfilmVSmin.y)).abs();
    println!(" areaVS {:?}", areaVS);
    let camera = PerspectiveCamera {
        focal_length: 1e6,
        far: 1000.0,
        near: 1e-3,
        fovy: 45.0,
        filmres,
        screenToRasterMatrix: screenToRaster,
        rasterToCameraMatrix,
        perspectiveMatrix,
        rasterToScreenMatrix,
        cameraToWorldMatrix: Matrix4::identity(),
        areaVS: areaVS,
    };
    let cameraInstance = PerspectiveCamera::new(1e-3, 1000.0, 45.0,  filmres);
    println!(" cameraInstance.area {:?}", cameraInstance.area());

   let rec =  cameraInstance.pdfWemission(&Ray::<f64>::new(
        Point3f::new(0.0, 0.0, 0.0),
        Vector3f::new(0.0, 0.0, 1.0),
    ));
    println!("{:?}", rec);
    let r = cameraInstance.get_ray(&(6.0,256.0));
}








#[test]

pub fn debug_camera_look() {
    let filmres: (u32, u32) = (32, 32);
    let cameranew = PerspectiveCamera::new(1e-3, 1000.0, 45.0, filmres);  
   let cameralook =  PerspectiveCamera::from_lookat(Point3f::new(0.0,0.0,0.0),Point3f::new(0.0,0.0,11.0),1e-3,1000.0,45.0,filmres);
   
   for (i ,j) in itertools::iproduct!(1..32,1..32)  {
       let p =  (i as f64 / 32.0,j as f64 / 32.0);
       let rlook =     cameralook.get_ray(&p);
       let rnew =     cameranew.get_ray(&p);
       println!("{:?},{:?}", rlook.origin == rnew.origin,  rlook.direction == rnew.direction );
       
        for (ii ,jj) in itertools::iproduct!(0..3,0..4)  { 
            let psample =  (ii as f64 / 32.0,jj as f64 / 32.0);
           let hit  = HitRecord::from_point(Point3f::new(psample.0,psample.1,1.0), Vector3f::new(0.0,0.0,0.0));
           let recoutlook =  cameralook.sample_importance(RecordSampleImportanceIn { praster: Point3f::new(p.0, p.1, 0.0), psample, hit  });
           let recoutnew =  cameranew.sample_importance(RecordSampleImportanceIn { praster: Point3f::new(p.0, p.1, 0.0), psample, hit  });
            // println!("{:?} {}", recoutlook.weight == recoutnew.weight, recoutlook.pdf ==  recoutnew.pdf);
       //      println!("{:?} ,{:?}   ", recoutlook.n , recoutnew.n );
            assert!(recoutlook.weight == recoutnew.weight );
            assert!(recoutlook.pdf ==  recoutnew.pdf );
           assert!(recoutlook.n == recoutnew.n ); 
           assert!(recoutlook.ray.unwrap().direction == recoutnew.ray.unwrap().direction );
           assert!( recoutlook.ray.unwrap().origin ==  recoutnew.ray.unwrap().origin);
           assert!(  cameralook.pdfWemission(&recoutlook.ray.unwrap()) == cameranew.pdfWemission(&recoutnew.ray.unwrap()) );

        }
    //    cameralook.sample_importance(RecordSampleImportanceIn { praster: (), psample: (), hit: () })
   }
     
      

}






#[test]

pub fn debug_camera_look1() {
    let w = 8;
    let filmres: (u32, u32) = (w,w);
   
   let cameralook =  PerspectiveCamera::from_lookat(Point3f::new(0.0,1.0,0.0),Point3f::new(0.0,0.0,1.0),1e-2,1000.0,45.0,filmres);
   
   for (i ,j) in itertools::iproduct!(0..w,0..w)  {
    let j = 4;
    let i = 4;
       let p =  (j as f64 / w as f64,i as f64 / w as f64);
       let rlook =     cameralook.get_ray(&p);
    //    let rnew =     cameranew.get_ray(&p);
       println!("{} {} {:?} \n   {:?},\n     {:?}",  j,i, p,  rlook.origin  ,  rlook.direction  );
       
    //     for (ii ,jj) in itertools::iproduct!(0..3,0..4)  { 
    //         let psample =  (ii as f64 / 32.0,jj as f64 / 32.0);
    //        let hit  = HitRecord::from_point(Point3f::new(psample.0,psample.1,1.0), Vector3f::new(0.0,0.0,0.0));
    //        let recoutlook =  cameralook.sample_importance(RecordSampleImportanceIn { praster: Point3f::new(p.0, p.1, 0.0), psample, hit  });
    //     //    let recoutnew =  cameranew.sample_importance(RecordSampleImportanceIn { praster: Point3f::new(p.0, p.1, 0.0), psample, hit  });
    //         // println!("{:?} {}", recoutlook.weight == recoutnew.weight, recoutlook.pdf ==  recoutnew.pdf);
    //    //      println!("{:?} ,{:?}   ", recoutlook.n , recoutnew.n );
    //     //     assert!(recoutlook.weight == recoutnew.weight );
    //     //     assert!(recoutlook.pdf ==  recoutnew.pdf );
    //     //    assert!(recoutlook.n == recoutnew.n ); 
    //     //    assert!(recoutlook.ray.unwrap().direction == recoutnew.ray.unwrap().direction );
    //     //    assert!( recoutlook.ray.unwrap().origin ==  recoutnew.ray.unwrap().origin);
    //     //    assert!(  cameralook.pdfWemission(&recoutlook.ray.unwrap()) == cameranew.pdfWemission(&recoutnew.ray.unwrap()) );

    //     }
    //    cameralook.sample_importance(RecordSampleImportanceIn { praster: (), psample: (), hit: () })
   }
     
      

}

















  // -1.5258789062500000e-05
  const   MINUS_ZERO :f64= hexf::hexf64!("-0x1.0p-32") as  f64 ;
  // +1.5258789062500000e-05
  const    PLUS_ZERO :f64 = hexf::hexf64!("0x1.0p-32") as  f64 ;


  
  
fn is_zero_fixed(f:f64)->f64{
   if is_zero(f) { 0.0 } else { f }
   
}
fn is_zero(f:f64)->bool{
  
    if  f.signum()==1.0 as f64 {
         return  f<PLUS_ZERO 
   } else {
    // if f.signum()==-1.0 as f64
        return   f>MINUS_ZERO
   }
   
}
#[test]
pub fn testing_is_zero() {
    let f =  -1e-17 as f64;
    let f1 =  -1e-7 as f64; // debe ser 0
    let f2 =  -1e-4 as f64; // no es  0
    let plusf =  1e-17 as f64;
    let plusf1 =  1e-7 as f64; // debe ser 0
    let plusf2 =  1e-4 as f64; // no es  0
  
    assert!(is_zero(f)==true);
    assert!(is_zero(f1)==true);
    assert!(is_zero(f2)==false);
    assert!(is_zero(plusf)==true);
    assert!(is_zero(plusf1)==true);
    assert!(is_zero(plusf2)==false);
//    println!("{:?}", is_zero(f));
//    println!("{:?}", is_zero(f1));
//    println!("{:?}", is_zero(f2));
//    println!("{:?}", is_zero(plusf));
//    println!("{:?}", is_zero(plusf1));
//    println!("{:?}", is_zero(plusf2));
}
