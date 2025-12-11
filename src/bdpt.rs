use std::borrow::{Cow, BorrowMut};
use std::cell::{RefCell, RefMut};
use std::fmt::Debug;
use std::ops::Deref;
use std::rc::Rc;
use std::sync::Arc;

use cgmath::{Vector3, Point3, Vector1, InnerSpace};
use float::Sqrt;
use num_traits::Float;
use palette::Srgb;

use crate::Cameras::{PerspectiveCamera, RecordSampleImportanceIn};
use crate::Lights::{Light, LigtImportanceSampling, IsAreaLight, RecordSampleLightIlumnation, PdfEmission, RecordPdfEmissionIn, GetEmission};
use crate::integrator::isBlack;
use crate::materials::{RecordSampleIn, Pdf, IsSpecular, Fr};
use crate::{Point3f, Vector3f};
use crate::sampler::Sampler;
use crate::primitives::prims::{Ray, IntersectionOclussion, IsEmptySpace};
use crate::primitives::prims::HitRecord;
 use crate::raytracerv2::{Scene, interset_scene, interset_scene_primitives, interset_scene_bdpt};//
use crate::materials::SampleIllumination;
use crate::Lights::SampleEmission;
use crate::Lights::RecordSampleLightEmissionIn ;
use crate::Lights::RecordSampleLightEmissionOut ;
 
use crate::Lights::IsBackgroundAreaLight ;
use crate::Lights::SampleLightIllumination;





#[derive( Clone,  Debug)]
pub struct ModifiedVtxAttr {
   pdfrev : f64,
  pub  delta: bool,
  pub  has_change: bool,
  pub sampled : Option<Rc<PathTypes>>
//  pub currentVtx : Option<impl PathTypes>,
}

impl  ModifiedVtxAttr  {
    pub fn new(pdfrev: f64, delta: bool, sampledVtx : PathTypes) -> Self { 
       
        Self { pdfrev, delta , has_change:true  , sampled:None
            // sampled:None
         } 
    }
    pub fn from_zero(   ) -> Self {
        Self { pdfrev: 0.0, delta : false , has_change:false, sampled:None }
     }
    pub fn from_vtx(  sampledVtx : PathTypes) -> Self { 
        
        Self { pdfrev : 0.0,  delta : false , has_change:true  , sampled:Some(Rc::new(sampledVtx)) } 
    }
    pub fn from_vtx_pdfrev(  sampledVtx : PathTypes ,  pdfrev : f64) -> Self { 
        
        Self { pdfrev  ,  delta : false , has_change:true  , sampled:Some(Rc::new(sampledVtx)) } 
    }
    pub fn from_pdf_delta(  pdfrev : f64, delta : bool) -> Self { 
        
        Self { pdfrev  ,  delta  , has_change:true ,sampled:None } 
    }
    pub fn from_pdf(  pdfrev : f64, delta : bool) -> Self { 
        
        Self { pdfrev  ,  delta  , has_change:true, sampled:None } 
    }
    pub fn clear( )->Self{
        Self { pdfrev : 0.0  , delta : false  , has_change:false ,sampled:None } 
    }

    /// Get a reference to the modified vtx attr's delta.
    pub fn delta(&self) -> bool {
        self.delta
    }

    /// Get a mutable reference to the modified vtx attr's delta.
    pub fn delta_mut(&mut self) -> &mut bool {
        self.has_change = true;
        &mut self.delta
    }

    /// Set the modified vtx attr's delta.
    pub fn set_delta(&mut self, delta: bool) {
        self.has_change = true;
        self.delta = delta;
    }

    /// Get a mutable reference to the modified vtx attr's pdfrev.
    pub fn pdfrev_mut(&mut self) -> &mut f64 {
        &mut self.pdfrev
    }

    /// Get a reference to the modified vtx attr's has change.
    pub fn has_change(&self) -> bool {
        self.has_change
    }
   
}

 
pub struct List<T>{
    head : Link<T>,
}

type Link<T> = Option<Box<Node<T>>>;
struct  Node<T>{
    elem: T,
    next : Link<T>,
}

impl<T> List<T>{
    pub fn new()->Self{
        List{head:None}
    }
    pub fn push(&mut self, t :T) {
        let newnode = Box::new(
            Node{
                elem:t,
                next:self.head.take()
            }
        );
        self.head = Some(newnode);
    }
    pub fn pop(&mut self)->Option<T>{
        self.head.take().map(|node|{
            self.head = node.next;
            node.elem
        })
    }
    pub fn peek(&self) -> Option<&T> {
        self.head.as_ref().map(|node| {
            &node.elem
        })
    }

    pub fn peek_mut(&mut self) -> Option<&mut T> {
        self.head.as_mut().map(|node| {
            &mut node.elem
        })
    }

}


impl<T> Drop for List<T> {
    fn drop(&mut self) {
        let mut cur_link = self.head.take();
        while let Some(mut boxed_node) = cur_link {
            cur_link = boxed_node.next.take();
        }
    }
}
pub struct IntoIter<T>(List<T>);
impl<T> List<T >{
    pub fn into_iter(self)->IntoIter<T>{
        IntoIter(self)
    }
}


impl <T>  Iterator for IntoIter<T>{
    type Item = T;
    fn next(&mut self)->Option<Self::Item>{
        self.0.pop()
    }
}

























#[derive( Clone, Copy ,Debug)]
pub enum Mode{
    
     from_camera, // Radiance, 
   from_light //  Importance  
}
#[derive( Clone , Debug)]
pub struct PathVtx(
    
    Vector3f, // normal interaciont
    Point3f,  // point interaction
    Srgb,      // beta 
    (f64, f64), /*pdfnext, pdfprev*/
     Option<HitRecord<f64>>, 
     Mode ,// transport mode,
    //  Option<Rc<ModifiedVtxAttr>>,
    Rc<ModifiedVtxAttr>, 
    Option<bool> ,    // es la distribucion que representa el vertice una distribucion de prob delta?
    Option<Light> , // este vertice es un arealight.
    );
impl  PathVtx {
    pub fn new_pathvtx(hit:HitRecord<f64>,transport:Srgb, pdfnext:f64,mode:Mode)->Self{ 
        PathVtx(   hit.normal ,hit.point, transport,(pdfnext,0.0), Some(hit),mode, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), None)
    }
    pub fn new_pathvtx1(hit:HitRecord<f64>,transport:Srgb, pdfnext:f64,pdfrev:f64,mode:Mode, light : Option<Light>)->Self{ 
        PathVtx(   hit.normal ,hit.point, transport,(pdfnext,pdfrev), Some(hit),mode, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), light)
    }
  
    pub fn from_zero()->Self{ 
        PathVtx( Vector3f::new(0.0,0.0,0.0) ,Point3f::new(0.0,0.0,0.0), Srgb::default(),(0.0,0.0), None,Mode::from_camera, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), None)
    }
    // remove!
    pub fn from_zero1(f:f64)->Self{ 
        PathVtx( Vector3f::new(0.0,0.0,0.0) ,Point3f::new(0.0,0.0,0.0), Srgb::default(),(0.0,f), None,Mode::from_camera, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), None)
    }
}

#[derive( Clone , Debug)]
pub  struct PathLightVtx (
    Vector3f,   //  n()
    Point3f,  // p()  
    Srgb, // beta 
    (f64, f64)  ,/*pdfnext, pdfprev*/
    Option<HitRecord<f64>>, 
   Option< Light>,
    Rc<ModifiedVtxAttr>,
    bool // is_endpoint_path()
);
impl  PathLightVtx {
    pub fn new_lightpathvtx(light:  Light, n : Vector3f, Lemission : Srgb,  pdf : f64)->Self{
        PathLightVtx( 
            n,
            light.clone().getPositionws(),
            Lemission,
            (pdf,0.0), /*pdfnext, pdfprev*/
            None,
            Some(light),
            Rc::new(ModifiedVtxAttr::from_zero()),
            false
        )
    }
    pub fn new_lightpathvtx1(light:  Light, n : Vector3f, Lemission : Srgb,  pdf : f64, poslight : Option<Point3f> )->Self{
        let poslight=match poslight {
            Some(p)=>p,
            None=>light.getPositionws()
        };
        PathLightVtx( 
            n,
            poslight,
            Lemission,
            (pdf,0.0), /*pdfnext, pdfprev*/
            None,
            Some(light),
            Rc::new(ModifiedVtxAttr::from_zero()),
            false
        )
    }
    pub fn from_hitrecord(light:  Light, hit : HitRecord<f64>, Lemission : Srgb,  pdf : f64)->Self{
       
        PathLightVtx( 
            hit.normal,
            hit.point,
            Lemission,
            (pdf,0.0), /*pdfnext, pdfprev*/
            Some(hit),
            Some(light)
            ,
            Rc::new(ModifiedVtxAttr::from_zero()),
            false
        )
    }
    pub fn from_endpoint(  r : Ray<f64> , transport : Srgb,  pdf : f64)->Self{
        PathLightVtx( 
           -r.direction.normalize(),
            r.at(1.0),
            transport,
            (pdf,0.0), /*pdfnext, pdfprev*/
            None,
           None
            ,
            Rc::new(ModifiedVtxAttr::from_zero()), 
            true // is a endpoint
        )
    }
}

#[derive( Clone , Debug)]
struct PathCameraVtx (
    f64,
    Vector3f, Point3f, Srgb,  /*pdfnext, pdfprev*/(f64, f64),  
     PerspectiveCamera,
      Option<Ray<f64>>,
      Option<HitRecord<f64>>,
      Rc<ModifiedVtxAttr>

    );
impl  PathCameraVtx  {
    pub fn new_camerapathvtx(camera:  PerspectiveCamera)->Self{
      
        PathCameraVtx(0.0,Vector3::new(0.0,0.0,0.0),Point3::new(0.0,0.0,0.0), Srgb::new(1.00,1.0,1.0),(0.0,0.0), camera,None, None, Rc::new(ModifiedVtxAttr::from_zero()))
    }
    pub fn new_camerapath_init(camera:  PerspectiveCamera, r : Ray<f64>)->Self{
      
        PathCameraVtx(0.0,Vector3::new(0.0,0.0,0.0),r.origin, Srgb::new(1.00,1.0,1.0),(0.0,0.0), camera,Some(r), None, Rc::new(ModifiedVtxAttr::from_zero()))
    }
    pub fn new_camerapath_init_with_hitpoint(camera:  PerspectiveCamera, r : Ray<f64>, hit : HitRecord<f64>)->Self{
      
        PathCameraVtx(0.0,Vector3::new(0.0,0.0,0.0),Point3::new(0.0,0.0,0.0), Srgb::new(0.0,0.0,0.0),(0.0,0.0), camera,Some(r), Some(hit),Rc::new(ModifiedVtxAttr::from_zero()))
    }
    pub fn new_camerapath_init_(camera:  PerspectiveCamera, r : Ray<f64>,  beta : Srgb , pdf : f64 )->Self{
     
        PathCameraVtx(
                0.0,
                r.direction.normalize(), 
                r.origin,  // en mis weight se pide el origen de la camara en world space 
        Srgb::new(beta.red /(pdf as f32),beta.green /(pdf as f32),beta.blue /(pdf as f32)),(0.0,0.0), camera,Some(r), None, Rc::new(ModifiedVtxAttr::from_zero()))
    }
 }


 
 pub  trait BdptVtx {
    fn n(&self)->Vector3f;
    fn p(&self)->Point3f;
    fn pdf(&self, scene:&Scene<f64>,  prev : Option<&PathTypes>,  next : &PathTypes)->f64;
    fn get_pdfrev(&self)->f64;
    fn get_pdfnext(&self)->f64;
    //fn isbackgroundlight(&self)->bool;
    fn fr(&self, nxt : PathTypes)->Srgb;
    fn solidarea(&self, nxt : Vector3f)->bool;
   
  
    fn transport(&self) ->&Srgb;
    fn set_pdfrev(& mut self, pdf:f64);
    fn set_pdfnext(& mut self, pdf:f64);
    fn get_hit(&  self)->Option<HitRecord<f64>>;
    fn get_light(&  self)->Option<Light>;
    fn set_modified_vtx_attr(&mut  self, data:ModifiedVtxAttr);
    fn has_modified_attr(&self)->bool;
    fn get_modified_attr(&self)->&ModifiedVtxAttr;
    fn get_emission(&self, scene:&Scene<f64>,prev: &PathTypes)->Srgb;
   
}

trait LightPdfFromOrigin {
    fn pdf_light(&self, vtxext : &PathTypes )->(
            f64    //prob pos
            ,f64 // prob dir
            ,f64 //probChoice
        );
}
trait LightPdfFromDir {
    fn pdf_light_dir(&self, vtxext : &PathTypes )->f64;
}
#[derive( Clone , Debug)]
pub enum PathTypes{
    PathCameraVtxType( PathCameraVtx ),
    PathLightVtxType(PathLightVtx),
    PathVtxType(PathVtx),
   None
}
 
trait  EmissionDbpt {
    fn emission(&self);
}
trait  SolidArea {
    fn solidarea(&self);
}
pub
trait  IsLightBdpt {
    fn is_light(&self)->bool;
}
trait  IsDeltaLight {
    fn is_deltalight(&self)->bool;
}
trait  CreateVtx  {
    fn createCameraVtxType(camera : PerspectiveCamera, r :Ray<f64>, wemission : Srgb)->PathTypes   ;
    fn createLightVtxType(light:Light, n : Vector3f, Lemission : Srgb,  pdf : f64)->PathTypes;
    fn createSurfaceVtxType(hit:HitRecord<f64>,transport:Srgb, pdfnext:f64,mode:Mode)->PathTypes ;
}

impl   CreateVtx  for PathTypes{
    fn createCameraVtxType(camera :  PerspectiveCamera , r :Ray<f64>,wemission : Srgb ) ->PathTypes {
        PathTypes::PathCameraVtxType(PathCameraVtx::new_camerapathvtx(camera ))
    }
    fn createLightVtxType(light:Light, n : Vector3f, Lemission : Srgb,  pdf : f64) ->PathTypes {
        PathTypes::PathLightVtxType(PathLightVtx::new_lightpathvtx(light,n,  Lemission, pdf))
    }
    fn createSurfaceVtxType(hit:HitRecord<f64>,transport:Srgb, pdfnext:f64,mode:Mode) ->PathTypes  {
        PathTypes::PathVtxType(PathVtx::new_pathvtx(hit,transport, pdfnext, mode))
    }
    
}
 trait   IsAreaLightBdpt{
    fn is_arealight(&self)->bool  ;

 }
 impl  IsAreaLightBdpt for PathLightVtx {
     fn is_arealight(&self)->bool  {
       let light =  self.5.as_ref().unwrap();
       light.is_arealight() || light.is_background_area_light()
    }
 }
impl   IsLightBdpt for PathTypes{
    fn is_light(&self)->bool {
         match self {
             Self:: PathCameraVtxType(p)=>false,
             Self::PathLightVtxType(p)=>p.is_light(),
             Self::PathVtxType(p)=>p.is_light(),
             _=>panic!(),
         }
    }
}
impl   IsLightBdpt for PathLightVtx {
    fn is_light(&self)->bool {
        
        true
        
    }
}
impl   IsLightBdpt for PathCameraVtx {
    fn is_light(&self)->bool {
        
        false
        
    }
}
impl   IsLightBdpt for PathVtx {
    fn is_light(&self)->bool {
        self.get_light().is_some()
    }
}




impl   IsDeltaLight for PathTypes{
    fn is_deltalight(&self)->bool {
         match self {
             Self:: PathCameraVtxType(l)=>false,
             Self::PathLightVtxType(pl)=>pl.is_deltalight(),
             Self::PathVtxType(l)=>false,
             _=>panic!(),
         }
    }
}
impl   IsDeltaLight for PathLightVtx {
    fn is_deltalight(&self)->bool {
        !self.5.as_ref().is_none() && !self.5.as_ref().unwrap().is_arealight() 
    }
}


trait IsDistributionDelta  {
    fn is_distribution_delta(&self)->bool  ;
}


impl   IsDistributionDelta for PathTypes{
    fn is_distribution_delta(&self)->bool {
         match self {
             Self:: PathCameraVtxType(l)=>false,
             Self::PathLightVtxType(pl)=>false,
             Self::PathVtxType(l)=>false,
             _=>panic!(),
         }
    }
}
impl  IsDistributionDelta for PathLightVtx {
     fn is_distribution_delta(&self) ->bool {
         false
     }
}





trait   IsConnectable{
    fn is_connectable(&self)->bool ;
}
impl   IsConnectable  for PathTypes {
    fn is_connectable(&self) ->bool {
        match self {
            Self:: PathCameraVtxType(l)=>true,
            Self::PathLightVtxType(l)=>{
                l.is_light() ||  l.is_arealight()
            },
            Self::PathVtxType(l)=>true,
            _=>panic!(),
        }
    
    }
}



trait  IsOnSurface {
    fn is_on_surface(&self)->bool;
}
impl   IsOnSurface for PathTypes{
    fn is_on_surface(&self) ->bool  {
         match self {
             Self:: PathCameraVtxType(l)=>false,
             Self::PathLightVtxType(l)=>l.5.as_ref().unwrap().is_arealight(),
             Self::PathVtxType(l)=>true,
             _=>panic!(),
         }
    }
}
impl   IsOnSurface for PathCameraVtx{
    fn is_on_surface(&self) ->bool  {
          false
    }
}
impl   IsOnSurface for PathVtx{
    fn is_on_surface(&self) ->bool  {
          true
    }
}
impl   IsOnSurface for PathLightVtx {
    fn is_on_surface(&self) ->bool  {
        // self.5.unwrap().is_arealight()
          false // si es un area light esto es true
    }
}
 

trait  ConvertDensityBdpt {
    fn convert(&self,  pdf : f64, other: & PathTypes )->f64;
}
impl  ConvertDensityBdpt for PathTypes {
    fn convert(&self, pdf : f64,  other: & PathTypes )   ->f64{
         match self {
             Self:: PathCameraVtxType(l)=>l.convert( pdf,  other),
             Self::PathLightVtxType(l)=>l.convert( pdf,   other),
             Self::PathVtxType(l)=>l.convert(  pdf,  other),
             _=>panic!(),
         }
    }
}
impl  ConvertDensityBdpt for PathCameraVtx{
    fn convert(&self, pdf : f64,  other: & PathTypes ) ->f64{
        let mut pdfrev = pdf;
        let v =other.p()- self.p()  ;
        let d2 = v.magnitude2();
        if d2 == 0.0 {
            return 0.0
        }
        let inv = 1.0 /  d2;
        if other.is_on_surface(){
            let b =  v *    d2.sqrt();;
            let b = b.normalize();
            // println!("{:?}",prev.n());
            // println!("{:?}",b);
            pdfrev *= other.n() .dot(b).abs();
        }
       pdfrev*inv 
    }
}
impl  ConvertDensityBdpt for PathLightVtx {
    fn convert(&self, pdfold : f64, other:& PathTypes ) ->f64{
        let mut pdf :f64 = pdfold;
        let mut v = other.p() - self.p();
        let l2 = v.magnitude2();
       if  l2 == 0.0 { return 0.0}
       let invl2 = 1.0/ l2;
    
      
      if other.is_on_surface(){
       let b =  v *    invl2.sqrt();;
        pdf *= other.n() .dot(b).abs();
      }
       pdf*invl2
    }
}
impl  ConvertDensityBdpt for PathVtx{
    fn convert(&self, pdfold : f64 , other:& PathTypes ) ->f64{
        let mut pdf :f64 = pdfold;
        let mut v = other.p() - self.p();
        let l2 = v.magnitude2();
       if  l2 == 0.0 {}
       let invl2 = 1.0/ l2;
    
      
      if other.is_on_surface(){
       let b =  v *    invl2.sqrt();
        pdf *= other.n() .dot(b).abs();
      }
       pdf*invl2

         
    }
}








// trait  PdfBdpt {
//     fn pdf(&self);
// }
// impl   PdfBdpt for PathTypes{
//     fn pdf(&self)   {
//          match self {
//              Self:: PathCameraVtxType(l)=>l.convert(),
//              Self::PathLightVtxType(l)=>l.convert(),
//              Self::PathVtxType(l)=>l.convert(),
//          }
//     }
// }
// impl   PdfBdpt for PathCameraVtx{
//     fn pdf(&self) {
         
//     }
// }
// impl   PdfBdpt for PathLightVtx{
//     fn pdf(&self) {
         
//     }
// }
// impl   PdfBdpt for PathVtx{
//     fn pdf(&self) {
         
//     }
// }





















// trait  PdfLight{
//     fn pdf_light(&self)->f64;
// }

// impl<'a>   PdfLight for PathLightVtx<'a> {
//    fn pdf_light(&self) ->(f64,f64) {
//       (0.0,0.0)
//    }
// }
//  pub

// trait  IsBckLight {
//     fn is_bck_light(&self)->bool;
// }
 
// impl  IsBckLight for PathTypes{
//     fn is_bck_light(&self)->bool {
//          match self {
//              Self:: PathCameraVtxType(l)=>false,
//              Self::PathLightVtxType(l)=>{
//                 let  mut isbck = false;
//                 // is endpoint light vtx?
//                 isbck |= l.7; 
//                 if l.is_light(){
//                     isbck |=l.get_light().unwrap().is_background_area_light();
//                 }
//                 return  isbck;;
//              },
//              Self::PathVtxType(l)=>false,
//          }
//     }
// }

impl  BdptVtx for PathTypes {
    fn n(&self)->Vector3<f64> {
        match self {
            Self:: PathCameraVtxType(v)=>v.n(),
            Self::PathLightVtxType(v)=>v.n(),
            Self::PathVtxType(v)=>v.n(),
            _=>panic!()
        } 
    }
    fn p(&self) ->Point3<f64>{
        match self {
            Self:: PathCameraVtxType(v)=>v.p(),
            Self::PathLightVtxType(v)=>v.p(),
            Self::PathVtxType(v)=>v.p(),
            _=>panic!()
        } 
    }
    fn pdf(&self, scene:&Scene<f64>, prev :Option<&PathTypes>,  next : &PathTypes) ->f64 {
        match self {
            Self:: PathCameraVtxType(v)=>v.pdf(scene, prev, next),
            Self::PathLightVtxType(v)=>v.pdf(scene, prev, next),
            Self::PathVtxType(v)=>v.pdf(scene, prev, next),
            _=>panic!()
        } 
    }
    fn get_pdfnext(&self) ->f64 {
        match self {
            Self:: PathCameraVtxType(v)=>v.get_pdfnext(),
            Self::PathLightVtxType(v)=>v.get_pdfnext(),
            Self::PathVtxType(v)=>v.get_pdfnext(),
            _=>panic!()
        } 
    }
    fn get_pdfrev(&self) ->f64 {
        match self {
            Self:: PathCameraVtxType(v)=>v.get_pdfrev(),
            Self::PathLightVtxType(v)=>v.get_pdfrev(),
            Self::PathVtxType(v)=>v.get_pdfrev(),
            _=>panic!()
        } 
    }
    fn transport(&self) ->&Srgb {
        match self {
            Self:: PathCameraVtxType(v)=>v.transport(),
            Self::PathLightVtxType(v)=>v.transport(),
            Self::PathVtxType(v)=>v.transport(),
            _=>panic!()
        } 
    }
    

    fn fr(&self, nxt : PathTypes)->Srgb {
        match self {
            Self:: PathCameraVtxType(v)=>v.fr(nxt),
            Self::PathLightVtxType(v)=>v.fr(nxt),
            Self::PathVtxType(v)=>v.fr(nxt),
            _=>panic!()
        } 
    }

    fn solidarea(&self, nxt : Vector3<f64>)->bool {
        todo!()
    }
    fn set_pdfrev(& mut self, pdf:f64){ 

        match self {
            Self:: PathCameraVtxType(v)=>v.set_pdfrev(pdf),
            Self::PathLightVtxType(v)=>v.set_pdfrev(pdf),
            Self::PathVtxType(v)=>v.set_pdfrev(pdf),
            _=>panic!()
        } 
    }
    fn set_pdfnext(& mut self, pdf:f64){
        match self {
            Self:: PathCameraVtxType(v)=>v.set_pdfnext(pdf),
            Self::PathLightVtxType(v)=>v.set_pdfnext(pdf),
            Self::PathVtxType(v)=>v.set_pdfnext(pdf),
            _=>panic!()
        } 
    }
    fn get_hit(&  self)->Option<HitRecord<f64>>{
        match self {
            Self:: PathCameraVtxType(v)=>v.get_hit( ),
            Self::PathLightVtxType(v)=>v.get_hit( ),
            Self::PathVtxType(v)=>v.get_hit( ),
            _=>panic!()
        } 
    }
    fn get_light(&  self) ->Option<Light> {
        match self {
            Self:: PathCameraVtxType(v)=>{
                None
            },
            Self::PathLightVtxType(v)=>v.get_light(),
            Self::PathVtxType(v)=> {v.get_light()}
            _=>panic!()
        } 
    }
    fn set_modified_vtx_attr(&mut  self, data:ModifiedVtxAttr) {
        match self {
            Self:: PathCameraVtxType(v)=>v.set_modified_vtx_attr(data),
            Self::PathLightVtxType(v)=>v.set_modified_vtx_attr(data),
            Self::PathVtxType(v)=>v.set_modified_vtx_attr(data),
            _=>panic!()
        } 
    }
    fn has_modified_attr(&self) ->bool {
        match self {
            Self:: PathCameraVtxType(v)=>v.has_modified_attr(),
            Self::PathLightVtxType(v)=>v.has_modified_attr(),
            Self::PathVtxType(v)=>v.has_modified_attr(),
            _=>panic!()
        } 
    }
    fn get_modified_attr(&self) ->&ModifiedVtxAttr{
        match self {
            Self:: PathCameraVtxType(v)=>v.get_modified_attr(),
            Self::PathLightVtxType(v)=>v.get_modified_attr(),
            Self::PathVtxType(v)=>v.get_modified_attr(),
            _=>panic!()
        } 
    }
    fn get_emission(&self, scene:&Scene<f64>,prev: &PathTypes)->Srgb{
        match self {
            Self:: PathCameraVtxType(v)=>Srgb::new(0.0,0.0,0.0),
            Self::PathLightVtxType(v)=>v.get_emission(scene, prev),
            Self::PathVtxType(v)=>{v.get_emission(scene, prev)},
            _=>panic!()
        } 
    }
    // fn  is_endpoint_path(&self)->bool {
    //     match self {
    //         Self:: PathCameraVtxType(v)=>false,
    //         Self::PathLightVtxType(v)=>false,
    //         Self::PathVtxType(v)=>{v.get_emission(scene, prev)},
           
    //     } 
    // }
}

impl   LightPdfFromOrigin for PathTypes {
    fn pdf_light(&self, vtxext : &PathTypes ) ->(f64 , f64,   f64){
        match self {
            Self:: PathCameraVtxType(l)=>panic!(" LightPdfFromOrigin "),
            Self::PathLightVtxType(l)=>l.pdf_light(vtxext),
            Self::PathVtxType(l)=>l.pdf_light(vtxext),
            _=>panic!()
        } 
    }
}

impl   LightPdfFromDir for PathTypes {
    fn pdf_light_dir(&self, vtxext : &PathTypes ) ->  f64 {
        match self {
            Self:: PathCameraVtxType(l)=>panic!(" LightPdfFromOrigin "),
            Self::PathLightVtxType(l)=>l.pdf_light_dir(vtxext),
            Self::PathVtxType(l)=>l.pdf_light_dir(vtxext),
            _=>panic!()
        } 
    }
}

impl   BdptVtx for PathCameraVtx {
    fn n(&self)->Vector3<f64> {
        self.1
    }
    fn p(&self) ->Point3<f64>{
        self.2
    }
    fn pdf(&self, scene:&Scene<f64>, prev :Option<&PathTypes>,  next : &PathTypes) ->f64 {
        // raycamera = next.p() surface, self.p() - camera "raster"
        let mut  vn = next.p() - self.p();
        if vn.magnitude2() == 0.0 {
            return 0.0;
        }
        vn =  vn.normalize();
         
         // este point es cargado cuando hacer  new_camerapath_init_ dentro de cameraTracerStrategy()
       let pray =  self.2;
       let (pdfPosimpotance, pdfdirimporance) =  self.5.pdfWemission(&Ray::<f64>::new(pray, vn));
     //  let pdfarea = self.convert(pdfdirimporance,next );
     
     self.convert(pdfdirimporance,next )
    }
    fn get_pdfnext(&self) ->f64 {
       self.4.0
    }
    fn get_pdfrev(&self) ->f64 {
        self.4.1
    }
    fn transport(&self) ->&Srgb {
        &self.3
    }
    

    fn fr(&self, nxt : PathTypes)->Srgb {
        todo!()
    }

    fn solidarea(&self, nxt : Vector3<f64>)->bool {
        todo!()
    }
    fn set_pdfrev(& mut self, pdf:f64){
        self.4.1 = pdf;
    }
    fn set_pdfnext(& mut self, pdf:f64) {
            self.4.0 = pdf;
    }
    fn get_hit(&  self) ->Option<HitRecord<f64>> {
        self.7
    }
    fn get_light(&  self) ->Option<Light> {
            None
    }
    fn set_modified_vtx_attr(&mut  self, data :ModifiedVtxAttr) {
        self.8=Rc::new(data)
    }
    fn has_modified_attr(&self) ->bool {
        self.8.has_change()
    }
    
    fn get_modified_attr(&self) ->&ModifiedVtxAttr {
        &self.8
    }
    fn get_emission(&self, scene:&Scene<f64>, prev: &PathTypes) ->Srgb {
        Srgb::new(0.0,0.0, 0.0)
    }
}
impl   BdptVtx for PathLightVtx {
    fn n(&self)->Vector3<f64> {
        self.0
    }
    fn p(&self) ->Point3<f64>{
        self.1
    }
    fn pdf(&self, scene:&Scene<f64>, prev : Option<&PathTypes>,  next : &PathTypes) ->f64 {
       let invdist = 1.0/ (next.p() - self.p()).magnitude2();
       let  pdf = self. pdf_light(&next);
       let mut  pdfdir = pdf.1;

    //   if  self.is_bcklight(){}
       if next.is_on_surface(){
        pdfdir *=  (next.p() - self.p()).normalize().dot(  next.n()).abs()
      
       }
   
        pdfdir *invdist // pdflight
       
    }
    fn get_pdfnext(&self) ->f64 {
       self.3.0
    }
    fn get_pdfrev(&self) ->f64 {
        self.3.1
    }
    fn transport(&self) ->&Srgb {
        &self.2
    }

    
    

    fn fr(&self, nxt : PathTypes)->Srgb {
    //     let v = nxt.p()-self.p();
    //     if v.magnitude2()==0.0{
    //        return  Srgb::new(0.0,0.0,0.0);
    //     }
    //     v=v.normalize();
    //    let hit =  self.4.unwrap();
    //    let currenthit= self.4.unwrap();

    //   let prev = currenthit.prev.unwrap();
      // hit.material.fr(prev, nxt);
        todo!()
    }

    fn solidarea(&self, nxt : Vector3<f64>)->bool {
        todo!()
    }
    fn set_pdfrev(& mut self, pdf:f64){
        self.3.1 = pdf;
    }
    fn set_pdfnext(& mut self, pdf:f64) {
        self.3.0 = pdf;
    }
    fn get_hit(&  self) ->Option<HitRecord<f64>> {
       self.4
    }
    fn get_light(&self) ->Option<Light> {
        self.5.clone()
    }
    fn set_modified_vtx_attr(&mut  self, data :ModifiedVtxAttr) {
        self.6=Rc::new(data)
    }
    fn has_modified_attr(&self) ->bool {
 // self.6.unwrap().has_change()
     
     self.6.has_change()
    
        // (*self.6.unwrap()).has_change()
       //  self.6.unwrap().as_ref(). has_change()
    }
    fn get_modified_attr(&self) ->&ModifiedVtxAttr {
        &self.6
    }
    fn get_emission(&self, scene:&Scene<f64>, v: &PathTypes) ->Srgb {
        // self.is_bc
        if self.is_endpoint_path() {
            return  Srgb::new(0.0,0.0, 0.0)
        }
       
        let light  = self.5.as_ref().unwrap();
        if light.is_arealight() {
            let r = Ray::new(Point3f::new(0.0,0.0, 0.0), Vector3f::new(0.0,0.0,0.0));
           return   light.get_emission( self.get_hit(), &r);
           
        }
        
        Srgb::new(0.0,0.0, 0.0)
    }
}
pub trait  isEndpointPath {
    fn is_endpoint_path(&self)->bool;
}
impl isEndpointPath for PathLightVtx {
    fn is_endpoint_path(&self)->bool {
        self.7
    }
}
impl    LightPdfFromOrigin for PathLightVtx  {
    
     fn pdf_light(&self, vext:&PathTypes) ->(f64 , f64, f64){
      let light = self.5.as_ref().unwrap();
      let  mut v = vext.p() - self.p();
    if  v.magnitude2() == 0.0 {
        return (0.0,0.0, 0.0);
    }
    v = v.normalize();
   
    let r = RecordPdfEmissionIn{
        n:self.n(),
        ray:Some(Ray::new(self.p(), v))
    };
 

     let recout = light.pdf_emission(r);
     (recout.pdfpos, recout.pdfdir, 1.0)
    }
}
impl    LightPdfFromOrigin for PathVtx  {
    
    fn pdf_light(&self, vext:&PathTypes) ->(f64 , f64, f64){
        
    let norm = self.n();
    let pt = self.p();
     let lightarea = self.8.as_ref().unwrap();
     let  mut v = vext.p() - self.p();
   if  v.magnitude2() == 0.0 {
       return (0.0,0.0, 0.0);
   }
   v = v.normalize();

   let discreteselectlight = 1.0;

   let r = RecordPdfEmissionIn{
       n:self.n(),
       ray:Some(Ray::new(self.p(), v))
   };

    let recout = lightarea.pdf_emission(r);
    (recout.pdfpos, recout.pdfdir, discreteselectlight)
   }
}


impl    LightPdfFromDir for PathLightVtx  {
    
    fn pdf_light_dir(&self, vext:&PathTypes) ->f64  {
        panic!("PathLightVtx::LightPdfFromDir")
//      let light = self.5.as_ref().unwrap();
//      let  mut v = vext.p() - self.p();
//    if  v.magnitude2() == 0.0 {
//        return (0.0,0.0, 0.0);
//    }
//    v = v.normalize();
//    panic!("piensa en como funciona esta mierda");
//    let r = RecordPdfEmissionIn{
//        n:Vector3 { x: 0.0, y: 0.0, z: 0.0 },
//        ray:Some(Ray::new(Point3f::new(0.0,0.0,0.0), Vector3f::new(0.0,0.0,0.0)))
//    };

//     let recout = light.pdf_emission(r);
//     (recout.pdfpos, recout.pdfdir, 1.0)
   }
}



impl    LightPdfFromDir for PathVtx  {
    
    fn pdf_light_dir(&self, next:&PathTypes) ->f64  {
     
      
         let lightarea = self.8.as_ref().unwrap();
         let  mut v = next.p() - self.p();
         let d2 = v.magnitude2() ;
       if d2 == 0.0 {return 0.0;}
       let mut pdf = 0.0;
       v = v.normalize();
       if lightarea.is_background_area_light(){
            panic!("require implementation");
            // calcule world bound box radius.
            // calcule disk area inverse .
        // return pdf
       }
    
       let discreteselectlight = 1.0;
    
       let r = RecordPdfEmissionIn{
           n:self.n(),
           ray:Some(Ray::new(self.p(), v))
       };
    
        let recout = lightarea.pdf_emission(r);
        pdf = recout.pdfdir/d2;
        if self.is_on_surface(){ 
           let nn =  self.n();
        //    println!("{:?}, {:?}", v, nn);
            pdf  =next.n().dot(v).abs() * pdf ;}
        pdf

   }
}

















impl   BdptVtx for PathVtx{
    fn n(&self)->Vector3<f64> {
        self.0
    }
    fn p(&self) ->Point3<f64>{
        self.1
    }
    fn pdf(&self, scene:&Scene<f64>, prev : Option<&PathTypes>,  next : &PathTypes) ->f64 {
        let mut  vn = next.p() - self.p();
        if vn.magnitude2() == 0.0 {
            return 0.0;
        }
        vn =  vn.normalize();

        let mut  vprevn = prev.unwrap().p() - self.p();
        if vprevn.magnitude2() == 0.0 {
            return 0.0;
        }
        vprevn =  vprevn.normalize();
        
        let hit = self.4.unwrap_or_else(||{panic!("vtx path degenerated. it  should have a hit record")});
       let pdf = hit.material.pdf( vprevn ,vn);
       self.convert(pdf, next)
    }
    
    fn get_pdfnext(&self) ->f64 {
       self.3.0
    }
    fn get_pdfrev(&self) ->f64 {
        self.3.1
    } 
    fn transport(&self) ->&Srgb {
        
        &self.2
    }
    fn fr(&self, nxt : PathTypes)->Srgb {
        
        let itc = self.4.unwrap();
        let mut vnext =nxt.p()- self.p()  ;
        let d2 = vnext.magnitude2();
        if d2 == 0.0 {
            return Srgb::new(0.0,0.0,0.0);
        }
       vnext =  vnext.normalize();
        
        let res = itc.material.fr(itc.prev.unwrap(), vnext);
     res
    } 
    fn solidarea(&self, nxt : Vector3<f64>)->bool {
        todo!()
    }
    fn set_pdfrev(& mut self, pdf:f64){

        self.3.1 = pdf;
         
    }
    fn set_pdfnext(& mut self, pdf:f64) {
        self.3.0 = pdf;
    }
    fn get_hit(&  self) ->Option<HitRecord<f64>> {
        self.4
    }
    fn get_light(&  self) ->Option<Light> {
       self.8 .clone()
    }
    fn set_modified_vtx_attr(&mut  self, data :ModifiedVtxAttr) {
        self.6= Rc::new(data);
    }
    fn has_modified_attr(&self) ->bool {
        self.6.has_change()
    
    }
    fn get_modified_attr(&self) ->&ModifiedVtxAttr {
        &self.6
    }
    fn get_emission(&self, scene:&Scene<f64>, prev: &PathTypes) ->Srgb {
       let isbcksource  =  self.8.as_ref().unwrap().is_background_area_light();
        let isemissionsource = self.is_light() && (self.8.as_ref().unwrap().is_arealight() || isbcksource);
        if isemissionsource {
            let  mut v = prev.p() - self.p();
            if  v.magnitude2() == 0.0 {
                return   Srgb::new(0.0,0.0, 0.0)
            }
            v = v.normalize();
            if isbcksource{
                todo!("implemented bck source")
            }
            let light = self.8.as_ref().unwrap();
            let r = Ray::new(Point3f::new(0.0,0.0, 0.0), v);
            let w  = self. get_hit().unwrap();
            
            return  light.get_emission(self.get_hit(), &r);
        }
        Srgb::new(0.0,0.0, 0.0)
    }
}

pub fn convert_static(  pdf : f64,current:   &PathTypes , prev:   &PathTypes )   ->f64{
    let mut pdfrev = pdf;
    let v =prev.p()- current.p()  ;
    let d2 = v.magnitude2();
    if d2 == 0.0 {
        return 0.0
    }
    let inv = 1.0 /  d2;
    if prev.is_on_surface(){
        let b =  v *    d2.sqrt();;
        let b = b.normalize();
        // println!("{:?}",prev.n());
        // println!("{:?}",b);
        pdfrev *= prev.n() .dot(b).abs();
    }
   pdfrev*inv 
 
}

  
pub fn walk( r: &Ray<f64>,paths :&mut  Vec<PathTypes>, beta:Srgb, pdfdir:f64,  scene:&Scene<f64>, sampler: &mut Box<dyn Sampler>,mode:Mode, maxdepth:usize)->usize{
    let mut transport = beta;
   let mut pdfnext =  pdfdir;
   let mut pdfrev =  0.0;
   let mut rcow = Cow::Borrowed(r);
   let mut depth = 1;
   while true {
    let tr = rcow.clone();
        let (mut hitop, light ) = interset_scene_bdpt(&tr, scene);
        
        if let Some(hiit) = hitop {
            let mut hitcow = Cow::Borrowed(&hiit); 
           
            paths.push( PathTypes::PathVtxType(PathVtx::new_pathvtx1(*hitcow, transport ,pdfnext,0.0, mode, light)));
            let newpdf = convert_static(pdfnext ,&paths[depth-1] , &paths[depth]  );
            ( &mut paths[depth]).set_pdfnext( newpdf);
            if depth >= maxdepth {

               return depth+1;
            }
            let psample = sampler.get2d();
            let recout =    hitcow.material.sample(RecordSampleIn::from_hitrecord(*hitcow, &rcow, psample));
            if recout.checkIfZero(){break;} 
            transport = recout.compute_transport(transport, &hitcow);
            pdfrev = hitcow.material.pdf(recout.next,hitcow.prev.unwrap());
            pdfnext  = recout.pdf;
            


            let newr =  Ray::<f64>::new(hitcow.point, recout.next);
            rcow = Cow::Owned(newr); 
        }else {
           match mode{
            Mode::from_camera=>{
                paths.push(PathTypes::PathLightVtxType(PathLightVtx::from_endpoint(*rcow,  transport, pdfnext)));
            },
            Mode::from_light=>{}
           }
          //  
           break;;
        }
        let updatepdf =  convert_static(pdfrev,&paths[depth], &paths[depth-1]);
        (& mut paths[depth-1]).set_pdfrev(updatepdf);
         depth = depth +1;
   
   }
  depth

}
 
pub fn init_camera_path(scene: & Scene<f64>,pfilm:&(f64,f64), paths :&mut  Vec<PathTypes>,camera :   PerspectiveCamera ,sampler: &mut Box<dyn Sampler>, maxdepth:usize) ->usize{
   
    let beta = Srgb::new(1.0,1.0,1.0);
    let r = camera.get_ray(pfilm);
   let (pdfpos, pdfdir) = camera.pdfWemission(&r);

 let cameravtx = PathTypes::PathCameraVtxType(PathCameraVtx::new_camerapath_init( camera, r));
 paths.push(cameravtx);

   let depth = walk(&r, paths,beta, pdfdir, scene, sampler, Mode::from_camera,maxdepth-1);
   depth 
 
 
}
 
 
pub fn init_light_path(scene:&Scene<f64>, paths :&mut  Vec<PathTypes>,sampler: &mut Box<dyn Sampler>,  maxdepth:usize)->usize{
    // sample light emision,get first ray
    // call walk
 
   let l =  &scene.lights[0];
   let pdflightselect = 1.0;
   let s = sampler.get2d();
    //  println!("{:?}",s );
   let recout = l.sample_emission(RecordSampleLightEmissionIn ::from_sample(sampler ));
  let a =  recout.n.dot(recout.ray.unwrap().direction).abs() / (recout.pdfdir*recout.pdfpos*pdflightselect);
  let beta =  LigtImportanceSampling::mulScalarSrgb(recout.Lemission,  a as f32);

  let lightVtx =  PathTypes::PathLightVtxType(PathLightVtx::from_hitrecord(l.to_owned() ,  HitRecord::from_point(recout.ray.unwrap().origin, recout.n), recout.Lemission,   recout.pdfpos * pdflightselect )) ;
   paths.push(lightVtx );
 
  walk(&recout.ray.unwrap(), paths,beta, recout.pdfdir, scene, sampler, Mode::from_light,maxdepth-1)+1
 
}









 
 

pub fn mergePath(a :&PathTypes, b:&PathTypes, mode:Mode){
    //    let lab =   a.fr(*b);
    //    let la  = a.transport();
    //   let lb =  b.transport();
    //   if a.is_on_surface(){ 
    //  //    let a = recout.0.dot(newvtx.n());
           
    //   }
}



pub fn cameraTracerStrategy( 
    s : i32,
    scene:&Scene<f64>, 
    pfilm:&Point3f, 
    
    sampler: &mut Box<dyn Sampler>, 
    camera :   PerspectiveCamera,
    lightpath :&   Vec<PathTypes>,
    camerapath :&   Vec<PathTypes>,
   
)->(Srgb, Option<PathTypes>){
  
    let vtxlight =  &lightpath[s as usize-1];
 
    if vtxlight.is_connectable(){
       let hit =  vtxlight.get_hit().unwrap();
       let psample =  sampler.get2d();
       let recout =  camera.sample_importance(RecordSampleImportanceIn{hit,praster:*pfilm,psample}); 
       if  recout.checkIfZero() {return (Srgb::new(0.0,0.0,0.0), None)}
       let newvtx = PathTypes::PathCameraVtxType(PathCameraVtx::new_camerapath_init_ (camera, recout.ray.unwrap(),   recout.weight , recout.pdf));
        //merge path (newvtx, othervtx)
       let transportfromcamera =   newvtx.transport();
       let fr = vtxlight.fr(newvtx.clone());
       let transportfromlight =   vtxlight.transport() ;
       let first  = LigtImportanceSampling::mulSrgb(transportfromcamera.clone(), fr);
            let mut  res  = LigtImportanceSampling::mulSrgb(first, transportfromlight.clone());
            if vtxlight.is_on_surface() {
                 
               res =  LigtImportanceSampling::mulScalarSrgb(res,  recout.n.dot(vtxlight .n()).abs() as f32);
              
            } 
              
           
            // println!("{:?}", res);
           return ( res,Some( newvtx ));
       }
       (Srgb::new(0.0,0.0,0.0),None)

    
}
pub fn emissionDirectStrategy( t : i32,  
    scene:&Scene<f64>, 
   
 
   sampler: &mut Box<dyn Sampler>, 
    
   lightpath :& Vec<PathTypes>,
   camerapath :&  Vec<PathTypes>,)->Srgb{

// aqui llega un momento en que si bien el vtx es una luz es un endpoint.
// y debe ser interpretado como un bck light, tengo he metido un flag diceindo que 
// hay sido creado omo un endpoint. me queda al llamar a emission obtener un v.is_endpoint()
// y si es cierto entonces hacer  Le sobre bck lighs
    
    let vtxcamerapath =  &camerapath[t as usize -1];
    vtxcamerapath.is_light();
    
    vtxcamerapath.is_deltalight();
    if(vtxcamerapath.is_light() ){
       let L = vtxcamerapath.get_emission(scene,&camerapath[t as usize -2]); 
       let transportfromcamera =   vtxcamerapath.transport();  
       return LigtImportanceSampling::mulSrgb(transportfromcamera.clone(), L);
    }
    Srgb::new(0.0,0.0,0.0)
   }
pub fn lightTracerStrategy( t : i32,  
     scene:&Scene<f64>, 
    
  
    sampler: &mut Box<dyn Sampler>, 
     
    lightpath :& Vec<PathTypes>,
    camerapath :&  Vec<PathTypes>,) ->(Srgb, Option<PathTypes>){
        pub fn checkIfZero(w:Srgb, pdf : f64)->bool{
      
            w.red == 0.0 && w.green == 0.0 && w.blue == 0.0 && pdf == 0.0
        }
        let pdfselectedlight = 1.0;
        let  mut newvtx :   PathTypes ;
        let vtxcamerapath =  &camerapath[t as usize -1];
        // esto no se!
    //    if  vtxcamerapath.is_light(){
    //     return  (  Srgb::new(0.0,0.0,0.0), None);
    //    }
        
        if  vtxcamerapath.is_connectable(){
            let u = sampler.get2d();
            let ulight = sampler.get2d();
            

     

            
           let light =  &scene.lights[0];
          
           let recout = light.sampleIllumination(&RecordSampleLightIlumnation::new(ulight, vtxcamerapath.get_hit().unwrap()));
           if !checkIfZero(recout.1, recout.2){
             let pdfw =( 1.0/  (recout.2*pdfselectedlight)) as f32;
            
          
            
             newvtx =   PathTypes::PathLightVtxType(PathLightVtx::new_lightpathvtx1(light.to_owned(),  recout.0,
             Srgb::new(recout.1.red * pdfw , recout.1.green * pdfw , recout.1.blue * pdfw  ), 0.0, recout.4) );
             let (pdfpos , pdfdir, pdfchoice) = newvtx.pdf_light(vtxcamerapath); 
             newvtx.set_pdfnext(pdfpos*pdfchoice);
            //merge path 
            let transportfromcamera =   vtxcamerapath.transport();
            let fr = vtxcamerapath.fr(newvtx.clone());
            let transportfromlight =   newvtx.transport() ;
            let first  = LigtImportanceSampling::mulSrgb(transportfromcamera.clone(), fr);
            let mut  res  = LigtImportanceSampling::mulSrgb(first, transportfromlight.clone());
            if vtxcamerapath.is_on_surface() {
               res =  LigtImportanceSampling::mulScalarSrgb(res,  recout.3.unwrap().direction.dot(vtxcamerapath .n()).abs() as f32);
              
            } 
            //query for occlussion. from surface in camera path to light point
          if !isBlack(res){
                if let Some((isoccluded,_, _, __)) = scene.intersect_occlusion(&recout.3.unwrap(), 
                       
                        &newvtx.p() //  &light.getPositionws() // point in light source . in area light case must be used sample points
                    ) {
                    if isoccluded {
                        res = Srgb::new(0.0,0.0,0.0);
                    }
                 
                }
           }
            
           
            // println!("{:?}", res);
           ( res,Some( newvtx ))
           }else{
          (  Srgb::new(0.0,0.0,0.0), None)
           } 
        } else {
            (  Srgb::new(0.0,0.0,0.0), None)
        }

} 
 

pub fn geometricterm( scene:&Scene<f64> , v : &PathTypes, v1 : &PathTypes   )->f64{
   
    
    let mut vray =v.p()- v1.p()  ;
   //  println!("      v.p {:?} v1.p{:?}", v.p(), v1.p());
    let d2 =  vray.magnitude2();
    if d2 == 0.0 {
        return 0.0;
    }
    let mut  inv = 1.0 /  d2;
    vray=vray*inv.sqrt();
  
    if v.is_on_surface(){
      
       inv *= v.n().dot(vray).abs();
    }
    if v1.is_on_surface(){
        inv *= v1.n().dot(vray).abs();
    } 
    let isempty = scene.is_empty_space(v.p(), v1.p());
    if !isempty {
       return  0.0;
    }
   
    inv
   
}
   
pub
struct ReqWeights{
    newqs : Option<PathTypes>, 
    deltaqs :Option<bool> ,
    newpt : Option<PathTypes>,
     deltapt:Option<bool>, 
     ptPdfRev:Option<f64>, 
     ptminusPdfRev:Option<f64>, 
     qspdfRev:Option<f64>, 
     qsminuspdfRev:Option<f64>
}

impl ReqWeights {
    pub fn new(newqs: Option<PathTypes>, deltaqs: Option<bool>, newpt: Option<PathTypes>, deltapt: Option<bool>, ptPdfRev: Option<f64>, ptminusPdfRev: Option<f64>, qspdfRev: Option<f64>, qsminuspdfRev: Option<f64>) -> Self { 
        Self { newqs, deltaqs, newpt, deltapt, ptPdfRev, ptminusPdfRev, qspdfRev, qsminuspdfRev } }
}

pub fn recompute( s:i32, t:i32, scene:&Scene<f64>, sample:Option<PathTypes>, qs:Option<&PathTypes>, pt:Option<&PathTypes>,qsminus:Option<&PathTypes>, ptminus:Option<&PathTypes>) ->ReqWeights{
 
   let mut qssample:Option<PathTypes> = None;
   let mut ptsample:Option<PathTypes> = None;
   if s == 1{
        qssample = sample;
   }else if t==1{
        ptsample = sample;
   }
    let  mut ptDelta = true;
    if t > 0 {
        ptDelta = false;
    }
    let  mut qsDelta = true;
    if s > 0 {
        qsDelta = false;
    }
    let mut  pdfptPdfRev = 0.0;
    
    if t > 0 {
        if s > 0 {
             // usamos el nuevo sample
           if qssample.is_some() {
            let  qssampleown =  qssample.clone().unwrap();
              pdfptPdfRev =qssampleown .pdf(scene, qsminus, pt.unwrap() );
           }else{
             pdfptPdfRev = qs.unwrap().pdf(scene, qsminus, pt.unwrap() );
           }
           
        }else{
            let (pdfpos, pdfdir, pdfselect ) = pt.unwrap().pdf_light(ptminus.unwrap());
            pdfptPdfRev = pdfpos * pdfselect;
        }
    }
  let mut ptMinusPdfRev= 0.0;
    if t > 1 {
        if s > 0 {
            ptMinusPdfRev= pt.unwrap().pdf(scene, Some(qs.unwrap()), ptminus.unwrap());
        }else{
             ptMinusPdfRev = pt.unwrap().pdf_light_dir(ptminus.unwrap());
            
        }
    }
    //if (qs) a6 = {&qs->pdfRev, pt->Pdf(scene, ptMinus, *qs)};
   let mut  qspdfRev = 0.0;
    if s> 0{
        // si va pero no me fuiooo hay que tener cuidado con los sample vtx especialemente cuando s > 0  ... o t >0 
        // porque usa el sample vtx en vez del que le pongo en la llamada al metodo
        // aqui podemos estar sampleando la camara. si asi es entonces tenemos que  obtener la pdf del ptsample
       if ptsample.is_some(){
        let ptcamera = ptsample.clone().unwrap();
        if qssample.is_some(){
            let  qssampleown =  qssample.clone().unwrap();
            qspdfRev =  ptcamera.pdf(scene, ptminus,  &qssampleown);
        }else{
             
            qspdfRev =  ptcamera.pdf(scene, ptminus, qs.unwrap());
        }
       

       }else {
        if qssample.is_some(){
            let  qssampleown =  qssample.clone().unwrap();
            qspdfRev = pt.unwrap().pdf(scene, ptminus, &qssampleown);
        }else{
            qspdfRev = pt.unwrap().pdf(scene, ptminus, qs.unwrap());
        }
        
       }
       
    }
    // if (qsMinus) a7 = {&qsMinus->pdfRev, qs->Pdf(scene, pt, *qsMinus)};
    let mut  qsminuspdfRev = 0.0;
    let mut qsminuspdfRevopt:Option<f64> = None;
    if s > 1 {
        qsminuspdfRev =  qs.unwrap().pdf(scene, pt, qsminus.unwrap());
        qsminuspdfRevopt = Some(qsminuspdfRev);
    }
    if false{
        println!("  pdfptPdfRev {} ", pdfptPdfRev);
        println!("  ptMinusPdfRev {} ", ptMinusPdfRev);
        println!("  qspdfRev {} ",  qspdfRev);
        println!("  qsminuspdfRev {} ",  qsminuspdfRev);
    }
    
      ReqWeights::new(qssample , Some(qsDelta), ptsample, Some(ptDelta),  Some(pdfptPdfRev),Some( ptMinusPdfRev), Some(qspdfRev),qsminuspdfRevopt)

  
   

}







pub fn update(v:  & mut PathTypes  , attr : ModifiedVtxAttr  ){ 
     v.set_modified_vtx_attr(attr); 
}

 

fn is_zero_then(t :f64 , f :f64)->f64{
    if t == 0.0 {
        f
    }else{
        t
    }

}
pub fn compute_ri( s:i32, t:i32,pathlight : & [PathTypes], pathcamera: &[PathTypes])->f64{
    let mut  ri_acum : f64 = 0.0;
    let mut  ri : f64 = 1.0;
   
    for tt in (1..=(t-1)as usize).rev() {
     
         let vtx = &pathcamera[tt];
         
        if vtx.has_modified_attr() {
          let modattr =  vtx.get_modified_attr();
          let mut  pdfrev =0.0;
          // caso en t == 1 
          if modattr.sampled.is_some(){
            pdfrev =   modattr. sampled.as_ref().unwrap().get_pdfrev();
   
          }else{
             pdfrev = modattr.pdfrev;
          }
         
         let pdfnext = vtx.get_pdfnext();
         
        //   println!("cameraVertices[{}].pdfRev {}", tt, pdfrev);
        //   println!("cameraVertices[{}].pdfFwd {}", tt , pdfnext);
         ri*= is_zero_then(pdfrev, 1.0) /is_zero_then(pdfnext, 1.0);
         
        }else{
            // println!("cameraVertices[{}].pdfRev {}", tt, vtx.get_pdfrev());
            // println!("cameraVertices[{}].pdfFwd {}", tt ,vtx.get_pdfnext());
            ri *= vtx.get_pdfrev() / vtx.get_pdfnext();
          
        }
        ri_acum+=ri;
    }
    // si estamos calculando s==0 entonces no necesitamos calcular mas
    if s==0  {
        return    1.0 /(1.0 + ri_acum);
    }
    let mut  rilight : f64 = 1.0;
    for ss in (0..=(s-1)as usize).rev() {
        let vtx = &pathlight[ss];   
        
        
        if vtx.has_modified_attr() {
            let modattr =  vtx.get_modified_attr();
            let mut  pdfrev =0.0;
            // caso en s== 1 
            if modattr.sampled.is_some(){
                pdfrev = modattr.pdfrev;
              }else{
                 pdfrev = modattr.pdfrev;
              }
            let pdfnext = vtx.get_pdfnext();
            
           
            rilight*=  is_zero_then(pdfrev, 1.0) / is_zero_then(pdfnext, 1.0) ;
       
           }else{
           
                let pdfrev = vtx.get_pdfrev();
                let pdfnext = vtx.get_pdfnext();
               
                rilight *=   is_zero_then(pdfrev, 1.0) / is_zero_then(pdfnext, 1.0) ;
           
         

           }
           let mut delta_iminus1  = false;
           if ss > 0 { 
            // cuando esta en un pt de luz   tengo que consultar si la distribucion es delta.

            delta_iminus1 =  pathlight[ss-1].is_distribution_delta();

           }else {
            delta_iminus1 = pathlight[0].is_deltalight();
           }
           // pathlight if is a delta light (ie point, pojection) this path is "erased"
           let  delta_i = pathlight[ss].get_modified_attr().delta() ;
      
           if !delta_i && !delta_iminus1 {
            ri_acum+=rilight;
       
           }
          
    }
 //    println!("s {} t {}   ri_acum {} ",s, t ,ri_acum );
    1.0 /(1.0 + ri_acum)

}
 

pub fn compute_weights1(s:i32, t:i32, scene:&Scene<f64>, sample:PathTypes,pathlight : &mut [PathTypes], pathcamera: &mut[PathTypes])->f64{
    if  s + t == 2  {return 1.0}
    // let range =  &((s-2) as usize..(s-1) as usize);
    let mut lights:Option<&[PathTypes]> = None;
    let mut cameraArr:Option<&[PathTypes]> = None;
    if s == 0{
        lights = None;
    }else if s == 1{
        // only extract one vertices. first : qs-2, sq-1
         lights =Some(&pathlight[ 0 as usize..=(0) as usize]);
    }else{
        // extract two vertices. first : qs-2, sq-1, 
        // after that swap the order of vertices
        lights =Some(&pathlight[(s-2) as usize..=(s-1) as usize]);
    }

    if t == 0{
        cameraArr = None;
    }else if t == 1{ 
        // only extract one vertices. first : qs-2, sq-1
        cameraArr =Some(&pathcamera[ 0 as usize..=(0) as usize]);
    }else{
        // extract two vertices. first : qs-2, sq-1, 
        // after that swap the order of vertices
        cameraArr =Some(&pathcamera[(t-2) as usize..=(t-1) as usize]);
    }
    let mut res  = ReqWeights::new(None,None,None,None,None,None,None,None );;
    let mut lenlights = 0;
    let mut  lencameraarr = 0;
    if lights.is_some(){
        lenlights =  lights.unwrap().len();
    }
    if cameraArr.is_some(){
        lencameraarr = cameraArr.unwrap().len();
    }
    
   
    if  lights.is_none() && !cameraArr.is_none() {
        let cam = cameraArr.unwrap(); 
        if lencameraarr>1 {
             
            res=    recompute(s, t, scene, Some(sample), None,Some(&cam[1]), None, Some(&cam[0]));
        } 
    } else if !cameraArr.is_none() && !lights.is_none() {
        let ls = lights.unwrap(); 
        let cam = cameraArr.unwrap(); 
        if lenlights == 1 { 
            if lencameraarr == 1 {
               res =  recompute(s, t, scene, Some(sample), Some(&ls[0]),Some(&cam[0]), None, None);
            }else{
              res =   recompute(s, t, scene, Some(sample), Some(&ls[0]),Some(&cam[1]), None, Some(&cam[0]));
            }
        }else{ // lightminus, light
            if lencameraarr == 1 {
                res =     recompute(s, t, scene, Some(sample), Some(&ls[1]),Some(&cam[0]), Some(&ls[0]), None);
            }else{
             res=    recompute(s, t, scene, Some(sample), Some(&ls[1]),Some(&cam[1]), Some(&ls[0]), Some(&cam[0]));
            }
        }

    }
    // el problema esta en que en cuando hacemos t=6 tenemos que variar t=5 y t=4.
    // y solo esos vertices. sin embargo aui tb se varia t=3 de modo que
    // cuando hacemos la suma se usa el vtx 3 variado...
    // pero no veo donde aqui se varian
    
        
 
        if s == 1{
        
            update(&mut pathlight[(s-1) as usize],    ModifiedVtxAttr::from_vtx_pdfrev(res.newqs.unwrap(), res.qspdfRev.unwrap()) );
        }else if s>1{
            update(&mut pathlight[(s-1) as usize], ModifiedVtxAttr::from_pdf_delta(res.qspdfRev.unwrap(), res.deltaqs.unwrap()));
            update(&mut pathlight[(s-2) as usize], ModifiedVtxAttr::from_pdf_delta(res.qsminuspdfRev.unwrap(), false));
        }
       
       
        if t == 1{ // sampled camera 
            update(&mut pathcamera[(t-1) as usize], ModifiedVtxAttr::from_vtx_pdfrev( res.newpt.unwrap(),  res.ptPdfRev.unwrap()));
        }else{ // regular linked list

            update(&mut pathcamera[(t-1) as usize], ModifiedVtxAttr::from_pdf_delta( res.ptPdfRev.unwrap(),  res.deltapt.unwrap()));
            update(&mut pathcamera[(t-2) as usize], ModifiedVtxAttr::from_pdf_delta( res.ptminusPdfRev.unwrap(), false));
        }
        
       
         
        if false{
            println!("s={},t={}",s,t);
            for i in 0..pathcamera.len(){
                println!("pathcamera[i].p {:?}", pathcamera[i].p());
    
                if  pathcamera[i].has_modified_attr(){
                    println!("  pdfrev : {:?}", pathcamera[i].get_modified_attr().pdfrev);
                    println!("  pdfnext : {:?}", pathcamera[i].get_pdfnext());
                     
                    
                }else{
                    println!("  pdfrev : {:?}", pathcamera[i].get_pdfrev());
                    println!("  pdfnext : {:?}", pathcamera[i].get_pdfnext());
                }
            }
    
            // for i in 0..pathlight.len(){
            //     println!("pathlight[i].p {:?}", pathlight[i].p());
    
            //     if  pathlight[i].has_modified_attr(){
            //         println!("  pdfrev : {:?}", pathlight[i].get_modified_attr().pdfrev);
            //         println!("  pdfnext : {:?}", pathlight[i].get_pdfnext());
                     
                    
            //     }else{
            //         println!("  pdfrev : {:?}", pathlight[i].get_pdfrev());
            //         println!("  pdfnext : {:?}", pathlight[i].get_pdfnext());
            //     }
            // }
        }
      
        let weight = compute_ri(s, t,pathlight, pathcamera);

        for p in pathcamera{
            p.set_modified_vtx_attr(ModifiedVtxAttr::clear());
        }
        for p in pathlight{
            p.set_modified_vtx_attr(ModifiedVtxAttr::clear());
        }
        
        weight
}
 
  
pub fn bidirectional(
    s:i32, 
    t:i32,
        scene:&Scene<f64>, 
        pfilm:&Point3f, 
      
       
        lightpath :&  Vec<PathTypes>,
        camerapath :&   Vec<PathTypes>
    )->Srgb{
        let vtxcamerapath =  &camerapath[t as usize-1];
        let vtxlight =  &lightpath[s as usize-1];
        if vtxcamerapath.is_connectable() && vtxlight.is_connectable(){
           let Ll = vtxlight.transport();
           let Lc=  vtxcamerapath.transport();
           let frc2l =  vtxcamerapath.fr(vtxlight.to_owned());
           let frl2c =  vtxlight.fr(vtxcamerapath.to_owned());

        //    println!(" radiance : {:?}", vtxcamerapath.fr(vtxlight.to_owned()));
        //    println!(" importance  : {:?}", vtxlight.fr(vtxcamerapath.to_owned()));

          
          let leftterm = LigtImportanceSampling::mulSrgb(*Ll, frl2c);
          let rightterm =  LigtImportanceSampling::mulSrgb(*Lc, frc2l);
          let mut  L =  LigtImportanceSampling::mulSrgb(leftterm, rightterm);
          if !isBlack(L){
            let g =  geometricterm(scene, &vtxlight, &vtxcamerapath);
                L  =  LigtImportanceSampling::mulScalarSrgb(L, g as f32);
                return  L;
          }else{
           return    Srgb::new(0.0, 0.0, 0.0);
          }
        
         
         
      
        }
        Srgb::new(0.0,0.0,0.0)
    }




 





