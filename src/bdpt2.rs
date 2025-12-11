
pub 
mod bdpt2{
    use core::panic;
    use std::borrow::{Cow, BorrowMut};
    use std::cell::{RefCell, RefMut};
    use std::fmt::Debug;
    use std::ops::Deref;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::time::Instant;
    
    use cgmath::{Vector3, Point3, Vector1, InnerSpace, Point2, Matrix3, Deg};
    use float::Sqrt;
    use num_traits::Float;
    use palette::Srgb;
    
    use crate::Cameras::{PerspectiveCamera, RecordSampleImportanceIn};
    use crate::Lights::{Light, LigtImportanceSampling, IsAreaLight, RecordSampleLightIlumnation, PdfEmission, RecordPdfEmissionIn, GetEmission, PointLight, New_Interface_AreaLight};
    use crate::imagefilm::{FilterFilmGauss, FilterFilm};
    use crate::integrator::{isBlack, IntergratorBaseType};
    use crate::materials::{RecordSampleIn, Pdf, IsSpecular, Fr, MaterialDescType, Plastic};
    use crate::primitives::{Disk, PrimitiveType, Cylinder};
    use crate::raytracerv2::{interset_scene_bdpt, new_interface_for_intersect_scene__bdpt, new_interface_for_intersect_scene__bdpt_walklight};
    use crate::samplerhalton::SamplerHalton;
    use crate::scene::Scene1;
    use crate::threapoolexamples1::{pararell, BBds2i, BBds2f};
    use crate::{Point3f, Vector3f, Point2i, Spheref64};
    use crate::sampler::{Sampler, SamplerType};
    use crate::primitives::prims::{Ray, IntersectionOclussion, IsEmptySpace};
    use crate::primitives::prims::HitRecord;
    // use crate::raytracerv2::{Scene, interset_scene, interset_scene_primitives, interset_scene_bdpt};
 
    use crate::materials::SampleIllumination;
    use crate::Lights::SampleEmission;
    use crate::Lights::RecordSampleLightEmissionIn ;
    use crate::Lights::RecordSampleLightEmissionOut ;
     
    use crate::Lights::IsBackgroundAreaLight ;
    use crate::Lights::SampleLightIllumination;

    use crate:: Config;
    





pub
    fn PlotChains(vpath: &Vec<PathTypes>) ->f64{
        let mut  ckcsumpdffwd = 0.0;
        let mut ckcsumpdfpdfRev = 0.0;
        let mut pat :PathTypes ;
        
        for (id, p) in vpath.iter().enumerate() {
            // println!(" vtx{} is light    {:?}",id,   p.is_light() );
          
            ;
            println!("\t{}, {:?}", id, p.transport());
            println!("\t\t  v.p()       {:?}", p.p());
            println!("\t\t  v.n()       {:?}", p.n());
            println!("\t\t  v.pdfFwd()  {:?}", p.get_pdfnext());
            println!("\t\t  v.pdfRev()  {:?}", p.get_pdfrev());
            ckcsumpdffwd+=p.get_pdfnext();
            ckcsumpdfpdfRev +=p.get_pdfrev(); 
        }
        // println!("                  ckcsumpdffwd {:?}", ckcsumpdffwd );
        // println!("                  ckcsumpdfpdfRev {:?}",ckcsumpdfpdfRev);
        ckcsumpdffwd + ckcsumpdfpdfRev
    }
    
























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
    Option<usize>,
    );
impl  PathVtx {
    pub fn new_pathvtx(hit:HitRecord<f64>,transport:Srgb, pdfnext:f64,mode:Mode)->Self{ 
        PathVtx(   hit.normal ,hit.point, transport,(pdfnext,0.0), Some(hit),mode, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), None, None)
    }
    pub fn new_pathvtx1(hit:HitRecord<f64>,transport:Srgb, pdfnext:f64,pdfrev:f64,mode:Mode, light : Option<Light>)->Self{ 
        PathVtx(   hit.normal ,hit.point, transport,(pdfnext,pdfrev), Some(hit),mode, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), light, None)
    }
    pub fn new_pathvtx_wilthlightid(hit:HitRecord<f64>,transport:Srgb, pdfnext:f64,pdfrev:f64,mode:Mode, lightid : Option<usize>)->Self{ 
        PathVtx(   hit.normal ,hit.point, transport,(pdfnext,pdfrev), Some(hit),mode, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), None, lightid)
    }
  
    pub fn from_zero()->Self{ 
        PathVtx( Vector3f::new(0.0,0.0,0.0) ,Point3f::new(0.0,0.0,0.0), Srgb::default(),(0.0,0.0), None,Mode::from_camera, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), None, None)
    }
    // remove!
    pub fn from_zero1(f:f64)->Self{ 
        PathVtx( Vector3f::new(0.0,0.0,0.0) ,Point3f::new(0.0,0.0,0.0), Srgb::default(),(0.0,f), None,Mode::from_camera, Rc::new(ModifiedVtxAttr::from_zero()), Some(false), None, None)
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
    bool ,// is_endpoint_path(),
   Option<usize> // light in scene.lights[]
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
            false, 
            None, 
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
            false, 
            None, 
        )
    }
    pub fn new_lightpathvtx2_lightid(lightid : usize,  n : Vector3f, Lemission : Srgb,  pdf : f64, poslight : Option<Point3f> , scene: & Scene1)->Self{
        
        let poslight=match poslight {
            Some(p)=>p,
            None=>scene.lights[lightid].getPositionws(),
        };
        PathLightVtx( 
            n,
            poslight,
            Lemission,
            (pdf,0.0), /*pdfnext, pdfprev*/
            None,
            None,
            Rc::new(ModifiedVtxAttr::from_zero()),
            false, 
            Some(lightid)
        )
    }
    pub fn from_hitrecord(light:  &Light, hit : HitRecord<f64>, Lemission : Srgb,  pdf : f64)->Self{
       
        PathLightVtx( 
            hit.normal,
            hit.point,
            Lemission,
            (pdf,0.0), /*pdfnext, pdfprev*/
            Some(hit),
            Some(light.to_owned()),
            
            Rc::new(ModifiedVtxAttr::from_zero()),
            false,
            Some(0),
        )
    }
    pub fn from_hitrecord_lightid(light:  &Light, hit : HitRecord<f64>, Lemission : Srgb,  pdf : f64, lightid : usize)->Self{
       
        PathLightVtx( 
            hit.normal,
            hit.point,
            Lemission,
            (pdf,0.0), /*pdfnext, pdfprev*/
            Some(hit),
           None,
            
            Rc::new(ModifiedVtxAttr::from_zero()),
            false,
            Some(lightid),
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
            true ,// is a endpoint,
            None, 
        )
    }
}



























#[derive( Clone , Debug)]
pub
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
    fn pdf(&self, scene:&Scene1,  prev : Option<&PathTypes>,  next : &PathTypes)->f64;
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
    fn get_emission(&self, scene:&Scene1,prev: &PathTypes)->Srgb;
   
}

trait LightPdfFromOrigin {
    fn pdf_light(&self, vtxext : &PathTypes, scene: &Scene1 )->(
            f64    //prob pos
            ,f64 // prob dir
            ,f64 //probChoice
        );
}
trait LightPdfFromDir {
    fn pdf_light_dir(&self, vtxext : &PathTypes , scene: & Scene1)->f64;
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
trait GetLightId {
    fn get_lightid(&self)->Option<usize >  ;
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
       //  self.get_light().is_some()
       self.9.is_some()
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


impl   GetLightId for PathTypes{
    fn get_lightid(&self)-> Option<usize> {
         match self {
             Self:: PathCameraVtxType(l)=>None, 
             Self::PathLightVtxType(pl)=>pl.get_lightid(),
             Self::PathVtxType(l)=>None,
             _=>panic!(),
         }
    }
}
impl   GetLightId  for PathLightVtx {

    fn get_lightid(&self)->Option<usize> {
        self.8
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
    fn is_on_surface(&self, scene:&Scene1)->bool;
}
impl   IsOnSurface for PathTypes{
    fn is_on_surface(&self,  scene:&Scene1) ->bool  {
         match self {
             Self:: PathCameraVtxType(l)=>false,
             Self::PathLightVtxType(l)=>scene.lights[l.8.unwrap()].is_arealight(),
             Self::PathVtxType(l)=>true,
             _=>panic!(),
         }
    }
}
impl   IsOnSurface for PathCameraVtx{
    fn is_on_surface(&self,  scene:&Scene1) ->bool  {
          false
    }
}
impl   IsOnSurface for PathVtx{
    fn is_on_surface(&self,  scene:&Scene1) ->bool  {
          true
    }
}
impl   IsOnSurface for PathLightVtx {
    fn is_on_surface(&self,  scene:&Scene1) ->bool  {
        // self.5.unwrap().is_arealight()
          false // si es un area light esto es true
    }
}
 

trait  ConvertDensityBdpt {
    fn convert(&self,  pdf : f64, other: & PathTypes ,  scene:&Scene1)->f64;
}
impl  ConvertDensityBdpt for PathTypes {
    fn convert(&self, pdf : f64,  other: & PathTypes,  scene:&Scene1 )   ->f64{
         match self {
             Self:: PathCameraVtxType(l)=>l.convert( pdf,  other, scene),
             Self::PathLightVtxType(l)=>l.convert( pdf,   other, scene),
             Self::PathVtxType(l)=>l.convert(  pdf,  other, scene),
             _=>panic!(),
         }
    }
}
impl  ConvertDensityBdpt for PathCameraVtx{
    fn convert(&self, pdf : f64,  other: & PathTypes ,  scene:&Scene1) ->f64{
        let mut pdfrev = pdf;
        let v =other.p()- self.p()  ;
        let d2 = v.magnitude2();
        if d2 == 0.0 {
            return 0.0
        }
        let inv = 1.0 /  d2;
        if other.is_on_surface(scene){
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
    fn convert(&self, pdfold : f64, other:& PathTypes ,  scene:&Scene1) ->f64{
        let mut pdf :f64 = pdfold;
        let mut v = other.p() - self.p();
        let l2 = v.magnitude2();
       if  l2 == 0.0 { return 0.0}
       let invl2 = 1.0/ l2;
    
      
      if other.is_on_surface(scene){
       let b =  v *    invl2.sqrt();;
        pdf *= other.n() .dot(b).abs();
      }
       pdf*invl2
    }
}
impl  ConvertDensityBdpt for PathVtx{
    fn convert(&self, pdfold : f64 , other:& PathTypes,  scene:&Scene1) ->f64{
        let mut pdf :f64 = pdfold;
        let mut v = other.p() - self.p();
        let l2 = v.magnitude2();
       if  l2 == 0.0 {}
       let invl2 = 1.0/ l2;
    
      
      if other.is_on_surface(scene){
       let b =  v *    invl2.sqrt();
        pdf *= other.n() .dot(b).abs();
      }
       pdf*invl2

         
    }
}






















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
    fn pdf(&self, scene:&Scene1, prev :Option<&PathTypes>,  next : &PathTypes) ->f64 {
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
    fn get_emission(&self, scene:&Scene1,prev: &PathTypes)->Srgb{
        match self {
            Self:: PathCameraVtxType(v)=>Srgb::new(0.0,0.0,0.0),
            Self::PathLightVtxType(v)=>{
              
                v.get_emission(scene, prev)},
            Self::PathVtxType(v)=>{
         
                v.get_emission(scene, prev)},
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
    fn pdf_light(&self, vtxext : &PathTypes, scene: &Scene1 ) ->(f64 , f64,   f64){
        match self {
            Self:: PathCameraVtxType(l)=>panic!(" LightPdfFromOrigin "),
            Self::PathLightVtxType(l)=>l.pdf_light(vtxext, scene),
            Self::PathVtxType(l)=>l.pdf_light(vtxext, scene),
            _=>panic!()
        } 
    }
}

impl   LightPdfFromDir for PathTypes {
    fn pdf_light_dir(&self, vtxext : &PathTypes , scene: & Scene1) ->  f64 {
        match self {
            Self:: PathCameraVtxType(l)=>panic!(" LightPdfFromOrigin "),
            Self::PathLightVtxType(l)=>l.pdf_light_dir(vtxext, scene),
            Self::PathVtxType(l)=>l.pdf_light_dir(vtxext, scene),
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
    fn pdf(&self, scene:&Scene1, prev :Option<&PathTypes>,  next : &PathTypes) ->f64 {
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
     
     self.convert(pdfdirimporance,next, scene )
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
    fn get_emission(&self, scene:&Scene1, prev: &PathTypes) ->Srgb {
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
    fn pdf(&self, scene:&Scene1, prev : Option<&PathTypes>,  next : &PathTypes) ->f64 {
       let invdist = 1.0/ (next.p() - self.p()).magnitude2();
       let  pdf = self. pdf_light(&next, scene);
       let mut  pdfdir = pdf.1;

    //   if  self.is_bcklight(){}
       if next.is_on_surface(scene){
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
    fn get_emission(&self, scene:&Scene1, v: &PathTypes) ->Srgb {
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
    
     fn pdf_light(&self, vext:&PathTypes, scene: &Scene1) ->(f64 , f64, f64){
    //   let light = self.5.as_ref().unwrap();
    let light = &scene.lights[self.8.unwrap()];
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
    
    fn pdf_light(&self, vext:&PathTypes, scene: &Scene1) ->(f64 , f64, f64){
        
    let norm = self.n();
    let pt = self.p();
    // panic!("tengo qeu usar scene para obtener la ref a la luz");
    
    let lightarea = &scene.lights[self.9.unwrap()];
    //  let lightarea = self.8.as_ref().unwrap();
    
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
    
    fn pdf_light_dir(&self, vext:&PathTypes, scene: & Scene1) ->f64  {
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
    
    fn pdf_light_dir(&self, next:&PathTypes, scene: & Scene1) ->f64  {
     
      
         let lightareaid = self.9.unwrap();
         let lightarea = &scene.lights[lightareaid];
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
        if self.is_on_surface(scene){ 
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
    fn pdf(&self, scene:&Scene1, prev : Option<&PathTypes>,  next : &PathTypes) ->f64 {
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
       self.convert(pdf, next, scene)
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
    fn get_emission(&self, scene:&Scene1, prev: &PathTypes) ->Srgb {
        // bdpt 2 
       // let isbcksource  =  self.9.as_ref().unwrap().is_background_area_light();
       let lightid = self.9.unwrap();
       let light = &scene.lights[lightid];
       let isbcksource = light.is_background_area_light();
        let isemissionsource = self.is_light() && (light.is_arealight() || isbcksource);
        if isemissionsource {
            let  mut v = prev.p() - self.p();
            if  v.magnitude2() == 0.0 {
                return   Srgb::new(0.0,0.0, 0.0)
            }
            v = v.normalize();
            if isbcksource{
                todo!("implemented bck source")
            }
           
            let r = Ray::new(Point3f::new(0.0,0.0, 0.0), v);
            let w  = self. get_hit().unwrap();
            
            return  light.get_emission(self.get_hit(), &r);
        }
        Srgb::new(0.0,0.0, 0.0)
    }
}

pub fn convert_static(  pdf : f64,current:   &PathTypes , prev:   &PathTypes , scene: & Scene1)   ->f64{
    let mut pdfrev = pdf;
    let v =prev.p()- current.p()  ;
    let d2 = v.magnitude2();
    if d2 == 0.0 {
        return 0.0
    }
    let inv = 1.0 /  d2;
    if prev.is_on_surface(scene){
        let b =  v *    d2.sqrt();;
        let b = b.normalize();
        // println!("{:?}",prev.n());
        // println!("{:?}",b);
        pdfrev *= prev.n() .dot(b).abs();
    }
   pdfrev*inv 
 
}









pub fn walk( r: &Ray<f64>,paths :&mut  Vec<PathTypes>, beta:Srgb, pdfdir:f64,  
    scene:&Scene1, 
    sampler: &mut SamplerType,
    mode:Mode, maxdepth:usize)->usize{
    let mut transport = beta;
   let mut pdfnext =  pdfdir;
   let mut pdfrev =  0.0;
   let mut rcow = Cow::Borrowed(r);
   let mut depth = 1;
   while true {
    let tr = rcow.clone();
        // let (mut hitop, light ) = interset_scene_bdpt(&tr, scene);
        let (mut hitop, light ) = new_interface_for_intersect_scene__bdpt(&tr, scene);
        
        if let Some(hiit) = hitop {
            let mut hitcow = Cow::Borrowed(&hiit); 
           
            paths.push( PathTypes::PathVtxType(PathVtx::new_pathvtx1(*hitcow, transport ,pdfnext,0.0, mode, light)));
            let newpdf = convert_static(pdfnext ,&paths[depth-1] , &paths[depth] , scene );
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
        let updatepdf =  convert_static(pdfrev,&paths[depth], &paths[depth-1], scene);
        (& mut paths[depth-1]).set_pdfrev(updatepdf);
         depth = depth +1;
   
   }
  depth

}




pub fn walklight( r: &Ray<f64>,paths :&mut  Vec<PathTypes>, beta:Srgb, pdfdir:f64,  
    scene:&Scene1, 
    sampler: &mut SamplerType,
    mode:Mode, maxdepth:usize)->usize{
    let mut transport = beta;
   let mut pdfnext =  pdfdir;
   let mut pdfrev =  0.0;
   let mut rcow = Cow::Borrowed(r);
   let mut depth = 1;
   while true {
    let tr = rcow.clone();
        // let (mut hitop, light ) = interset_scene_bdpt(&tr, scene);
        let (mut hitop, lightid ) = new_interface_for_intersect_scene__bdpt_walklight(&tr, scene);
        
        if let Some(hiit) = hitop {
            
            let mut hitcow = Cow::Borrowed(&hiit); 
            paths.push( PathTypes::PathVtxType(PathVtx::new_pathvtx_wilthlightid(*hitcow, transport ,pdfnext,0.0, mode, lightid)));
           
            let newpdf = convert_static(pdfnext ,&paths[depth-1] , &paths[depth] , scene );
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
        let updatepdf =  convert_static(pdfrev,&paths[depth], &paths[depth-1], scene);
        (& mut paths[depth-1]).set_pdfrev(updatepdf);
         depth = depth +1;
   
   }
  depth

}









pub fn init_camera_path(scene: & Scene1,pfilm:&(f64,f64), paths :&mut  Vec<PathTypes>,
camera :   PerspectiveCamera ,
sampler: & mut SamplerType,
maxdepth:usize) ->usize{
   
    let beta = Srgb::new(1.0,1.0,1.0);
    let r = camera.get_ray(pfilm);
   let (pdfpos, pdfdir) = camera.pdfWemission(&r);

 let cameravtx = PathTypes::PathCameraVtxType(PathCameraVtx::new_camerapath_init( camera, r));
 paths.push(cameravtx);

     let depth = walklight(&r, paths,beta, pdfdir, scene, sampler, Mode::from_camera,maxdepth-1);
    depth 
 
 
}






 
pub fn init_light_path(scene:&Scene1, paths :&mut  Vec<PathTypes>,
    sampler:  & mut SamplerType,
     maxdepth:usize)->usize{
    // sample light emision,get first ray
    // call walk
 
   let l =  &scene.lights[0];
  let pdflightselect = 1.0;
    let s = sampler.get2d();
//     //  println!("{:?}",s );
   let recout = l.sample_emission(RecordSampleLightEmissionIn ::from_sampletype(sampler ));
 let a =  recout.n.dot(recout.ray.unwrap().direction).abs() / (recout.pdfdir*recout.pdfpos*pdflightselect);
  let beta =  LigtImportanceSampling::mulScalarSrgb(recout.Lemission,  a as f32);

//   let lightVtx =  PathTypes::PathLightVtxType(
//     PathLightVtx::from_hitrecord(
//         l.to_owned() ,  
//         HitRecord::from_point(recout.ray.unwrap().origin, recout.n), 
//         recout.Lemission,   
//         recout.pdfpos * pdflightselect )) ;
// let light    = Light::New_Interface_AreaLightType(
//     New_Interface_AreaLight ::new(Srgb::new(4.0,4.0,4.0),
//     PrimitiveType::DiskType(Disk::new(Vector3::new( 0.0000,0.3,2.5000),Vector3::new(0.0, -1.0, 0.0),0.0, 0.5, 
//     MaterialDescType::NoneType)),4));
   
  let lightVtx =  PathTypes::PathLightVtxType(
    PathLightVtx::from_hitrecord_lightid(
         &l,  
        HitRecord::from_point(recout.ray.unwrap().origin, recout.n), 
        recout.Lemission,   
        recout.pdfpos * pdflightselect,
    0 )) ;
   paths.push(lightVtx );
 
let ra =  Ray::new(Point3f::new(0.,0.,0.), Vector3f::new(0.0,0.0,1.0));
walklight(&recout.ray.unwrap(), paths,beta, recout.pdfdir, scene, sampler, Mode::from_light,maxdepth-1)+1
  
}































pub fn lightTracerStrategy1<S:Sampler>( t : i32,  
    scene:&Scene1, 
   
 
   sampler: &mut S, 
    
   lightpath :& Vec<PathTypes>,
   camerapath :&  Vec<PathTypes>,) ->(Srgb, Option<PathTypes>){
    pub fn checkIfZero(w:Srgb, pdf : f64)->bool{
      
        w.red == 0.0 && w.green == 0.0 && w.blue == 0.0 && pdf == 0.0
    }
    let pdfselectedlight = 1.0;
    let  mut newvtx :   PathTypes ;
    let vtxcamerapath =  &camerapath[t as usize -1];





    if  vtxcamerapath.is_connectable(){
        let u = sampler.get2d();
        let ulight = sampler.get2d();
        // println!("u {:?}", u);
        // println!("ulight {:?}", ulight);

 

        
       let light =  &scene.lights[0];
      
       let recout = light.sampleIllumination(&RecordSampleLightIlumnation::new(ulight, vtxcamerapath.get_hit().unwrap()));
       if !checkIfZero(recout.1, recout.2){
         let pdfw =( 1.0/  (recout.2*pdfselectedlight)) as f32;
        
      
        
         newvtx =   PathTypes::PathLightVtxType(
            PathLightVtx::new_lightpathvtx2_lightid(0,  recout.0,Srgb::new(recout.1.red * pdfw , recout.1.green * pdfw , recout.1.blue * pdfw  ), 0.0, recout.4, scene) );
       let (pdfpos , pdfdir, pdfchoice) = newvtx.pdf_light(vtxcamerapath, scene); 
          newvtx.set_pdfnext(pdfpos*pdfchoice);

        
 //        merge path 
        let transportfromcamera =   vtxcamerapath.transport();
        let fr = vtxcamerapath.fr(newvtx.clone());
        let transportfromlight =   newvtx.transport() ;
        let first  = LigtImportanceSampling::mulSrgb(transportfromcamera.clone(), fr);
        let mut  res  = LigtImportanceSampling::mulSrgb(first, transportfromlight.clone());
        if vtxcamerapath.is_on_surface(scene) {
           res =  LigtImportanceSampling::mulScalarSrgb(res,  recout.3.unwrap().direction.dot(vtxcamerapath .n()).abs() as f32);
          
        } 


        //query for occlussion. from surface in camera path to light point
      if !isBlack(res){
            if let Some((isoccluded,_, _)) = scene.intersect_occlusion(&recout.3.unwrap(), 
                    &newvtx.p() //  &light.getPositionws() // point in light source . in area light case must be used sample points
                ) {
                if isoccluded {
                    res = Srgb::new(0.0,0.0,0.0);
                }
             
            }
       }
        
       
        // println!("{:?}", res);
      ( res,Some( newvtx ))
   

    //   (  Srgb::new(0.0,0.0,0.0), None)


       }else{
      (  Srgb::new(0.0,0.0,0.0), None)
       } 
    } else {
        (  Srgb::new(0.0,0.0,0.0), None)
    }
     
      
     
   }


pub fn emissionDirectStrategy1<S:Sampler>( t : i32,  
    scene:&Scene1, 
   
 
   sampler: &mut  S, 
    
   lightpath :& Vec<PathTypes>,
   camerapath :&  Vec<PathTypes>,)->Srgb{
 
    Srgb::new(0.0,0.0,0.0)
   }



pub fn cameraTracerStrategy1<S:Sampler>( 
    s : i32,
    scene:&Scene1, 
    pfilm:&Point3f, 
    
    sampler:  &mut  S, 
    camera :   PerspectiveCamera,
    lightpath :&   Vec<PathTypes>,
    camerapath :&   Vec<PathTypes>,
   
)->(Srgb, Option<PathTypes>){
     
    (Srgb::new(0.0,0.0,0.0),None)
}


pub fn cameraTracerStrategy( 
    s : i32,
    scene:&Scene1, 
    pfilm:&Point3f, 
    
    sampler: &mut SamplerType, 
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
            if vtxlight.is_on_surface(scene) {
                 
               res =  LigtImportanceSampling::mulScalarSrgb(res,  recout.n.dot(vtxlight .n()).abs() as f32);
              
            } 
              
           
            // println!("{:?}", res);
           return ( res,Some( newvtx ));
       }
       (Srgb::new(0.0,0.0,0.0),None)

    
}
pub fn emissionDirectStrategy( t : i32,  
    scene:&Scene1, 
   
 
   sampler: &mut  SamplerType, 
    
   lightpath :& Vec<PathTypes>,
   camerapath :&  Vec<PathTypes>,)->Srgb{

// aqui llega un momento en que si bien el vtx es una luz es un endpoint.
// y debe ser interpretado como un bck light, tengo he metido un flag diceindo que 
// hay sido creado omo un endpoint. me queda al llamar a emission obtener un v.is_endpoint()
// y si es cierto entonces hacer  Le sobre bck lighs
    
    let vtxcamerapath =  &camerapath[t as usize -1];
      vtxcamerapath.is_light();
    
    // vtxcamerapath.is_deltalight();
    if(vtxcamerapath.is_light() ){
       let L = vtxcamerapath.get_emission(scene,&camerapath[t as usize -2]); 
       
       let transportfromcamera =   vtxcamerapath.transport();  
       return LigtImportanceSampling::mulSrgb(transportfromcamera.clone(), L);
    }
    Srgb::new(0.0,0.0,0.0)
   }














pub fn lightTracerStrategy( t : i32,  
     scene:&Scene1, 
    
  
    sampler: &mut SamplerType, 
     
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
            
          
            
             newvtx =   PathTypes::PathLightVtxType(
                PathLightVtx::new_lightpathvtx2_lightid(0,  recout.0,Srgb::new(recout.1.red * pdfw , recout.1.green * pdfw , recout.1.blue * pdfw  ), 0.0, recout.4, scene) );
           let (pdfpos , pdfdir, pdfchoice) = newvtx.pdf_light(vtxcamerapath, scene); 
              newvtx.set_pdfnext(pdfpos*pdfchoice);

            
     //        merge path 
            let transportfromcamera =   vtxcamerapath.transport();
            let fr = vtxcamerapath.fr(newvtx.clone());
            let transportfromlight =   newvtx.transport() ;
            let first  = LigtImportanceSampling::mulSrgb(transportfromcamera.clone(), fr);
            let mut  res  = LigtImportanceSampling::mulSrgb(first, transportfromlight.clone());
            if vtxcamerapath.is_on_surface(scene) {
               res =  LigtImportanceSampling::mulScalarSrgb(res,  recout.3.unwrap().direction.dot(vtxcamerapath .n()).abs() as f32);
              
            } 


            //query for occlussion. from surface in camera path to light point
          if !isBlack(res){
                if let Some((isoccluded,_, _)) = scene.intersect_occlusion(&recout.3.unwrap(), 
                        &newvtx.p() //  &light.getPositionws() // point in light source . in area light case must be used sample points
                    ) {
                    if isoccluded {
                        res = Srgb::new(0.0,0.0,0.0);
                    }
                 
                }
           }
            
           
            // println!("{:?}", res);
          ( res,Some( newvtx ))
       

        //   (  Srgb::new(0.0,0.0,0.0), None)


           }else{
          (  Srgb::new(0.0,0.0,0.0), None)
           } 
        } else {
            (  Srgb::new(0.0,0.0,0.0), None)
        }
         
} 
 

pub fn geometricterm( scene:&Scene1 , v : &PathTypes, v1 : &PathTypes   )->f64{
   
    
    let mut vray =v.p()- v1.p()  ;
   //  println!("      v.p {:?} v1.p{:?}", v.p(), v1.p());
    let d2 =  vray.magnitude2();
    if d2 == 0.0 {
        return 0.0;
    }
    let mut  inv = 1.0 /  d2;
    vray=vray*inv.sqrt();
  
    if v.is_on_surface(scene){
      
       inv *= v.n().dot(vray).abs();
    }
    if v1.is_on_surface(scene){
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

pub fn recompute( s:i32, t:i32, scene:&Scene1, sample:Option<PathTypes>, qs:Option<&PathTypes>, pt:Option<&PathTypes>,qsminus:Option<&PathTypes>, ptminus:Option<&PathTypes>) ->ReqWeights{
  if(s==1 && t==2){
print!("");
  }
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
            let (pdfpos, pdfdir, pdfselect ) = pt.unwrap().pdf_light(ptminus.unwrap(),scene);
            pdfptPdfRev = pdfpos * pdfselect;
        }
    }
  let mut ptMinusPdfRev= 0.0;
    if t > 1 {
        if s > 0 {
            ptMinusPdfRev= pt.unwrap().pdf(scene, Some(qs.unwrap()), ptminus.unwrap());
        }else{
             ptMinusPdfRev = pt.unwrap().pdf_light_dir(ptminus.unwrap(), scene);
            
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

pub fn compute_ri( s:i32, t:i32, scene: & Scene1,pathlight : & [PathTypes], pathcamera: &[PathTypes])->f64{
   let mut  ri_acum : f64 = 0.0;
   let mut  ri : f64 = 1.0;
  if(s==1 && t ==2){
    // print!("");
  }
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
       //     delta_iminus1 = pathlight[0].is_deltalight();// 
       // es una solucion de mierda...pero si no. tengo que usar el id de la luz
       // para sacarlo de la scene y comprobar que esta luz no es delta
       // cuando esta luz es delta este path no influye
         let lightid =     pathlight[0].get_lightid().unwrap();
         delta_iminus1 =     !scene.lights[lightid].is_arealight() && !scene.lights[lightid].is_background_area_light();
            
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


pub fn compute_weights1(s:i32, t:i32, scene:&Scene1, sample:PathTypes,pathlight : &mut [PathTypes], pathcamera: &mut[PathTypes])->f64{
   if  s + t == 2  {return 1.0}
   if(s==1&& t==2){
    print!("");
   }
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
     
       let weight = compute_ri(s, t,scene,pathlight, pathcamera);

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
        scene:&Scene1, 
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




 
























 
pub
struct BdptIntegrator{
    scene : Scene1 ,
    sampler : SamplerType,
    film:pararell::Film1,
    camera : PerspectiveCamera,
    config: Config,
  
}
impl BdptIntegrator {
    pub fn get_config(&self)->&Config{
        &self.config
    }
    pub fn new(scene : Scene1,   sampler : SamplerType, film: pararell::Film1, camera : PerspectiveCamera, config:Config)->Self{
        BdptIntegrator {scene :  scene  , sampler, film, camera, config}
    }
    pub fn preprocess(& self ){}
   pub fn integrate(&self, pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, mut samplerhalton: &mut SamplerType)->Srgb{ 
   
    let maxdepth: i32 = 6;
    let cameraInstance = self.get_camera();
    let mut pathlight: Vec<PathTypes> = vec![];
    let mut pathcamera: Vec<PathTypes> = vec![];
   init_camera_path(scene,pfilm,&mut pathcamera,*cameraInstance,&mut  samplerhalton,maxdepth as usize,);
    init_light_path(scene, &mut pathlight, &mut  samplerhalton, maxdepth as usize);
   
    let ncamera = pathcamera.len();
 
    let nlights  =  pathlight.len();
    let pnewpointsample = pfilm;

       
    let mut L_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_bdir_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_cam_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut L_lightStra_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    let mut ckcsumpdTOTAL = 0.0;



    
    let mut Lsum = Srgb::new(0.0, 0.0, 0.0);
    let mut  Llightstrategy= Srgb::new(0.0, 0.0, 0.0);
    let mut  Lcamerastrategy= Srgb::new(0.0, 0.0, 0.0);
    let mut  LbirectionalStrategy= Srgb::new(0.0, 0.0, 0.0);
    let mut L_PX_TOTAL = Srgb::new(0.0, 0.0, 0.0);
    
    for t in 1..=ncamera as i32 {
        for s in 0..=nlights as i32 {
            let depth: i32 = s + t - 2;
        
            let pfilm = Point3f::new(pfilm.0, pfilm.1,0.0);
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
                if true{
                    let L = emissionDirectStrategy(
                        t,
                        scene,
                        &mut  samplerhalton,
                        &pathlight,
                        &pathcamera,
                    );
                    if !isBlack(L){
                        let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                        let weight =compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                        Lsum = LigtImportanceSampling::addSrgb(LigtImportanceSampling::mulScalarSrgb(L, weight as f32),Lsum,);
                    }
                  
                }
            } else if t == 1 { 
                if   false { 
                        let (L, sampledvtx) = cameraTracerStrategy(
                            s,
                            scene,
                            &Point3f::new(pnewpointsample.0,pnewpointsample.1, 0.0),
                            &mut  samplerhalton,
                            *cameraInstance,
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
            
                    }
                    
                }
            } else {
                 
                if true{
                    
                    let Lbidirectional = bidirectional(s,t,scene,&Point3f::new(pnewpointsample.0,pnewpointsample.1, 0.0),&pathlight,&pathcamera,);
                    if !isBlack(Lbidirectional) {
                        let sampledvtx = PathTypes::PathVtxType(PathVtx::from_zero());
                        let weight = compute_weights1(s, t, scene, sampledvtx, &mut pathlight, &mut pathcamera);
                        let LPath = LigtImportanceSampling::mulScalarSrgb(Lbidirectional, weight as f32);
                        Lsum = LigtImportanceSampling::addSrgb( LPath,Lsum,);
                       
                    }
                  
                     
                }
                
            }
        } 
    }
    L_PX_TOTAL =     LigtImportanceSampling::addSrgb(L_PX_TOTAL, Lsum);
  







    L_PX_TOTAL
         
     
    }
    pub fn get_res(&self){}
    pub fn get_scene(&self)-> &Scene1 { &self.scene }
    pub fn get_spp(&self)->u32{self.sampler.get_spp()}
   pub fn get_film(&self)->&pararell::Film1{ &self.film}
   pub fn get_camera(&self)->&PerspectiveCamera{ &self.camera}
   pub fn get_sampler(&self)->&SamplerType{ &self.sampler}
}





pub fn main_bdpt2(){
    let h  = 512.0;
    let   w  = h * 1.77777777778;
      let res:(usize,usize) = (w as usize, h as usize);
      let spp =2048;
      let config = Config::from_args(spp, w as i64, h as i64, 1, 
          Some("newdir1".to_string()), 
            Some("bdpt_sphere_3_emiteddirect.png")
         
      );
    //   estimate light no funciona! mirar a ver que ocurre!
    
      let scene  = Scene1::make_scene(
              res.0 as usize, 
              res.1 as usize, 
              vec![
                Light::New_Interface_AreaLightType ( New_Interface_AreaLight::new( Srgb::new(40.0,40.0,40.0),PrimitiveType::SphereType( Spheref64::new(Vector3::new(0.0000, 0.30,3.00000),1.0,   MaterialDescType::PlasticType(Plastic::from_albedo(0.90,0.90,0.0)))),4)),
                // Light::New_Interface_AreaLightType( New_Interface_AreaLight::new(  Srgb::new(42.0,42.0,42.0),PrimitiveType::CylinderType(   Cylinder::new(Vector3::new( 0.0000, 0.30,3.30000),Matrix3::from_angle_y(Deg( 90.0)),-1.0, 1.0, 0.150, MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)))),4)),
               // Light::PointLight(PointLight{iu: 4.0,positionws: Point3::new(0.0000,0.2,2.5000),color: Srgb::new(24.190,24.190, 24.190),})
                // Light::New_Interface_AreaLightType(New_Interface_AreaLight::new(Srgb::new(42.0,42.0,42.0),PrimitiveType::DiskType(Disk::new(Vector3::new( 0.0000,0.4,3.5000),Vector3::new(0.0, -1.0, 0.0),0.0, 0.7, 
                // MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)))),4)),
   
              ], 
              vec![
               PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.0,-0.60,3.41), 0.40, MaterialDescType::PlasticType(Plastic::from_albedo(0.90,0.90,0.0)))),
               PrimitiveType::SphereType(Spheref64::new(Vector3f::new(-0.70,-0.8,3.00), 0.20, MaterialDescType::PlasticType(Plastic::from_albedo(0.90,0.0,0.90)))),
             PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.70,-0.80,3.0), 0.20, MaterialDescType::PlasticType(Plastic::from_albedo(0.0,0.90,0.90)))),
             PrimitiveType::DiskType( Disk::new(Vector3::new( 0.0000,-1.0,0.0000),Vector3::new(0.0, 1.0, 0.0),0.0, 1000.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00))))
              ], 
              spp.clone() as usize,1);
  
             
      // let rray =      Ray::new(Point3f::new(0.,0.1,0.),Vector3f::new(0.0,-1.0,0.0));
      // let reshit = new_interface_for_intersect_scene(&rray, &scene);
    //   PrimitiveType::DiskType( Disk::new(Vector3::new( 0.0000,-1.0,0.0000),Vector3::new(0.0, 1.0, 0.0),0.0, 100.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00))));
    // panic!("");
    // tengo que hacer estimateDirect
    // y depues mirar a ver que ocurre con trace camera
      let sampler  = SamplerType::SamplerHaltonType(SamplerHalton::new(&Point2i::new(0, 0),&Point2i::new(w as i64, h as i64),spp.clone() as i64,false,));
      let cameraInstance =  PerspectiveCamera::from_lookat(Point3f::new(0.0,0.000, 0.0), Point3f::new(0.0,0.0,1.00), 1e-2, 100.0, 75.0, (res.0 as u32, res.1 as u32));
      let filtergauss  =  FilterFilm::FilterFilmGaussType(FilterFilmGauss::new((2.0,2.0),3.0));
      let mut film  = pararell::Film1::new(BBds2i::from(BBds2f::<f64>::new(Point2::new(0.0,0.0,),Point2::new(res.0 as f64 ,res.1 as f64 ))) , filtergauss);  
    
      let mut integrator =  IntergratorBaseType::BdptIntegratorType(BdptIntegrator::new(scene, sampler, film, cameraInstance,config));
     integrator.integrate_par();
}






#[test]
pub fn debug_main_bdpt2(){

    main_bdpt2();
}






}// mod bpdt2