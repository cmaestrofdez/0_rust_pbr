
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
extern crate cgmath;

#[macro_export]
macro_rules! assert_delta {
    ($x:expr, $y:expr, $d:expr) => {
        if !($x - $y < $d || $y - $x < $d) {
            panic!();
        }
    };
}

use crate::Lights::LigtImportanceSampling;
use crate::samplerhalton::ONE_MINUS_EPSILON_f64;
use crate::texture::{Texture2Df64f64MapUV, Texture2D, Mapping2d, Texture2DSRgbMapConstant, Texture2DSRgbMapUV, TexType, EvalTexture, MapUV, MapConstant};
use crate::volumentricpathtracer::volumetric::Colorf64;
use crate::{primitives, Lambertian, Vector3f};
use crate::primitives::prims::{self, HitRecord};
use crate::primitives::prims::{IntersectionOclussion, Ray,Sphere};
use crate::raytracerv2::{clamp, lerp};
use crate::raytracerv2::Scene;
use cgmath::*;
use cgmath::{Deg, Euler, Quaternion};
use num_traits::cast::cast;
use num_traits::Float;
use num_traits::Pow;
use palette::rgb::Srgb;
use rand::Rng;
use std::f32;
use std::f64;
use std::iter::zip;
use crate::texture::Texture2DMapUV;
///
/// let rotation = Quaternion::from(Euler {
///     x: Deg(90.0),
///     y: Deg(45.0),
///     z: Deg(15.0),
/// });


/**
 *  Point2f uOffset = 2.f * u - Vector2f(1, 1);

    // Handle degeneracy at the origin
    if (uOffset.x == 0 && uOffset.y == 0) return Point2f(0, 0);

    // Apply concentric mapping to point
    Float theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y)) {
        r = uOffset.x;
        theta = PiOver4 * (uOffset.y / uOffset.x);
    } else {
        r = uOffset.y;
        theta = PiOver2 - PiOver4 * (uOffset.x / uOffset.y);
    }
    return r * Point2f(std::cos(theta), std::sin(theta));
 */
pub fn ConcentricSampleDisk(u: &(f64, f64))->(f64,f64){

     let offset0 = 2.0 * u.0 -1.0;
     let offset1 = 2.0 * u.1 -1.0;
     if (offset0 == 0.0 && offset1 == 0.0) {return (0.0, 0.0);}
     
     if offset0.abs() >  offset1.abs() {
      let   r = offset0;
       let  theta = (offset1 / offset0 )*f64::consts::FRAC_PI_4;
       ( r *  theta.cos() , r *  theta.sin())
     }else{
        let   r = offset1;
        let  theta = f64::consts::FRAC_PI_2 - (offset0 / offset1)*f64::consts::FRAC_PI_4;
       
        ( r *  theta.cos() , r *  theta.sin())
     }
 
     
}
pub fn CosSampleH2(u: &(f64, f64))->Vector3<f64>{
   let sp = ConcentricSampleDisk(u);
   let z = (1.0 - sp.0 * sp.0 - sp.1 * sp.1).max(0.0).sqrt();
   Vector3::new(sp.0, sp.1, z)

}
pub fn CosSampleH2Pdf(u:   f64)-> f64 {
    f64::consts::FRAC_1_PI * u
 
 }
/***
 *      z|
 *       |
 *       |
 *   y /  \ x
 *
 */

fn UniformSampleSemisphereVec(u: &(f64, f64)) -> Vector3<f64> {
    let z = u.0;
    let r = f64::max(0.0, (1.0 - (z * z))).sqrt();
    let phi = 2.0 * f64::consts::PI * u.1;
    let (s, c) = phi.sin_cos();
    let x = r * s;
    let y = r * c;
    Vector3::new(x, y, z).normalize()
}
pub
fn UniformSampleSemispherePdf() -> f64 {
    f64::consts::FRAC_PI_2
}
pub fn UniformSampleSemisphere(u: &(f64, f64)) -> (Vector3<f64>, f64) {
    (UniformSampleSemisphereVec(u), UniformSampleSemispherePdf())
}



// Vector3f UniformSampleSphere(const Point2f &u) {
//     Float z = 1 - 2 * u[0];
//     Float r = std::sqrt(std::max((Float)0, (Float)1 - z * z));
//     Float phi = 2 * Pi * u[1];
//     return Vector3f(r * std::cos(phi), r * std::sin(phi), z);
// }



fn UniformSampleSphereVec(u: &(f64, f64)) -> Vector3<f64> {
    let z = 1.0-2.0*u.0;
    let r = f64::max(0.0, (1.0 - (z * z))).sqrt();
    let phi = 2.0 * f64::consts::PI * u.1;
    let (s, c) = phi.sin_cos();
    let y  = r * s;
    let x = r * c;
    Vector3::new(x, y, z).normalize()
}
pub
fn UniformSampleSpherePdf() -> f64 {
    // inv4pi
    0.07957747154594766788
}
pub fn UniformSampleSphere(u: &(f64, f64)) -> (Vector3<f64>, f64) {
    (UniformSampleSphereVec(u), UniformSampleSpherePdf())
}








pub
fn UniformSampleConeVec(u: &(f64, f64), costhetamax: f64) -> Vector3<f64> {
    let costheta = (1.0 - u.0) + u.0 * costhetamax;
    let sintheta = (1.0 - costheta * costheta).sqrt();
    let phi = u.1 * 2.0 * f64::consts::PI;
    let (cosphi, sinphi) = phi.sin_cos();
    Vector3::new(cosphi * sintheta, sinphi * sintheta, costheta)
}
pub
fn UniformSampleConePdf(costhetamax: f64) -> f64 {
    let pdf = 1.0 / (2.0 * f64::consts::PI * (1.0 - costhetamax));
    pdf
}
pub
fn UniformSampleCone(u: &(f64, f64), costhetamax: f64) -> (Vector3<f64>, f64) {
    (
        UniformSampleConeVec(u, costhetamax),
        UniformSampleConePdf(costhetamax),
    )
}

// esto permite test test_sample_sphere poniendolo en el sampleo

pub fn AbsCosTheta<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    w.z.abs()
}
pub fn CosTheta<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    w.z
}
pub fn Cos2Theta<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    w.z * w.z
}

pub fn Sin2Theta<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
   let s =  Cos2Theta(w);
  let c2 =  cast::<Scalar,f64>(s).unwrap();
 
    cast::<f64, Scalar>(1.0 - c2.max(0.0)).unwrap()
}

pub fn SinTheta<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    Sin2Theta(w).sqrt()
}

pub fn TanTheta<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    SinTheta(w) / CosTheta(w)
}

pub fn Tan2Theta<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    Sin2Theta(w) / Cos2Theta(w)
}

pub fn CosPhi<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    let sintheta = SinTheta(w);
    
    if sintheta == cast::<f64, Scalar>(0.0).unwrap() {
        cast::<f64, Scalar>(1.0).unwrap()
    } else {
        clamp(
            w.x / sintheta,
            cast::<f64, Scalar>(-1.0).unwrap(),
            cast::<f64, Scalar>(1.0).unwrap(),
        )
    }
}

pub fn SinPhi<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    let sintheta = SinTheta(w);
    if sintheta == cast::<f64, Scalar>(0.0).unwrap() {
        cast::<f64, Scalar>(0.0).unwrap()
    } else {
        clamp(
            w.y / sintheta,
            cast::<f64, Scalar>(-1.0).unwrap(),
            cast::<f64, Scalar>(1.0).unwrap(),
        )
    }
}

pub fn Cos2Phi<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    CosPhi(w) * CosPhi(w)
}

pub fn Sin2Phi<Scalar: BaseFloat>(w: &Vector3<Scalar>) -> Scalar {
    SinPhi(w) * SinPhi(w)
}



pub fn SphTheta(w : Vector3<f64>)->f64{
    clamp(w.z , -1.0, 1.0).acos()
}
pub  fn SphPhi(w : Vector3<f64>)->f64{
   let mut phi =  w.y .atan2(w.x) ;
   if phi<0.0 {
        phi+=2.0 * std::f64::consts::PI;
   }
   phi
         
}
pub fn SphericalCoords(w:Vector3<f64>)->(f64,f64){
    (SphTheta(w), SphPhi(w))
}










pub
trait FrameShadingUtilites<Scalar>{
    fn sin2phi(self:Self)->Scalar;
    fn sinphi(self:Self)->Scalar;
    fn cos2phi(self:Self)->Scalar;
    fn cosphi(self:Self)->Scalar;
    fn tantheta(self:Self)->Scalar;
    fn tan2theta(self:Self)->Scalar;
    fn abscostheta(self:Self)->Scalar;
    fn reflect(self:Self, other : Self)->Self;
    fn sameSemisphere(self:Self, other : Self)->bool;
    fn clamp(self:Self, low : Scalar, hight : Scalar)->Self;
}
impl <Scalar:BaseFloat>  FrameShadingUtilites<Scalar> for Vector3<Scalar>  {
    fn sameSemisphere(self:Self, other : Self)->bool{
        self.z * other.z >num_traits::cast::<f64, Scalar>(0.0).unwrap()
    }
    fn reflect(self:Self, other : Self) ->Self {
        let s =  self.dot(other)*num_traits::cast::<f64, Scalar>(2.0).unwrap();
        Vector3::new(-self.x + s*other.x,-self.y + s*other.y,-self.z + s*other.z)
      
    }
   fn sin2phi(self:Self)->Scalar {
    
       Sin2Phi(&self)
      
   }

   fn sinphi(self:Self)->Scalar  {
  
       SinPhi(&self)
   }

   fn cos2phi(self:Self)->Scalar  {
       Cos2Phi(&self)
   }

   fn cosphi(self:Self)->Scalar  {
       CosPhi(&self)
   }

   fn abscostheta(self:Self)->Scalar  {
    
       AbsCosTheta(&self)
   }

   fn tantheta(self:Self)->Scalar {
       TanTheta(&self)
   }

   fn tan2theta(self:Self)->Scalar {
       Tan2Theta(&self)
   }
    fn clamp(self:Self, low : Scalar, hight : Scalar) ->Self {
       Vector3::new( clamp(self.x, low, hight),clamp(self.y, low, hight),clamp(self.z, low, hight))
    }
}
 

#[derive(Debug, Clone, Copy)]
pub enum MicroFaceDistribution {
    BeckmannDistribution(BeckmannDistribution),
    TrowbridgeReitzDistribution( TrowbridgeReitzDistribution),
}

#[derive(Debug, Clone, Copy)]
pub struct BeckmannDistribution {
   pub ax :f64, 
   pub ay :f64,

    // R  : Srgb,
}
impl BeckmannDistribution {
    // rougness
    pub fn new( rx :f64, ry :f64 )->BeckmannDistribution{
        BeckmannDistribution{ax:beckman_alpha(rx),ay: beckman_alpha(ry)}
    }
    pub fn beckman_alpha( r :f64 )->f64{
        beckman_alpha(r)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TrowbridgeReitzDistribution {
   pub ax :f64, 
   pub ay :f64,
    // R  : Srgb,
}


impl TrowbridgeReitzDistribution{
    pub fn new(rx:f64, ry:f64)->TrowbridgeReitzDistribution{
        TrowbridgeReitzDistribution{ax : trowbridge_alpha(rx), ay :trowbridge_alpha(ry)}
    }
       
}
 
struct RecordSampleDistributionIn{
    u : (f64, f64),
    prev : Vector3<f64>,

}
struct RecordSampleDistributionOut{
    mid : Vector3<f64>
}

trait IMicroDistribution {
    fn D(&self, wh: &Vector3<f64> ) -> f64;
    fn G(&self, wprevlocal: &Vector3<f64>, wnextlocal: &Vector3<f64>) -> f64;
    fn sample(&self , recordin : &RecordSampleDistributionIn )->RecordSampleDistributionOut;
    fn pdf(&self, prev : &Vector3<f64>, mid : &Vector3<f64> )->f64;
}

impl IMicroDistribution for MicroFaceDistribution {
    fn D(&self, wh: &Vector3<f64>) -> f64 {
        match *self {
            MicroFaceDistribution::BeckmannDistribution(mi) => mi.D(wh),
            MicroFaceDistribution::TrowbridgeReitzDistribution(trow)=>trow.D(wh)
        }
    }
    fn G(&self, wprevlocal: &Vector3<f64>, wnextlocal: &Vector3<f64>) -> f64 {
        match *self {
            MicroFaceDistribution::BeckmannDistribution(mi) => mi.G(wprevlocal, wnextlocal),
             MicroFaceDistribution::TrowbridgeReitzDistribution(trow)=>trow.G(wprevlocal, wnextlocal)
        }
    }
    fn sample(&self, recordin : &RecordSampleDistributionIn) ->RecordSampleDistributionOut {
        match *self {
            MicroFaceDistribution::BeckmannDistribution(mi) => mi.sample(&RecordSampleDistributionIn{prev:recordin.prev,u:recordin.u}),
             MicroFaceDistribution::TrowbridgeReitzDistribution(trow)=>trow.sample(&RecordSampleDistributionIn{prev:recordin.prev,u:recordin.u})
        }
    }
    fn pdf(&self, prev : &Vector3<f64>, mid : &Vector3<f64> )->f64{
        match *self {
            MicroFaceDistribution::BeckmannDistribution(mi) => mi.pdf(prev, mid),
             MicroFaceDistribution::TrowbridgeReitzDistribution(trow)=>trow.pdf(prev, mid)
        }
    }
    
    
   
}

impl IMicroDistribution for BeckmannDistribution {
    fn D(&self, wh: &Vector3<f64> ) -> f64 {
        
        beckman_D(wh,self. ax,self. ay)
    }
    fn G(&self, vprev: &Vector3<f64>, vnext: &Vector3<f64> ) -> f64 {
        beckman_G(vprev, vnext, self.ax, self.ay)
    }
    
    fn sample(&self, recordin : &RecordSampleDistributionIn)->RecordSampleDistributionOut {
        RecordSampleDistributionOut { mid: Vector3::new(0.0,0.0,0.0) }
    }
    fn pdf(&self, prev : &Vector3<f64>, mid : &Vector3<f64>) ->f64 {
        0.0
    }
}

impl IMicroDistribution for  TrowbridgeReitzDistribution {
    fn D(&self, wh: &Vector3<f64> ) -> f64 {
        trowbridge_D(wh, self.ax,self. ay) 
    }
    fn G(&self, vprev: &Vector3<f64>, vnext: &Vector3<f64> ) -> f64 {
        trowbridge_G(vprev, vnext, self.ax, self.ay)
       
    }
    fn pdf(&self, prev : &Vector3<f64>, mid : &Vector3<f64>)->f64 {
        self.D(mid) * AbsCosTheta(mid)
        
    }
    fn sample(&self, recordin : &RecordSampleDistributionIn)->RecordSampleDistributionOut {

        if self.ax == self.ay {
            let phi = (2.0 * f64::consts::PI)*recordin.u.1;
           let sc =  phi.sin_cos();

            // sample symetrict TrowbridgeReitzDistribution
            let tan2theta = self.ax * self.ay *(recordin.u.0/(1.0-recordin.u.0));
            let costheta  = 1.0 /(1.0+tan2theta).sqrt();
           let sintheta = ( 1.0 - costheta*costheta).max(0.0).sqrt();
           let mut vmid = Vector3::new( 
            sintheta * sc.1,
            sintheta * sc.0,
            costheta

           );
           if recordin.prev.dot(vmid) < 0.0{
            return  RecordSampleDistributionOut{mid:-vmid}
           }else{
            return  RecordSampleDistributionOut{mid:vmid}
           }
        } else{
          let  phi = ( (self.ay /  self.ax) * ( 2.0 * f64::consts::PI * recordin.u.1 + 0.5 * f64::consts::PI).tan()).atan();
          let sc =  phi.sin_cos();
          let a2x = self.ax *self.ax;
          let a2y = self.ay *self.ay;
          let a2 = 1.0/ ( sc.1*sc.1) / a2x +  ( sc.0*sc.0) / a2y ;
          let tan2theta = a2 *(recordin.u.0/(1.0-recordin.u.0));
          let costheta  = 1.0 /(1.0+tan2theta).sqrt();
          let sintheta = ( 1.0 - costheta*costheta).max(0.0).sqrt();
          let mut vmid = Vector3::new( 
            sintheta * sc.1,
            sintheta * sc.0,
            costheta

           );
           if recordin.prev.dot(vmid) < 0.0{
            return  RecordSampleDistributionOut{mid:-vmid}
           }else{
            return  RecordSampleDistributionOut{mid:vmid}
           }
        }
    }
    
}

 
 

fn fresnel_aprox(dotHI: f64, eta : f64)->f64{

    let d = (1.0 - eta) * (1.0 -dotHI).powf(5.0);
    eta + d
}


#[derive(Debug, Clone, Copy)]
pub struct MicroFacetReflection{
  pub    frame: Option<FrameShading>,
  pub  distri : MicroFaceDistribution,
  pub fresnel : Fresnel
}
impl Pdf for MicroFacetReflection {
    fn pdf(&self, prev: Vector3<f64>, next: Vector3<f64>) -> f64{
        
        self.distri.pdf(&prev, &next) /( 4.0 *prev.dot(next))
    }
}

impl Fr  for MicroFacetReflection{
    fn fr(&self, vprev: Vector3<f64>, vnext: Vector3<f64>) -> Srgb {
        let ci = AbsCosTheta(&vnext);
        let co = AbsCosTheta(&vprev);
        if ci== 0.|| co == 0. {return Srgb::new( 0f32,0f32,0f32)}
        
        let mut wh = ( vnext+ vprev);
        if wh.x == 0.0 && wh.y == 0.0 && wh.z ==0.0 {
            return Srgb::new( 0f32,0f32,0f32)
        }
        wh = wh.normalize();
        let F =  self.fresnel.eval(wh.dot(vnext)); //fresnel_aprox(wh.dot(vnext),0.1024);
        let D =  self.distri.D(&wh);
        let G =   self.distri.G(&vprev, &vnext);
        let fr = ((F*D * G )/(4.0 *co*ci)) as f32;
          
        Srgb::new( fr, fr,  fr)
    }
}
impl SampleIllumination  for MicroFacetReflection{
    fn sample(&self, record: RecordSampleIn ) -> RecordSampleOut  {
        

        let bsdf = self.frame.unwrap();
        let vprevlocal = bsdf.to_local(&record.prevW);
       if  vprevlocal.z == 0.0 {}
       let recordout =  self.distri.sample(&RecordSampleDistributionIn{prev:vprevlocal, u : record.sample.unwrap()});
       let nextlocal = vprevlocal.reflect( recordout.mid);
       if !nextlocal.sameSemisphere(vprevlocal) {
        return RecordSampleOut{
            f:Srgb::new( 0f32,0f32,0f32),
            newray:None,
            next:Vector3::new(0.0,0.0,0.0),
            pdf:0.0,
        }
       }
     
       let pdf  = self.distri.pdf(&vprevlocal, &recordout.mid) /( 4.0 *vprevlocal.dot(recordout.mid));
         let f = self.fr(vprevlocal, nextlocal);

        let nextworld = bsdf.to_world(&nextlocal);
        RecordSampleOut{
            f ,
            newray:Some(Ray::new(record.pointW, nextworld)),
            next:nextworld,
            pdf ,
        }
    }
}
// asi se instancia
fn testff(){
    let microfacet = MicroFacetReflection{
        
        frame:Some(FrameShading::default()),
        distri : MicroFaceDistribution::BeckmannDistribution(BeckmannDistribution::new(0.1, 0.1)),
        fresnel:Fresnel::FresnelNop(FresnelNop{})
    };
    microfacet.fr(Vector3::new(0.0,0.0,1.0), Vector3::new(0.0, 0.0, 1.0));
    Vector3::from_value(0.0).tan2theta();
}

pub

fn fresnel_aprox_disney(c:f64)->f64{
    let f =clamp(1.0-c,0.0, 1.0);
    (f*f)*(f*f)*f
 }
 fn fresnelCondutor(c:f64,etai:f64, etat:f64, k:f64)->f64{
    let ci = clamp(c, -1.0, 1.0);
    let costhetai2 = ci*ci;
    let sinthetai2 = 1.0-costhetai2;
    let eta = etat / etai;
    let eta2 = eta * eta;
    let etak = k / etai;
    let etak2 =etak * etak;
    let t0 = eta2 - etak2- sinthetai2;
    let a2b2 = (t0*t0+4.0*eta2*etak).sqrt();
    let a = (0.5 *(a2b2+t0)).sqrt();
    let  t1 = a2b2 + costhetai2;
    let  t2 = 2.0 * ci*a;
    let rs = (t1-t2)/(t1+t2);
    let t3 = costhetai2*a2b2+sinthetai2*sinthetai2;
    let t4  = t2 * sinthetai2;
    let rp = rs *(t3-t4)/(t3+t4);
   ( rp + rs ) * 0.5
 }



 pub fn fr_dielectric(cos_theta_i: f64, eta_i: f64, eta_t: f64) -> f64 {
    let mut cos_theta_i = clamp(cos_theta_i, -1.0, 1.0);
     
    let entering: bool = cos_theta_i > 0.0;
    // use local copies because of potential swap (otherwise eta_i and
    // eta_t would have to be mutable)
    let mut local_eta_i = eta_i;
    let mut local_eta_t = eta_t;
    if !entering {
        std::mem::swap(&mut local_eta_i, &mut local_eta_t);
        cos_theta_i = cos_theta_i.abs();
    }
    // compute _cos_theta_t_ using Snell's law
    let sin_theta_i = (0.0 as f64)
        .max(1.0 as f64 - cos_theta_i * cos_theta_i)
        .sqrt();
    let sin_theta_t = local_eta_i / local_eta_t * sin_theta_i;
    // handle total internal reflection
    if sin_theta_t >= 1.0 as f64 {
        return 1.0 as f64;
    }
    let cos_theta_t  = (0.0 as f64)
        .max(1.0 as f64 - sin_theta_t * sin_theta_t)
        .sqrt();
    let r_parl: f64 = ((local_eta_t * cos_theta_i) - (local_eta_i * cos_theta_t))
        / ((local_eta_t * cos_theta_i) + (local_eta_i * cos_theta_t));
    let r_perp: f64 = ((local_eta_i * cos_theta_i) - (local_eta_t * cos_theta_t))
        / ((local_eta_i * cos_theta_i) + (local_eta_t * cos_theta_t));
    (r_parl * r_parl + r_perp * r_perp) / 2.0
}
 
pub
 fn frSclick(r0:f64, c:f64)->f64{
     lerp(fresnel_aprox_disney(c), r0, 1.0)
 }
 //peje : eta = 1.5 air->plastic
 
pub fn transform_eta_to_air_to_r0(eta:f64)->f64{
     (eta-1.0).sqrt() /  (eta+1.0).sqrt()
 }
 
 
pub fn fresnel_diffuse_disney(wprev: &Vector3<f64>, wnext: &Vector3<f64>)->f64{
    let fin  =fresnel_aprox_disney(AbsCosTheta(wprev));
  let fout =   fresnel_aprox_disney( AbsCosTheta(wnext) );
   f64::consts::FRAC_1_PI*( 1.0-  fin /  2.0) * (1.0-  fout /  2.0)
 }






 pub fn beckman_alpha(r: f64) -> f64 {
    let x = r.max(1e-3).ln( );
   
    1.62142 + 0.819955 * x + 0.1734 * x * x + 0.0171201 * x * x * x + 0.000640711 * x * x * x * x
}
pub fn beckman_D(wh: &Vector3<f64>, ax: f64, ay: f64) -> f64 {
    let t = Tan2Theta(&wh);
    if Tan2Theta(&wh).is_infinite() {
        return 0.0;
    }

    let cos4theta = Cos2Theta(&wh) * Cos2Theta(&wh);
    let c2phi = Cos2Phi(wh) ;
    let s2phi = Sin2Phi(wh) ;
    let elip = Cos2Phi(wh) / (ax * ax) + Sin2Phi(wh) / (ay * ay);
   let t = -Tan2Theta(&wh);
    let r = t*elip;
    let tan = Float::exp(r  );
   let et = Float::exp(-Tan2Theta(&wh)   );
   

    tan / (f64::consts::PI * ax * ay * cos4theta)
}




pub fn Lambda_beckmann(w: &Vector3<f64>, ax: f64, ay: f64) -> f64 {
    let abst = TanTheta(w).abs();
    if abst.is_infinite() {
        return 0.0;
    }
    let alp = (Cos2Phi(w) * ax * ax + Sin2Phi(w) * ay * ay).sqrt();
    let a = 1.0 / (alp * abst);
    if  a >=1.6{
        return 0.0;
    }
    (1.0 - 1.259 * a + 0.396 * a * a) / (3.535 * a + 2.181 * a * a)
}
pub fn beckman_G(wprev: &Vector3<f64>, wnext: &Vector3<f64>,ax: f64, ay: f64)-> f64 {
    1.0 /(1.0+Lambda_beckmann(wprev, ax, ay)+Lambda_beckmann(wnext, ax, ay))
}

















pub fn trowbridge_alpha(r: f64) -> f64 {
    let x = r.max(1e-3).ln( );
   
    1.62142 + 0.819955 * x + 0.1734 * x * x + 0.0171201 * x * x * x +0.000640711 * x * x * x * x
}
pub fn trowbridge_D(wh: &Vector3<f64>, ax: f64, ay: f64) -> f64 {
    if Tan2Theta(&wh).is_infinite() {
        return 0.0;
    }
    let elip = Cos2Phi(wh) / (ax * ax) + Sin2Phi(wh) / (ay * ay);
    let t = Tan2Theta(&wh) * elip;

    let cos4theta = Cos2Theta(&wh) * Cos2Theta(&wh);
    let t2 = (1.0 + t) * (1.0 + t);

    1.0 / (f64::consts::PI * ax * ay * cos4theta * t2)
}


pub fn lambda_trowbridge(w: &Vector3<f64>, ax: f64, ay: f64) -> f64{
    let abst = TanTheta(w).abs();
    if abst.is_infinite() {
        return 0.0;
    }
    let alp = (Cos2Phi(w) * ax * ax + Sin2Phi(w) * ay * ay).sqrt();
    let a2 = (alp*abst)*(alp*abst);
    (-1.0 + (1.0+a2).sqrt())/2.0
}
pub fn trowbridge_G(wprev: &Vector3<f64>, wnext: &Vector3<f64>,ax: f64, ay: f64)-> f64 {
    1.0 /(1.0+lambda_trowbridge(wprev, ax, ay)+lambda_trowbridge(wnext, ax, ay))
}



pub fn phong(vprev : Vector3<f64>,vnext : Vector3<f64>)->f64{
    let mut wh = ( vnext+ vprev);
    if wh.x == 0.0 && wh.y == 0.0 && wh.z ==0.0 {
        return 0.0
    }
    wh = wh.normalize();
    let w = AbsCosTheta(&wh);
    w.powf(50.0)
}
pub fn orenNayar(vprev : Vector3<f64>,vnext : Vector3<f64>, sigmaDeg :f64)->f64{

    // una vez...pero
    let s = sigmaDeg.to_radians();
    let ss = s*s;
   let A =  1.0 - (ss /(2.0-(ss+0.33)));
   let B = 0.45 * ss /(ss+0.09);
     
    let sinthetaprev = SinTheta(&vprev);
    let sinthetanext = SinTheta(&vnext);
    let mut c : Option<f64> = None;
    if sinthetanext > 1e-4 && sinthetaprev > 1e-4{
       let snext =  SinPhi(&vnext) ; let cnext = CosPhi(&vnext);
       let sprev =   SinPhi(&vprev) ; let cprev =  CosPhi(&vprev);
       let mut ctemp = cnext * cprev + snext *sprev;
      c = Some( ctemp.max(0.0));

    }
    let mut sinalpha :Option<f64> = None;
    let mut  tnalpha:Option<f64> = None;
    if(AbsCosTheta(&vnext)>AbsCosTheta(&vprev)){
        sinalpha = Some(sinthetaprev);
        tnalpha =  Some(sinthetanext / AbsCosTheta(&vnext));
    }else{

        sinalpha = Some(sinthetanext);
        tnalpha =  Some(sinthetaprev / AbsCosTheta(&vprev));
    }
(A+  B * c.unwrap() * sinalpha.unwrap() * tnalpha.unwrap())*f64::consts::FRAC_1_PI  

}
pub fn blinn_D(wh:&Vector3<f64>, a :f64)->f64{
     let costheta = AbsCosTheta(&wh);
     costheta.powf(a) * (a +2.0) * f64::consts::FRAC_2_PI
}
pub fn blinn_G(vprev : &Vector3<f64>,vnext : &Vector3<f64>,vh :& Vector3<f64> )->f64{
   let acprev =  AbsCosTheta(&vprev);
   let acnext =   AbsCosTheta(&vnext);
   let ach = AbsCosTheta(&vh);
   let odoth =vprev.dot(*vh);
   1.0.min((2.0* ach * acprev / odoth).min(2.0*ach * acnext / odoth))
}
pub  fn microfacet_blinn(vprev : Vector3<f64>,vnext : Vector3<f64>,e :f64)->f64{
    let ci = AbsCosTheta(&vnext);
    let co = AbsCosTheta(&vprev);
    if ci== 0.|| co == 0. {
        return 0.0
    }


    let mut wh = ( vnext+ vprev);
    if wh.x == 0.0 && wh.y == 0.0 && wh.z ==0.0 {
        return 0.0
    }
    wh = wh.normalize();
    let G = blinn_G(&vprev, &vnext, &wh);
    let D = blinn_D(&wh,  2.0);
    D*G/(4.0 * ci * co)
}
pub fn microfacet_beckmann(vprev : Vector3<f64>,vnext : Vector3<f64>, ax :f64, ay:f64)->f64{
    let fresne = Fresnel::FresnelSchlickApproximation(FresnelSchlickApproximation::from_eta(1.5));
    let ci = AbsCosTheta(&vnext);
    let co = AbsCosTheta(&vprev);
    if ci== 0.|| co == 0. {
        return 0.0
    }


    let mut wh = ( vnext+ vprev);
    if wh.x == 0.0 && wh.y == 0.0 && wh.z ==0.0 {
        return 0.0
    }
    wh = wh.normalize();
    let F =  fresnel_aprox(wh.dot(vnext),0.1024);
    let G =  beckman_G(&vprev, &vnext, ax, ay);
    let D =  beckman_D(&wh,  ax, ay);
   
    (D*G)/(4.0 *co*ci)
   
}















#[derive(Debug, Clone, Copy)]
pub enum Fresnel {
    FresnelSchlickApproximation(FresnelSchlickApproximation),
    FresnelNop(FresnelNop),
    FresnelCondutor(FresnelConductor),
    FresnelDielectricType(FresnelDielectric),
}

pub trait Eval {
    fn eval(&self, c:f64) -> f64;
}

impl Eval for Fresnel {
    fn eval(&self, c:f64) -> f64 {
        // let zero = cast::<f64, Scalar>(0.0).unwrap();
        // zero
        match *self {
            Fresnel::FresnelSchlickApproximation(fresnel) => fresnel.eval(c),
            Fresnel::FresnelNop(fresnel) => fresnel.eval(c),
            Fresnel::FresnelCondutor(fresnel)=>fresnel.eval(c),
            Fresnel::FresnelDielectricType(fresnel)=>fresnel.eval(c)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FresnelSchlickApproximation {
    eta: f64,
}
impl FresnelSchlickApproximation {
    pub fn new(etaincoming: f64, etatransmitted: f64) -> FresnelSchlickApproximation {
        let n = (etaincoming - etatransmitted) / (etaincoming + etatransmitted);

        FresnelSchlickApproximation { eta: n * n }
    }
    /**
     * eta_t (trans)
     * ----
     * eta_i (incoming)
     */
    pub fn from_eta(eta: f64) -> FresnelSchlickApproximation {
        FresnelSchlickApproximation { eta }
    }
}

impl Eval for FresnelSchlickApproximation {
    fn eval(&self, c:f64) -> f64 {
        let d = (1.0 - self.eta) * (1.0 - c.max(0.0).min(1.0)).powf(5.0);
        self.eta + d
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FresnelNop {
    
}
impl  Default for FresnelNop {
    fn default() -> Self {
        FresnelNop {}
    }
}


impl Eval for FresnelNop {
    fn eval(&self, c:f64) -> f64 {
       1.0
    }
}




#[derive(Debug, Clone, Copy)]
pub struct FresnelConductor {
    pub etai:f64,
    pub etat:f64,
    pub k:f64,
    
}

impl FresnelConductor{
    pub fn new(etai:f64, etat:f64, k:f64)->FresnelConductor{
        // ejemplo .: etai = 1 , etat = 1.31, k =0,[0,10 )
        FresnelConductor {etai, etat, k}
    }

}
impl  Default for FresnelConductor {
    fn default() -> Self {
        FresnelConductor ::new(1.0 ,1.3, 1.0)
    }
}

impl  Eval for FresnelConductor {
    fn eval(&self, c:f64) -> f64 {
        fresnelCondutor(c.abs(), self.etai, self.etat, self.k)
    }
}

#[test]
pub fn frenselConductorTest(){
    let frcond  = Fresnel::FresnelCondutor(FresnelConductor::default());
    for ii in  0..150{
      let iii =   ii as f64 / 100.0;
      println!("{}, frcond {}", iii, frcond.eval(iii));
      
    }
}





#[derive(Debug, Clone, Copy)]
pub struct FresnelDielectric {
    pub etai:f64,
    pub etat:f64,
  
    
}

impl FresnelDielectric{
    pub fn new(etai:f64, etat:f64 )->FresnelDielectric{
        // ejemplo .: etai = 1 , etat = 1.31, k =0,[0,10 )
        FresnelDielectric {etai, etat }
    }

}
impl  Default for FresnelDielectric{
    fn default() -> Self {
        FresnelDielectric::new(1.0 ,1.3 )
    }
}

impl  Eval for FresnelDielectric {
    fn eval(&self, c:f64) -> f64 {
        fr_dielectric(c , self.etai, self.etat) 
    }
}
 





























#[derive(Debug, Clone, Copy)]
pub struct FrameShading {
    pub n: Vector3<f64>,
    pub tn: Vector3<f64>,
    pub bn: Vector3<f64>,
}

impl  Default for FrameShading {
    fn default() -> Self {
        
        FrameShading {
            n: Vector3::new(0.0,0.0,0.0),
            tn: Vector3::new(0.0,0.0,0.0),
            bn: Vector3::new(0.0,0.0,0.0),
        }
    }
}

impl  FrameShading {
    pub fn new(n: Vector3<f64>, tn: Vector3<f64>, bn: Vector3<f64>) -> FrameShading {
        FrameShading { n, tn, bn }
    }
    pub fn to_local(&self, v: &Vector3<f64>) -> Vector3<f64> {
        Vector3::new(v.dot(self.bn), v.dot(self.tn), v.dot(self.n))
    }
    pub fn to_world(&self, v: &Vector3<f64>) -> Vector3<f64> {
        Vector3::new(
            self.bn.x * v.x + self.tn.x * v.y + self.n.x * v.z,
            self.bn.y * v.x + self.tn.y * v.y + self.n.y * v.z,
            self.bn.z * v.x + self.tn.z * v.y + self.n.z * v.z,
        )
    }
    pub fn random(min: f64, max: f64) -> Vector3<f64> {
        let mut rng = rand::thread_rng();
        Vector3::new(
            rng.gen_range(min..max),
            rng.gen_range(min..max),
            rng.gen_range(min..max),
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BsdfType {
    None,
    Lambertian(LambertianBsdf ),
    LambertianDisneyBsdf(LambertianDisneyBsdf ),
    MicroFacetReflection(MicroFacetReflection),
    SpecReflection(SpecReflection),
    FresnelTransmissionReflectionType(FresnelTransmissionReflection),
    GlassBsdfType(GlassBsdf),

    // Metal(Metal),
}






pub trait IsSpecular  {
    fn is_specular(&self) -> bool ;
}
 impl IsSpecular for BsdfType{
    fn is_specular(&self) -> bool {
        match *self {
            BsdfType::SpecReflection(b)=>true,
            BsdfType::GlassBsdfType(b)=>true,
            BsdfType::FresnelTransmissionReflectionType(b)=>true,
            _=>false
        }
    }
 }




#[derive(Debug, Clone, Copy)]
pub struct LambertianBsdf  {
    pub frame: Option<FrameShading >,
    pub albedo:  Srgb ,
}
impl  LambertianBsdf  {
    pub fn new(bsdf: FrameShading , albedo: Srgb) -> LambertianBsdf  {
        LambertianBsdf {
           frame: Some(bsdf),
            albedo
        }
    }
    pub fn from_albedo(albedo: Srgb) -> LambertianBsdf {
        LambertianBsdf { frame: None, albedo }
    }
    pub fn from_lum(s: f64) -> LambertianBsdf  {
        LambertianBsdf {
            frame: None,
            albedo: Srgb::new(
                s as f32,
                s as f32,
               s as f32,
           )
        }
    }
   pub fn internal_pdf(&self, prevlocal: Vector3<f64>, nextlocal: Vector3<f64>) -> f64 {
        if prevlocal.z == 0.0{return 0.0}
 //       if prevlocal.dot(nextlocal) < 0.0 {return 0.0}
      if prevlocal.z * nextlocal.z > 0.0 {
       nextlocal.abscostheta() * f64::consts::FRAC_1_PI
      }else{0.0}
      
  }
}



impl  Default for LambertianBsdf  {
    fn default() -> Self {
        LambertianBsdf {
            frame: None,
            albedo: Srgb::new(1.0, 1.0, 1.0)
        }
    }
}





#[derive(Debug, Clone, Copy)]
pub struct LambertianDisneyBsdf  {
    pub frame: Option<FrameShading >,
    pub albedo: Srgb,
}
impl  LambertianDisneyBsdf {
    pub fn new(frame: FrameShading , albedo: Srgb) -> LambertianDisneyBsdf  {
        LambertianDisneyBsdf {
           frame: Some(frame),
            albedo,
        }
    }
    pub fn from_albedo(albedo: Srgb) -> LambertianDisneyBsdf {
        LambertianDisneyBsdf { frame: None, albedo }
    }
    pub fn from_lum(s: f64) -> LambertianDisneyBsdf  {
        LambertianDisneyBsdf {
            frame: None,
            albedo: Srgb::new(
                 s as f32,
                 s as f32,
                s as f32,
            ),
        }
    }
}



impl  Default for LambertianDisneyBsdf  {
    fn default() -> Self {
        LambertianDisneyBsdf {
            frame: None,
            albedo: Srgb::new(1.0, 1.0, 1.0),
        }
    }
}


impl 
SampleIllumination  for  LambertianDisneyBsdf 
{
    fn sample(&self, mut record: RecordSampleIn ) -> RecordSampleOut  {
        use crate::primitives::prims::Ray;

        //TODO: hay que hacerlo sobre el cos de la normal
        let mut rng = rand::thread_rng();
 

        let bsdf = self.frame.unwrap();
        let wprevlocal = bsdf.to_local(&record.prevW);

        let min =  0.0 ;
        let max =  1.0 ;

        let tuple = (
        min..max ,
            min..max ,
        );
        let sampleCone = UniformSampleCone(&(tuple.0.start, tuple.1.start), 0.91);

        let newdirlocal = Vector3::new(
            sampleCone.0.x,
            sampleCone.0.y,
            sampleCone.0.z,
          
        )
        .normalize();
        //     let f = self.fr(wprevlocal, newdirlocal);

        let nextworld = bsdf.to_world(&newdirlocal);

        return RecordSampleOut {
            newray: Some(Ray::new(record.pointW, nextworld)),
            f: Srgb::new(1.0, 0.1, 0.5),
            next: nextworld,
            pdf:  sampleCone.1 ,
        };
    }
}
impl  Fr for  LambertianDisneyBsdf 
{
    fn fr(&self, prevlocal: Vector3<f64>, nextlocal: Vector3<f64>) -> Srgb {
          if prevlocal.z == 0.0{return Srgb::new(0.0,0.0,0.0)}
        if prevlocal.dot(nextlocal) < 0.0 {return Srgb::new(1.0,0.0,0.0)}
        let fr_diffuse_disney =  fresnel_diffuse_disney(&prevlocal, &nextlocal) as f32;
     
        Srgb::new( self.albedo.red*fr_diffuse_disney,  self.albedo.green * fr_diffuse_disney,  self.albedo.blue * fr_diffuse_disney  )
 
    }
}










#[derive(Debug, Clone,  Copy)]
pub struct SpecReflection  {
    pub frame: Option<FrameShading >,
    pub albedo: Option<Srgb>,
    pub fresnel : Option<Fresnel>,
}
impl SpecReflection   {
    pub fn new(bsdf: FrameShading , albedo: Srgb) ->  SpecReflection  {
        SpecReflection {
           frame: Some(bsdf),
            albedo:Some(albedo),
            fresnel:None,
        }
    }
    pub fn from_albedo(albedo: Srgb) ->  SpecReflection {
        SpecReflection{ frame: None, albedo:Some(albedo) , fresnel:None}
    }
     
    pub fn from_none( ) ->  SpecReflection {
        SpecReflection{ frame: None, albedo:None, fresnel:None }
    }
    pub fn inner_sample(&self, mut record: RecordSampleIn)-> RecordSampleOut{

        let bsdf = self.frame.unwrap();
        let wprevlocal = bsdf.to_local(&record.prevW);

        
        let newdirlocal = Vector3::new(
            -wprevlocal .x,
             -wprevlocal .y,
 
             wprevlocal .z,
              
           
         );
         let R = self.albedo.unwrap();
         let cosprev = CosTheta(&newdirlocal) as f32;
         // fresnel->eval(cosprev)
  //        c.abs(), self.etai, self.etat
        let a =  self.fresnel.unwrap().eval(newdirlocal.z);
        let b  = AbsCosTheta(&newdirlocal);
         let fresnel_term = self.fresnel.unwrap().eval(newdirlocal.z) as f32;
         let abscosnext = AbsCosTheta(&newdirlocal) as f32;
        let res =  Srgb ::new (fresnel_term *R.red / abscosnext,fresnel_term *R.green/ abscosnext,fresnel_term *R.blue/ abscosnext);

        let nextworld = bsdf.to_world(&newdirlocal);
       
        let r =  Ray::new(record.pointW, nextworld);
        
        return RecordSampleOut {
            newray: Some(r),
            f:  res,
            next:nextworld,
            pdf: 1.0 ,
        };
         
    }
}



impl  Default for SpecReflection   {
    fn default() -> Self {
        SpecReflection  {
            fresnel:None,
            frame: None,
            albedo: Some(Srgb::new(1.0, 1.0, 1.0))
        }
    }
}




impl 
SampleIllumination  for  SpecReflection
{
    fn sample(&self, mut record: RecordSampleIn ) -> RecordSampleOut  {
        use crate::primitives::prims::Ray;

      
         
 

        let bsdf = self.frame.unwrap();
        let wprevlocal = bsdf.to_local(&record.prevW);

        


        let newdirlocal = Vector3::new(
           -wprevlocal .x,
            -wprevlocal .y,

            wprevlocal .z,
             
          
        );
        
        let R = self.albedo.unwrap();
        let cosprev = CosTheta(&newdirlocal) as f32;
        // fresnel->eval(cosprev)
        let fresnel = 1.0;
        let abscosnext = AbsCosTheta(&newdirlocal) as f32;
       let res =  Srgb ::new (fresnel *R.red / abscosnext,fresnel *R.green/ abscosnext,fresnel *R.blue/ abscosnext);
        
        
        //     let f = self.fr(wprevlocal, newdirlocal);

        let nextworld = bsdf.to_world(&newdirlocal);
       
        let r =  Ray::new(record.pointW, nextworld);
        
        return RecordSampleOut {
            newray: Some(r),
            f:  res,
            next:nextworld,
            pdf: 1.0 ,
        };
    }
}
impl  Fr for  SpecReflection 
{
    fn fr(&self, prevlocal: Vector3<f64>, nextlocal: Vector3<f64>) -> Srgb {
       Srgb ::new (0.0,0.0,0.0) 
        
    }
}













#[derive(Debug, Clone,  Copy)]
pub struct SpecTransmission  {
    pub frame: Option<FrameShading >,
    pub albedo: Option<Srgb>,
    pub fresnel : Option<Fresnel>,
    pub eta_a : f64,
    pub eta_b : f64,
}
impl  SpecTransmission    {
    pub fn new(bsdf: FrameShading , albedo: Srgb, eta_a : f64 , eta_b : f64) ->   SpecTransmission   {
        SpecTransmission  {
           frame: Some(bsdf),
            albedo:Some(albedo),
            fresnel:None,
            eta_a,eta_b
        }
    }
    pub fn from_albedo(albedo: Srgb, eta_a : f64 , eta_b : f64) ->   SpecTransmission  {
        SpecTransmission { frame: None, albedo:Some(albedo) , fresnel:None, eta_a, eta_b}
    }
     
    pub fn from_none( ) ->   SpecTransmission {
        SpecTransmission { frame: None, albedo:None, fresnel:None ,eta_a:0.0,eta_b:0.0 }
    }
    fn refract(next : Vector3<f64>, n : Vector3<f64>, eta : f64 )->(bool, Vector3<f64>){
       let costi = next.dot(n);
       let sin2ti = (1.0-costi*costi).max(0.0);
       let sin2tt = eta*eta *sin2ti;
       if sin2tt>=1.0 {
        //criticalpoint
        return (false, Vector3::new(0.0,0.0,0.0))
       }
     let cost =   ( 1.0 - sin2tt ).sqrt();
      let  b =eta *costi - cost;
      let nn = Vector3::new(n.x * b , n.y * b, n.z *b);
      let a  = Vector3::new(-eta * next.x, - eta * next.y,-eta * next.z);
      let r = Vector3::new (a.x + nn.x, a.y + nn.y, a.z + nn.z );
      (true, r )

    }
    pub fn inner_sample(&self, mut recordin: RecordSampleIn) -> RecordSampleOut {
        let bsdf = self.frame.unwrap();
        let prev = bsdf.to_local(&recordin.prevW);
        let mut eta_t: f64;
        let mut eta_i: f64;
        if prev.z > 0.0 {
            eta_i = self.eta_a;
            eta_t = self.eta_b;
        }else{
            eta_t = self.eta_a;
            eta_i = self.eta_b;
        }
        fn face(n : Vector3<f64>, v : Vector3<f64>)-> Vector3<f64> {
            if n.dot(v) < 0.0{
                -n 
            }else { 
                 n 
                }
        };
        let n  =face(Vector3::new(0.0,0.0,1.0), prev);
        let rs = Self::refract(prev, n, eta_i / eta_t );
        if  !rs.0  {
            return RecordSampleOut ::default(); // default initialice the struture with ray = None
        }

        let R = self.albedo.unwrap();
        let cosprev = CosTheta(&rs.1)  ;
         
       let a =  self.fresnel.unwrap().eval(cosprev);
       let b  = AbsCosTheta(&rs.1) as f32;
        let fresnel_term = self.fresnel.unwrap().eval(rs.1.z) as f32;
       let term = 1.0 - fresnel_term ;

       let mut res : Srgb; 
       // radiance mode 
       if true {
        let symmetry = (  (eta_i*eta_i) /     (eta_t*eta_t) ) as f32;
         res  = Srgb::new( symmetry * self.albedo.unwrap() .red *term / b,symmetry *self.albedo.unwrap() .green * term / b,symmetry *self.albedo.unwrap() .blue  * term / b )
       }else{ // importance mode
          res  = Srgb::new(self.albedo.unwrap() .red *term / b,self.albedo.unwrap() .green* term / b,self.albedo.unwrap() .blue  * term / b )
       }
       let next = bsdf.to_world(&rs.1);
      
      
  RecordSampleOut {
            newray: Some( Ray::new(recordin.pointW,   next) ),
            f:  res,
            next   ,
            pdf: 1.0 ,
        } 
         
    }
}

 
 
/*

   let bsdf = self.frame.unwrap();
        let wprevlocal = bsdf.to_local(&record.prevW);

        
        let newdirlocal = Vector3::new(
            -wprevlocal .x,
             -wprevlocal .y,
 
             wprevlocal .z,
              
           
         );
         let R = self.albedo.unwrap();
         let cosprev = CosTheta(&newdirlocal) as f32;
         // fresnel->eval(cosprev)
  //        c.abs(), self.etai, self.etat
        let a =  self.fresnel.unwrap().eval(newdirlocal.z);
        let b  = AbsCosTheta(&newdirlocal);
         let fresnel_term = self.fresnel.unwrap().eval(newdirlocal.z) as f32;
         let abscosnext = AbsCosTheta(&newdirlocal) as f32;
        let res =  Srgb ::new (fresnel_term *R.red / abscosnext,fresnel_term *R.green/ abscosnext,fresnel_term *R.blue/ abscosnext);

        let nextworld = bsdf.to_world(&newdirlocal);
       
        let r =  Ray::new(record.pointW, nextworld);
        
        return RecordSampleOut {
            newray: Some(r),
            f:  res,
            next:nextworld,
            pdf: 1.0 ,
        };*/





#[derive(Debug, Clone, Copy)]
pub struct  FresnelTransmissionReflection { 
    pub frame: Option<FrameShading >,
    pub R : Option<Srgb>,
    pub T : Option<Srgb>,
    pub fresnel : Option<Fresnel>,
    pub eta_a : f64,
    pub eta_b : f64,
}
impl  FresnelTransmissionReflection   {
    pub fn new(frame: FrameShading , R : Srgb ,T : Srgb, eta_a : f64 , eta_b : f64) ->   Self   {
        let fresneldielectric_reflection = Fresnel::FresnelDielectricType(FresnelDielectric::new(eta_a, eta_b));
        FresnelTransmissionReflection  {
           frame: Some(frame), 
           R  :Some(R) ,
          T  :Some(T),
            fresnel:Some(fresneldielectric_reflection),
            eta_a,eta_b
        }
    }
    pub fn from_albedo(albedo: Srgb, eta_a : f64 , eta_b : f64) ->   Self   {
        FresnelTransmissionReflection  { 
            frame: None , 
            fresnel:None, eta_a, eta_b,   
            R :Some(Srgb::new(0.0,0.0,0.0)),
            T :Some(Srgb::new(0.0,0.0,0.0)),}
    }
     
     
    fn refract(next : Vector3<f64>, n : Vector3<f64>, eta : f64 )->(bool, Vector3<f64>){
        let costi = next.dot(n);
       let sin2ti = (1.0-costi*costi).max(0.0);
       let sin2tt = eta*eta *sin2ti;
       if sin2tt>=1.0 {
        //criticalpoint
        return (false, Vector3::new(0.0,0.0,0.0))
       }
     let cost =   ( 1.0 - sin2tt ).sqrt();
      let  b =eta *costi - cost;
      let nn = Vector3::new(n.x * b , n.y * b, n.z *b);
      let a  = Vector3::new(-eta * next.x, - eta * next.y,-eta * next.z);
      let r = Vector3::new (a.x + nn.x, a.y + nn.y, a.z + nn.z );
      (true, r )

    }
    pub fn inner_sample(&self, mut recordin: RecordSampleIn) ->RecordSampleOut {
        
        let bsdf = self.frame.unwrap();
        let prev = bsdf.to_local(&recordin.prevW);
        let fresnel_term = self.fresnel.unwrap().eval(prev.z);
        if recordin.sample.unwrap().0 < fresnel_term  { 
            self.inner_sample_reflection(fresnel_term, recordin)
        }else{
            self.inner_sample_transmision(fresnel_term, recordin)
        }







    }
    pub fn inner_sample_reflection(&self, F:f64,  mut recordin: RecordSampleIn)->RecordSampleOut {
        let bsdf = self.frame.unwrap();
        let prev = bsdf.to_local(&recordin.prevW);
        let R = self.R.unwrap();
        let next = Vector3::new(-prev .x,-prev .y,prev .z, );
         let abscosnext = AbsCosTheta(&next) as f32;
         let res =  Srgb ::new (F as f32 *R.red / abscosnext,F as f32*R.green/ abscosnext,F as f32 *R.blue/ abscosnext);
        let nextw = bsdf.to_world(&next);
        RecordSampleOut {newray: Some( Ray::new(recordin.pointW,   nextw) ),f:  res,next  :  nextw  ,pdf :  F} 
    }
    pub fn inner_sample_transmision(&self, F:f64,  mut recordin: RecordSampleIn) ->RecordSampleOut {
        let bsdf = self.frame.unwrap();
        let prev = bsdf.to_local(&recordin.prevW);
        let T = self.T.unwrap();
        let mut eta_t: f64;
        let mut eta_i: f64;
        if prev.z > 0.0 {
            eta_i = self.eta_a;
            eta_t = self.eta_b;
        }else{
            eta_t = self.eta_a;
            eta_i = self.eta_b;
        }
        fn face(n : Vector3<f64>, v : Vector3<f64>)-> Vector3<f64> {
            if n.dot(v) < 0.0{-n }else { n }
        };
        let n  =face(Vector3::new(0.0,0.0,1.0), prev);
        let rs = Self::refract(prev, n, eta_i / eta_t );
        if  !rs.0  {
       //     return RecordSampleOut ::default(); // default initialice the struture with ray = None
        }

       
        let cosprev = CosTheta(&rs.1)  ; 
       let b  = AbsCosTheta(&rs.1) as f32;
   
       let term = (1.0 - F ) as f32;

       let mut res : Srgb; 
       // radiance mode 
       if true {
        let symmetry = (  (eta_i*eta_i) /     (eta_t*eta_t) ) as f32;
         res  = Srgb::new( symmetry * T .red *term / b,symmetry *T .green * term / b,symmetry *T.blue * term / b )
       }else{ // importance mode
          res  = Srgb::new(T.red *term / b,T.green * term / b,T .blue  * term / b )
       }
       let next = bsdf.to_world(&rs.1); 
       RecordSampleOut {
                    newray: Some( Ray::new(recordin.pointW,   next) ),
                    f:  res,
                    next   ,
                    pdf: term as f64,
                } 
                
            }
}




impl 
SampleIllumination  for  FresnelTransmissionReflection 
{
    fn sample(&self, mut record: RecordSampleIn ) -> RecordSampleOut  {
        use crate::primitives::prims::Ray;

      self.inner_sample(record)
         
  
    }
}
impl  Fr for   FresnelTransmissionReflection 
{
    fn fr(&self, prevlocal: Vector3<f64>, nextlocal: Vector3<f64>) -> Srgb {
       Srgb ::new (0.0,0.0,0.0) 
        
    }
}
impl  Pdf for   FresnelTransmissionReflection 
{
    fn pdf(&self, prev: Vector3<f64>, next: Vector3<f64>) -> f64 {
       0.0 
    }
}
     





#[derive(Debug, Clone, Copy)]
pub struct RecordSampleIn  {
    pub prevW: Vector3<f64>,
    pub pointW: Point3<f64>,
    pub sample: Option<(f64, f64)>
    
}

impl RecordSampleIn{
    pub fn from_hitrecord(hit : HitRecord<f64>, r : &Ray<f64>, sample:(f64, f64))->  RecordSampleIn{
        RecordSampleIn{
            pointW: hit.point,
            prevW: -r.direction,
            sample : Some(sample),
        }
    }
    
   
}
impl Default for RecordSampleIn {
    fn default() -> Self {
        RecordSampleIn {
            prevW: Vector3::from_value(0.0),
            pointW: Point3::from_value(0.0),
            sample : None,
        
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RecordSampleOut  {
    pub next: Vector3<f64>,
    pub newray: Option<Ray<f64>>,
    pub f: Srgb,
    pub pdf: f64,
}
impl RecordSampleOut {
    pub fn checkIfZero(&self)->bool{
        self.f.red == 0.0 && self.f.green == 0.0 && self.f.blue == 0.0 && self.pdf == 0.0
    }
    pub fn compute_transport(&self, transportold:Srgb,  hit :&HitRecord<f64>)->Srgb{
        let absdot = hit.normal.dot(self.next).abs();
        let pdf = self.pdf;
        let a = absdot / pdf;
        let fr = self.f;
        let b =  LigtImportanceSampling::mulScalarSrgb(fr, a as f32);
         LigtImportanceSampling::mulSrgb( transportold , b )
    }
    pub fn compute_transport_color(&self, transportold:Colorf64,  hit :&HitRecord<f64>)->Colorf64{
        let absdot = hit.normal.dot(self.next).abs();
        let pdf = self.pdf;
        let a = absdot / pdf;
        let fr = self.f;
       let b =  Colorf64::new(fr.red as f64, fr.green as f64, fr.blue as f64) * a;
        //let b =  LigtImportanceSampling::mulScalarSrgb(fr, a as f32);
        transportold * b 
    }
}
impl  Default for RecordSampleOut  {
    fn default() -> Self {
        RecordSampleOut {
            newray: None,
            next: Vector3::from_value(0.0),
            f: Srgb::new(0.0, 0.0, 0.0),
            pdf: 0.0,
        }
    }
}











pub trait SampleIllumination  {
    fn sample(&self, record: RecordSampleIn  )-> RecordSampleOut ;
    // fn pdf(&self, record: RecordSampleIn  )-> RecordSampleOut ;
}


pub trait Fr :Sized {
    fn fr(&self, prev: Vector3<f64>, next: Vector3<f64>) -> Srgb;
}

pub trait Pdf :Sized {
    fn pdf(&self, prev: Vector3<f64>, next: Vector3<f64>) -> f64;
}
impl  SampleIllumination  for BsdfType 
{
    fn sample(&self, record: RecordSampleIn ) -> RecordSampleOut  {
        
        match *self {
            BsdfType::Lambertian(l) => l.sample(record), // Bsdf ::LambertianBsdf(l)=>l.sample(record)
            BsdfType::MicroFacetReflection(micro)=>{micro.sample(  record ) },
            BsdfType::SpecReflection(s)=>{s.sample(record)},
            BsdfType::LambertianDisneyBsdf(l)=> l.sample(record),
            BsdfType::GlassBsdfType(glass)=>glass.sample(record),
            BsdfType::FresnelTransmissionReflectionType(f)=>f.sample(record),
            BsdfType::None=>panic!("SampleIllumination  for BsdfType")
        }
    }
}

impl  Fr  for BsdfType 
{
    fn fr(&self, prev: Vector3<f64>, next: Vector3<f64>) -> Srgb {
        // if prev.z == 0.0{return Srgb::new(0.0,0.0,0.0)}
     //     if prev.dot(next) < 0.0 {return Srgb::new(1.0,0.0,0.0)}
       
       
        match *self {
            BsdfType::Lambertian(l) => {
                let frame = l.frame.unwrap();
                let reflect = frame.n.dot(prev) * frame.n.dot(next) > 0.0;
                let prev = frame.to_local(&prev).normalize();
                let next  = frame.to_local(&next).normalize();
                if !reflect {return Srgb::new(0.0,0.0,0.0)};

                l.fr( prev, next )
            },
            BsdfType::MicroFacetReflection(m)=>{
                let frame = m.frame.unwrap();
                m.fr( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())

            },
            BsdfType::LambertianDisneyBsdf(l)=>{
                let frame = l.frame.unwrap();
             
                l.fr( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())

            }
            BsdfType::SpecReflection(s)=>{
                let frame = s.frame.unwrap();
                s.fr( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())
            }
            BsdfType::GlassBsdfType(s)=>{
                let frame = s.frame.unwrap();
                s.fr( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())
            }
         
            BsdfType::FresnelTransmissionReflectionType(s)=>{
                let frame = s.frame.unwrap();
                s.fr( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())
            }
            BsdfType::None=>{
                panic!("BsdfType::Fr has throw a method!    ")
            }
         //  _=>Srgb::new(0.0,0.0,0.0)
        }
    }
}



impl  Pdf  for BsdfType 
{
    fn pdf(&self, prev: Vector3<f64>, next: Vector3<f64>) -> f64 {
      //    if prev.z == 0.0{return 0.0}
     //     if prev.dot(next) < 0.0 {return Srgb::new(1.0,0.0,0.0)}
       
       
        match *self {
            BsdfType::Lambertian(l) => {
                let frame = l.frame.unwrap();
                l.pdf( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())
            },
            BsdfType::MicroFacetReflection(m)=>{
                let frame = m.frame.unwrap();
                let prevlocal = frame.to_local(&prev).normalize();
                let nextlocal = frame.to_local(&next).normalize();
                let h = ( prevlocal + nextlocal ) .normalize();
            
                m.distri.pdf(&prevlocal, &h)/ (4.0 * prevlocal.dot(h))
             

            },
            BsdfType::GlassBsdfType(m)=>{
                let frame = m.frame.unwrap();
                let prevlocal = frame.to_local(&prev).normalize();
                let nextlocal = frame.to_local(&next).normalize();
           
                m.pdf( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())
               
             

            },
            BsdfType::FresnelTransmissionReflectionType(m)=>{
                panic!("pdf!"); 
            },
            BsdfType::LambertianDisneyBsdf(l)=>{
                panic!("pdf!");
                // let frame = l.frame.unwrap();
             
                // l.fr( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())

            }
            BsdfType::SpecReflection(s)=>{
                panic!("pdf!");
                // let frame = s.frame.unwrap();
                // s.fr( frame.to_local(&prev).normalize(),  frame.to_local(&next).normalize())
            }
            BsdfType::None=>{
                panic!("BsdfType::Fr has throw a method!    ")
            }
         //  _=>Srgb::new(0.0,0.0,0.0)
        }
    }
}

impl 
SampleIllumination  for LambertianBsdf 
{
    fn sample(&self, mut record: RecordSampleIn ) -> RecordSampleOut  {
        use crate::primitives::prims::Ray;

        //TODO: hay que hacerlo sobre el cos de la normal
        //  let mut rng = rand::thread_rng();
 

        let bsdf = self.frame.unwrap();
        let wprevlocal = bsdf.to_local(&record.prevW); 
        
        
        let mut newdirlocal =   CosSampleH2(&record.sample.unwrap());
       
        if   wprevlocal.z <  0.0  { newdirlocal.z*=-1.0; }
        let f = self.fr(wprevlocal, newdirlocal);

        let pdf =  self.internal_pdf(wprevlocal, newdirlocal);
        let nextworld = bsdf.to_world(&newdirlocal);
        let samesemisphere = wprevlocal.z * newdirlocal.z > 0.0;

        return RecordSampleOut {
            newray: Some(Ray::new(record.pointW, nextworld)),
            f ,
            next: nextworld,
            pdf
        };
    }
}
impl  Fr for LambertianBsdf 
{
    fn fr(&self, prevlocal: Vector3<f64>, nextlocal: Vector3<f64>) -> Srgb {
            if prevlocal.z == 0.0{return Srgb::new(0.0,0.0,0.0)}

          
            
            Srgb::new( self.albedo.red * f32::consts::FRAC_1_PI, self.albedo.green * f32::consts::FRAC_1_PI,self.albedo.blue * f32::consts::FRAC_1_PI)
    }
}



impl Pdf for LambertianBsdf {
   fn pdf(&self, prev: Vector3<f64>, next: Vector3<f64>) -> f64 {
        //  if prev.z == 0.0{return 0.0}
        //  if prev.dot(next) < 0.0 {return 0.0}
       if prev.z * next.z > 0.0 {
        next.abscostheta() * f64::consts::FRAC_1_PI
       }else{0.0}
       
   }
}

pub
trait MaterialDescriptor{
     fn instantiate(  &self, p:&Point3<f64>, uv : &(f64, f64),  frame:&FrameShading)->BsdfType ;

} 
 
#[derive(Debug, Clone)]
pub enum MaterialDescType{
    PlasticType(Plastic),
    MetaType(Metal),
    MirrorType(Mirror),
    GlassType(Glass),
    Glass2Type(Glass2),
    NoneType,
}
impl MaterialDescriptor for MaterialDescType{
    fn instantiate(&self, p:&Point3<f64>,uv : &(f64, f64),frame:&FrameShading) ->BsdfType {
        match self{
            MaterialDescType::PlasticType(  plastic)=>{
                plastic.instantiate(p,uv, frame)
             },
             MaterialDescType::MetaType(m)=>{
                m.instantiate(p,uv, frame)
             },
             MaterialDescType::MirrorType(mirror)=>{
                mirror.instantiate(p, uv, frame)
             }
             MaterialDescType::GlassType(glass)=>{
                glass.instantiate(p, uv, frame)
             }
             MaterialDescType::Glass2Type(glass2)=>{
                glass2.instantiate(p, uv, frame)
             }
             MaterialDescType::NoneType =>{
                BsdfType::None
             }
             _ => {panic!("Material Descriptor can`t be none")},
        }
    }
}
 


#[derive(Debug, Clone )]
pub struct Plastic{
 pub   R : Texture2DSRgbMapConstant,
 pub   ft : Texture2DSRgbMapConstant,
 pub dumb : Option<TexType>,
 pub Rimg :Option<TexType>
}


impl Plastic{
    pub fn from_albedo(r : f64, g : f64 , b :f64)->Plastic{
        Plastic{
            
            R : Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(1.2, 1.1, 1.0)),
            ft :Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),
            dumb:Some(TexType::Texture2DSRgbMapConstant(Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(r as f32, g as f32, b as f32)))),
            Rimg:Some(TexType::Texture2DSRgbMapUV(Texture2D::<Srgb,MapUV>:: createTexture2DSRgbMapUVImg())),
           
        }
    }
}
impl  Default for Plastic {
    fn default() -> Self {
        Plastic{
            R : Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 1.1, 1.0)),
            ft :Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)),
            dumb:Some(TexType::Texture2DSRgbMapConstant(Texture2D::<Srgb,MapConstant>::createConstantText(&Srgb::new(2.1, 0.1, 1.0)))),
            Rimg:Some(TexType::Texture2DSRgbMapUV(Texture2D::<Srgb,MapUV>::createTexture2DSRgbMapUV())),
        }
      
    }
}
impl MaterialDescriptor for Plastic{
      fn instantiate(  &self,intersectionRecord:&Point3<f64>,uv : &(f64, f64),frame:&FrameShading) ->BsdfType {
       
        let albedo  = self.dumb.as_ref().unwrap().eval(&Point3::new(uv.0,uv.1,0.0));
   
       //  BsdfType::LambertianDisneyBsdf(LambertianDisneyBsdf{albedo,frame:Some(*frame)})
     BsdfType::Lambertian(LambertianBsdf{albedo,frame:Some(*frame)})
         
    }
}














#[derive(Debug, Clone )]
pub struct Metal{
 pub   roughtu: Option<TexType>,
 pub   roughtv: Option<TexType>,
 pub   eta: Option<TexType>,
 pub k :Option<TexType>
}

impl   Metal {
    pub fn from_constant_rought(rx:f64, ry:f64)->Self{
        Metal{
            // cuanto mas cerca al 0 esta mas brilla. como es una bsdf solo no se puede combinar lo todo con un lambert y demas...!
            roughtu:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&rx))),
            roughtv:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&ry))),
            //  eta:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&0.01))),
            //  k:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&0.01))),
             eta:Some(TexType::None),
             k:Some(TexType::None),
        }
    }
}
impl  Default for Metal {
    fn default() -> Self {
        Metal{
            // cuanto mas cerca al 0 esta mas brilla. como es una bsdf solo no se puede combinar lo todo con un lambert y demas...!
            roughtu:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&0.1001))),
            roughtv:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&0.1001))),
            //  eta:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&0.01))),
            //  k:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&0.01))),
             eta:Some(TexType::None),
             k:Some(TexType::None),
        }
      
    }
}
impl MaterialDescriptor for Metal{
    fn instantiate(  &self,intersectionRecord:&Point3<f64>,uv : &(f64, f64),frame:&FrameShading) ->BsdfType {
     
        let rougx= self.roughtu.as_ref().unwrap().eval(intersectionRecord).red;
        let rougy= self.roughtv.as_ref().unwrap().eval(intersectionRecord).red;
      //   let fresnel = Fresnel::FresnelCondutor(FresnelConductor::new(1.0, self.eta.as_ref().unwrap().eval(intersectionRecord).red as f64, self.k.as_ref().unwrap().eval(intersectionRecord).red as f64));
        let micro_trosky = BsdfType::MicroFacetReflection(MicroFacetReflection{
            frame:Some(*frame),
            distri : MicroFaceDistribution::TrowbridgeReitzDistribution(TrowbridgeReitzDistribution::new(rougx as f64, rougy as f64)),
             fresnel:Fresnel::FresnelNop(FresnelNop::default())
         //   fresnel
        });
        return  micro_trosky;
  }
}











#[derive(Debug, Clone )]
pub struct Mirror{
 pub   R: Option<TexType>,
  
}

impl  Default for Mirror {
    fn default() -> Self {
        Mirror{
            R:Some(TexType::Texture2Df64MapConstant(Texture2D::<f64,MapConstant>::createTexture2Df64MapConstant(&1.0))),
            
        }
      
    }
}
impl MaterialDescriptor for Mirror{
    fn instantiate(  &self,intersectionRecord:&Point3<f64>,uv : &(f64, f64),frame:&FrameShading) ->BsdfType {
     
       let R =  self.R.as_ref().unwrap().eval(intersectionRecord) ;
       let brdf =  SpecReflection{
        frame:Some(*frame),
        albedo : Some(R),
        fresnel:None,
       };
 
        
       BsdfType::SpecReflection(brdf)
  }
}





#[derive(Debug, Clone,Copy )]
pub struct GlassBsdf{
    pub frame: Option< FrameShading >,
    pub specular_reflection : Option<SpecReflection>,
    pub specular_transmision : Option<SpecTransmission>,
}
impl GlassBsdf{
    pub fn new()->Self{
        GlassBsdf {
            frame : None,
            specular_reflection : None,  
            specular_transmision  :  None, 
        }
    }
    pub fn from_frame(frame: FrameShading, R : Srgb, T : Srgb, eta : f64)->Self{
       let fresneldielectric_reflection = Fresnel::FresnelDielectricType(FresnelDielectric::new(1.0, eta));
       let fresneldielectric_transmission = Fresnel::FresnelDielectricType(FresnelDielectric::new(1.0, eta));
        GlassBsdf {
            frame : Some(frame),
            specular_reflection : Some( SpecReflection{fresnel:Some(fresneldielectric_reflection),frame:Some(frame), albedo:Some(R)}), 
            specular_transmision  :Some( SpecTransmission{eta_a:1.0, eta_b:eta,fresnel:Some(fresneldielectric_transmission),frame:Some(frame), albedo:Some(T)}),  }
    }
}

impl SampleIllumination for GlassBsdf{
     fn sample(&self, recordin: RecordSampleIn) -> RecordSampleOut {
        let ncomponents  =2 as f64;
        let psample =  recordin.sample.unwrap();
        let  selectedcomponent = (psample.0*ncomponents).floor().min(ncomponents -1.0) ;
        let remapsample0 =( psample.0 * ncomponents - selectedcomponent ).min(ONE_MINUS_EPSILON_f64);
        let  mut recordout ; 
        let renewrecord =  RecordSampleIn{pointW:recordin.pointW, prevW: recordin.prevW,sample:Some((remapsample0, psample.1))};
        if selectedcomponent as i32 == 0 {

          recordout=  self. specular_reflection .unwrap().inner_sample(renewrecord);
        }else{
            recordout= self. specular_transmision .unwrap().inner_sample(renewrecord);
        }
        recordout.pdf /= ncomponents;
        recordout
        
     }
 }
 impl Pdf for GlassBsdf{
    fn pdf(&self, prev: Vector3<f64>, next: Vector3<f64>) -> f64 {
        0.0
    }
 }
 impl Fr for GlassBsdf{
     fn fr(&self, prev: Vector3<f64>, next: Vector3<f64>) -> Srgb {
         Srgb::new(0.0,0.0,0.0)
     }
 }


#[derive(Debug, Clone,Copy )]
pub struct Glass{
    pub   R : f64,
    pub   T: f64,
    pub   bsdf :  GlassBsdf, 
}


impl  Default for Glass {
    fn default() -> Self {
        Glass{
            R:1.0,
            T:1.0,
            bsdf :  GlassBsdf::new(),
        }
      
    }
}

impl MaterialDescriptor for Glass{
    fn instantiate(  &self,intersectionRecord:&Point3<f64>,uv : &(f64, f64),frame:&FrameShading) ->BsdfType {
       
        
       let  R = Srgb::new(0.1,0.1,0.1);
        let T = Srgb::new(1.0,1.0,1.0);
         BsdfType::GlassBsdfType( GlassBsdf::from_frame(*frame, R, T,1.5)) 
  }
}






#[derive(Debug, Clone,Copy )]
pub struct Glass2{
    pub   R : f64,
    pub   T: f64,
    
}
impl Glass2 {
    pub fn new()->Glass2{
        Glass2{R:1.0, T : 1.0}
    }
}

 

impl MaterialDescriptor for Glass2{
    fn instantiate(  &self,intersectionRecord:&Point3<f64>,uv : &(f64, f64),frame:&FrameShading) ->BsdfType {
       
        
       let  R = Srgb::new(1.0,1.0,1.0);
        let T = Srgb::new(1.0,1.0,1.0);
         BsdfType::FresnelTransmissionReflectionType(  FresnelTransmissionReflection::new(*frame, R, T , 1.0, 1.5)) 
  }
}

