#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(non_snake_case)]
#![allow(unused_mut)]
#![allow(unused_assignments)]
#![allow(unused_imports)]
#![allow(unreachable_code)]
#![allow(unused_parens)]
use std::{process::Output, rc::Rc, sync::mpsc::Receiver};

use image::{io::Reader as ImageReader, GenericImageView, Pixel};
use std::io::Cursor;
use std::path::Path;

use crate::{
    assert_delta,
    integrator::{updateL, estimatelights, new_interface_estimatelights, isBlack},
    materials::{
        AbsCosTheta, BeckmannDistribution, BsdfType, Fr, FrameShading, Fresnel, FresnelNop,
        MaterialDescType, MicroFaceDistribution, MicroFacetReflection, Pdf, Plastic,
        RecordSampleIn, SampleIllumination, SphPhi, SphTheta, SphericalCoords,
        TrowbridgeReitzDistribution, UniformSampleConePdf, UniformSampleSemisphere, UniformSampleSphere, UniformSampleSpherePdf, CosSampleH2, CosSampleH2Pdf,
    },
    primitives::{
        prims::{self, HitRecord, IntersectionOclussion, Ray, Sphere, coord_system},
        Disk, Plane, PrimitiveIntersection, PrimitiveType, Cylinder,
    },
    raytracerv2::{clamp, powerHeuristic, Camera},
    raytracerv2::{interset_scene, Scene},
    sampler::{Sampler, custom_rng::CustomRng, SamplerUniform, SamplerType},
    Metal, Vector3f, Point3f, scene::Scene1, threapoolexamples1::{new_interface_for_intersect_scene, new_interface_for_intersect_occlusion, new_interface_for_intersect_scene_surfaces}, Spheref64,
};
use cgmath::{Decomposed, Deg, InnerSpace, Point3, Quaternion, Transform, Vector3, MetricSpace, Matrix3, AbsDiffEq};
use num_traits::Float;
use palette::{Srgb, white_point::A, LinSrgb};
use rand::Rng;

#[derive(Clone, Debug)]
pub struct Distribution1d {
    pub data: Vec<f64>,
    pub cdf: Vec<f64>,
    pub norm: f64,
}
impl Distribution1d {
    pub fn new(a: &Vec<f64>) -> Distribution1d {
        let mut cdf = vec![0.0; a.len() + 1];

        cdf[0] = 0.0;
        for (i, x) in a.iter().enumerate() {
            if i == 0 {continue;}
            cdf[i] = cdf[i - 1] + a[i - 1] / (a.len() as f64);
        }
        cdf[a.len()] = cdf[a.len() - 1] + a[a.len() - 1] / (a.len() as f64);
        let norm = cdf[a.len()];
        for x in cdf.iter_mut() {
            let f = (*x / norm);
            *x = f;
        }
        //    println!("{:?}",cdf);
        Distribution1d {
            cdf,
            data: a.clone(),
            norm,
        }
    }
    pub fn find(&self, u: f64) -> i32 {
        self.cdf.iter().position(|x| !(x <= &u)).unwrap() as i32 - 1
    }
    // pos , offset, pdf
    pub fn sample(&self, u: f64) -> (i32, f64, f64) {
        let off = self.find(u);
        let mut du = u - self.cdf[off as usize];

        // si crece
        let difinter = self.cdf[(off + 1) as usize] - self.cdf[off as usize];
        if difinter > 0.0 {
            du /= difinter;
        }
        let pdf = self.data[off as usize] as f64 / self.norm;

        (off, ((off as f64) + du) / self.data.len() as f64, pdf)
    }
}

/**
         *
Initializing a vector in a struct
         https://users.rust-lang.org/t/initializing-a-vector-in-a-struct/5442/2
         */

#[derive(Clone, Debug)]
pub struct Distribution2d {
    pub vcondicional: Vec<Distribution1d>,
    pub vmarginal: Distribution1d,
    // vmarginal :  Distribution1d ;
}
impl Distribution2d {
    pub fn init_condicional(
        vcondicional: &mut Vec<Distribution1d>,
        lumdata: &Vec<f64>,
        w: usize,
        h: usize,
    ) {
        for i in 0..h {
            vcondicional.push(Distribution1d::new(&Vec::from_iter(
                lumdata[w * i..w * (i + 1)].iter().cloned(),
            )));
        }
    }
    pub fn init_marginal(vcondicional: &mut Vec<Distribution1d>) -> Vec<f64> {
        vcondicional.into_iter().map(|c| c.norm).collect()
    }
    pub fn new(lumdata: &Vec<f64>, w: usize, h: usize) -> Distribution2d {
        let mut vcondicional: Vec<Distribution1d> = vec![];
        Self::init_condicional(&mut vcondicional, lumdata, w, h);
        let marginal = Self::init_marginal(&mut vcondicional);
        // let r = Vec::from_iter(lumdata[w*i..w*(i+1)].iter().cloned());
        // Distribution1d::new(&Vec::from_iter(lumdata[w*i..w*(i+1)].iter().cloned()))
        // println!("{:?}", r);

        Distribution2d {
            vmarginal: Distribution1d::new(&marginal),
            vcondicional: vcondicional,
        }
    }
    // offset cond ,offset mar , pdf = pdfcond * pdfmar
    pub fn sample(&self, u: &(f64, f64)) -> (f64, f64, f64) {
        let samplemarginal = self.vmarginal.sample(u.1);
        // println!("{:?}", samplemarginal);
        let condi = &self.vcondicional[samplemarginal.0 as usize];
        let sampleconditional = condi.sample(u.0);
        let pdf = samplemarginal.2 * sampleconditional.2;
        (sampleconditional.1, samplemarginal.1, pdf)
    }
    pub fn get_pdf(&self, u: &(f64, f64)) -> f64 {
        // select column
        let conditionalEntry = (self.vcondicional[0].data.len() as f64 * u.0) as usize;
        let conditionalEntry = clamp(conditionalEntry, 0, self.vcondicional[0].data.len() - 1);
        let marginalEntry = (self.vmarginal.data.len() as f64 * u.1) as usize;
        let marginalEntry = clamp(marginalEntry, 0, self.vmarginal.data.len() - 1);

        self.vcondicional[marginalEntry].data[conditionalEntry] / self.vmarginal.norm
    }
}

#[test]
pub fn distribution_test() {
    let v: Vec<f64> = vec![1.0; 10];
    let vf64: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let dis = Distribution1d::new(&vf64);
    for i in 0..32 {
        let fi = i as f64 / 32.0;
        let samplef = dis.sample(fi);
        println!(
            "i {} buck {} pdf {} offset {}",
            i, samplef.0, samplef.2, samplef.1
        );
        // println!("i {}, fi {}, offset {}", i, fi, offset);
        //   mas pruebas ...pero creo que va bien...objetivo: la envlight
    }
}
#[test]
pub fn distribution2d_test() {
    let v: Vec<f64> = vec![1.0; 10];
    let vf64: Vec<f64> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0, 9.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    ];
    let vp = vec![1.0; 20];
    let dis2d = Distribution2d::new(&vf64, 10, 5);
    for iy in 0..32 {
        for ix in 0..32 {
            let fy = iy as f64 / 32.0;
            let fx = ix as f64 / 32.0;
            //    let iy = 0;
            //    let ix = 1;
            let i = iy * 32 + ix;
            // let sample2d =  dis2d.sample(&(fx, fy));
            // println!(" ix,iy: {},{},  off cond  {}, off  marg {}, pdf(A|B) : {} ",ix, iy, sample2d.0,sample2d.1,sample2d.2);
            //    dis2d.get_pdf(&(fx, fy));
            println!(" ix,iy: {},{}, pdf {}", ix, iy, dis2d.get_pdf(&(fx, fy)));
        }
    }

    println!("");
}









#[derive(Debug, Clone, Copy)]
pub struct RecordSampleLightEmissionIn  {
    pub psample0:(f64, f64),
    pub psample1:(f64, f64),
}
impl RecordSampleLightEmissionIn{
    pub fn from_sample(  sampler: &mut Box<dyn Sampler>) ->Self{
        let   u0 = sampler.get2d();
        let   u1 = sampler.get2d();
 
        RecordSampleLightEmissionIn{
            psample0: u0,
            psample1:u1
        }
    }
    pub fn from_sampletype(  sampler: &mut SamplerType) ->Self{
        let   u0 = sampler.get2d();
        let   u1 = sampler.get2d();
 
        RecordSampleLightEmissionIn{
            psample0: u0,
            psample1:u1
        }
    }
    pub fn from_sample2dx2d(  u0 :(f64, f64) , u1:(f64, f64)) ->Self{
     
 
        RecordSampleLightEmissionIn{
            psample0: u0,
            psample1:u1
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RecordSampleLightEmissionOut  {
    pub Lemission : Srgb,
    pub  pdfpos : f64, 
    pub pdfdir : f64, 
    pub  n : Vector3f,
    pub  ray : Option<Ray<f64>>
}


impl RecordSampleLightEmissionOut{
    pub fn from_hitrecord(  Lemission : Srgb,pdfpos : f64, pdfdir : f64,n : Vector3f,ray : Option<Ray<f64>>) ->Self{
        RecordSampleLightEmissionOut{Lemission, pdfpos, pdfdir, n, ray}
    }
}

pub trait SampleEmission  {
    fn sample_emission(&self, record: RecordSampleLightEmissionIn  )-> RecordSampleLightEmissionOut  ;
    
} 










#[derive(Debug, Clone, Copy)]
pub struct RecordPdfEmissionIn  {
 
    pub  n : Vector3f,
    pub  ray : Option<Ray<f64>>
}


impl  RecordPdfEmissionIn{
    pub fn from_hitrecord(   n : Vector3f,ray : Option<Ray<f64>>) ->Self{
        RecordPdfEmissionIn{  n, ray}
    }
}


#[derive(Debug, Clone, Copy)]
pub struct RecordPdfEmissionOut  {
 
    pub  pdfpos : f64,
    pub  pdfdir :  f64,
}


impl  RecordPdfEmissionOut {
   pub fn new( pdfpos : f64 ,  pdfdir:f64 )->Self{
    RecordPdfEmissionOut { pdfpos, pdfdir }

   }
}
impl Default for RecordPdfEmissionOut {
    fn default() -> Self {
        RecordPdfEmissionOut { pdfpos:0.0, pdfdir:0.0 }
    }
}
pub trait PdfEmission  {
    fn pdf_emission(&self , rec : RecordPdfEmissionIn )-> RecordPdfEmissionOut ;
    
}

















pub struct RecordSampleLightIlumnation {
    psample: (f64, f64),
    hit: HitRecord<f64>,
}

impl RecordSampleLightIlumnation {
    pub fn new(psample: (f64, f64), hit: HitRecord<f64>) -> RecordSampleLightIlumnation {
        RecordSampleLightIlumnation { psample, hit }
    }
    pub fn from_hit(hit: HitRecord<f64>) -> RecordSampleLightIlumnation {
        RecordSampleLightIlumnation {
            psample: (0.0, 0.0),
            hit,
        }
    }
}
pub trait SampleLightIllumination: IsAreaLight {
    // vnext, ilumination pdflight, ray occlusion
    fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    );
    fn getPositionws(&self) -> Point3<f64>;
}

pub trait NumSamples {
    fn num_samples(&self) -> u32;
}
pub trait IsAreaLight {
    fn is_arealight(&self) -> bool;
}
pub trait IsAmbientLight {
    fn is_ambientlight(&self) -> bool;
}


pub trait IsBackgroundAreaLight {
    fn is_background_area_light(&self) -> bool;
}

pub trait GetShape {
    fn get_shape(&self) -> Option<Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>>;
}

pub trait GetEmission {
    fn get_emission(&self, hit: Option<HitRecord<f64>>, r: &Ray<f64>) -> Srgb;
}

pub trait PdfIllumination {
    fn pdfIllumination(&self, hit: &HitRecord<f64>, scene: &Scene<f64>, next: Vector3<f64>) -> f64;
}
 
#[derive(Clone, Debug)]
pub enum Light {
    PointLight(PointLight<f64>),
    SpotLightType(SpotLight),
    AreaLightType(AreaLight),
    AreaLightDiskType(AreaLightDisk),
    AreaLightSphere(AreaLightSphere),
    BackgroundAreaLightType(BackgroundAreaLight),
    AmbientLightType(AmbientLight),
    New_Interface_AreaLightType(New_Interface_AreaLight),
    NoneLightType,
}


impl NumSamples for Light {
    fn num_samples(&self) -> u32 {
        match self {
            Light::New_Interface_AreaLightType(a)=>a.num_samples(),
            _ => panic!("Light:: not yet? implented area li"),
        }
        // Light::PointLight(l) => l.is_arealight(),
        // Light::SpotLightType(a) => false,
        // Light::AreaLightType(a) => a.is_arealight(),
        // Light::AreaLightDiskType(a) => a.is_arealight(),
        // Light::BackgroundAreaLightType(a) => a.is_arealight(),
        // Light::AmbientLightType(a)=>a.inner_is_arealight(),
        // Light::AreaLightSphere(a)=>true,
     
    } 
}
impl IsAreaLight for Light {
    fn is_arealight(&self) -> bool {
        match &*self {
            Light::PointLight(l) => l.is_arealight(),
            Light::SpotLightType(a) => false,
            Light::AreaLightType(a) => a.is_arealight(),
            Light::AreaLightDiskType(a) => a.is_arealight(),
            Light::BackgroundAreaLightType(a) => a.is_arealight(),
            Light::AmbientLightType(a)=>a.inner_is_arealight(),
            Light::AreaLightSphere(a)=>true,
            Light::New_Interface_AreaLightType(a)=>a.is_arealight(),
            _ => panic!("Light:: not yet? implented area li"),
        }
    }
}
impl IsAmbientLight for   Light {
    fn is_ambientlight(&self) -> bool {
        match &*self {
            Light::PointLight(l) => false,
            Light::SpotLightType(a) => false,
            Light::AreaLightType(a) => false,
            Light::AreaLightDiskType(a) => false,
            Light::BackgroundAreaLightType(a) => false,
            Light::AmbientLightType(a)=>true,
            Light::AreaLightSphere(a)=>false,
            Light::New_Interface_AreaLightType(a)=>false,
            _ => panic!("Light:: not yet?  "),
        }
    }
}

impl IsBackgroundAreaLight for Light {
    fn is_background_area_light(&self) -> bool {
        match &*self {
            Light::PointLight(l) => false,
            Light::SpotLightType(a) => false,
            Light::AreaLightType(a) => false,
            Light::AreaLightDiskType(a) => false,
            Light::BackgroundAreaLightType(a) => true,
            Light::AmbientLightType(a)=>false,
            Light::AreaLightSphere(a)=>false,
            Light::New_Interface_AreaLightType(a)=>false,
            _ => panic!("Light:: not yet?  "),
        }
    }
}

impl GetShape for Light {
    fn get_shape(&self) -> Option<Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> {
        match &*self {
            Light::PointLight(l) => panic!("point lihgt is  not a area light"),
            Light::SpotLightType(l) => panic!("spot lihgt is  not a area light"),
            Light::AreaLightType(a) => a.get_shape(),
            Light::AreaLightDiskType(a) => a.get_shape(),
            Light::BackgroundAreaLightType(a) => a.get_shape(),
            Light::AmbientLightType(a)=>  panic!("spot lihgt is  not a area light"),
            Light::AreaLightSphere(a)=>a.get_shape(),
            Light::New_Interface_AreaLightType(a)=>a.get_shape(),
            _ => panic!("Light::sampleIllumination"),
 
        }
    }
}
impl GetShape1 for Light {
    fn get_shape1(&self) ->&PrimitiveType{
        match  self {
            // Light::PointLight(l) => panic!("point lihgt is  not a area light"),
            // Light::SpotLightType(l) => panic!("spot lihgt is  not a area light"),
            // Light::AreaLightType(a) => a.get_shape(),
            // Light::AreaLightDiskType(a) => a.get_shape(),
            // Light::BackgroundAreaLightType(a) => a.get_shape(),
            // Light::AmbientLightType(a)=>  panic!("spot lihgt is  not a area light"),
            // Light::AreaLightSphere(a)=>a.get_shape(),
            Light::New_Interface_AreaLightType(a)=>a.get_shape1(),
            _ => panic!("Light::sampleIllumination"),
 
        }
    }
}


impl PdfIllumination1 for Light {
    fn pdfIllumination1(&self, hit: &HitRecord<f64>, scene: &Scene1, next: Vector3<f64>) -> f64 {
        match &*self {
            // Light::PointLight(l) => panic!("point light is  not a area light"),
            // Light::SpotLightType(l) => panic!("spot oint light is  not a area light"),
            // Light::AreaLightType(a) => a.pdfIllumination(hit, scene, next),
            // Light::AreaLightDiskType(a) => a.pdfIllumination(hit, scene, next),
            // Light::BackgroundAreaLightType(a) => a.pdfIllumination(hit, scene, next),
            // Light::AmbientLightType(a)=>  panic!("Ambient light is  not a area light pdf.  "),
            // Light::AreaLightSphere(a)=>a.pdfIllumination(hit, scene, next),
            Light::New_Interface_AreaLightType(a)=>a.pdfIllumination1(hit, scene, next),
            _ => panic!("Light:: not yet?  "),
        }
    }
}

 
impl GetEmission for Light {
    fn get_emission(&self, hit: Option<HitRecord<f64>>, r: &Ray<f64>) -> Srgb {
        match &*self {
            Light::PointLight(l) => panic!("point lihgt is  not a area light"),
            Light::SpotLightType(l) => panic!("spot lihgt is  not a area light "),
            Light::AreaLightType(a) => a.get_emission(hit, r),
            Light::AreaLightDiskType(a) => a.get_emission(hit, r),
            Light::BackgroundAreaLightType(a) => a.get_emission(hit, r),
            Light::AmbientLightType(a)=>  panic!("Ambient lihgt is  not a area light, lacks  emission"),
            Light::AreaLightSphere(a)=>a.get_emission(hit,r),
            Light::New_Interface_AreaLightType(a)=>a.get_emission(hit, r),
            _ => panic!("Light::sampleIllumination"),
        }
    }
}

impl PdfIllumination for Light {
    fn pdfIllumination(&self, hit: &HitRecord<f64>, scene: &Scene<f64>, next: Vector3<f64>) -> f64 {
        match &*self {
            Light::PointLight(l) => panic!("point light is  not a area light"),
            Light::SpotLightType(l) => panic!("spot oint light is  not a area light"),
            Light::AreaLightType(a) => a.pdfIllumination(hit, scene, next),
            Light::AreaLightDiskType(a) => a.pdfIllumination(hit, scene, next),
            Light::BackgroundAreaLightType(a) => a.pdfIllumination(hit, scene, next),
            Light::AmbientLightType(a)=>  panic!("Ambient light is  not a area light pdf.  "),
            Light::AreaLightSphere(a)=>a.pdfIllumination(hit, scene, next),
            // Light::New_Interface_AreaLightType(a)=>a.pdfIllumination(hit, scene, next),
            _ => panic!("Light:: not yet?  "),
        }
    }
}

impl SampleLightIllumination for Light {
    fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        match &*self {
            Light::PointLight(l) => l.sampleIllumination(phit),
            Light::SpotLightType(l) => l.sampleIllumination(phit),
            Light::AreaLightType(a) => a.sampleIllumination(phit),
            Light::AreaLightDiskType(a) => a.sampleIllumination(phit),
            Light::BackgroundAreaLightType(a) => a.sampleIllumination(phit),
            Light::AmbientLightType(a)=>  a.inner_sampleIllumination(phit),
            Light::AreaLightSphere(a)=>a.sampleIllumination(phit),
            Light::New_Interface_AreaLightType(a)=>a.sampleIllumination(phit),
            _ => panic!("Light:: not yet?  "),
        }
    }
    fn getPositionws(&self) -> Point3<f64> {
        match &*self {
            Light::PointLight(l) => l.getPositionws(),
            Light::SpotLightType(l) => l.getPositionws(),
            Light::AreaLightType(a) => panic!(""),
            Light::AreaLightDiskType(a) => panic!(""),
            Light::BackgroundAreaLightType(a) => a.getPositionws(),
            Light::AmbientLightType(a)=>  panic!(""),
            Light::AreaLightSphere(a)=>panic!(""),
            Light::New_Interface_AreaLightType(a)=>panic!("Light::New_Interface_AreaLightType getPositionws "),
            _ => panic!("Light:: not yet?  "),
        }
    }
}








impl SampleEmission for Light {
    fn sample_emission(&self, record: RecordSampleLightEmissionIn) -> RecordSampleLightEmissionOut {
        match self {
            Light::PointLight(p)=>p.sample_emission(record),
            Light::AreaLightDiskType(p)=>p.sample_emission(record),
            Light::New_Interface_AreaLightType(p)=>p.sample_emission(record),
            _=>todo!()
        }
    }
}



impl PdfEmission for Light {
    fn pdf_emission(&self , rec :  RecordPdfEmissionIn) -> RecordPdfEmissionOut {
        match self {
                   Light::PointLight(p)=>p.pdf_emission(rec),
                   Light::AreaLightDiskType(p)=>p.pdf_emission(rec),
                   Light::New_Interface_AreaLightType(p)=>p.pdf_emission(rec),
                    _=>todo!()
        }
    }
    // fn sample_emission(&self, record: RecordSampleLightEmissionIn) -> RecordSampleLightEmissionOut {
    //     match &*self {
    //         Light::PointLight(p)=>p.sample_emission(record),
    //         _=>todo!()
    //     }
    // }
}

























#[derive(Debug, Clone, Copy)]
pub struct AmbientLight{
    pub world: Decomposed<Vector3<f64>, Quaternion<f64>>,
    pub local: Decomposed<Vector3<f64>, Quaternion<f64>>,
    pub color: Srgb,
  

}
impl AmbientLight{
    pub fn new(direction :Vector3<f64>,  radiance : Srgb )->Self{
       
        let t1 = Decomposed {
            disp: Vector3::new(0.0,0.0,0.0),
            rot: Quaternion::from_arc(Vector3::new(0.0, 0.0, 1.0), direction, None),
            scale: 1.0,
        };
        
        AmbientLight{
            world: t1,
            local: t1.inverse_transform().unwrap(),
            color:radiance,
            
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
        pub fn getPositionws(&self) -> Point3<f64> {
            Point3::new(0.0,0.0,0.0)
        }
        pub fn inner_sampleIllumination(
            &self,
            phit: &RecordSampleLightIlumnation,
        ) ->  (
            Vector3<f64>,
            Srgb,
            f64,
            Option<Ray<f64>>,
            Option<Point3<f64>>,
        ) {

            // guardamos la rotation . no la direccion. entonces neceistamos hacer la rotatcion. por ser consistentes...
            let direction = self.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0));
            let occlusionRay = Ray::new(phit.hit.point, direction);
         
            (direction.normalize(), self.color, 1.0,Some(occlusionRay), Some(   occlusionRay.at(1000.0)))

        }
        
        pub fn inner_is_arealight(&self) -> bool {
            false
        }
}
#[derive(Debug, Clone, Copy)]
pub struct SpotLight {
    pub world: Decomposed<Vector3<f64>, Quaternion<f64>>,
    pub local: Decomposed<Vector3<f64>, Quaternion<f64>>,
    pub color: Srgb,
    pub positionws: Point3<f64>,
    pub cosmin: f64,
    pub cosmax: f64,
}
impl SpotLight {
    pub fn from_light(
        translation: Vector3<f64>,
        direction: Vector3<f64>,
        lemission: Srgb,
        cosmindegree: f64,
        difdegree: f64,
    ) -> Light {
        Light::SpotLightType(SpotLight::new(
            translation,
            direction,
            lemission,
            cosmindegree,
            difdegree,
        ))
    }
    pub fn new(
        translation: Vector3<f64>,
        direction: Vector3<f64>,
        lemission: Srgb,
        cosmindegree: f64,
        difdegree: f64,
    ) -> Self {
        let t1 = Decomposed {
            disp: translation,
            rot: Quaternion::from_arc(Vector3::new(0.0, 0.0, 1.0), direction, None),
            scale: 1.0,
        };
        SpotLight {
            world: t1,
            local: t1.inverse_transform().unwrap(),
            positionws: t1.transform_point(Point3::new(0.0, 0.0, 0.0)),
            color: lemission,
            cosmax: (cosmindegree - difdegree).to_radians().cos(),
            cosmin: cosmindegree.to_radians().cos(),
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

    //v is normal
    pub fn fallout(&self, v: Vector3<f64>) -> f64 {
        let localspace = self.trafo_vec_to_local(v);
        // println!("{:?}", localspace);
        let costheta = localspace.z;
        if costheta < self.cosmin {
            return 0.0;
        } else if costheta > self.cosmax {
            return 1.0;
        } else {
            let costheta = (costheta - self.cosmin) / (self.cosmax - self.cosmin);
            costheta * costheta * costheta * costheta * costheta
        }
    }
    pub fn getPositionws(&self) -> Point3<f64> {
        self.positionws
    }
    pub fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        let v = (self.positionws - phit.hit.point);
        let d = v.magnitude2();
        let dir = v.normalize();
        let fall = self.fallout(-dir) / d;
        let illumination = Srgb::new(
            self.color.red * fall as f32,
            self.color.green * fall as f32,
            self.color.blue * fall as f32,
        );
        let occlray = Ray::new(
            phit.hit.point,
            (self.positionws - phit.hit.point).normalize(),
        );
        (dir, illumination, 1.0, Some(occlray), None)
    }
    pub fn inner_sampleIllumination_debug(
        &self,
        phit: &Point3<f64>,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        let v = (self.positionws - phit);
        let d = v.magnitude2();
        let dir = v.normalize();
        let fall = self.fallout(-dir) / d;
        let illumination = Srgb::new(
            self.color.red * fall as f32,
            self.color.green * fall as f32,
            self.color.blue * fall as f32,
        );
        let occlray = Ray::new(*phit, (self.positionws - phit).normalize());
        (dir, illumination, 1.0, Some(occlray), None)
    }
    pub fn inner_is_arealight(&self) -> bool {
        false
    }
}

#[test]
fn test_spotlight() {
     
    {
        let vdirspot = (Point3::new(0.0, 1.0, 0.0) - Point3::new(0.10, 0.0, 0.0)).normalize();
        let spot = SpotLight::new(
            Vector3::new(0.0, 1.0, 0.0),
            Vector3::new(0.0, -1.0, 0.0),
            Srgb::new(1.0, 1.0, 1.0),
            30.0,
            10.0,
        );
        for i in 0..100 {
            println!(
                "{:?}",
                spot.inner_sampleIllumination_debug(&Point3::new(i as f64 / 100.0, 0.0, 0.0))
                    .1
                    .red
            );
        }
    }
    {
        let vdirspot = (Point3::new(0.0, 1.0, 0.0) - Point3::new(0.10, 0.0, 0.0));
        let spot = SpotLight::new(
            Vector3::new(0.0, 1.0, 0.0),
            vdirspot,
            Srgb::new(1.0, 1.0, 1.0),
            30.0,
            10.0,
        );
        for i in 0..100 {
            println!(
                "{:?}",
                spot.inner_sampleIllumination_debug(&Point3::new(i as f64 / 100.0, 0.0, 0.0))
                    .1
                    .red
            );
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct PointLight<Scalar> {
    pub iu: Scalar,
    pub color: Srgb,
    pub positionws: Point3<Scalar>,
}
impl SampleLightIllumination for PointLight<f64> {
    fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        let v = (self.positionws - phit.hit.point);
        let d = v.magnitude2();
        let dir = v.normalize();
        let illumination = Srgb::new(
            self.color.red / d as f32,
            self.color.green / d as f32,
            self.color.blue / d as f32,
        );
        let occlray = Ray::new(
            phit.hit.point,
            (self.positionws - phit.hit.point).normalize(),
        );
        (dir, illumination, 1.0, Some(occlray), None)
    }
    fn getPositionws(&self) -> Point3<f64> {
        self.positionws
    }
}

impl<Scalar> IsAreaLight for PointLight<Scalar> {
    fn is_arealight(&self) -> bool {
        false
    }
}

impl SampleEmission for PointLight<f64> {
    fn sample_emission(&self, record: RecordSampleLightEmissionIn) -> RecordSampleLightEmissionOut {
        let sampleu = UniformSampleSphere(&record.psample0);
         
      
        let r = Ray::<f64>::new(Point3::new(self.positionws.x  ,self.positionws.y ,self.positionws.z ),  sampleu.0);
    
    
        RecordSampleLightEmissionOut::from_hitrecord(   
            self.color,
            1.0, 
            sampleu.1, 
            r.direction.normalize(), 
            Some( r )
        )
    }
    
}

impl <Scalar> PdfEmission for PointLight<Scalar> {
    fn pdf_emission(&self ,rec :  RecordPdfEmissionIn  ) -> RecordPdfEmissionOut {
        // uniform sample Sphere Inv4Pi = 0.07957747154594766788;
        RecordPdfEmissionOut::new(0.0,  0.07957747154594766788)
    } 
    
}








#[derive(Clone, Debug)]
pub struct AreaLight {
    pub color: Srgb,
    pub positionws: Point3<f64>,
    pub plane: Plane,
}
impl AreaLight {
    pub fn new(
        translation: Vector3<f64>,
        direction: Vector3<f64>,
        halfw: f64,
        halfh: f64,
    ) -> AreaLight {
        AreaLight {
            color: Srgb::new(0.5, 0.5, 0.5),
            positionws: Point3::new(0.0, 0.0, 0.0),
            plane: Plane::new(
                translation,
                direction,
                halfw,
                halfh,
                MaterialDescType::NoneType,
            ),
        }
    }
    pub fn new_emission(
        translation: Vector3<f64>,
        direction: Vector3<f64>,
        halfw: f64,
        halfh: f64,
        lemission: Srgb,
    ) -> AreaLight {
        AreaLight {
            color: lemission,
            positionws: Point3::new(translation.x, translation.y, translation.z),
            plane: Plane::new(
                translation,
                direction,
                halfw,
                halfh,
                MaterialDescType::NoneType,
            ),
        }
    }
}
impl IsAreaLight for AreaLight {
    fn is_arealight(&self) -> bool {
        true
    }
}
// fn  get_shape(&self)->Option<
// Box<dyn PrimitiveIntersection<Output=HitRecord<f64>>>
// >;
impl GetShape for AreaLight {
    fn get_shape(&self) -> Option<Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> {
        Some(Box::new(self.plane.clone()))
    }
}

impl GetEmission for AreaLight {
    fn get_emission(&self, hit: Option<HitRecord<f64>>, r: &Ray<f64>) -> Srgb {
        //   self.plane.sample(psample)
        //  tengo  que poner la luz mirando al suelo  de modo que de bien

        if hit.unwrap().prev.unwrap().dot(hit.unwrap().normal) > 0.0 {
            Srgb::new(1.0, 1.0, 1.0)
        } else {
            Srgb::new(0.0, 0.0, 0.0)
        }
    }
}

impl PdfIllumination for AreaLight {
    fn pdfIllumination(&self, hit: &HitRecord<f64>, scene: &Scene<f64>, next: Vector3<f64>) -> f64 {
        let r = Ray::new(hit.point, next);
        let hitlight = interset_scene(&r, &scene).unwrap();
        let distance2 = (hit.point - hitlight.point).magnitude2();
        let solidangle = (-next).dot(hit.normal).abs();
        let a = self.plane.area();
        let pdf = distance2 / (solidangle * a);
        pdf
    }
}

impl SampleLightIllumination for AreaLight {
    fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        let psample = self.plane.sample(phit.psample);
        // println!("pt  in plane   {:?}" , psample.0);
        //   println!("dir in plane   {:?}" , psample.1);
        //   println!("pdf          : {:?}" , psample.2);
        let ptplane = psample.0;

        let ptnormal = psample.1;
        let mut pdf = psample.2;

        let mut vnext = (ptplane - phit.hit.point);
        if vnext.magnitude2() == 0.0 {
            pdf = 0.0;
        } else {
            vnext = vnext.normalize();
            // println!("{:?}" ,vnext);
            let num = (ptplane - phit.hit.point).magnitude2();
            let w = -vnext;
            let den = w.dot(ptnormal).abs();
            let pdfrat = num / (den);

            if pdfrat == std::f64::INFINITY {
                pdf = 0.0;
            } else {
                pdf = pdf * pdfrat
            }
        }
        let vnext = (ptplane - phit.hit.point).normalize();
        //  L(normal, vnext);
        let Ld = self.color;

        // (Vector3::new(0.0,0.0,0.0),Srgb::new(0.0,0.0,0.0), 0.0,None)
        (vnext, Ld, pdf, None, Some(psample.0))
    }
    fn getPositionws(&self) -> Point3<f64> {
        self.positionws
    }
}



#[derive(Clone, Debug)]
pub struct New_Interface_AreaLight {
    pub color: Srgb, 
    pub shape : PrimitiveType,
    pub num_samples:u32,
    
}
impl  New_Interface_AreaLight  {
    pub fn new( lemission: Srgb,shape : PrimitiveType,num_samples:u32) ->  New_Interface_AreaLight  {
        New_Interface_AreaLight { color:lemission, shape, num_samples }
        
    }
}



impl  NumSamples for New_Interface_AreaLight {
    fn num_samples(&self) -> u32 {
        self.num_samples
    } 
}



impl IsAreaLight for New_Interface_AreaLight {
    fn is_arealight(&self) -> bool {
        true
    }
}


impl GetShape for New_Interface_AreaLight {
    fn get_shape(&self) -> Option<Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> {
       None
    }
}




impl GetEmission for New_Interface_AreaLight  {
    fn get_emission(&self, hit: Option<HitRecord<f64>>, r: &Ray<f64>) -> Srgb {
        //   self.plane.sample(psample)
        //  tengo  que poner la luz mirando al suelo  de modo que de bien

        if hit.unwrap().prev.unwrap().dot(hit.unwrap().normal) > 0.0 {
            self.color
            
        } else {
            Srgb::new(0.0, 0.0, 0.0)
        }
    }
}
pub
trait PdfIllumination1    {
    fn pdfIllumination1(&self, hit: &HitRecord<f64>, scene: &Scene1, next: Vector3<f64>) -> f64 ;
}
 
 
pub fn generic_shape_pdfillumination(hit: &HitRecord<f64>, scene: &Scene1, next: Vector3<f64>, areashape:f64) -> f64{
    let r = Ray::new(hit.point, next);
    let ohitlight = new_interface_for_intersect_scene(&r, &scene);
    match ohitlight{
        Some( hitlight)=>{
            let distance2 = (hit.point - hitlight.point).magnitude2();
            
            let solidangle = (-next).dot(hitlight.normal).abs();
            let a = areashape;
            let pdf = distance2 / (solidangle * a);
            pdf
        },
        None => {
             0.0
        } 
    }
   
}
impl PdfIllumination1 for New_Interface_AreaLight {
    fn pdfIllumination1(&self, hit: &HitRecord<f64>, scene: &Scene1, next: Vector3<f64>) -> f64 {
        //      next = ray
        //         /
        // hit.p__/______________
    //     let r = Ray::new(hit.point, next);
    //     hay que hacer el pdf PdfIllumination en la sphere en pbrt
    //   Sphere  304  Float Sphere::Pdf(const Interaction &ref, const Vector3f &wi) const {
    //    match  self.shape {
    //         PrimitiveType::SphereType(s)=>{s.pdfIllumination()},
    //         PrimitiveType::CylinderType || PrimitiveType::DiskType =>{
    //             si esto es una spehere tenemos que calcular la pdf de la manera de la esphere si lo es de otro modo usamos el generico 
    //             que es este codigo. 
    //              let ohitlight = new_interface_for_intersect_scene(&r, &scene);
    //              match ohitlight{
    //                      Some( hitlight)=>{
         
         
    //                          let distance2 = (hit.point - hitlight.point).magnitude2();
    //                          let solidangle = (-next).dot(hit.normal).abs();
    //                          let a = self.shape.area();
    //                          let pdf = distance2 / (solidangle * a);
    //                          pdf
    //                      },
    //                      None => {
    //                          0.0
    //                      } 
    //              }
    //         }
    //        _=>0.
    //    } 
        match  &self.shape {
            PrimitiveType::CylinderType(c) => {
                generic_shape_pdfillumination(hit, scene, next, c.area())
             //   todo!()
            },
            PrimitiveType::DiskType(s) =>{
                generic_shape_pdfillumination(hit, scene, next, s.area())
               // todo!()
            },
            PrimitiveType::SphereType(p) => p.pdfIllumination1(hit, scene, next),
        }
       
        // mide la distancia desde la superficie con el angulo next hasta la luz c
    }
}

pub trait GetShape1 {
    fn get_shape1(&self) -> &PrimitiveType;
}

impl GetShape1 for New_Interface_AreaLight {
    fn get_shape1(&self) ->&PrimitiveType {
      &self.shape
    }
}





impl SampleLightIllumination for New_Interface_AreaLight {
    fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        if let  PrimitiveType::SphereType(shape)  =  &self.shape  {
            let psample = self.shape.sample_ref(phit.hit.point,phit.psample);
            let ptplane = psample.0; 
            let ptnormal = psample.1;
            let vnext = (ptplane - phit.hit.point).normalize(); 
            (ptnormal,self.color ,psample.2 , Some( Ray::new(phit.hit.point, vnext) ), Some(psample.0))
        }else{
            // disk and cilinder onversion solid angle to area measure

            let psample = self.shape.sample_ref(phit.hit.point,phit.psample);
       
       
            let ptplane = psample.0;
    
            let ptnormal = psample.1.normalize();
            let mut pdf = psample.2;
            let  mut Ld = self.color; 
            let mut vnext = (ptplane - phit.hit.point);
            if vnext.magnitude2() == 0.0 {
                pdf = 0.0;
            } else {
                vnext = vnext.normalize();
               
                let num = (ptplane - phit.hit.point).magnitude2();
                let w = -vnext;
                let den = w.dot(ptnormal);
                if den <= 0.0 {
                    Ld = Srgb::new(0.0, 0.0, 0.0);
                }
                //---- convert area!!-----
                let den = w.dot(ptnormal).abs();
                let pdfrat = num / (den);
    
                if pdfrat == std::f64::INFINITY {
                    pdf = 0.0;
                } else {
                    pdf = pdf * pdfrat
                }
            }
            let vnext = (ptplane - phit.hit.point).normalize();  
            (ptnormal, Ld, pdf, Some( Ray::new(phit.hit.point, vnext) ), Some(psample.0))
        }

         
    //     let psample = self.shape.sample_ref(phit.hit.point,phit.psample);
       
       
    //     let ptplane = psample.0;

    //     let ptnormal = psample.1;
    //     let mut pdf = psample.2;
    //     let  mut Ld = self.color; 
    //     let mut vnext = (ptplane - phit.hit.point);
    //     if vnext.magnitude2() == 0.0 {
    //         pdf = 0.0;
    //     } else {
    //         vnext = vnext.normalize();
           
    //         let num = (ptplane - phit.hit.point).magnitude2();
    //         let w = -vnext;
    //         let den = w.dot(ptnormal);
    //         if den <= 0.0 {
    //             Ld = Srgb::new(0.0, 0.0, 0.0);
    //         }
    //         //---- convert area!!-----
    //         let den = w.dot(ptnormal).abs();
    //         let pdfrat = num / (den);

    //         if pdfrat == std::f64::INFINITY {
    //             pdf = 0.0;
    //         } else {
    //             pdf = pdf * pdfrat
    //         }
    //     }
    //     let vnext = (ptplane - phit.hit.point).normalize(); 
        
       
    //     //TODO: esta normal latengo que meter en el constructor de disk
    // //    if self.shape.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0)).normalize().dot(-vnext) < 0.0{
    // //     Ld = Srgb::new(0.0,0.0,0.0);
    // //    }
     
    //     (ptnormal, Ld, pdf, Some( Ray::new(phit.hit.point, vnext) ), Some(psample.0))
    }
    fn getPositionws(&self) -> Point3<f64> {
        panic!("")
    }
}



impl SampleEmission for New_Interface_AreaLight{
    fn sample_emission(&self, record: RecordSampleLightEmissionIn  )-> RecordSampleLightEmissionOut {
   
        // (  Point3f::default_epsilon(), Vector3f::default_epsilon() , 0.0);
       let recordshape = self.shape.sample(record.psample0);
   // let recordshape =  (  Point3f::new(0.0,0.0,0.0), Vector3f::new(0.0,0.0,0.0) , 0.0);;
        
        let n = recordshape.1.normalize();
        let v =   CosSampleH2(&record.psample1);
       let pdfdir =  CosSampleH2Pdf(v.z);
   
       let ( v0, v1 ) = coord_system(n);
       let dir = v.x * v0 + v.y * v1 + v.z * n;
    
      let  mut L : Srgb;
      if n.dot(dir) > 0.0 {
         L = self.color;
       
      }else{
         L = Srgb::new(0.0,0.0,0.0);
      }
     
        RecordSampleLightEmissionOut{
            Lemission:L,
            pdfdir,
            pdfpos:recordshape.2,
            n:recordshape.1.normalize(),
             ray:Some(Ray::new(recordshape.0, dir.normalize())),
        }
    }
}

 
impl  PdfEmission for New_Interface_AreaLight{
    fn pdf_emission(&self , rec :  RecordPdfEmissionIn ) -> RecordPdfEmissionOut {
       
       
       let z =rec.n.dot(rec.ray.unwrap().direction);
       RecordPdfEmissionOut::new( 1.0/self.shape.area(), CosSampleH2Pdf(z));
       RecordPdfEmissionOut::new( 1.0/self.shape.area(), CosSampleH2Pdf(rec.n.dot(rec.ray.unwrap().direction) )) 
    } 
    
}

















#[derive(Clone, Debug)]
pub struct AreaLightDisk {
    pub color: Srgb,
    pub positionws: Point3<f64>,
    pub disk: Disk,
    
}
impl AreaLightDisk {
    pub fn new(
        translation: Vector3<f64>,
        direction: Vector3<f64>,
        radius: f64,
        lemission: Srgb,
    ) -> AreaLightDisk {
        AreaLightDisk {
            color: lemission,
            positionws: Point3::new(0.0, 0.0, 0.0),
            disk: Disk::new(
                translation,
                direction,
                0.0,
                radius,
                MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0))
            ),
           
        }
    }
}
impl IsAreaLight for AreaLightDisk {
    fn is_arealight(&self) -> bool {
        true
    }
}
// esto es lo que tengo que variar de momento...
// yo creo escribo un  inteface generico y ya vere como lo hago depues

impl GetShape for AreaLightDisk {
    fn get_shape(&self) -> Option<Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> {
        Some(Box::new(self.disk.clone()))
    }
}

impl GetEmission for AreaLightDisk {
    fn get_emission(&self, hit: Option<HitRecord<f64>>, r: &Ray<f64>) -> Srgb {
        //   self.plane.sample(psample)
        //  tengo  que poner la luz mirando al suelo  de modo que de bien

        if hit.unwrap().prev.unwrap().dot(hit.unwrap().normal) > 0.0 {
            self.color
            
        } else {
            Srgb::new(0.0, 0.0, 0.0)
        }
    }
}

impl PdfIllumination for AreaLightDisk {
    fn pdfIllumination(&self, hit: &HitRecord<f64>, scene: &Scene<f64>, next: Vector3<f64>) -> f64 {
        //      next = ray
        //         /
        // hit.p__/______________
        let r = Ray::new(hit.point, next);
        let ohitlight = interset_scene(&r, &scene);
        match ohitlight{
                Some( hitlight)=>{


                    let distance2 = (hit.point - hitlight.point).magnitude2();
                    let solidangle = (-next).dot(hit.normal).abs();
                    let a = self.disk.area();
                    let pdf = distance2 / (solidangle * a);
                    pdf
                },
                None => {
                    0.0
                } 
        }
        
       
        // mide la distancia desde la superficie con el angulo next hasta la luz c
    }
}

impl SampleLightIllumination for AreaLightDisk {
    fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        let psample = self.disk.sample(phit.psample);
       
        let ptplane = psample.0;

        let ptnormal = psample.1;
        let mut pdf = psample.2;

        let mut vnext = (ptplane - phit.hit.point);
        if vnext.magnitude2() == 0.0 {
            pdf = 0.0;
        } else {
            vnext = vnext.normalize();
           
            let num = (ptplane - phit.hit.point).magnitude2();
            let w = -vnext;
            let den = w.dot(ptnormal).abs();
            let pdfrat = num / (den);

            if pdfrat == std::f64::INFINITY {
                pdf = 0.0;
            } else {
                pdf = pdf * pdfrat
            }
        }
        let vnext = (ptplane - phit.hit.point).normalize(); 
        
        let  mut Ld = self.color; 
        //TODO: esta normal latengo que meter en el constructor de disk
       if self.disk.trafo_vec_to_world(Vector3::new(0.0,0.0,1.0)).normalize().dot(-vnext) < 0.0{
        Ld = Srgb::new(0.0,0.0,0.0);
       }
     
        (ptnormal, Ld, pdf, Some( Ray::new(phit.hit.point, vnext) ), Some(psample.0))
    }
    fn getPositionws(&self) -> Point3<f64> {
        panic!("")
    }
}



impl SampleEmission for AreaLightDisk{
    fn sample_emission(&self, record: RecordSampleLightEmissionIn  )-> RecordSampleLightEmissionOut {
        let recordshape = self.disk.sample(record.psample0);
        
        let n = recordshape.1.normalize();
        let v = CosSampleH2(&record.psample1);
       let pdfdir = CosSampleH2Pdf(v.z);
   
       let ( v0, v1 ) = coord_system(n);
       let dir = v.x * v0 + v.y * v1 + v.z * n;
    
      let  mut L : Srgb;
      if n.dot(dir) > 0.0 {
         L = self.color;
       
      }else{
         L = Srgb::new(0.0,0.0,0.0);
      }
     
        RecordSampleLightEmissionOut{
            Lemission:L,
            pdfdir,
            pdfpos:recordshape.2,
            n:recordshape.1.normalize(),
            ray:Some(Ray::new(recordshape.0, dir.normalize())),
        }
    }
}




impl  PdfEmission for AreaLightDisk{
    fn pdf_emission(&self , rec :  RecordPdfEmissionIn ) -> RecordPdfEmissionOut {
       
       
       let z =rec.n.dot(rec.ray.unwrap().direction);
       RecordPdfEmissionOut::new( 1.0/self.disk.area(), CosSampleH2Pdf(z));
       RecordPdfEmissionOut::new( 1.0/self.disk.area(), CosSampleH2Pdf(rec.n.dot(rec.ray.unwrap().direction) ))
        
       
      
    } 
    
}


#[derive(Clone, Debug)]
pub struct AreaLightSphere {
    pub color: Srgb,
    pub positionws: Point3<f64>,
    pub sphere: Sphere<f64>,
}
impl AreaLightSphere{
    pub fn new(
        translation: Vector3<f64>,
       
        scale: f64,
        lemission: Srgb,
    ) -> AreaLightSphere {
        
        AreaLightSphere {
            color: lemission,
            positionws: Point3::new(0.0, 0.0, 0.0),
         sphere:Sphere::new(translation, scale, MaterialDescType::NoneType)
        }
    }
}
impl IsAreaLight for AreaLightSphere {
    fn is_arealight(&self) -> bool {
  
        true
    }
}

impl GetShape for AreaLightSphere {
    fn get_shape(&self) -> Option<Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> {
    
        Some(Box::new(self.sphere.clone()))
    }
}

impl GetEmission for AreaLightSphere {
    fn get_emission(&self, hit: Option<HitRecord<f64>>, r: &Ray<f64>) -> Srgb {
        
        //   self.plane.sample(psample)
        //  tengo  que poner la luz mirando al suelo  de modo que de bien

        if hit.unwrap().prev.unwrap().dot(hit.unwrap().normal) > 0.0 {
            self.color
        } else {
            Srgb::new(0.0, 0.0, 0.0)
        }
    }
}

/*
 Float sinThetaMax2 = radius * radius / DistanceSquared(ref.p, pCenter);
    Float cosThetaMax = std::sqrt(std::max((Float)0, 1 - sinThetaMax2));
    return UniformConePdf(cosThetaMax); */
impl PdfIllumination for AreaLightSphere {
    fn pdfIllumination(&self, hit: &HitRecord<f64>, scene: &Scene<f64>, next: Vector3<f64>) -> f64 { 
        let r = Ray::new(hit.point, next);
        let ohitlight = interset_scene(&r, &scene);
        match ohitlight{
                Some( hitlight)=>{  
                    let distance2 =  self.getPositionws().distance2(hit.point); 
                    let a = self.sphere.area();
                    let sintheta2  =  (self.sphere.radius*self.sphere.radius) / distance2; 
                    let costheta2  = (1.0-sintheta2).max(0.0).sqrt(); 
                    UniformSampleConePdf(costheta2) 
                },
                None => {
                    0.0
                } 
        }  
    }
}

impl SampleLightIllumination for AreaLightSphere {
    fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>, // normal vector in point sampled in sphere with phit.point2d sample
        Srgb,
        f64,
        Option<Ray<f64>>, //from phit.point to sample point in sphere
        Option<Point3<f64>>,
    ) {
 
        
        let psample = self.sphere.sample(  phit.psample);
        
        let ptsphere = psample.0;
        let ptnormal = psample.1;
        let mut pdf = psample.2;

        let mut vnext = (ptsphere - phit.hit.point);
        if vnext.magnitude2() == 0.0 ||  psample.2==0.0 {
          
            (vnext, Srgb::new(0.0,0.0,0.0), 0.0, None, Some(psample.0))
        } else {
            vnext = vnext.normalize();
            
            let vnext = (ptsphere - phit.hit.point).normalize();
        //  L(normal, vnext);
             let mut  Ld =  Srgb::new(0.0,0.0,0.0);
            // let w = -vnext;
            let den = vnext.dot(-ptnormal) > 0.0 ;
            if den {
                Ld  = self.color;
            }
           ;
            (vnext, Ld, pdf,  Some(Ray::new(phit.hit.point,vnext)), Some(psample.0))

            
        }
      

        // (Vector3::new(0.0,0.0,0.0),Srgb::new(0.0,0.0,0.0), 0.0,None)
        //(vnext, Ld, pdf, None, Some(psample.0))
    }
    fn getPositionws(&self) -> Point3<f64> {
       self.sphere.trafo_to_world(Point3::new(0.0,0.0,0.0))
    }
}






































#[derive(Debug, Clone)]
pub struct BackgroundAreaLight {
    pub emissionMap: Vec<Srgb>,
    pub radianceMap: Vec<f64>,
    pub distribution: Distribution2d,
    pub w: usize,
    pub h: usize,
    pub world: Decomposed<Vector3<f64>, Quaternion<f64>>,
    pub local: Decomposed<Vector3<f64>, Quaternion<f64>>,
    // pub  positionws: Point3<f64>,
}
impl BackgroundAreaLight {
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
    pub fn init_radianceMap(
        emissionMap: &Vec<Srgb>,
        radianceMap: &mut Vec<f64>,
        w: usize,
        h: usize,
    ) {
        // let mut pixels = vec![0; scene.width * scene.height * 3];
        //  let bands: Vec<(usize, &mut [u8])> = pixels.chunks_mut(scene.width * 3).enumerate().collect();
        for iy in 0..h {
            let v = (iy as f64 + 0.5) / h as f64;
            let sintheta = (std::f64::consts::PI * v).sin();
            for ix in 0..w {
                let u = (ix as f64 + 0.5) / (w as f64);
                // aqui deberia meter coords u, v + 0.5
                // but i want  to keep it simple
                let rgb = emissionMap[iy * w + ix];
                let L = rgb.red * 0.2126 + rgb.green * 0.7152 + rgb.blue * 0.0732;
                //     println!("x, y,  {},{} {} , sintheta {}",u,v, L, L* sintheta as f32);
                radianceMap[iy * w + ix] = (L as f64) * sintheta;
            }
        }
    }
    pub fn init_ditribution(radianceMap: &Vec<f64>, w: usize, h: usize) -> Distribution2d {
        Distribution2d::new(radianceMap, w, h)
    }
    pub fn from_file(filename: &str, multiplier: Srgb) -> BackgroundAreaLight {
        let i = ImageReader::open(filename).unwrap();
        let mut ii = i.with_guessed_format().unwrap().decode().unwrap();

        let mut radianceMap = vec![0.0; (ii.height() * ii.width()) as usize];
        let mut lemissionMap: Vec<Srgb> =
            vec![Srgb::new(0.0, 0.0, 0.0); (ii.height() * ii.width()) as usize];

        for p in ii.pixels() {
            let x = p.0;
            let y = p.1;
            let co = p.2.to_rgba();
            let channels = co.channels();

            let c = channels[0] as f32 / 255.0;
            let c1 = channels[1] as f32 / 255.0;
            let c2 = channels[2] as f32 / 255.0;

            lemissionMap[(ii.width() * (y) + x) as usize] = Srgb::new(
                c * multiplier.red,
                c1 * multiplier.green,
                c2 * multiplier.blue,
            );
        }

        let mut radianceMap = vec![0.0; (ii.width() * ii.height()) as usize];
        Self::init_radianceMap(
            &lemissionMap,
            &mut radianceMap,
            ii.width() as usize,
            ii.height() as usize,
        );
        let distribution =
            Self::init_ditribution(&radianceMap, ii.width() as usize, ii.height() as usize);

        let t1 = Decomposed {
            disp: Vector3::new(0.0, 0.0, 0.0),
            rot: Quaternion::from_arc(
                Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
                None,
            ),
            scale: 1.0,
        };
        BackgroundAreaLight {
            emissionMap: lemissionMap,
            radianceMap,
            distribution,
            w: ii.width() as usize,
            h: ii.height() as usize,
            local: t1.inverse_transform().unwrap(),
            world: t1,
        }
    }

    pub fn new(emissionMap: Vec<Srgb>, w: usize, h: usize) -> BackgroundAreaLight {
        let mut radianceMap = vec![0.0; w * h];
        Self::init_radianceMap(&emissionMap, &mut radianceMap, w, h);
        let distribution = Self::init_ditribution(&radianceMap, w, h);

        let t1 = Decomposed {
            disp: Vector3::new(0.0, 0.0, 0.0),
            rot: Quaternion::from_arc(
                Vector3::new(0.0, 0.0, 1.0),
                Vector3::new(0.0, 0.0, 1.0),
                None,
            ),
            scale: 1.0,
        };
        BackgroundAreaLight {
            emissionMap: emissionMap,
            radianceMap,
            distribution,
            w,
            h,
            local: t1.inverse_transform().unwrap(),
            world: t1,
        }
    }
    fn texel(&self, s: usize, t: usize) -> Srgb {
        let x = clamp(s, 0, self.w - 1);
        let y = clamp(t, 0, self.h - 1);
        self.emissionMap[y * self.w + x]
    }
    fn auxmultsrgb(m: f64, srgb: Srgb) -> Srgb {
        Srgb::new(
            m as f32 * srgb.red,
            m as f32 * srgb.green,
            m as f32 * srgb.blue,
        )
    }
    fn auxsumrgb(srgb0: Srgb, srgb1: Srgb, srgb2: Srgb, srgb3: Srgb) -> Srgb {
        Srgb::new(
            srgb0.red + srgb1.red + srgb2.red + srgb3.red,
            srgb0.green + srgb1.green + srgb2.green + srgb3.green,
            srgb0.blue + srgb1.blue + srgb2.blue + srgb3.blue,
        )
    }
    fn lookup(&self, stlook: &(f64, f64)) -> Srgb {
        let st = (
            (stlook.0 * self.w as f64) - 0.5,
            (stlook.1 * self.h as f64) - 0.5,
        );
        let sfloor = st.0.floor();
        let tfloor = st.1.floor();
        let ds = st.0 - sfloor;
        let dt = st.1 - tfloor;
        let sfloor = st.0.floor() as usize;
        let tfloor = st.1.floor() as usize;
        let tx00 = Self::auxmultsrgb((1.0 - ds) * (1.0 - dt), self.texel(sfloor, tfloor));
        let tx01 = Self::auxmultsrgb((1.0 - ds) * (dt), self.texel(sfloor, tfloor + 1));
        let tx10 = Self::auxmultsrgb((ds) * (1.0 - dt), self.texel(sfloor + 1, tfloor));
        let tx11 = Self::auxmultsrgb((ds) * (dt), self.texel(sfloor + 1, tfloor + 1));
        let L = Self::auxsumrgb(tx00, tx01, tx10, tx11);
        L
    }
    pub fn inner_sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        // offset cond ,offset mar , pdf = pdfcond * pdfmar
        let samplep = self.distribution.sample(&phit.psample);
        let theta = samplep.1 * std::f64::consts::PI;
        let phi = samplep.0 * 2.0 * std::f64::consts::PI;
        let sc_theta = theta.sin_cos();
        let sc_phi = phi.sin_cos();
        let vnext = Vector3::new(sc_theta.0 * sc_phi.1, sc_theta.0 * sc_phi.0, sc_theta.1);
        let mut pdf = samplep.2 / (2.0 * std::f64::consts::PI * std::f64::consts::PI * sc_theta.0);
        if sc_theta.0 == 0.0 {
            pdf = 0.0
        }
        let vnextWow = self.trafo_vec_to_world(vnext);

        let r = Ray::new(phit.hit.point, vnextWow);

        let target = r.at(1000.0);
        //    let ycoord = (  samplep.1 * (self.h-1) as f64) as usize;
        //    let xcoord = (  samplep.0 * (self.w-1) as f64) as usize;
        //    let L  = self.emissionMap[ycoord*self.w + xcoord];

        let s = (samplep.0 * self.w as f64) - 0.5;
        let t = (samplep.1 * self.h as f64) - 0.5;
        let sfloor = s.floor();
        let tfloor = t.floor();
        let ds = s - sfloor;
        let dt = t - tfloor;
        let sfloor = s.floor() as usize;
        let tfloor = t.floor() as usize;

        let tx00 = Self::auxmultsrgb((1.0 - ds) * (1.0 - dt), self.texel(sfloor, tfloor));
        let tx01 = Self::auxmultsrgb((1.0 - ds) * (dt), self.texel(sfloor, tfloor + 1));
        let tx10 = Self::auxmultsrgb((ds) * (1.0 - dt), self.texel(sfloor + 1, tfloor));
        let tx11 = Self::auxmultsrgb((ds) * (dt), self.texel(sfloor + 1, tfloor + 1));
        let L = Self::auxsumrgb(tx00, tx01, tx10, tx11);

        // radianceMap[iy*self.w+ix]
        (
            self.trafo_vec_to_world(vnext),
            L,
            pdf,
            Some(r),
            Some(target),
        )
    }
    pub fn inner_pdf_illumination(&self, next: Vector3<f64>) -> f64 {
        let vnext = self.trafo_vec_to_local(next);
        let thetaphi = SphericalCoords(vnext);
        let sintheta = thetaphi.0.sin();
        if sintheta == 0.0 {
            return 0.0;
        }
        let u = (
            thetaphi.1 * (1.0 / (std::f64::consts::PI * 2.0)),
            thetaphi.0 * (1.0 / (std::f64::consts::PI)),
        );
        self.distribution.get_pdf(&u)
            / (2.0 * std::f64::consts::PI * std::f64::consts::PI * sintheta)
    }
    pub fn get_emission(&self, hit: Option<HitRecord<f64>>, r: &Ray<f64>) -> Srgb {
        let vdir = self.trafo_vec_to_local(r.direction);
        let st = (
            SphericalCoords(vdir).1 * (1.0 / (std::f64::consts::PI * 2.0)),
            SphericalCoords(vdir).0 * (1.0 / (std::f64::consts::PI)),
        );
        //  println!("{:?}",r.direction);
        //  println!("{:?}", vdir);
        //  println!("{:?}", st);
        self.lookup(&st)
        // let xcoord = (  st.0 * (self.w-1) as f64) as usize; // phi
        // let ycoord = (  st.1 * (self.h-1) as f64) as usize; // theta
        // let L  = self.emissionMap[ycoord*self.w + xcoord];
        // L
    }
    pub fn get_emission_debug(&self, r: Ray<f64>) -> Srgb {
        let vdir = self.trafo_vec_to_local(r.direction);
        let st = (
            SphericalCoords(vdir).1 * (1.0 / (std::f64::consts::PI * 2.0)),
            SphericalCoords(vdir).0 * (1.0 / (std::f64::consts::PI)),
        );

        let xcoord = (st.0 * (self.w - 1) as f64) as usize; // phi
        let ycoord = (st.1 * (self.h - 1) as f64) as usize; // theta
        let L = self.emissionMap[ycoord * self.w + xcoord];
        L
    }
}

impl IsAreaLight for BackgroundAreaLight {
    fn is_arealight(&self) -> bool {
        false
    }
}

impl GetShape for BackgroundAreaLight {
    fn get_shape(&self) -> Option<Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> {
        None
    }
}

impl GetEmission for BackgroundAreaLight {
    fn get_emission(&self, hit: Option<HitRecord<f64>>, r: &Ray<f64>) -> Srgb {
        self.get_emission(hit, r)
    }
}

impl PdfIllumination for BackgroundAreaLight {
    fn pdfIllumination(&self, hit: &HitRecord<f64>, scene: &Scene<f64>, next: Vector3<f64>) -> f64 {
        self.inner_pdf_illumination(next)
    }
}

impl SampleLightIllumination for BackgroundAreaLight {
    fn getPositionws(&self) -> Point3<f64> {
        Point3::new(0.0, 0.0, 0.0)
    }
    fn sampleIllumination(
        &self,
        phit: &RecordSampleLightIlumnation,
    ) -> (
        Vector3<f64>,
        Srgb,
        f64,
        Option<Ray<f64>>,
        Option<Point3<f64>>,
    ) {
        self.inner_sampleIllumination(phit)
    }
}
#[test]
fn test_BackgroundAreaLight() {
    let vf64: Vec<Srgb> = vec![
        Srgb::new(0.10, 0.10, 0.10),
        Srgb::new(0.20, 0.20, 0.20),
        Srgb::new(0.30, 0.30, 0.30),
        Srgb::new(0.40, 0.40, 0.40),
        Srgb::new(0.50, 0.50, 0.50),
        Srgb::new(0.60, 0.60, 0.60),
        Srgb::new(0.70, 0.70, 0.70),
        Srgb::new(0.80, 0.80, 0.80),
        Srgb::new(0.10, 0.10, 0.10),
        Srgb::new(0.20, 0.20, 0.20),
        Srgb::new(0.30, 0.30, 0.30),
        Srgb::new(0.40, 0.40, 0.40),
        Srgb::new(0.50, 0.50, 0.50),
        Srgb::new(0.60, 0.60, 0.60),
        Srgb::new(0.70, 0.70, 0.70),
        Srgb::new(0.80, 0.80, 0.80),
        Srgb::new(0.10, 0.10, 0.10),
        Srgb::new(0.20, 0.20, 0.20),
        Srgb::new(0.30, 0.30, 0.30),
        Srgb::new(0.40, 0.40, 0.40),
        Srgb::new(0.50, 0.50, 0.50),
        Srgb::new(0.60, 0.60, 0.60),
        Srgb::new(0.70, 0.70, 0.70),
        Srgb::new(0.80, 0.80, 0.80),
        Srgb::new(0.10, 0.10, 0.10),
        Srgb::new(0.20, 0.20, 0.20),
        Srgb::new(0.30, 0.30, 0.30),
        Srgb::new(0.40, 0.40, 0.40),
        Srgb::new(0.50, 0.50, 0.50),
        Srgb::new(0.60, 0.60, 0.60),
        Srgb::new(0.70, 0.70, 0.70),
        Srgb::new(0.80, 0.80, 0.80),
    ];

    // let inflight = BackgroundAreaLight::new(  vf64, 8,4);

    let inflight = BackgroundAreaLight::new(vec![Srgb::new(1.0, 1.0, 1.0)], 1, 1);

    for iy in 0..32 {
        for ix in 0..32 {
            // let ix=0;
            // let iy=1;

            let fx = ix as f64 / 32.0;
            let fy = iy as f64 / 32.0;
            // println!("{}", fy);
            //    let iy = 0;
            //    let ix = 1;
            //  let res = inflight.inner_sampleIllumination(&(fx, fy));
            // let pdfinneril =  inflight.inner_pdf_illumination(res.0);
            // let color = inflight.get_emission_debug(Ray::new(Point3::new(0.0,0.0,0.0), res.0));
            //     println!(" ix,iy: {},{}, pdf {} L {:?} wi {:?} pdf_illu: {} , color: {:?}",ix, iy, res.2 , res.1, res.0 , pdfinneril , color );
            //      assert_delta!(res.2 , pdfinneril, 0.00001);
            //     // assert!(res.1.red == color.red);
            //     assert_delta!(res.1.red  , color.red, 0.00001);
            // }
        }
    }
}

#[derive(Clone, Debug)]
pub struct LigtImportanceSampling {}
impl LigtImportanceSampling {
    pub
    fn mulScalarSrgb(a: Srgb, b: f32) -> Srgb {
        Srgb::new(a.red * b, a.green * b, a.blue * b)
    }

    pub fn addSrgb(a: Srgb, b: Srgb) -> Srgb {
        Srgb::new(a.red + b.red, a.green + b.green, a.blue + b.blue)
    }
pub
    fn mulSrgb(a: Srgb, b: Srgb) -> Srgb {
        Srgb::new(a.red * b.red, a.green * b.green, a.blue * b.blue)
    }
    pub
    fn mulLinearSrgb(a:  LinSrgb<f64>, b: Srgb) -> Srgb {
        Srgb::new(a.red as f32 * b.red, a.green   as f32 * b.green, a.blue  as f32 * b.blue)
    }

    pub fn mulVecBySrgb(v: Vec<f64>, b: Srgb) ->  Vec<f64>{
       vec! [ v[0]  * b.red as f64, v[1]  * b.green as f64 , v[2]   * b.blue as f64]
    }
    pub fn addVecBySrgb(v: Vec<f64>, b: Srgb) ->  Vec<f64>{
        vec! [ v[0]  + b.red as f64, v[1]  + b.green as f64 , v[2]   + b.blue as f64]
     }
    pub fn estimatelights(
        ray: &Ray<f64>,
        scene: &Scene<f64>,
        hit: &HitRecord<f64>,
        arealight: Light,
        usamplelight: &(f64, f64),
        refineWithSamplebsdf: bool,
    ) -> Srgb {
        Self::samplelight(ray, scene, hit, arealight, usamplelight)
    }

    pub fn estimate_area_light<'a>(
        r: &Ray<f64>,
        light: &Light,
        scene: &Scene<'a, f64>,
        depth: i32,
        hit: HitRecord<f64>,
        sampler: &mut Box<dyn Sampler>,
        nsamples: u32,
    )  -> Srgb {
        let mut rng = CustomRng::new();
        rng.set_sequence(0);
        let mut newLd: Vec<f32> = vec![0.0; 3];
        let vecsamples = generateSamplesWithSamplerGather(sampler, nsamples);

        for isample in vecsamples.into_iter() {
            
          //   rng.uniform_float();
            // let slight = (rng.uniform_float(), rng.uniform_float());
            // let sbsdf = (rng.uniform_float(), rng.uniform_float());
            //  println!("{:?}" ,slight);
            //  println!("{:?}" ,sbsdf);
          
            // let resnewe =LigtImportanceSampling::samplelight(&r, &scene, &hit, light.clone(), &slight);
            // let bsdfsample =LigtImportanceSampling::samplebsdf(&r, &scene, &hit, light.clone(), &sbsdf);
              let resnewe =LigtImportanceSampling::samplelight(&r, &scene, &hit, light.clone(), &isample.0);
            let bsdfsample =LigtImportanceSampling::samplebsdf(&r, &scene, &hit, light.clone(), &isample.1);
            //  println!("          {:?}", LigtImportanceSampling::addSrgb(resnewe , bsdfsample));
            updateL(
                &mut newLd,
                LigtImportanceSampling::addSrgb(resnewe, bsdfsample),
            );
            // println!("--> {:?}", newLd );
           
        } // end all lights loop
     
        
        newLd[0] = newLd[0] / (nsamples as f32);
        newLd[1] = newLd[1] / (nsamples as f32);
        newLd[2] = newLd[2] / (nsamples as f32);
        // println!("Ldfinal {:?}", newLd );
        Srgb::new(newLd[0], newLd[1], newLd[2])
    }

    pub fn estimate_bck_area_light<'a>(
        r: &Ray<f64>,
        light: &Light,
        scene: &Scene<'a, f64>,
     
        hit: HitRecord<f64>,
        sampler: &mut Box<dyn Sampler>,
        nsamples: u32,
    ) -> Srgb {
        let mut newLd: Vec<f32> = vec![0.0; 3];
        let vecsamples = generateSamplesWithSamplerGather(sampler, nsamples);
        for isample in vecsamples.into_iter() {

            let resnewe =
                LigtImportanceSampling::samplelight(&r, &scene, &hit, light.clone(), &isample.0);
            let bsdfsample =
                LigtImportanceSampling::samplebsdf(&r, &scene, &hit, light.clone(), &isample.1);
            //  println!("          {:?}", LigtImportanceSampling::addSrgb(resnewe , bsdfsample));
            updateL(
                &mut newLd,
                LigtImportanceSampling::addSrgb(resnewe, bsdfsample),
            );
        } // end all lights loop
        newLd[0] = newLd[0] / (nsamples as f32);
        newLd[1] = newLd[1] / (nsamples as f32);
        newLd[2] = newLd[2] / (nsamples as f32);
        // println!("Ldfinal {:?}", newLd )
        Srgb::new(newLd[0], newLd[1], newLd[2])
    }

    pub fn samplelight(
        ray: &Ray<f64>,
        scene: &Scene<f64>,
        hit: &HitRecord<f64>,
        arealight: Light,
        usamplelight: &(f64, f64),
    ) -> Srgb {
        let samplelight =
            arealight.sampleIllumination(&RecordSampleLightIlumnation::new(*usamplelight, *hit));
        if true {
            //arealight.is_background_area_light()
            let pointinLight = samplelight.4.unwrap();
            let occlusionRay = Ray::new(hit.point, (pointinLight - hit.point).normalize());
            let occ = scene
                .intersect_occlusion(&occlusionRay, &pointinLight)
                .unwrap();
            if occ.0 == true {
                return Srgb::new(0.0, 0.0, 0.0);
            }
        }
 
        // let vnext = samplelight.0;
      
        // let reflect = hit.normal.dot(-ray.direction) * hit.normal.dot(vnext) > 0.0;
        let vnext = samplelight.3.unwrap().direction;
        let reflect = hit.normal.dot(-ray.direction) * hit.normal.dot(samplelight.3.unwrap().direction) > 0.0;
        let mut bsdf_fr;
        let mut bsdf_pdf = 0.0;
        if reflect == true {
            bsdf_fr = hit.material.fr(-ray.direction, samplelight.3.unwrap().direction);
            let absdot = vnext.dot(hit.normal).abs();
            bsdf_fr = Self::mulScalarSrgb(bsdf_fr, absdot as f32);

            bsdf_pdf = hit.material.pdf((-ray.direction).normalize(), samplelight.3.unwrap().direction);
        } else {
            return  Srgb::new(0.0, 0.0, 0.0);
        }

        let w = powerHeuristic(samplelight.2, bsdf_pdf);

        let ratio = (w / samplelight.2);
        //  Ld += f * Li * weight / lightPdf;
        let fpartial = Self::mulSrgb(samplelight.1, bsdf_fr);
        let res = Self::mulScalarSrgb(fpartial, ratio as f32);
 
        if false {
            println!("psample:  {:?}", usamplelight.clone());
            println!("               wi:  {:?}", vnext);
            println!("               Li:  {:?}", samplelight.1);
            println!("               pdf Light:  {:?}", samplelight.2);
            println!("                 bsdf f:  {:?}", bsdf_fr);
            println!("                 bsdf pdf:  {:?}", bsdf_pdf);
            println!("                  powerHeuristic:  {:?}", w);
            println!("                 Ld res,{:?}", res);
        }

        return res;
    }

    pub fn samplebsdf(
        ray: &Ray<f64>,
        scene: &Scene<f64>,
        hit: &HitRecord<f64>,
        arealight: Light,
        usamplebsdf: &(f64, f64),
    ) -> Srgb {
        let recout = hit
            .material
            .sample(RecordSampleIn::from_hitrecord(*hit, ray, *usamplebsdf));

        let absdot = recout.next.dot(hit.normal).abs();
        let bsdfsamplef = LigtImportanceSampling::mulScalarSrgb(recout.f, absdot as f32);

        fn isBlack(c: Srgb) -> bool {
            c.red == 0.0 && c.green == 0.0 && c.blue == 0.0
        }
        if !isBlack(bsdfsamplef) && recout.pdf > 0.0 {
            let pdflight = arealight.pdfIllumination(hit, scene, recout.next);

            if pdflight > 0.0 {
                let Lemited = arealight.get_emission(Some(*hit), &recout.newray.unwrap());
                let w = powerHeuristic(recout.pdf, pdflight);

                let aux = LigtImportanceSampling::mulSrgb(bsdfsamplef, Lemited);
                return LigtImportanceSampling::mulScalarSrgb(aux, (w / recout.pdf) as f32);
            }
        }
        Srgb::new(0.0, 0.0, 0.0)
    }







    pub fn samplelight1(
        ray: &Ray<f64>,
        scene: &Scene1,
        hit: &HitRecord<f64>,
        light: Light,
        usamplelight: &(f64, f64),
    ) -> Srgb {
        let samplelight =
            light.sampleIllumination(&RecordSampleLightIlumnation::new(*usamplelight, *hit));
            // std::cout << "      EstimateDirect uLight:" << uLight << " -> Li: " << Li << ", wi: "<< wi << ", pdf: " << lightPdf;
       //      println!("      EstimateDirect uLight {:?} wi {:?} pdf:  {:?}",samplelight.1,  samplelight.3.unwrap().direction, samplelight.2);
        if true {
            //arealight.is_background_area_light()
            let pointinLight = samplelight.4.unwrap();
            let occlusionRay = Ray::new(hit.point, (pointinLight - hit.point).normalize());
            let occ = new_interface_for_intersect_occlusion(&occlusionRay, scene,&pointinLight).unwrap();
            // let occ = scene
            //     .intersect_occlusion(&occlusionRay, &pointinLight)
            //     .unwrap();
            if occ.0 == true {
                return Srgb::new(0.0, 0.0, 0.0);
            }
        }

        let vnext = samplelight.3.unwrap().direction;
        let reflect = hit.normal.dot(-ray.direction) * hit.normal.dot(vnext) > 0.0;
        let mut bsdf_fr;
        let mut bsdf_pdf = 0.0;
            if reflect == true {
            bsdf_fr = hit.material.fr(-ray.direction, vnext);
            let absdot = vnext.dot(hit.normal).abs();
            bsdf_fr = Self::mulScalarSrgb(bsdf_fr, absdot as f32);

            bsdf_pdf = hit.material.pdf((-ray.direction).normalize(), vnext);
        } else {
            return  Srgb::new(0.0, 0.0, 0.0);
        }

        let w = powerHeuristic(samplelight.2, bsdf_pdf);

        let ratio = (w / samplelight.2);
        //  Ld += f * Li * weight / lightPdf;
        let fpartial = Self::mulSrgb(samplelight.1, bsdf_fr);
        let res = Self::mulScalarSrgb(fpartial, ratio as f32);

        let vnext = samplelight.0;
        if false {
            // println!("psample:  {:?}", usamplelight.clone());
            // println!("               wi:  {:?}", vnext);
            // println!("               Li:  {:?}", samplelight.1);
            //
             println!("                 bsdf f:  {:?}", bsdf_fr);
           
             println!("                 powerHeuristic:  {:?}", w);
             println!("                 Ld res,{:?}", res);
         
             println!("                 pdf Light:  {:?}", samplelight.2);
             println!("                 bsdf pdf:  {:?}", bsdf_pdf);
        }

        return res;
    }
    pub fn samplebsdf1(
        ray: &Ray<f64>,
        scene: &Scene1,
        hit: &HitRecord<f64>,
         light: Light,
        usamplebsdf: &(f64, f64),
    ) -> Srgb {

        let recout = hit
            .material
            .sample(RecordSampleIn::from_hitrecord(*hit, ray, *usamplebsdf));
       
        let absdot = recout.next.dot(hit.normal).abs();
        let bsdfsamplef = LigtImportanceSampling::mulScalarSrgb(recout.f, absdot as f32);
       // println!(" BSDF / phase sampling f {:?} {:?}" ,   bsdfsamplef ,  recout.pdf );
        fn isBlack(c: Srgb) -> bool {
            c.red == 0.0 && c.green == 0.0 && c.blue == 0.0
        }
        if !isBlack(bsdfsamplef) && recout.pdf > 0.0 {
            let pdflight = light.pdfIllumination1(hit, scene, recout.next);
           
            let fromsurfacetolight =  new_interface_for_intersect_scene(&recout.newray.unwrap(), scene);
            
            // fromsurfacetolight.unwrap().is_emision.unwrap() esto deberia decir si la luz del parametro light es la misma que la luz que hemos encontrado en la intersection!
            if pdflight > 0.0 && fromsurfacetolight.is_some() && fromsurfacetolight.unwrap().is_emision.unwrap(){
                let Lemited = light.get_emission(Some(*hit), &recout.newray.unwrap());
                let w = powerHeuristic(recout.pdf, pdflight);

                let aux = LigtImportanceSampling::mulSrgb(bsdfsamplef, Lemited);
                return LigtImportanceSampling::mulScalarSrgb(aux, (w / recout.pdf) as f32);
            }
        }
        Srgb::new(0.0, 0.0, 0.0)
    }

   
}

pub fn generateSamplesDeterministic(nsamples: u32) -> Vec<(f64, f64)> {
    let mut vec: Vec<(f64, f64)> = vec![];
    for y in 0..nsamples {
        for x in 0..nsamples {
            let fx = (x as f64) / (nsamples as f64);
            let fy = (y as f64) / (nsamples as f64);
            vec.push((fx, fy))
        }
    }
    vec
}

pub fn generateSamplesWithSamplerGather(
    sampler: &mut Box<dyn Sampler>,
    nsamples: u32,
) -> Vec<((f64, f64), (f64, f64))> {
    let mut vec: Vec<((f64, f64), (f64, f64))> = vec![];
    for sample in 1..nsamples {
        let slight = sampler.get2d();
        let sbsdf = sampler.get2d();
        vec.push((slight, sbsdf))
    }
    vec
}

pub fn generateSamplesWithSamplerGather1(
    sampler: & mut SamplerType,
    nsamples: u32,
) -> Vec<((f64, f64), (f64, f64))> {
    let mut vec: Vec<((f64, f64), (f64, f64))> = vec![];
    for sample in 1..nsamples {
        let slight = sampler.get2d();
        let sbsdf = sampler.get2d();
        vec.push((slight, sbsdf))
    }
    vec
}

// mock samples for testing and debug proposites
pub fn generateSamplesRandomVector(nsamples: u32) -> Vec<(f64, f64)> {
    let mut rng = rand::thread_rng();
    let mut vec: Vec<(f64, f64)> = vec![];
    for y in 0..nsamples {
        for x in 0..nsamples {
            vec.push((rng.gen::<f64>(), rng.gen::<f64>()))
        }
    }
    vec
}
pub fn generateSamplesTexels(nsamples: u32) -> Vec<Srgb> {
    let mut rng = rand::thread_rng();
    let mut vec: Vec<Srgb> = vec![];
    for y in 0..nsamples {
        let d = y as f32 / (nsamples) as f32;

        vec.push(Srgb::new(1.0, 1.0, 1.0));
    }
    vec
}

#[test]
fn testing_area_light_background() {
    let vectexels = generateSamplesTexels(8 * 8);

    let bcklight =
        Light::BackgroundAreaLightType(BackgroundAreaLight::new(vectexels, 8_usize, 8_usize));

    fn addSrgb(a: Srgb, b: Srgb) -> Srgb {
        Srgb::new(a.red + b.red, a.green + b.green, a.blue + b.blue)
    }

    fn mulSrgb(a: Srgb, b: Srgb) -> Srgb {
        Srgb::new(a.red * b.red, a.green * b.green, a.blue * b.blue)
    }
    fn mulScalarSrgb(a: Srgb, b: f32) -> Srgb {
        Srgb::new(a.red * b, a.green * b, a.blue * b)
    }

    let camera = Camera::new(
        Point3::new(0.0, 0.0, 1.5),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        1.0f64 / 1.0f64,
    );

    let arealightdisk = Light::AreaLightDiskType(AreaLightDisk::new(
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        1.0,
        Srgb::new(0.5, 0.5, 0.5),
    ));
    let disk = Disk::new(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        0.0,
        100.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );
    let diskptr = &(Box::new(disk) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);

    let plane2 = Plane::new(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );
    let planeptr2 = &(Box::new(plane2) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);
    let primitivesIntersections: Vec<&Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> =
        vec![diskptr];

    let primitives = vec![];
    let scene: Scene<f64> = Scene::make_scene1(
        128,
        128,
        vec![],
        primitives,
        primitivesIntersections,
        1,
        1,
        None,
    );

    let ray = Ray::new(
        Point3::new(0.001, 1.0, 1.0),
        Vector3::new(0.0, -1.0, 0.0).normalize(),
    );

    if let Some(hit) = interset_scene(&ray, &scene) {
        let mut newLd: Vec<f32> = vec![0.0; 3];
        let vecsamples = generateSamplesDeterministic(32);
        for isample in vecsamples.into_iter() {
            let p = isample;
            //  let p = (0.4375, 0.03125);

            let resnewe =
                LigtImportanceSampling::samplelight(&ray, &scene, &hit, bcklight.clone(), &p);
            updateL(&mut newLd, resnewe);
            if false {
                //    let resnewe  = LigtImportanceSampling::samplebsdf(&ray, &scene, &hit,arealightdisk.clone(),&psamplebsdf);
                //    updateL(&mut newLd,resnewe);
            }
        }
        println!(
            "       RESULTADO FINALLLL!          Ld res,{:?}",
            newLd[0] / ((32 * 32) as f32)
        );
        println!("");
    }
}
 
#[test]
fn testing_area_light_new_interface() {
    let mut sampleruniform :  Box<dyn Sampler> =  Box::new(SamplerUniform::new(((0, 0), (1 as u32,1 as u32)),32, false, Some(0)));
   
   








        
    if false{
        
   


        let mut sampler  = SamplerType::UniformType( SamplerUniform::new(((0, 0), (8 as u32, 8 as u32)),8,false,Some(0)) );
        // let arealightsphere = Light::AreaLightSphere(AreaLightSphere::new(Vector3::new(0.0,3.0,0.0),1.0, Srgb::new(1.0,1.0,1.0)));
 
    // let  c = Cylinder::new(Vector3f::new(0.0, 0.0,  3.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType);
 //   New_Interface_AreaLight ::new(  Srgb::new(1.0,1.0,1.0),PrimitiveType::CylinderType(   Cylinder::new(Vector3::new( 0.0000,4.0,0.0000),Matrix3::from_angle_x(Deg( 0.0)),-1.0, 1.0, 1.0,MaterialDescType::NoneType)));
        let scene  = Scene1::make_scene(
            8,8 ,
            
            vec![
                 Light::New_Interface_AreaLightType(
                    New_Interface_AreaLight ::new(  Srgb::new(1.0,1.0,1.0),
                    PrimitiveType::CylinderType(   Cylinder::new(Vector3::new( -1.0000,8.0,-2.0000),Matrix3::from_angle_x(Deg( 0.0)),-1.0, 1.0, 1.0,MaterialDescType::NoneType)),1)
  
                    //New_Interface_AreaLight ::new(  PrimitiveType::CylinderType(   Cylinder::new(Vector3::new( 0.0000,4.0,0.0000),Matrix3::from_angle_x(Deg( 0.0)),-1.0, 1.0, 1.0,MaterialDescType::NoneType)))
             //  New_Interface_AreaLight ::new( Srgb::new(1.0,1.0,1.0), PrimitiveType::SphereType( Sphere::new(Vector3::new( 0.0000,5.0,0.0000),2.0,   MaterialDescType::NoneType)))),
           //   New_Interface_AreaLight ::new( Srgb::new(1.0,1.0,1.0), PrimitiveType::DiskType(   Disk::new(Vector3::new( 0.0000,4.0,0.0000),Vector3::new(0.0, -1.0, 0.0),0.0, 2.0, MaterialDescType::NoneType))))
              
                // Light::New_Interface_AreaLightType(New_Interface_AreaLight ::new(Srgb::new(1.0,1.0,1.0),PrimitiveType::SphereType( Disk::new(Vector3::new( 0.0000,1.0,0.0000),Vector3::new(0.0, -1.0, 0.0),0.0, 100.0, MaterialDescType::NoneType)))))
                //  Light::PointLight(PointLight {iu: 4.0,positionws: Point3::new(0.00004,2.20,0.0),color: Srgb::new(35.190,35.190, 35.190),}
            )
                ], 
            vec![
                // PrimitiveType::SphereType(Spheref64::new(Vector3f::new(-0.5,0.0,2.0), 0.40, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,0.0)))),
                //    PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.5,0.0,2.0), 0.40, MaterialDescType::PlasticType(Plastic::from_albedo(0.0,1.0,1.0)))),
                    PrimitiveType::DiskType( Disk::new(Vector3::new( 0.00001, 0.0000,0.00001),Vector3::new(0.0, 1.0, 0.0),0.0, 100.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00))))
                ], 
            1,1);

            let nsamples =1;
            let ray = Ray::new(
                Point3::new(0.001, 0.10, 0.001),
                Vector3::new(0.0, -1.0, 0.0).normalize(),
            );
            if let Some(hit) = new_interface_for_intersect_scene(&ray, &scene) {
                let mut newLd: Vec<f32> = vec![0.0; 3];
                let vecsamples = generateSamplesDeterministic(nsamples);
                
                for isample in 0..1024 {
                    // let p = isample;  
                  
                    sampleruniform.get1d();
                    let psample  = sampleruniform.get2d();
                    println!("psample {:?}", psample);
                    // let p =  psample;
                    let p = psample;
                  
        // new_interface_estimatelights(&scene, hit, r, &mut sampler)
       // mañana tengo que comprobar distintos samples...distintas distancias a la esferea. hacer el sampleo para el disco y despues el cindro!
                    let mut accpartial:Srgb=Srgb::new(0.,0.,0.);

                    let resnewe = LigtImportanceSampling::samplelight1(&ray, &scene, &hit, scene.lights[0].clone(), &p);
                        updateL(&mut newLd, resnewe); 
                        accpartial = LigtImportanceSampling::addSrgb(accpartial, resnewe);
                    let isbsdfsampleable =   scene.lights[0].is_arealight() ||   scene.lights[0].is_background_area_light() ;

                    if  isbsdfsampleable{
                        let resnewe  = LigtImportanceSampling::samplebsdf1(&ray, &scene, &hit,scene.lights[0].clone(),&p);
                        if !isBlack(resnewe){
                            updateL(&mut newLd,resnewe);
                            accpartial = LigtImportanceSampling::addSrgb(accpartial, resnewe);
                            // println!("bsdf sample : {:?} res {:?}",p, resnewe.red);
                        } 
                   } 
                   println!(" {}   accpartial   {} ",isample,  accpartial .red);
                
                }
                   
                println!("       Ld {}", newLd[0]);
            
        
        
            }
        
    }
    if false {
        
        let mut sampler  = SamplerType::UniformType( SamplerUniform::new(((0, 0), (8 as u32, 8 as u32)),8,false,Some(0)) );
        // let arealightsphere = Light::AreaLightSphere(AreaLightSphere::new(Vector3::new(0.0,3.0,0.0),1.0, Srgb::new(1.0,1.0,1.0)));
    // let  c = Cylinder::new(Vector3f::new(0.0, 0.0,  3.0), mat3rot,-1.0, 1.0,1.0, materials::MaterialDescType::NoneType);
 //   New_Interface_AreaLight ::new(  Srgb::new(1.0,1.0,1.0),PrimitiveType::CylinderType(   Cylinder::new(Vector3::new( 0.0000,4.0,0.0000),Matrix3::from_angle_x(Deg( 0.0)),-1.0, 1.0, 1.0,MaterialDescType::NoneType)));
        let scene  = Scene1::make_scene(
            8,8 ,
            
            vec![
           //     Light::New_Interface_AreaLightType ( New_Interface_AreaLight ::new( Srgb::new(1.0,1.0,1.0), PrimitiveType::SphereType( Sphere::new(Vector3::new( 2.0000,4.0,2.0000),1.0,   MaterialDescType::NoneType))))
               //  Light::New_Interface_AreaLightType(New_Interface_AreaLight ::new(  Srgb::new(1.0,1.0,1.0),PrimitiveType::CylinderType(   Cylinder::new(Vector3::new(  0., 0.0, 0.),Matrix3::from_angle_x(Deg( 0.0)),-1.0, 1.0, 1.0,MaterialDescType::NoneType)))  ),
  
           
               Light::New_Interface_AreaLightType (  
                        New_Interface_AreaLight ::new( Srgb::new(1.0,1.0,1.0), 
                        PrimitiveType::DiskType(   Disk::new(Vector3::new( 0.0000,4.0,0.0000),Vector3::new(0.0, -1.0, 0.0),0.0, 1.0, MaterialDescType::NoneType)),1))
              
                // Light::New_Interface_AreaLightType(New_Interface_AreaLight ::new(Srgb::new(1.0,1.0,1.0),PrimitiveType::SphereType( Disk::new(Vector3::new( 0.0000,1.0,0.0000),Vector3::new(0.0, -1.0, 0.0),0.0, 100.0, MaterialDescType::NoneType)))))
                //  Light::PointLight(PointLight {iu: 4.0,positionws: Point3::new(0.00004,2.20,0.0),color: Srgb::new(35.190,35.190, 35.190),}
          
                ], 
            vec![
                // PrimitiveType::SphereType(Spheref64::new(Vector3f::new(-0.5,0.0,2.0), 0.40, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,0.0)))),
                //    PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.5,0.0,2.0), 0.40, MaterialDescType::PlasticType(Plastic::from_albedo(0.0,1.0,1.0)))),
                   PrimitiveType::DiskType( Disk::new(Vector3::new( 0.00001, 0.0000,0.00001),Vector3::new(0.0, 1.0, 0.0),0.0, 100., MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00))))
                ], 
            1,1);

            let nsamples =1;
            let ray = Ray::new(
                Point3::new(2.001, 0.10, 2.001),
                Vector3::new(0.0, -1.0, 0.0).normalize(),
            );
            if let Some(hit) = new_interface_for_intersect_scene(&ray, &scene) {
                let mut newLd: Vec<f32> = vec![0.0; 3];
                
             
                for isample in 0..1024 {
                    // let p = isample;  
                  
                    sampleruniform.get1d();
                    let psample  = sampleruniform.get2d();
                    println!("psample {:?}", psample);
                  let p =  psample;
                    // let p = psample;
                    
                  
        // new_interface_estimatelights(&scene, hit, r, &mut sampler)
       // mañana tengo que comprobar distintos samples...distintas distancias a la esferea. hacer el sampleo para el disco y despues el cindro!
                    let mut accpartial:Srgb=Srgb::new(0.,0.,0.);

                    let resnewe = LigtImportanceSampling::samplelight1(&ray, &scene, &hit, scene.lights[0].clone(), &p);
                        updateL(&mut newLd, resnewe); 
                        accpartial = LigtImportanceSampling::addSrgb(accpartial, resnewe);
                    let isbsdfsampleable =   scene.lights[0].is_arealight() ||   scene.lights[0].is_background_area_light() ;

                    if  isbsdfsampleable{
                        let resnewe  = LigtImportanceSampling::samplebsdf1(&ray, &scene, &hit,scene.lights[0].clone(),&p);
                        if !isBlack(resnewe){
                            updateL(&mut newLd,resnewe);
                            accpartial = LigtImportanceSampling::addSrgb(accpartial, resnewe);
                            // println!("bsdf sample : {:?} res {:?}",p, resnewe.red);
                        } 
                   } 
                   println!(" {}   accpartial   {} ",isample,  accpartial .red);
                
                }
                   
                println!("       Ld {}", newLd[0]);
            
        
        
            }
        
    }



    // estimate lights 
    if true {
        let mut sampler  = SamplerType::UniformType( SamplerUniform::new(((0, 0), (8 as u32, 8 as u32)),8,false,Some(0)) );
        let scene  = Scene1::make_scene(
            8,8 ,
            
            vec![
                // Light::New_Interface_AreaLightType ( New_Interface_AreaLight::new( Srgb::new(1.0,1.0,1.0),PrimitiveType::SphereType( Sphere::new(Vector3::new( 0.0000,4.0,0.0000),1.0,   MaterialDescType::NoneType)),8)),
                Light::New_Interface_AreaLightType(New_Interface_AreaLight ::new(Srgb::new(4.0,4.0,4.0),PrimitiveType::DiskType(Disk::new(Vector3::new( 0.0000,0.3,2.5000),Vector3::new(0.0, -1.0, 0.0),0.0, 0.5, MaterialDescType::NoneType)),4)),
           
      //         Light::New_Interface_AreaLightType (  New_Interface_AreaLight ::new( Srgb::new(1.0,1.0,1.0),PrimitiveType::DiskType(   Disk::new(Vector3::new( 0.0000,4.0,0.0000),Vector3::new(0.0, -1.0, 0.0),0.0, 2.0, MaterialDescType::NoneType)),8))
              
                // Light::New_Interface_AreaLightType(New_Interface_AreaLight ::new(Srgb::new(1.0,1.0,1.0),PrimitiveType::SphereType( Disk::new(Vector3::new( 0.0000,1.0,0.0000),Vector3::new(0.0, -1.0, 0.0),0.0, 100.0, MaterialDescType::NoneType)))))
                //  Light::PointLight(PointLight {iu: 4.0,positionws: Point3::new(0.00004,2.20,0.0),color: Srgb::new(35.190,35.190, 35.190),}
          
                ], 
            vec![
                // PrimitiveType::SphereType(Spheref64::new(Vector3f::new(-0.5,0.0,2.0), 0.40, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,0.0)))),
                //    PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.5,0.0,2.0), 0.40, MaterialDescType::PlasticType(Plastic::from_albedo(0.0,1.0,1.0)))),
              //     PrimitiveType::DiskType( Disk::new(Vector3::new( 0.00001, 0.0000,0.00001),Vector3::new(0.0, 1.0, 0.0),0.0, 100., MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00))))
             PrimitiveType::SphereType(Spheref64::new(Vector3f::new(0.0,-0.7,2.50), 0.20, MaterialDescType::PlasticType(Plastic::from_albedo(0.0,1.0,1.0)))),
              PrimitiveType::DiskType( Disk::new(Vector3::new( 0.0000,-1.0,0.0000),Vector3::new(0.0, 1.0, 0.0),0.0, 100.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.00,1.00))))
                ], 
            1,1);
           
        let ray = Ray::new(Point3::new(0.001, -0.950,2.5),Vector3f::new(0.,-1.0,0.0 ));
        if let Some(hit) = new_interface_for_intersect_scene(&ray, &scene) {
            if hit.is_emision.unwrap(){
                println!("->{:?}",   hit.emission);
                
            }
            let Ld =  new_interface_estimatelights(&scene, hit, &ray, &mut sampler, None, false);
            
            println!("->{:?}", Ld);
           
        }
    }

}

#[test]
fn testing_area_light() {
    let camera = Camera::new(
        Point3::new(0.0, 0.0, 1.5),
        Point3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        90.0,
        1.0f64 / 1.0f64,
    );

    let arealight = Light::AreaLightType(AreaLight::new(
        Vector3::new(0.0, 1.0, 0.0),
        Vector3::new(0.0, -1.0, 0.0),
        5.0,
        5.0,
    ));
    let plane2 = Plane::new(
        Vector3::new(0.0, 0.0, 0.0),
        Vector3::new(0.0, 1.0, 0.0),
        1.0,
        1.0,
        MaterialDescType::PlasticType(Plastic::from_albedo(1.0, 1.0, 1.0)),
    );
    let planeptr2 = &(Box::new(plane2) as Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>);
    let primitivesIntersections: Vec<&Box<dyn PrimitiveIntersection<Output = HitRecord<f64>>>> =
        vec![planeptr2];

    let primitives = vec![];
    let scene = &Scene::make_scene1(
        128,
        128,
        vec![
            Light::PointLight(PointLight {
                iu: 4.0,
                positionws: Point3::new(1.15, 1.15, 1.15),
                color: Srgb::new(2.20, 2.20, 23.9420),
            }),
            //   arealight
        ],
        primitives,
        primitivesIntersections,
        1,
        1,
        None,
    );
    fn addSrgb(a: Srgb, b: Srgb) -> Srgb {
        Srgb::new(a.red + b.red, a.green + b.green, a.blue + b.blue)
    }

    fn mulSrgb(a: Srgb, b: Srgb) -> Srgb {
        Srgb::new(a.red * b.red, a.green * b.green, a.blue * b.blue)
    }
    fn mulScalarSrgb(a: Srgb, b: f32) -> Srgb {
        Srgb::new(a.red * b, a.green * b, a.blue * b)
    }

    //    falla dos cosas ...
    //    la primera es el FrameShading del plano... con se como se puede hacer la orientacion...
    //    solucion 2. poner la normal tbn usando la esfera que se que va bien

    let dir = (Point3::new(0.0, 0.0, 0.) - Point3::new(0.00, 1.0, 2.0)).normalize();
    let ray = Ray::new(Point3::new(0.0, 1.0, 2.0), dir);
    if let Some(hit) = interset_scene(&ray, &scene) {
        let mut Ld: Vec<f32> = vec![0.0; 3];
        for isample in 0..32 {
            let p = (isample as f64 / 32.0, isample as f64 / 32.0);
            let samplelight =
                arealight.sampleIllumination(&RecordSampleLightIlumnation::new(p, hit));
            let vnext = samplelight.0;
            println!("psample:  {:?}", p);
            println!("               wi:  {:?}", vnext);
            println!("               Li:  {:?}", samplelight.1);
            println!("               pdf Light:  {:?}", samplelight.2);

            //  let pdflight = samplelight.2;
            let bsdf_fr = hit.material.fr(-ray.direction, vnext);
            println!("                 bsdf fr:  {:?}", bsdf_fr);
            vnext.dot(hit.normal).abs();
            let bsdf_pdf = hit.material.pdf(-ray.direction, vnext);
            println!("                 bsdf pdf:  {:?}", bsdf_pdf);
            let w = powerHeuristic(samplelight.2, bsdf_pdf);
            println!("                  powerHeuristic:  {:?}", w);
            let ratio = (w / samplelight.2);
            //  Ld += f * Li * weight / lightPdf;
            let fpartial = mulSrgb(samplelight.1, bsdf_fr);
            let res = mulScalarSrgb(fpartial, ratio as f32);

            println!("                 Ld res,{:?}", res);

            updateL(&mut Ld, res);
        }
    }
}
