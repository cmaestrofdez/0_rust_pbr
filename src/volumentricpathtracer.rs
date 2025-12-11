 

use std::{fs::File, io::Read};



use once_cell::sync::Lazy;
// JUSTIFICACION
// la idea que tengo es en el main comprobar que hay una volumetric path  y si es asi llenar un array statico
// para tener en el grid medium. El objetivo es poder dejar en paz la enumeracion y no cambiar 
// el trato de Copy para no tener que refactorizar miles de lineas de codigo...literalmente ...miles
// esto se  sigue  de https://dev.to/rustyoctopus/generating-static-arrays-during-compile-time-in-rust-10d8
use palette::Srgb;





pub mod volumetric{

    use cgmath::Decomposed;
    use cgmath::InnerSpace;
    use cgmath::MetricSpace;
    use cgmath::Point3;
    use cgmath::Quaternion;
    use cgmath::Transform;
    use itertools::iproduct;
    use num::complex::ComplexFloat;
  
    use once_cell::sync::Lazy;
    use palette::LinSrgb;
    use palette::Srgb;
 
    use cgmath::Vector3;


    use crate::Cameras::PerspectiveCamera;
    use crate::Cameras::RecordSampleImportanceIn;
    use crate::Config;
    use crate::Lights::Light;
    use crate::Lights::IsAmbientLight;
    use crate::Lights::LigtImportanceSampling;
    use crate::Lights::New_Interface_AreaLight;
    use crate::Lights::NumSamples;
    use crate::Lights::PointLight;
    use crate::Lights::generateSamplesWithSamplerGather1;
    use crate::Lum;
    use crate::Matrix4f;
    use crate::Point2f;
    use crate::Point2i;
    use crate::Point3f;
    use crate::Point3i;
    use crate::Spheref64;
    use crate::Vector3f;
    use crate::Vector3i;
    use crate::integrator::isBlack;
    use crate::integrator::new_interface_estimatelights;
    use crate::primitives::Bounds3f;
    use crate::primitives::PrimitiveType;
    use crate::primitives::prims::HitRecord;
    use crate::primitives::prims::Ray;
    use crate::primitives::prims::coord_system;
    use crate::raytracerv2::lerp;
    use crate::raytracerv2::lerp_color;
    use crate::sampler::Sampler;
    use crate::sampler::SamplerType;
    use crate::sampler::SamplerUniform;
    use crate::scene::Scene1;
    use crate::threapoolexamples1::BBds3i64;
    use crate::threapoolexamples1::new_interface_for_intersect_occlusion;
    use crate::threapoolexamples1::new_interface_for_intersect_scene;
    use crate::materials;
    use crate::materials::BsdfType;
  
    use crate::materials::Eval;
    use crate::materials::Fr;
    use crate::materials::Fresnel;
    use crate::materials::FresnelSchlickApproximation;
    use crate::materials::IsSpecular;
    use crate::materials::MaterialDescType;
    use crate::materials::Metal;
    use crate::materials::Mirror;
    use crate::materials::Plastic;
    use crate::materials::RecordSampleIn;
    use crate::materials::RecordSampleOut;
    use crate::materials::SampleIllumination;
    use crate::threapoolexamples1::new_interface_for_intersect_scene_for_medium;
    use crate::threapoolexamples1::new_interface_for_intersect_scene_surfaces;
 
    use std::cell::Cell;
    use std::cell::RefCell;
    use std::f64::consts::PI;
    use std::io::BufReader;
    use std::ops::AddAssign;
 
 
    use std::fs::File;
    use std::io::prelude::*;

         pub const n_width_voxels:i64=32;
         pub const n_height_voxels:i64=32;
         pub const n_depth_voxels:i64=32;
         pub const n_total_voxels:i64=n_depth_voxels*n_height_voxels*n_width_voxels;

   
pub
const  functionPtr :fn(i64,i64,i64, (i64, i64,i64) )->usize= |x, y, z, dims|{((dims.2 * dims.1 * z)+(dims.0 * y)+x)as usize};
pub
static   STATIC_DATA_ARRAY_DENSITY_GRID  : Lazy<Vec<Colorf64>>=  Lazy::new(||{
    let mut density :Vec<Colorf64>  = vec![Colorf64::new(1.0,1.0,1.0); n_total_voxels as usize]; 
    // let mut file = File::create("foo.txt").unwrap();
    // let mut buf_reader = BufReader::new(file);
    // let mut contents = String::new();
    // buf_reader.read_to_string(&mut contents).unwrap();
    // println!("{}", contents);
    for (x, y, z) in  iproduct!(0..n_width_voxels,0..n_height_voxels, 0..n_depth_voxels){
        let offset =   functionPtr(0,0,0,(32,32,32));
        if   x==2&&y==2&&z==2{
        
            density[offset] = Colorf64::new(0.5, 0.5,0.5);
        } 
        // println!(" {} {} {}", x, y, z);
    }
    println!("Lazy inialization for GRID {},{},{}",n_width_voxels,n_height_voxels, n_depth_voxels );
    density
});
     


    #[derive(Debug, Clone )]
    pub
    struct PhaseFunction{}
    
    
enum RecordSampleType {
    None,
    RecordMediumOut(RecordSampleOutMedium),
    RecordSurfaceOut(RecordSampleOut),
}
 




 

    #[derive(Debug, Clone  , Copy )]
     pub 
    enum MediumType{
        None,
        HomogeneousType(HomogeneousGrid),
        GridType(RGBGrid),
    }


    


    impl MediumType{
        pub fn phase_medium(c : f64, medium_G : f64)->f64{

            let term =   1.0 +  medium_G* medium_G + 2.0 * medium_G  * c;
           ( 1.0 -  medium_G* medium_G) / (term * term.sqrt()) *0.07957747154594766788 //inv4pi
           
        }
        pub fn transmission(&self, scene: &Scene1, sampler: &mut SamplerType, psource:&Point3f, ptarget:&Point3f)->Colorf64{
            match self{
                MediumType::HomogeneousType(grid)=>{grid.transmission(scene, sampler, psource, ptarget)}
                MediumType::GridType(grid)=>{grid.transmission(scene, sampler, psource, ptarget)}
                _=>panic!("nop for other  transmision type  "),
            }
        }
        pub fn get_g_asymetrical_parameter(&self)->f64{
            match self{
                MediumType::HomogeneousType(grid)=>{grid.g},
                MediumType::GridType(grid)=>{grid.g},
                _=>panic!("")
            }
        }
        pub fn sample(&self, record :  RecordSampleInMedium )->RecordSampleOutMedium {
            match self{
                MediumType::HomogeneousType(grid)=>{grid.sample(record) },
                MediumType::GridType(grid)=>{grid.sample(record) },
                _=>panic!("")
            }
        }
    }
    pub type MediumOutsideInside = (MediumType,MediumType);

    
 
  
    
pub
    type Colorf64 = LinSrgb<f64>;
    
 
    
    #[derive(Debug, Clone )]
    pub struct RecordSampleOutMedium {
        L :Colorf64,
       pub  nextray:Ray<f64>,
        prev : Vector3f,
      pub  pinteraction : Point3f,
        pdfinteraction : f64, 
        medium_G : f64,
        pub    sampleinmedium : bool,
        phase_f :Colorf64,
        medium:MediumType,

    }
    impl RecordSampleOutMedium {
        pub fn new( L :Colorf64,
            nextray:Ray<f64>,
            prev : Vector3f,
            pinteraction : Point3f,
            pdfinteraction : f64, 
            
            medium_G : f64,
            sampleinmedium :bool,
            
            )->Self{
            RecordSampleOutMedium { 
                L , 
                nextray  ,
                prev,
                pdfinteraction ,
                pinteraction ,
                medium_G,
               sampleinmedium ,
               phase_f:Colorf64::default(),
               medium:MediumType::None,
               
               
            }
        }
        pub fn new_medium( L :Colorf64,
            nextray:Ray<f64>,
            prev : Vector3f,
            pinteraction : Point3f,
            pdfinteraction : f64, 
            
            medium_G : f64,
            sampleinmedium :bool,
            medium : MediumType,
            
            )->Self{
            RecordSampleOutMedium { 
                L , 
                nextray  ,
                prev,
                pdfinteraction ,
                pinteraction ,
                medium_G,
               sampleinmedium ,
               phase_f:Colorf64::default(),
               medium
               
               
            }
        }
       
        pub fn sample_phase(&mut self , u :&(f64, f64)){
            
            let mut ctheta = 0.0;
            if self.medium_G.abs()<1e-3{
                ctheta =  1.0 - 2.0 * u.0 ;
            }else{
               
             let r =    (1.0-self.medium_G  *self.medium_G) /(1.0 + self.medium_G -2.0 * self.medium_G*u.0);
             ctheta =   -(1.0 + self.medium_G*self.medium_G - r *r )/(2.0*self.medium_G);
            }
            let stetha = (1.0-ctheta*ctheta).max(0.0).sqrt();
            let phi = 2.0 * PI * u.1;
            let uv = coord_system(self.prev);
            
            let sc = phi.sin_cos();
            (sc.1 * stetha)*uv.0;
            (sc.0 * stetha)*uv.1;
            sc.1 * self.prev;

          let next =  (sc.1 * stetha)*uv.0 + (sc.0 * stetha)*uv.1 + ctheta * self.prev;

        //   MediumType::phase_medium(c, medium_G); 
          let term =   1.0 + self.medium_G*self.medium_G + 2.0 *self.medium_G  * sc.1;
          let num =  ( 1.0- self.medium_G*self.medium_G) *0.07957747154594766788; //inv4pi
         let fr =  num /  ( term*term.sqrt());
 

    
         self.nextray= Ray ::new_with_state_medium(self.pinteraction, next ,self.medium);
        }    
    }
    impl Default for RecordSampleOutMedium{
        fn default() -> Self {
        
           
            RecordSampleOutMedium { 
                L: LinSrgb::<f64>::new(0.,0.0,0.0), 
                nextray: Ray::<f64>::new(Point3f::new(0.0, 0.0, 0.0), Vector3f::new(0.0, 0.0, 0.0)) ,
                prev : Vector3f::new(0.0,0.0,0.0),
                pdfinteraction:0.0,
                pinteraction:Point3f::new(0.0,0.0,0.0),
                medium_G:0.0,
                sampleinmedium:false,
    
                phase_f:Colorf64::default(),
                medium:MediumType::None,
               
            }
        }

    }

    
   
    pub struct RecordSampleInMedium <'a>{
        hit : HitRecord<f64>,
        r : Ray<f64>,
         sampler:(f64, f64),
         sam : &'a mut SamplerType,
    }
    impl <'a>RecordSampleInMedium <'a>{
        pub fn from_hit(  r : Ray<f64>, hit : HitRecord<f64>,sampler:(f64, f64), sam :&'a mut  SamplerType, )->Self{

            RecordSampleInMedium{hit, r ,sampler, sam :sam }
        }
    }


    
    #[derive(Debug, Clone , Copy)]
    pub
    struct HomogeneousGrid {
        pub coef_absorption : Colorf64,
        pub coef_transmission : Colorf64,
        pub coef_scattering : Colorf64, 
        pub g :f64,
    }
    impl HomogeneousGrid{
        pub fn new(sigma_absorption:Colorf64, sigma_scattering : Colorf64, g_isotropic:f64)->Self{
            HomogeneousGrid{
                coef_absorption : sigma_absorption,
               coef_transmission:sigma_scattering  +sigma_absorption, 
               coef_scattering : sigma_scattering,
              g : g_isotropic
            }
        }
        pub fn transmission(&self, scene: &Scene1, sampler: &mut SamplerType,psource:&Point3f,ptarget:&Point3f)->Colorf64{
            // new_interface_for_intersect_occlusion
            

            let directionToPtarget = ptarget-psource;
            let mut r =  Ray::new (*psource, directionToPtarget);
            let mut tr = Colorf64::new(1.0, 1.0,1.0);
            while true {
                let mut hitop = new_interface_for_intersect_scene_for_medium (&mut r , scene); 
                if hitop.is_none() {break};
                
                let hit = hitop.unwrap();
                
                if hit.is_surface_interaction()  {
                    match hit.material {
                        BsdfType::None =>{ }
                        _=>{return  Colorf64::new(0.0, 0.0,0.0);}
                    }
                }
             
                let thit = hit.t; 
                // el rayo sobrepasa la distancia entre p0,ptarget cuando el parametro es mayor que 1. la direccion no esta normalizedo
                // esto ocurre cuando el punto de luz esta dentro del medio.
                // thit=1-0.0001 epsilon
                // 0.999899983
                if thit>0.999899983{
                    let tmax = (r.direction.magnitude() ).min(f64::MAX);
                    let a  = (self.coef_transmission*-tmax); 
                    tr*=   Colorf64::new  ( a.red.exp(),a.green.exp(),a.blue.exp());
                   return  tr;
                }
                let tmax = (r.direction.magnitude() * thit).min(f64::MAX);
                let a  = (self.coef_transmission*-tmax);
                
                tr*=   Colorf64::new  ( a.red.exp(),a.green.exp(),a.blue.exp());
                r = Ray::new(hit.point, directionToPtarget);
                
            }
            tr
        }

        /**
         * //     let directionToPtarget = ptarget-psource;
        //   let mut r =  Ray::new_with_medium_active (*psource, directionToPtarget);
        //    let mut tr = Colorf64::new(1.0, 1.0,1.0);
        //   while true {
        //     // tengo que meter aqui si esta en un medio el beam,edicir en prim.oculssion ver como sacar esa informacion
        //     let (ishit, thit, phit, medium) =  new_interface_for_intersect_occlusion(&r, scene,ptarget).unwrap();
           
        //     let tmax = (r.direction.magnitude() * thit).min(f64::MAX);
        //    tr*= Colorf64::new( ( -self.st[0]*tmax).exp(),  (-self.st[1]*tmax).exp(),  (-self.st[2]*tmax).exp()) ;
        //    if !ishit {break};
        //    r = Ray::new(phit, directionToPtarget);
          
        //   }
        //   tr
         */
        pub fn sample(&self, record :  RecordSampleInMedium )->RecordSampleOutMedium { 
            self.sample_homogeneous( record.r,     record.hit.t   ,& record.sampler , &record.hit.point ) 
        }
        fn select_channel(&self,usample:f64, c : Colorf64)->f64{
            let ith = (usample*3.0).clamp(0.0, 2.0) as usize;
            if ith == 1 {return c.red;} else if ith==2 {return  c.green;}else {return  c.blue;}
        }

        pub
        fn sample_homogeneous(&self, r:Ray<f64>,tmax:f64,  u:&(f64, f64) , phit : &Point3f )->RecordSampleOutMedium{
           let ithch = (u.0*3.0).clamp(0.0, 2.0) as usize;
           let st_ith   =  self.select_channel(u.0, self.coef_transmission);
      
             let sampleddistance = -(( 1.0 - u.1)).ln() / st_ith;
            let d =   (sampleddistance / r.direction.magnitude()).min(tmax);
            let mut  a  = self.coef_transmission*(-d)* r.direction.magnitude();
      
            let transmision =Colorf64::new  ( a.red.exp(),a.green.exp(),a.blue.exp());
           

   
         
            if sampleddistance < tmax {   
               let  tr =       transmision * self.coef_transmission;
                let tr =  Vector3f::new(tr.red, tr.green, tr.blue);
                let  mut rec = RecordSampleOutMedium::default();
           //    println!("{:?}", tr);
                let mut pdf = Vector3f::new(1.0,1.0,1.0).dot(tr) / 3.0;
              //  println!("{:?}", pdf);
                if pdf == 0_f64 {
    
                    pdf = 1_f64;
                }
                let density  =  transmision * self.coef_scattering  / pdf;
            
            //        println!("{:?}", transmision);
             //       println!("{:?}", density);
                    RecordSampleOutMedium::new_medium(

                        density,
                        Ray::<f64>::new(Point3f::new(0.0, 0.0, 0.0), Vector3f::new(0.0, 0.0, 0.0)),
                       -r.direction,
                        r.at(d), 
                        pdf,
                        
                        // Some(Box::new(self)),
                        self.g,
                        true,
                        MediumType::HomogeneousType(*self)

                    )

               }else{
                let tr =  Vector3f::new(transmision.red, transmision.green, transmision.blue);
                let mut pdf = Vector3f::new(1.0,1.0,1.0).dot(tr) / 3.0;
                if pdf == 0_f64 { 
                    pdf = 1_f64;
                }
                let density  =  transmision   / pdf;
                    RecordSampleOutMedium::new(
                        density,
                        Ray::<f64>::new(Point3f::new(0.0, 0.0, 0.0), Vector3f::new(0.0, 0.0, 0.0)),
                        -r.direction,
                        *phit, 
                        pdf,
                        self.g,
                        false,
                        
                        
                    )
               }
          
           
        }

    }


 


    #[derive(Debug, Clone , Copy )]
    pub
    struct RGBGrid {
        pub coef_absorption : Colorf64,
        pub coef_transmission : Colorf64,
        pub coef_scattering : Colorf64, 
        pub g :f64,
        // pub n_width_voxels:i64, 
        // pub n_height_voxels:i64, 
        // pub n_depth_voxels:i64, 
       
        pub world: Decomposed<Vector3<f64>,Quaternion<f64>>,
        pub local: Decomposed<Vector3<f64>,Quaternion<f64>>,
        pub invDensity : Colorf64,
        
        
        pub bounds : BBds3i64,
        pub  map_voxel_linear_fn_ptr: fn(i64,i64,i64,(i64, i64, i64))->usize,
        
        
    }
    impl RGBGrid {
        pub fn new(  coef_absorption : Colorf64,
          
             coef_scattering : Colorf64, 
             g :f64,
 
            
             world: Decomposed<Vector3<f64>,Quaternion<f64>>,
             map:Option<fn(i64,i64,i64, (i64, i64, i64))->usize> ,
         )->RGBGrid{
            let coef_transmission = coef_absorption + coef_scattering;
            let bounds = BBds3i64::new(Point3i::new(0, 0, 0), Point3i::new(n_width_voxels, n_height_voxels, n_depth_voxels));
           let c  = STATIC_DATA_ARRAY_DENSITY_GRID.iter().fold(Colorf64::default(), |acc, b|{
           
            Colorf64::new(f64::max( acc.red, b.red) ,f64::max( acc.green, b.green), f64::max( acc.blue, b.blue)) 
        });
            let map_ptr_fn = match  map { None=>{
               let map_std :fn(i64,i64,i64, (i64, i64, i64))->usize =  |x, y, z, dims|{((dims.2 * dims.1 * z)+(dims.0 * y)+x)as usize};
               map_std
            }, 
            _=>{map.unwrap()}
            };
             let invDensity  =  Colorf64::new(1.0_f64/c.red ,  1.0_f64/c.green ,  1.0_f64/c.blue) ;
             RGBGrid{
                bounds,
                coef_absorption,
                coef_scattering,
                coef_transmission,
                g,
               
                world,
                local:world.inverse_transform().unwrap(),
                invDensity,
                map_voxel_linear_fn_ptr: functionPtr

             }
 
        }
          
        fn checkBoundsAgaintsNormalizedGrid(&self,rinmedium: &Ray<f64>)->(bool, f64, f64){
            let boundsNorm = Bounds3f::new(Point3f::new(0., 0., 0.), Point3f::new(1., 1., 1.));
            let rec =  boundsNorm.intersect(rinmedium, 0.0, f64::MAX);
            rec
        }
       
        fn load(&self,p:&Point2i)->Colorf64{  todo!()}

        pub fn sample(&self, record :  RecordSampleInMedium )->RecordSampleOutMedium { 
 
                let rlocal = Ray::new(self.trafo_to_local(record.r.origin, ),self.trafo_vec_to_local(record.r.direction));
                let (ishit, tmin, tmax) = self.checkBoundsAgaintsNormalizedGrid(&rlocal);
                if ishit {return RecordSampleOutMedium ::default();}
                self.tracking_grid(&rlocal , &record.r , tmin, tmax, record.sam);
                RecordSampleOutMedium ::default()
        }
        fn sample_inner(&self, r:  Ray<f64> ,   sam : & mut SamplerType,)->RecordSampleOutMedium { 
 
            let rlocal = Ray::new(self.trafo_to_local(r.origin, ),self.trafo_vec_to_local(r.direction));
            let (ishit, tmin, tmax) = self.checkBoundsAgaintsNormalizedGrid(&rlocal);
            if !ishit {return RecordSampleOutMedium ::default();}
            self.tracking_grid(&rlocal , &r,tmin, tmax, sam);
            RecordSampleOutMedium ::default()
        }
        pub fn transmission(&self, scene: &Scene1, sampler: &mut SamplerType, psource:&Point3f, ptarget:&Point3f)->Colorf64{
            self.transmission__inner(sampler, psource, ptarget)
        }
        fn transmission__inner(&self, sampler: &mut SamplerType,psource:&Point3f,ptarget:&Point3f)->Colorf64{
            let directionToPtarget = ptarget-psource;
            let lengthd = directionToPtarget.magnitude();
            let mut raymedium_innormalspace =  Ray::new (self.trafo_to_local(*psource), self.trafo_vec_to_local( directionToPtarget.normalize()));
           // let rlocal = Ray::new(self.trafo_to_local(record.r.origin, ),self.trafo_vec_to_local(record.r.direction));
            let (ishit, tmin, tmax) = self.checkBoundsAgaintsNormalizedGrid(&raymedium_innormalspace);
            if !ishit {return Colorf64::new(1.0, 1.0, 1.0)}
            let invDmax =  self.invDensity.y();
           // alta densidad (pejem: 100) , baja transmision (sigmat = 0.1) alta prob de tener un evento 
           // t = x / 100 , x =(0,-10), => sera bajo si densidad el alto. el camino sera mas sampleado  
           let ratio = invDmax / self.coef_transmission.y();
            let mut t = tmin;
            let mut Tr = Colorf64::new(1.0,1.0,1.0);
            loop {
           
            let texp = self.sampleExponential(    sampler.get1d())*ratio;
            
                t = t - texp;
               
               if t>tmax {break;}
            

               let d = self.triliarinterpolation(&raymedium_innormalspace, t);
         
               Tr *= 1.0 - ( d.y() * invDmax).max(0.0);
               println!("t {}   d {:?} Tr {}",t,  d.red, Tr.red);
               if (Tr.y()<0.1){
                let q = (1.0-Tr.y()).max(0.05);
                if (sampler.get1d() < q) { return Colorf64::new(0.0, 0.0, 0.0);}; 
                Tr/=( 1.0 - q );
                
               }
            }
            println!("   Tr {}",   Tr.red);
           Tr

        }
        /**
         * 
         * transform point in 0,1 to coords in voxel grid.
         */
        fn to_voxel_difs(&self, p:&Point3f)->(Point3i, Vector3<f64> ){
         let   vxelcoords = Point3f::new( 
            p.x * self.bounds.pmax.x as f64 -  0.5_f64,
            p.y * self.bounds.pmax.y as f64 -  0.5_f64,
            p.z * self.bounds.pmax.z as f64 -  0.5_f64
        );
        let   ivxelcoords = Point3i::new( 
            ( p.x * self.bounds.pmax.x as f64 -  0.5_f64).floor() as i64,
            (p.y * self.bounds.pmax.y as f64 -  0.5_f64).floor() as i64,
            (p.z * self.bounds.pmax.z as f64 -  0.5_f64).floor()  as i64
        );
       
         let   vxelcoords_floor = Point3f::new( 
           ( p.x * self.bounds.pmax.x as f64 -  0.5_f64).floor(),
            (p.y * self.bounds.pmax.y as f64 -  0.5_f64).floor(),
            (p.z * self.bounds.pmax.z as f64 -  0.5_f64).floor()
            );
            // println!("{:?}",p);
            // println!("{:?}",vxelcoords);
            // println!("{:?}",vxelcoords_floor);
            // println!("{:?}",ivxelcoords);
            (ivxelcoords, vxelcoords - vxelcoords_floor  )

        }

        fn load_density(&self, p : &Point3i)->Colorf64{ 
            
            // is outside from normalize box? then 0 return 
            if  !self.bounds.is_inside(p) { 
                // println!("it is not inside" );
                 return  Colorf64::new(0.0, 0.0, 0.0) ;
            }
            //  let offset =( ( self.bounds.pmax.y * p.z + p.y) +  (self.bounds.pmax.x * p.x) )as usize;
             let offset1 = (self.map_voxel_linear_fn_ptr)(p.x, p.y,p.z,(self.bounds.pmax.x, self.bounds.pmax.y, self.bounds.pmax.z));
            //  STATIC_DATA_ARRAY_GRID.len();
            // if offset1 > self.density.len() {panic!("load_density failed!!!")}
            if offset1 > STATIC_DATA_ARRAY_DENSITY_GRID.len() {panic!("load_density failed!!!")}
            
            STATIC_DATA_ARRAY_DENSITY_GRID[offset1]





              
        }
        
        /**
         * 
         * this point is p:&Point3f between [0, 1.]
         */
        fn triliarinterpolation_point(&self, p:&Point3f)->Colorf64{ 
            debug_assert!(p.x<1.0&&p.x>=0.0 &&
                p.y<1.0&&p.y>=0.0 && 
                p.z<1.0&&p.z>=0.0  
            
             );
            let (pi, pdifs) = self.to_voxel_difs(&p);
            
         let a =   lerp_color(pdifs.x, self.load_density(&(pi)), self.load_density(&( pi + Vector3i::new(1,0,0))));
          let b =    lerp_color(pdifs.x, self.load_density(&(pi+ Vector3i::new(0 ,1,0))), self.load_density(&( pi + Vector3i::new(1,1,0))));
          let c =   lerp_color(pdifs.x, self.load_density(&(pi+ Vector3i::new(0,0,1))), self.load_density(&( pi + Vector3i::new(1,0,1))));
          let d =   lerp_color(pdifs.x, self.load_density(&(pi+ Vector3i::new(0,1,1))), self.load_density(&( pi + Vector3i::new(1,1,1))));
          let e = lerp_color(pdifs.y, a, b);
          let f = lerp_color(pdifs.y, c, d);
         lerp_color(pdifs.z, e, f)
         }
        fn triliarinterpolation(&self,r: &Ray<f64>, t:f64)->Colorf64{  
            let p = r.at(t); 
            self.triliarinterpolation_point(&p)
          
        }
        fn create_scattering_event(&self, density : Colorf64, r :&Ray<f64>, pdf : f64,  t: f64)->RecordSampleOutMedium{  
            RecordSampleOutMedium::new(
                density,
                Ray::<f64>::new(Point3f::new(0.0, 0.0, 0.0), Vector3f::new(0.0, 0.0, 0.0)),
                -r.direction,
                r.at(t), 
                pdf,
                self.g,
                true,
                // MediumType::GridType(*self),
                // None
            )
        }
         fn tracking_grid(&self, raymedium_innormalspace : &Ray<f64>, rayworld : &Ray<f64>, tmin : f64, tmax : f64, sampler: &mut SamplerType )-> RecordSampleOutMedium{
            let mut t =tmin;
           let invDmax =  self.invDensity.y();
           // alta densidad (pejem: 100) , baja transmision (sigmat = 0.1) alta prob de tener un evento 
           // t = x / 100 , x =(0,-10), => sera bajo si densidad el alto. el camino sera mas sampleado  
           let ratio = invDmax / self.coef_transmission.y();

            loop {
           
            let texp = self.sampleExponential(    sampler.get1d())*ratio;
            
                t = t - texp;
                println!("t {} ", t);
               if t>tmax {break;}
            

               let d = self.triliarinterpolation(raymedium_innormalspace, t);
               println!(" d {:?} ", d);
               let sample1d = sampler.get1d();
               if d.y() * invDmax > sample1d { 
                println!(" scattering {:?} ", d); 
                 return  self.create_scattering_event( self.coef_scattering / self.coef_transmission ,rayworld, 0.0, t) ;
               };
            }
         
           RecordSampleOutMedium::new(
                Colorf64::new(1.0, 1.0, 1.0),
                Ray::<f64>::new(Point3f::new(0.0, 0.0, 0.0), Vector3f::new(0.0, 0.0, 0.0)),
                Vector3f::new(0., 0., 0.),
                Point3f::new(0., 0., 0.), 
                0.,
                // Some(Box::new(self)),
                self.g,
                false, 
            )
  
            
        }
        pub fn sampleExponential(&self,u:f64)->f64{
           ( 1.0 - u).ln() 
        }
        pub fn trafo_to_local(&self, pworld: Point3f) -> Point3f {
            self.local.transform_point(pworld)
        }
        pub fn trafo_to_world(&self, plocal: Point3f) -> Point3f {
            self.world.transform_point(plocal)
        }
        pub fn trafo_vec_to_local(&self, pworld: Vector3f) ->  Vector3f {
            self.local.transform_vector(pworld)
        }
        pub fn trafo_vec_to_world(&self, plocal:  Vector3f) ->  Vector3f {
            self.world.transform_vector(plocal)
        }

       
    }









#[test]
pub fn test_grid(){

    STATIC_DATA_ARRAY_DENSITY_GRID.len();
    
    // el grid debe superponerse a una esfera que cubre. de modo que no sobresalga es decir la grid debe tener de w = sphere.radius*2, el centro de la grid debe estar en el centro de la espera
    //cuando se hace la transformacion al medio el medio estara entre (0,1)x(0,1)x(0,1)
    let trafo =  Decomposed{
        disp:Vector3f::new(0.0, 0., 0.0),
        rot :Quaternion::from_arc(Vector3::new(0.0,0.0,1.0),Vector3f::new(0.0,0.0,0.0), None),
        scale:5.0
    };
    let nx   = 32_i64;
   let mut density_buffer = vec![ Colorf64::new(100.,100.,100.); (nx*nx*nx) as usize];
   let mapfn_ptr: fn(i64,i64,i64, BBds3i64)->usize=|x, y, z, b|{((b.pmax.y * z +y)+(b.pmax.x * x))as usize};
   let mapfn_ptr_st: fn(i64,i64,i64, BBds3i64)->usize=|x, y, z, b|{((b.pmax.x * b.pmax.y * z)+(b.pmax.x * y)+x)as usize};
   let gridvolumen =  BBds3i64::new(Point3i::new(0, 0, 0), Point3i::new(n_width_voxels, n_height_voxels, n_depth_voxels));
   for (x, y, z) in  iproduct!(gridvolumen.pmin.x..gridvolumen.pmax.x, gridvolumen.pmin.y..gridvolumen.pmax.y, gridvolumen.pmin.z..gridvolumen.pmax.z){
       let offset =   mapfn_ptr_st(x, y, z, gridvolumen);
       if   x==2&&y==2&&z==2{
        // println!("{}, {}, {}, {}", x, y, z ,offset);
      //  density_buffer[offset] = Colorf64::new(0.5, 0.5,0.5);
       } 
   }

 
 let rgbgrid =    RGBGrid::new(
    Colorf64::new(0.05, 0.05, 0.05),  
    Colorf64::new(0.05, 0.05, 0.05),  
    1.0,  
    
    // trafo, Some(mapfn_ptr_st));
     trafo, Some(functionPtr ));
    // rgbgrid.bounds.

    let ray = Ray::new(Point3f::new(0., 0.0, 0.), Vector3f::new(0., 0.0, 1.));
    let mut sampler  = SamplerType::UniformType( SamplerUniform::new(((0, 0), (32 as u32, 32 as u32)),1 as u32,false,Some(0)) );    
    rgbgrid.transmission__inner(  &mut sampler, &Point3f::new(0., 5., 0.0), &Point3f::new(0., 0., 5.0));
 


for (x, y, z) in  iproduct!(gridvolumen.pmin.x..gridvolumen.pmax.x, gridvolumen.pmin.y..gridvolumen.pmax.y, gridvolumen.pmin.z..gridvolumen.pmax.z){
    println!("{}, {}, {} {}", x, y, z, (rgbgrid.map_voxel_linear_fn_ptr)(x, y, z, (gridvolumen.pmax.x,gridvolumen.pmax.y,gridvolumen.pmax.z)));
    let pn = rgbgrid.load_density(&Point3i::new(x, y, z));
    // println!("{:?}", pn);
}
// cambio el metodo de acceso de pbrt y ya esta!
// rgbgrid.load_density(p);




let mediumgrid = MediumType::GridType(RGBGrid::new(
    Colorf64::new(0.05, 0.05, 0.05),  
    Colorf64::new(0.05, 0.05, 0.05),  
    1.0,  
    
    // trafo, Some(mapfn_ptr_st));
     trafo, Some(functionPtr )));
     
}







    


pub fn integrate_mock( pfilm:&(f64, f64), lights: &Vec<Light>,scene: &Scene1 ,depth: i32, sampler: &mut SamplerType)->Srgb{ 
 
    
    let mut r = Ray ::new(Point3f::new(0.,0.,0.), Vector3f::new(0.,0.0,1.) );
    
// mira lo del state a ver si funciona...
   let  mut paththrogh = Colorf64::new(1.0,1.0,1.0);
   let  mut Lpath = Colorf64::new(0.0,0.0,0.0);
    
    let mut stats_num_iterations_total = 0;
    let mut stats_num_iterations_surfaces = 0;
    let mut stats_num_iterations_volumes = 0;
    for mut idepth in 1..110  {

        
        let mut hitop = new_interface_for_intersect_scene_for_medium (&mut r , scene); 
         
        if hitop.is_none(){
            break;
        }
        let mut hit = hitop.unwrap(); 
        let mut   recout :RecordSampleOutMedium =RecordSampleOutMedium ::default();
       if r.medium.is_some(){
        let u = sampler.get2d();
      
        recout =  r.medium.unwrap().sample(RecordSampleInMedium::from_hit(r,hit, u, sampler));
        if !recout.sampleinmedium{// println!("no se samplea");
                                    }
        hit.point =  recout.pinteraction; 
        paththrogh *=recout.L;
         
       }
       if  recout.sampleinmedium{
                    sampler.get1d();sampler.get2d();sampler.get2d();
                    let Ld =  new_interface_estimatelights(scene, hit, &r, sampler,Some(&recout), false);
                    let ldf64 =  Colorf64::new(Ld.red as f64, Ld.green as f64, Ld.blue as f64);
                    Lpath+= ldf64*paththrogh;
                    println!("{} L  {:?}", idepth-1 ,Lpath.red);
                    let u1 = sampler.get2d(); 
                    recout.sample_phase(&u1);  
                    r =recout.nextray;
       }else{
         
         if(!hit.has_material()){
            r =  hit.spawn_ray(r.direction );   
            idepth=idepth-1; 
            continue;
         }
    
         if ! hit.material.is_specular(){
            sampler.get1d();sampler.get2d();sampler.get2d();
//  esto falla en el sample 
             let Ld = new_interface_estimatelights(scene, hit, &r, sampler, None, true);
           
             Lpath+=  Colorf64::new(Ld.red as f64, Ld.green as f64, Ld.blue as f64)*paththrogh;
            println!("{} L  {:?}",idepth-1,  Lpath.red);
         }

         let psample = sampler.get2d();
         let rec = hit.material.sample(RecordSampleIn::from_hitrecord(hit, &r, psample));   
        
         if  rec.checkIfZero()  {break;}
         paththrogh   =  rec.compute_transport_color(paththrogh, &hit); 
       
         r =  hit.spawn_ray(rec.next );  

       }
      
      
         
        
    }
 

    println!("{:?}", Lpath);
     Srgb::default()
}







#[test]
pub fn sampler_volumetrictest() {
   
   let b : Bounds3f = Bounds3f::new(Point3f::new(0.0, 0.0, 0.0), Point3f::new(1.0, 1.0, 3.0));
   let r = Ray::new(Point3f::new(0.0, 0.50, 0.0),Vector3f::new(0.0, 0.0, 2.0));
   let res = b.intersect(&r, 0_f64, 2.0_f64);
   let (_, tmin, tmax) = res;
   let pmin = r.at(tmin); let pmax = r.at(tmax);

  let outin: MediumOutsideInside  =   ( 
    MediumType::None,
    MediumType::HomogeneousType(HomogeneousGrid :: new( 
        Colorf64::new(0.5,0.5,0.5),
        Colorf64::new(0.5,0.5,0.5), 
        0.8)));
    let res = (512,512);
    let mut sampler  = SamplerType::UniformType( SamplerUniform::new(((0, 0), (res.0 as u32, res.1 as u32)),1 as u32,false,Some(0)) );
    let scene = Scene1::make_scene(
        res.0,
        res.1,
        vec![Light::PointLight(PointLight {
            iu: 4.0,
            positionws: Point3::new(0.0000, 6.0, 10.0000),
            color: Srgb::new(115.0,115.0,115.0),
        })],
        vec![
             PrimitiveType::SphereType(Spheref64::new_with_medium(Vector3f::new(0.0, 0., 15.0), 5.0,MaterialDescType::NoneType, outin)),
            PrimitiveType::SphereType(Spheref64::new_with_medium (Vector3f::new(0.0, 0., 15.0), 1.0, MaterialDescType::PlasticType(Plastic::from_albedo(1.0,1.0,1.0)),(outin.1,outin.0))),
          
             
        ],
        1,
        1,
    );
    integrate_mock(&(0.,0.), &scene.lights, &scene,1, &mut sampler);

    
}

}
